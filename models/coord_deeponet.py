from __future__ import annotations

from copy import deepcopy
from typing import Literal
import math

import numpy as np
import torch

from ._coordinate import (
    BaseVariableSampler,
    area_weighted_equal_base_variable_normalized_rmse_grid,
    field_metadata,
    field_norm_stats,
    latitude_weights,
    parse_field_name,
    t21_coordinate_features,
)
from ._common import resolve_torch_device

ActivationName = Literal["silu", "relu", "tanh"]


def _activation(name: ActivationName) -> torch.nn.Module:
    return {"silu": torch.nn.SiLU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}[name]()


def _tensor(x, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.array(x, copy=True), device=device, dtype=dtype)


class _MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_width: int, num_layers: int, activation: ActivationName) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = int(hidden_width)
        for i in range(int(num_layers)):
            layers += [torch.nn.Linear(int(in_dim) if i == 0 else width, width), _activation(activation)]
        layers.append(torch.nn.Linear(width if layers else int(in_dim), int(out_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DeepONet(torch.nn.Module):
    def __init__(
        self,
        branch_in_dim: int,
        trunk_in_dim: int,
        *,
        rank: int,
        branch_hidden_width: int,
        trunk_hidden_width: int,
        branch_num_layers: int,
        trunk_num_layers: int,
        activation: ActivationName,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.branch = _MLP(branch_in_dim, self.rank, branch_hidden_width, branch_num_layers, activation)
        self.trunk = _MLP(trunk_in_dim, self.rank, trunk_hidden_width, trunk_num_layers, activation)
        self.bias = _MLP(trunk_in_dim, 1, trunk_hidden_width, trunk_num_layers, activation)

    def forward(self, branch_x: torch.Tensor, trunk_x: torch.Tensor) -> torch.Tensor:
        return self.bias(trunk_x).squeeze(-1) + torch.sum(self.branch(branch_x) * self.trunk(trunk_x), dim=-1) / math.sqrt(self.rank)


class CoordDeepONet:
    def __init__(
        self,
        *,
        rank: int = 128,
        branch_hidden_width: int = 256,
        trunk_hidden_width: int = 256,
        branch_num_layers: int = 3,
        trunk_num_layers: int = 3,
        activation: ActivationName = "silu",
        batch_size: int = 32768,
        predict_chunk_size: int = 262_144,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "auto",
    ) -> None:
        self.rank = int(rank)
        self.branch_hidden_width = int(branch_hidden_width)
        self.trunk_hidden_width = int(trunk_hidden_width)
        self.branch_num_layers = int(branch_num_layers)
        self.trunk_num_layers = int(trunk_num_layers)
        self.activation = activation
        self.batch_size = int(batch_size)
        self.predict_chunk_size = int(predict_chunk_size)
        self.dtype = dtype
        self.device = resolve_torch_device(device)
        self._net: _DeepONet | None = None
        self.fit_stats_: dict | None = None

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        s: np.ndarray | torch.Tensor,
        Y: np.ndarray | torch.Tensor,
        *,
        field_mask: np.ndarray | torch.Tensor,
        field_names: list[str],
        n_sim_types: int | None = None,
        num_steps: int = 3000,
        lr: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        seed: int = 0,
        val_X: np.ndarray | torch.Tensor | None = None,
        val_s: np.ndarray | torch.Tensor | None = None,
        val_Y: np.ndarray | torch.Tensor | None = None,
        val_field_mask: np.ndarray | torch.Tensor | None = None,
        val_max_points: int | None = None,
        log_every: int = 100,
        early_stop_patience_evals: int | None = None,
        early_stop_min_delta: float = 0.0,
        hard_stop_step: int | None = None,
    ) -> None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self.field_names_ = list(field_names)
        self.field_meta_ = field_metadata(self.field_names_)
        self.n_sim_types_ = int(n_sim_types or (int(torch.as_tensor(s).max().item()) + 1))
        self.spatial_shape_ = tuple(int(x) for x in torch.as_tensor(Y).shape[-2:])

        X_t = _tensor(X, device=self.device, dtype=self.dtype)
        s_t = _tensor(s, device=self.device, dtype=torch.long)
        Y_t = _tensor(Y, device=self.device, dtype=self.dtype)
        fm_t = _tensor(field_mask, device=self.device, dtype=torch.bool)
        self._train_Y_np = np.asarray(Y, dtype=np.float32)
        self._train_field_mask_np = np.asarray(field_mask, dtype=bool)
        self._field_base_idx = torch.as_tensor(self.field_meta_.base_idx, device=self.device, dtype=torch.long)
        self._field_level = torch.as_tensor(self.field_meta_.level, device=self.device, dtype=self.dtype)
        lat_mu, lon_sin, lon_cos, lat_w = t21_coordinate_features(*self.spatial_shape_)
        self._lat_mu = torch.as_tensor(lat_mu, device=self.device, dtype=self.dtype)
        self._lon_sin = torch.as_tensor(lon_sin, device=self.device, dtype=self.dtype)
        self._lon_cos = torch.as_tensor(lon_cos, device=self.device, dtype=self.dtype)
        self._lat_weights = torch.as_tensor(latitude_weights(self.spatial_shape_[0]), device=self.device, dtype=self.dtype)
        self._field_mean, self._field_std = field_norm_stats(Y_t, fm_t, self._lat_weights)

        self._net = _DeepONet(
            X_t.shape[1] + self.n_sim_types_,
            len(self.field_meta_.bases) + 4,
            rank=self.rank,
            branch_hidden_width=self.branch_hidden_width,
            trunk_hidden_width=self.trunk_hidden_width,
            branch_num_layers=self.branch_num_layers,
            trunk_num_layers=self.trunk_num_layers,
            activation=self.activation,
        ).to(device=self.device, dtype=self.dtype)
        opt = torch.optim.AdamW(self._net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        sampler = BaseVariableSampler(fm_t, self.field_meta_, torch.as_tensor(lat_w, device=self.device), device=self.device)
        total_steps = min(int(num_steps), int(hard_stop_step)) if hard_stop_step is not None else int(num_steps)
        if total_steps < 1:
            raise ValueError("num_steps / hard_stop_step must leave at least one training step.")

        val_pack = self._validation_pack(val_X, val_s, val_Y, val_field_mask, val_max_points, int(seed) + 17)
        val_metric_name = "val_eq_base_nrmse" if val_max_points is None else "val_sample_nrmse"
        best_metric, best_state, best_step = float("inf"), None, None
        bad_evals = n_evals = steps_run = 0
        patience = None if early_stop_patience_evals is None else int(early_stop_patience_evals)
        train_mse = float("nan")
        early_stop = False

        for step in range(total_steps):
            n, f, la, lo = sampler.sample(self.batch_size, self.spatial_shape_[1], gen)
            target = (Y_t[n, f, la, lo] - self._field_mean[f]) / self._field_std[f]
            self._net.train()
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((self._net(self._branch_features(X_t, s_t, n), self._trunk_features(f, la, lo)) - target) ** 2)
            loss.backward()
            opt.step()
            train_mse = float(loss.detach().item())
            steps_run = step + 1
            if val_pack is None or (steps_run % int(log_every) and steps_run != total_steps):
                continue
            metric = self._validation_metric(*val_pack)
            n_evals += 1
            if metric < best_metric - float(early_stop_min_delta):
                best_metric, best_step, best_state, bad_evals = metric, steps_run, deepcopy(self._net.state_dict()), 0
            else:
                bad_evals += 1
            print(f"[coord_deeponet] iter {steps_run:>5}/{total_steps} | train_mse={train_mse:.6e} | {val_metric_name}={metric:.6f} | best={best_metric:.6f}")
            if patience is not None and bad_evals >= patience:
                early_stop = True
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        self.fit_stats_ = {
            "steps_requested": int(num_steps),
            "hard_stop_step": None if hard_stop_step is None else int(hard_stop_step),
            "steps_run": int(steps_run),
            "early_stop_triggered": bool(early_stop),
            "best_step": None if best_step is None else int(best_step),
            "best_val_equal_base_normalized_rmse_grid": None if best_step is None or val_max_points is not None else float(best_metric),
            "best_val_sampled_normalized_rmse": None if best_step is None or val_max_points is None else float(best_metric),
            "val_max_points": None if val_max_points is None else int(val_max_points),
            "n_evals": int(n_evals),
            "final_train_pointwise_mse": float(train_mse),
            "rank": int(self.rank),
            "branch_hidden_width": int(self.branch_hidden_width),
            "trunk_hidden_width": int(self.trunk_hidden_width),
            "branch_num_layers": int(self.branch_num_layers),
            "trunk_num_layers": int(self.trunk_num_layers),
            "activation": str(self.activation),
            "batch_size": int(self.batch_size),
            "optimizer": "AdamW",
            "optimizer_lr": float(lr),
            "optimizer_weight_decay": float(weight_decay),
            "target_normalization": "per_field_training_grid_latitude_weighted",
            "coordinate_encoding": "t21_lat_mu_lon_sincos",
            "sampling": "base_variable_first_area_weighted_latitude",
        }

    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor, s: np.ndarray | torch.Tensor, *, chunk_size: int | None = None) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Model not fitted.")
        self._net.eval()
        X_t = _tensor(X, device=self.device, dtype=self.dtype)
        s_t = _tensor(s, device=self.device, dtype=torch.long)
        n, f, h, w = X_t.shape[0], len(self.field_names_), *self.spatial_shape_
        out = np.empty((n, f, h, w), dtype=np.float32)
        size = int(chunk_size or self.predict_chunk_size)
        for start in range(0, out.size, size):
            idx = torch.arange(start, min(start + size, out.size), device=self.device)
            lo = idx % w
            la = (idx // w) % h
            fj = (idx // (h * w)) % f
            ni = idx // (f * h * w)
            pred = self._net(self._branch_features(X_t, s_t, ni), self._trunk_features(fj, la, lo)) * self._field_std[fj] + self._field_mean[fj]
            out.reshape(-1)[start : start + idx.numel()] = pred.detach().cpu().to(torch.float32).numpy()
        return torch.from_numpy(out)

    def _branch_features(self, X: torch.Tensor, s: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return torch.cat([X[n], torch.nn.functional.one_hot(s[n], self.n_sim_types_).to(dtype=self.dtype)], dim=1)

    def _trunk_features(self, f: torch.Tensor, la: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                torch.nn.functional.one_hot(self._field_base_idx[f], len(self.field_meta_.bases)).to(dtype=self.dtype),
                self._field_level[f, None],
                self._lat_mu[la, None],
                self._lon_sin[lo, None],
                self._lon_cos[lo, None],
            ],
            dim=1,
        )

    def _validation_pack(self, X, s, Y, field_mask, max_points: int | None, seed: int):
        if X is None or s is None or Y is None or field_mask is None:
            return None
        X_t = _tensor(X, device=self.device, dtype=self.dtype)
        s_t = _tensor(s, device=self.device, dtype=torch.long)
        if max_points is not None:
            Y_t = _tensor(Y, device=self.device, dtype=self.dtype)
            fm_t = _tensor(field_mask, device=self.device, dtype=torch.bool)
            gen = torch.Generator(device=self.device).manual_seed(int(seed))
            n, f, la, lo = BaseVariableSampler(fm_t, self.field_meta_, self._lat_weights, device=self.device).sample(
                int(max_points),
                self.spatial_shape_[1],
                gen,
            )
            return "sampled", X_t, s_t, Y_t, n, f, la, lo
        return (
            "exact",
            X_t,
            s_t,
            np.asarray(Y, dtype=np.float32),
            np.asarray(field_mask, dtype=bool),
        )

    @torch.no_grad()
    def _validation_metric(self, kind: str, X: torch.Tensor, s: torch.Tensor, Y, *args) -> float:
        if kind == "sampled":
            n, f, la, lo = args
            self._net.eval()
            pred = self._net(self._branch_features(X, s, n), self._trunk_features(f, la, lo))
            target = (Y[n, f, la, lo] - self._field_mean[f]) / self._field_std[f]
            return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())
        field_mask = args[0]
        pred = self.predict(X, s).numpy()
        return area_weighted_equal_base_variable_normalized_rmse_grid(
            pred,
            Y,
            self.field_names_,
            self._train_Y_np,
            self._train_field_mask_np,
            field_mask,
        )
