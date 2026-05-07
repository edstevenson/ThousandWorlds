from __future__ import annotations

from copy import deepcopy
import re
from typing import Literal

import numpy as np
import torch

from thousandworlds.field_spec import CANONICAL_FIELD_VARIABLES
from ._common import resolve_torch_device

ActivationName = Literal["tanh", "silu", "relu"]
_LEVEL_RE = re.compile(r"^(.+?)_(\d+)$")


def _activation(name: ActivationName) -> torch.nn.Module:
    return {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU, "relu": torch.nn.ReLU}[name]()


def parse_field_name(name: str) -> tuple[str, int]:
    match = _LEVEL_RE.match(name)
    return (match.group(1), int(match.group(2))) if match else (name, -1)


def _tensor(x, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.array(x, copy=True), device=device, dtype=dtype)


def _field_metadata(field_names: list[str]) -> tuple[list[str], np.ndarray, np.ndarray]:
    bases = [name for name in CANONICAL_FIELD_VARIABLES if any(parse_field_name(f)[0] == name for f in field_names)]
    base_to_idx = {name: i for i, name in enumerate(bases)}
    parsed = [parse_field_name(name) for name in field_names]
    max_level = max([level for _, level in parsed if level >= 0] or [0])
    base_idx = np.asarray([base_to_idx[base] for base, _ in parsed], dtype=np.int64)
    level = np.asarray([-1.0 if k < 0 else (0.0 if max_level == 0 else 2.0 * k / max_level - 1.0) for _, k in parsed], dtype=np.float32)
    return bases, base_idx, level


class _PointMLP(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_width: int, num_layers: int, activation: ActivationName) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        width = int(hidden_width)
        for i in range(int(num_layers)):
            layers += [torch.nn.Linear(int(in_dim) if i == 0 else width, width), _activation(activation)]
        layers.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CoordMLP:
    def __init__(
        self,
        *,
        hidden_width: int = 256,
        num_layers: int = 6,
        activation: ActivationName = "tanh",
        batch_size: int = 128,
        predict_chunk_size: int = 262_144,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "auto",
    ) -> None:
        self.hidden_width = int(hidden_width)
        self.num_layers = int(num_layers)
        self.activation = activation
        self.batch_size = int(batch_size)
        self.predict_chunk_size = int(predict_chunk_size)
        self.dtype = dtype
        self.device = resolve_torch_device(device)
        self._net: _PointMLP | None = None
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
        lr: float = 1.0e-4,
        weight_decay: float = 0.0,
        seed: int = 0,
        val_X: np.ndarray | torch.Tensor | None = None,
        val_s: np.ndarray | torch.Tensor | None = None,
        val_Y: np.ndarray | torch.Tensor | None = None,
        val_field_mask: np.ndarray | torch.Tensor | None = None,
        val_max_points: int = 200_000,
        log_every: int = 100,
        early_stop_patience_evals: int | None = None,
        early_stop_min_delta: float = 0.0,
        hard_stop_step: int | None = None,
    ) -> None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self.field_names_ = list(field_names)
        self.field_bases_, base_idx, level = _field_metadata(self.field_names_)
        self.n_sim_types_ = int(n_sim_types or (int(torch.as_tensor(s).max().item()) + 1))
        self.spatial_shape_ = tuple(int(x) for x in torch.as_tensor(Y).shape[-2:])

        X_t = _tensor(X, device=self.device, dtype=self.dtype)
        s_t = _tensor(s, device=self.device, dtype=torch.long)
        Y_t = _tensor(Y, device=self.device, dtype=self.dtype)
        fm_t = _tensor(field_mask, device=self.device, dtype=torch.bool)
        self._field_base_idx = torch.as_tensor(base_idx, device=self.device, dtype=torch.long)
        self._field_level = torch.as_tensor(level, device=self.device, dtype=self.dtype)
        self._lat = torch.linspace(-1.0, 1.0, self.spatial_shape_[0], device=self.device, dtype=self.dtype)
        self._lon = torch.linspace(-1.0, 1.0, self.spatial_shape_[1], device=self.device, dtype=self.dtype)
        self._fit_norm_stats(Y_t, fm_t)

        obs = torch.nonzero(fm_t, as_tuple=False)
        self._net = _PointMLP(
            X_t.shape[1] + self.n_sim_types_ + len(self.field_bases_) + 3,
            self.hidden_width,
            self.num_layers,
            self.activation,
        ).to(device=self.device, dtype=self.dtype)
        opt = torch.optim.RMSprop(self._net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        total_steps = min(int(num_steps), int(hard_stop_step)) if hard_stop_step is not None else int(num_steps)
        if total_steps < 1:
            raise ValueError("num_steps / hard_stop_step must leave at least one training step.")
        best_metric, best_state, best_step = float("inf"), None, None
        bad_evals = n_evals = steps_run = 0
        patience = None if early_stop_patience_evals is None else int(early_stop_patience_evals)
        train_mse = float("nan")
        val_pack = self._validation_pack(val_X, val_s, val_Y, val_field_mask, int(val_max_points), int(seed) + 17)
        early_stop = False

        for step in range(total_steps):
            n, f, la, lo = self._sample_points(obs, int(self.batch_size), gen)
            target = (Y_t[n, f, la, lo] - self._field_mean[f]) / self._field_std[f]
            self._net.train()
            opt.zero_grad(set_to_none=True)
            loss = torch.mean((self._net(self._features(X_t, s_t, n, f, la, lo)) - target) ** 2)
            loss.backward()
            opt.step()
            train_mse = float(loss.detach().item())
            steps_run = step + 1
            if val_pack is None or (steps_run % int(log_every) and steps_run != total_steps):
                continue
            metric = self._pointwise_mse(*val_pack)
            n_evals += 1
            if metric < best_metric - float(early_stop_min_delta):
                best_metric, best_step, best_state, bad_evals = metric, steps_run, deepcopy(self._net.state_dict()), 0
            else:
                bad_evals += 1
            print(f"[coord_mlp] iter {steps_run:>5}/{total_steps} | train_mse={train_mse:.6e} | val_mse={metric:.6e} | best={best_metric:.6e}")
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
            "best_val_pointwise_mse": None if best_step is None else float(best_metric),
            "n_evals": int(n_evals),
            "final_train_pointwise_mse": float(train_mse),
            "hidden_width": int(self.hidden_width),
            "num_layers": int(self.num_layers),
            "activation": str(self.activation),
            "batch_size": int(self.batch_size),
            "optimizer": "RMSprop",
            "optimizer_lr": float(lr),
            "optimizer_weight_decay": float(weight_decay),
        }

    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor, s: np.ndarray | torch.Tensor, *, chunk_size: int | None = None) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Model not fitted.")
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
            pred = self._net(self._features(X_t, s_t, ni, fj, la, lo)) * self._field_std[fj] + self._field_mean[fj]
            out.reshape(-1)[start : start + idx.numel()] = pred.detach().cpu().to(torch.float32).numpy()
        return torch.from_numpy(out)

    def _fit_norm_stats(self, Y: torch.Tensor, field_mask: torch.Tensor) -> None:
        means, stds = [], []
        for f in range(Y.shape[1]):
            vals = Y[field_mask[:, f], f].reshape(-1)
            vals = vals[torch.isfinite(vals)]
            means.append(vals.mean() if vals.numel() else Y.new_tensor(0.0))
            std = vals.std(unbiased=False) if vals.numel() else Y.new_tensor(1.0)
            stds.append(torch.where(std > 1.0e-6, std, Y.new_tensor(1.0)))
        self._field_mean = torch.stack(means).to(device=self.device, dtype=self.dtype)
        self._field_std = torch.stack(stds).to(device=self.device, dtype=self.dtype)

    def _sample_points(self, obs: torch.Tensor, batch_size: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = obs[torch.randint(obs.shape[0], (batch_size,), generator=gen, device=self.device)]
        return (
            pair[:, 0],
            pair[:, 1],
            torch.randint(self.spatial_shape_[0], (batch_size,), generator=gen, device=self.device),
            torch.randint(self.spatial_shape_[1], (batch_size,), generator=gen, device=self.device),
        )

    def _features(self, X: torch.Tensor, s: torch.Tensor, n: torch.Tensor, f: torch.Tensor, la: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                X[n],
                torch.nn.functional.one_hot(s[n], self.n_sim_types_).to(dtype=self.dtype),
                torch.nn.functional.one_hot(self._field_base_idx[f], len(self.field_bases_)).to(dtype=self.dtype),
                self._field_level[f, None],
                self._lat[la, None],
                self._lon[lo, None],
            ],
            dim=1,
        )

    def _validation_pack(self, X, s, Y, field_mask, max_points: int, seed: int):
        if X is None or s is None or Y is None or field_mask is None:
            return None
        X_t = _tensor(X, device=self.device, dtype=self.dtype)
        s_t = _tensor(s, device=self.device, dtype=torch.long)
        Y_t = _tensor(Y, device=self.device, dtype=self.dtype)
        fm_t = _tensor(field_mask, device=self.device, dtype=torch.bool)
        obs = torch.nonzero(fm_t, as_tuple=False)
        h, w = self.spatial_shape_
        total = int(obs.shape[0]) * h * w
        if total <= int(max_points):
            idx = torch.arange(total, device=self.device)
            pair = obs[idx // (h * w)]
            return X_t, s_t, Y_t, pair[:, 0], pair[:, 1], (idx // w) % h, idx % w
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        pair = obs[torch.randint(obs.shape[0], (int(max_points),), generator=gen, device=self.device)]
        return (
            X_t,
            s_t,
            Y_t,
            pair[:, 0],
            pair[:, 1],
            torch.randint(h, (int(max_points),), generator=gen, device=self.device),
            torch.randint(w, (int(max_points),), generator=gen, device=self.device),
        )

    @torch.no_grad()
    def _pointwise_mse(self, X: torch.Tensor, s: torch.Tensor, Y: torch.Tensor, n: torch.Tensor, f: torch.Tensor, la: torch.Tensor, lo: torch.Tensor) -> float:
        pred = self._net(self._features(X, s, n, f, la, lo))
        target = (Y[n, f, la, lo] - self._field_mean[f]) / self._field_std[f]
        return float(torch.mean((pred - target) ** 2).item())
