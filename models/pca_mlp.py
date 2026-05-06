from __future__ import annotations

from copy import deepcopy
import re
from typing import Literal

import torch

from ._torch_kernels import build_design_matrix, fit_ridge
from ._metrics import srmse_spectral
from ._ppca import PPCAFit, fit_ppca
from ._common import resolve_torch_device

ActivationName = Literal["silu", "relu"]
GROUP_ORDER = ("surface_temperature", "temperature", "specific_humidity", "cloud_fraction", "u", "v", "asr_cloudy", "olr_cloudy")
LEVEL_RE = re.compile(r"^(.+?)_(\d+)$")


def _activation(name: ActivationName) -> torch.nn.Module:
    if name == "silu":
        return torch.nn.SiLU()
    if name == "relu":
        return torch.nn.ReLU()
    raise ValueError(f"Unknown activation={name!r}; expected one of ['silu', 'relu'].")


def _group_key(name: str) -> str:
    match = LEVEL_RE.match(name)
    return match.group(1) if match else name


def _equal_group_mean(per_field: torch.Tensor, field_names: list[str]) -> torch.Tensor:
    grouped = []
    for key in GROUP_ORDER:
        vals = [per_field[i] for i, name in enumerate(field_names) if _group_key(name) == key and torch.isfinite(per_field[i])]
        if vals:
            grouped.append(torch.stack(vals).mean())
    return torch.stack(grouped).mean() if grouped else per_field.new_tensor(float("nan"))


class _ScoreMLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_width: int, *, activation: ActivationName) -> None:
        super().__init__()
        width = int(hidden_width)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(int(in_dim), width),
            _activation(activation),
            torch.nn.Linear(width, width),
            _activation(activation),
            torch.nn.Linear(width, int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PCAMLP:
    def __init__(
        self,
        *,
        latent_dim: int,
        hidden_width: int = 128,
        activation: ActivationName = "silu",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "auto",
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.hidden_width = int(hidden_width)
        self.activation = activation
        self.dtype = dtype
        self.device = resolve_torch_device(device)

        self.ppca_: PPCAFit | None = None
        self.linear_trend_: dict | None = None
        self.n_sim_types_: int | None = None
        self._net: _ScoreMLP | None = None
        self._in_dim: int | None = None
        self.field_names_: list[str] | None = None
        self.mlp_fit_stats_: dict[str, int | float | bool | None] | None = None

    def fit(
        self,
        X: torch.Tensor,
        s: torch.Tensor,
        Y: torch.Tensor,
        *,
        field_mask: torch.Tensor,
        sh_mask: torch.Tensor,
        linear_trend_cfg: dict | None = None,
        field_names: list[str] | None = None,
        ppca_iters: int = 50,
        num_steps: int = 3000,
        lr: float = 1.0e-3,
        weight_decay: float = 1.0e-3,
        seed: int = 0,
        val_X: torch.Tensor | None = None,
        val_s: torch.Tensor | None = None,
        val_Y: torch.Tensor | None = None,
        val_field_mask: torch.Tensor | None = None,
        log_every: int = 20,
        early_stop_metric: str = "norm_srmse_val_1e3",
        early_stop_patience_evals: int | None = None,
        early_stop_min_delta: float = 0.0,
        hard_stop_step: int | None = None,
    ) -> None:
        torch.manual_seed(int(seed))
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        Y = Y.to(device=self.device, dtype=self.dtype)
        field_mask = field_mask.to(device=self.device, dtype=torch.bool)
        sh_mask = sh_mask.to(device=self.device, dtype=torch.bool)
        self.field_names_ = None if field_names is None else list(field_names)
        if val_X is not None:
            val_X = val_X.to(device=self.device, dtype=self.dtype)
        if val_s is not None:
            val_s = val_s.to(device=self.device, dtype=torch.long)
        if val_Y is not None:
            val_Y = val_Y.to(device=self.device, dtype=self.dtype)
        if val_field_mask is not None:
            val_field_mask = val_field_mask.to(device=self.device, dtype=torch.bool)

        self.n_sim_types_ = int(s.max().item()) + 1
        has_val = val_X is not None and val_s is not None and val_Y is not None
        if early_stop_patience_evals is not None and not has_val:
            raise ValueError("early_stop_patience_evals requires validation tensors (val_X, val_s, val_Y).")
        metric = str(early_stop_metric).lower()
        if metric != "norm_srmse_val_1e3":
            raise ValueError(f"Unknown early_stop_metric={early_stop_metric!r} (expected 'norm_srmse_val_1e3').")

        eval_every = int(log_every)
        if eval_every <= 0:
            raise ValueError(f"log_every must be >= 1 (got {log_every}).")
        patience = None if early_stop_patience_evals is None else int(early_stop_patience_evals)
        patience = None if patience is None or patience <= 0 else patience
        min_delta = float(early_stop_min_delta)

        lt_cfg = linear_trend_cfg or {}
        if lt_cfg.get("enabled", False):
            design_in = lt_cfg.get("design", {}) or {}
            design_cfg = {
                "intercept": design_in.get("intercept", True),
                "inputs": design_in.get("inputs", True),
                "sim_onehot": design_in.get("sim_onehot", False),
            }
            lambda_reg = float(lt_cfg.get("lambda", 1.0e-3))
            H = build_design_matrix(X, s, n_sim_types=self.n_sim_types_, design_cfg=design_cfg)
            Gamma = fit_ridge(H, Y, lambda_reg=lambda_reg, field_mask=field_mask)
            Y = Y - torch.einsum("np,paf->naf", H, Gamma)
            Y = torch.where(field_mask.unsqueeze(1), Y, Y.new_zeros(()))
            self.linear_trend_ = {"Gamma": Gamma, "design_cfg": design_cfg, "lambda_reg": lambda_reg}
        else:
            self.linear_trend_ = None

        self.ppca_ = fit_ppca(
            Y,
            field_mask=field_mask,
            sh_mask=sh_mask,
            latent_dim=self.latent_dim,
            num_iters=int(ppca_iters),
            seed=int(seed),
            dtype=self.dtype,
        )
        Z_target = self.ppca_.Z.detach()

        Xin = self._build_inputs(X, s)
        Xin_val = None if not has_val else self._build_inputs(val_X, val_s)
        self._in_dim = int(Xin.shape[1])
        self._net = _ScoreMLP(
            in_dim=self._in_dim,
            out_dim=self.latent_dim,
            hidden_width=self.hidden_width,
            activation=self.activation,
        ).to(device=self.device, dtype=self.dtype)

        opt = torch.optim.AdamW(self._net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        best_metric = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_step: int | None = None
        n_evals = 0
        bad_evals = 0
        steps_run = 0
        early_stop = False
        train_mse = float("nan")
        schedule_steps = int(num_steps)
        total_steps = schedule_steps if hard_stop_step is None else min(schedule_steps, int(hard_stop_step))
        if total_steps < 1:
            raise ValueError(f"hard_stop_step must be >= 1 (got {hard_stop_step}).")

        if has_val:
            self._net.eval()
            with torch.no_grad():
                Z_val0 = self._net(Xin_val)
                metric0 = self._eval_norm_srmse_val_1e3(
                    self._predict_from_latents(Z_val0, X=val_X, s=val_s),
                    val_Y,
                    sh_mask=sh_mask,
                    field_mask=val_field_mask,
                )
                best_metric = metric0
                best_step = 0
                best_state = deepcopy(self._net.state_dict())
                n_evals = 1
                print(
                    f"[mlp] iter {0:>4}/{int(total_steps)} | val_norm_srmse_1e3={metric0:.6f} "
                    f"| best={best_metric:.6f} | bad={bad_evals}/{patience if patience is not None else '-'}"
                )

        for step_idx in range(int(total_steps)):
            self._net.train()
            opt.zero_grad(set_to_none=True)
            train_mse_t = torch.mean((self._net(Xin) - Z_target) ** 2)
            train_mse_t.backward()
            opt.step()
            train_mse = float(train_mse_t.detach().item())
            steps_run = step_idx + 1

            should_log = steps_run % eval_every == 0 or steps_run == int(total_steps)
            if not should_log:
                continue
            msg = f"[mlp] iter {steps_run:>4}/{int(total_steps)} | train_mse={train_mse:.6e}"
            if has_val:
                self._net.eval()
                with torch.no_grad():
                    Z_val = self._net(Xin_val)
                    metric_val = self._eval_norm_srmse_val_1e3(
                        self._predict_from_latents(Z_val, X=val_X, s=val_s),
                        val_Y,
                        sh_mask=sh_mask,
                        field_mask=val_field_mask,
                    )
                    n_evals += 1
                    improved = metric_val < (best_metric - min_delta)
                    if improved:
                        best_metric = metric_val
                        best_step = int(steps_run)
                        best_state = deepcopy(self._net.state_dict())
                        bad_evals = 0
                    elif patience is not None:
                        bad_evals += 1
                    msg = (
                        f"{msg} | val_norm_srmse_1e3={metric_val:.6f} | best={best_metric:.6f} "
                        f"| bad={bad_evals}/{patience if patience is not None else '-'}"
                    )
                if patience is not None and bad_evals >= patience:
                    early_stop = True
                    print(f"{msg} | early_stop=1")
                    break
            print(msg)

        if best_state is not None:
            self._net.load_state_dict(best_state)
        if has_val:
            print(f"[mlp] restore_best step={best_step} best_norm_srmse_1e3={best_metric:.6f}")

        self._net.eval()
        self.mlp_fit_stats_ = {
            "steps_requested": int(schedule_steps),
            "hard_stop_step": None if hard_stop_step is None else int(hard_stop_step),
            "effective_steps": int(total_steps),
            "steps_run": int(steps_run),
            "n_evals": int(n_evals),
            "early_stop_triggered": bool(early_stop),
            "best_norm_srmse_val_1e3": None if not has_val else float(best_metric),
            "best_step": None if best_step is None else int(best_step),
            "patience": None if patience is None else int(patience),
            "min_delta": float(min_delta),
            "log_every": int(eval_every),
            "final_train_mse": float(train_mse),
            "hidden_width": int(self.hidden_width),
            "activation": str(self.activation),
            "optimizer_lr": float(lr),
            "optimizer_weight_decay": float(weight_decay),
        }

    @torch.no_grad()
    def predict(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.ppca_ is None or self._net is None:
            raise RuntimeError("Model not fitted.")
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        Z_pred = self._predict_latents(X, s)
        return self._predict_from_latents(Z_pred, X=X, s=s)

    def _build_inputs(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.n_sim_types_ is None:
            raise RuntimeError("Missing n_sim_types_.")
        one_hot = torch.nn.functional.one_hot(s.long(), num_classes=self.n_sim_types_).to(dtype=X.dtype, device=X.device)
        return torch.cat([X, one_hot], dim=1)

    @torch.no_grad()
    def _predict_latents(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("MLP not fitted.")
        return self._net(self._build_inputs(X, s))

    @torch.no_grad()
    def _predict_from_latents(self, Z_pred: torch.Tensor, *, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.ppca_ is None:
            raise RuntimeError("PPCA not fitted.")
        mu = self.ppca_.mu.to(device=self.device, dtype=self.dtype)
        W = self.ppca_.loadings.to(device=self.device, dtype=self.dtype)
        y_flat = mu.reshape(-1)[None, :] + Z_pred @ W.reshape(-1, self.latent_dim).T
        Y_pred = y_flat.reshape(X.shape[0], mu.shape[0], mu.shape[1]).permute(0, 2, 1).contiguous()
        if self.linear_trend_ is None:
            return Y_pred
        if self.n_sim_types_ is None:
            raise RuntimeError("Missing n_sim_types_.")
        lt = self.linear_trend_
        Gamma = lt["Gamma"].to(device=Y_pred.device, dtype=Y_pred.dtype)
        H = build_design_matrix(X, s, n_sim_types=self.n_sim_types_, design_cfg=lt["design_cfg"])
        return Y_pred + torch.einsum("np,paf->naf", H, Gamma)

    @torch.no_grad()
    def _eval_norm_srmse_val_1e3(
        self,
        Y_pred: torch.Tensor,
        Y_val: torch.Tensor,
        *,
        sh_mask: torch.Tensor,
        field_mask: torch.Tensor | None,
    ) -> float:
        sh = sh_mask.to(device=Y_pred.device, dtype=Y_pred.dtype)
        y_hat = Y_pred * sh.unsqueeze(0)
        y_ref = Y_val.to(device=y_hat.device, dtype=y_hat.dtype) * sh.unsqueeze(0)
        fm = field_mask.to(device=y_hat.device, dtype=torch.bool) if field_mask is not None else None
        per_field = srmse_spectral(y_hat, y_ref, aggregate_only=False, field_mask=fm)[:-1]
        agg = _equal_group_mean(per_field, self.field_names_) if self.field_names_ is not None else per_field.nanmean()
        return float((agg * 1e3).item())
