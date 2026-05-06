from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal

import torch

from ._torch_kernels import apply_kernel, build_design_matrix, fit_ridge
from ._metrics import srmse_spectral
from ._ppca import PPCAFit, fit_ppca
from ._common import resolve_torch_device

KernelName = Literal["rbf", "matern32", "matern52"]
KernelMode = Literal["shared", "per_pc"]


def _corr_from_tril(tril: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    L = torch.tril(tril)
    S = L @ L.T + eps * torch.eye(L.shape[0], device=L.device, dtype=L.dtype)
    d = torch.sqrt(torch.diagonal(S))
    return S / (d[:, None] * d[None, :])


@dataclass
class _GPKernel:
    kernel: KernelName
    ell: torch.Tensor  # (d,)
    sigma_f: torch.Tensor  # ()
    sigma2_n: torch.Tensor  # ()
    r_sim: torch.Tensor  # (S,)
    corr: torch.Tensor  # (S,S)


class PPCAICM:
    def __init__(
        self,
        *,
        latent_dim: int,
        kernel: KernelName = "matern52",
        kernel_mode: KernelMode = "shared",
        jitter: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "auto",
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.kernel: KernelName = kernel
        self.kernel_mode: KernelMode = kernel_mode
        self.jitter = float(jitter)
        self.dtype = dtype
        self.device = resolve_torch_device(device)

        self.ppca_: PPCAFit | None = None
        self._X_train: torch.Tensor | None = None
        self._s_train: torch.Tensor | None = None
        self.n_sim_types_: int | None = None
        self.linear_trend_: dict | None = None
        self._sh_mask: torch.Tensor | None = None

        self._shared: _GPKernel | None = None
        self._L_shared: torch.Tensor | None = None
        self._alpha_shared: torch.Tensor | None = None

        self._per_pc: list[_GPKernel] | None = None
        self._alpha_per_pc: torch.Tensor | None = None  # (q,n)
        self.gp_fit_stats_: dict[str, Any] | None = None

    def fit(
        self,
        X: torch.Tensor,
        s: torch.Tensor,
        Y: torch.Tensor,
        *,
        field_mask: torch.Tensor,
        sh_mask: torch.Tensor,
        linear_trend_cfg: dict | None = None,
        ppca_iters: int = 50,
        gp_steps: int = 50,
        gp_lr: float = 5e-2,
        seed: int = 0,
        ell_init: str | float = "median_pairwise_dist",
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
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        Y = Y.to(device=self.device, dtype=self.dtype)
        field_mask = field_mask.to(device=self.device, dtype=torch.bool)
        sh_mask = sh_mask.to(device=self.device, dtype=torch.bool)
        self._sh_mask = sh_mask
        if val_X is not None:
            val_X = val_X.to(device=self.device, dtype=self.dtype)
        if val_s is not None:
            val_s = val_s.to(device=self.device, dtype=torch.long)
        if val_Y is not None:
            val_Y = val_Y.to(device=self.device, dtype=self.dtype)
        if val_field_mask is not None:
            val_field_mask = val_field_mask.to(device=self.device, dtype=torch.bool)

        self._X_train, self._s_train = X, s
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
            num_iters=ppca_iters,
            seed=seed,
            dtype=self.dtype,
        )

        self.gp_fit_stats_ = {}
        self.gp_fit_stats_["shared"] = self._fit_gp_shared(
            self.ppca_.Z,
            gp_steps=gp_steps,
            hard_stop_step=hard_stop_step,
            lr=gp_lr,
            seed=seed,
            sh_mask=sh_mask,
            ell_init=ell_init,
            val_X=val_X,
            val_s=val_s,
            val_Y=val_Y,
            val_field_mask=val_field_mask,
            log_every=eval_every,
            patience=patience,
            min_delta=min_delta,
        )
        if self.kernel_mode == "per_pc":
            self.gp_fit_stats_["per_pc"] = self._fit_gp_per_pc(
                self.ppca_.Z,
                gp_steps=gp_steps,
                hard_stop_step=hard_stop_step,
                lr=gp_lr,
                sh_mask=sh_mask,
                val_X=val_X,
                val_s=val_s,
                val_Y=val_Y,
                val_field_mask=val_field_mask,
                log_every=eval_every,
                patience=patience,
                min_delta=min_delta,
            )

    @torch.no_grad()
    def predict(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.ppca_ is None:
            raise RuntimeError("Model not fitted.")
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        Z_pred = self._predict_latents(X, s)  # (n_test,q)
        return self._predict_from_latents(Z_pred, X=X, s=s)

    @torch.no_grad()
    def predict_samples(
        self,
        X: torch.Tensor,
        s: torch.Tensor,
        *,
        n_post_samples: int,
        seed: int | None = None,
        include_gp_nugget: bool = True,
        include_ppca_noise: bool = True,
    ) -> torch.Tensor:
        if self.ppca_ is None:
            raise RuntimeError("Model not fitted.")
        if self._sh_mask is None:
            raise RuntimeError("Missing sh_mask.")
        M = int(n_post_samples)
        if M < 1:
            raise ValueError(f"n_post_samples must be >= 1 (got {M}).")
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        if seed is not None:
            torch.manual_seed(int(seed))

        Z_mean, Z_var = self._predict_latent_mean_and_var(X, s, include_gp_nugget=include_gp_nugget)
        eps = torch.randn((M, int(X.shape[0]), int(Z_mean.shape[1])), device=self.device, dtype=self.dtype)
        Z = Z_mean.unsqueeze(0) + eps * torch.sqrt(Z_var.clamp_min(0.0)).unsqueeze(0)

        mu = self.ppca_.mu.to(device=self.device, dtype=self.dtype)  # (F,A)
        W = self.ppca_.loadings.to(device=self.device, dtype=self.dtype)  # (F,A,q)
        y_flat = mu.reshape(-1)[None, :] + Z.reshape(M * int(X.shape[0]), -1) @ W.reshape(-1, self.latent_dim).T
        Y = y_flat.reshape(M, int(X.shape[0]), mu.shape[0], mu.shape[1]).permute(0, 1, 3, 2).contiguous()

        if self.linear_trend_ is not None:
            if self.n_sim_types_ is None:
                raise RuntimeError("Missing n_sim_types_.")
            lt = self.linear_trend_
            Gamma = lt["Gamma"].to(device=Y.device, dtype=Y.dtype)
            H = build_design_matrix(X, s, n_sim_types=self.n_sim_types_, design_cfg=lt["design_cfg"])
            Y = Y + torch.einsum("np,paf->naf", H, Gamma).unsqueeze(0)

        if include_ppca_noise:
            sigma2 = self.ppca_.sigma2.to(device=Y.device, dtype=Y.dtype)
            noise = torch.randn_like(Y) * torch.sqrt(sigma2)
            noise = noise * self._sh_mask.to(device=Y.device).to(dtype=Y.dtype).unsqueeze(0).unsqueeze(0)
            Y = Y + noise
        return Y

    @torch.no_grad()
    def _predict_from_latents(self, Z_pred: torch.Tensor, *, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.ppca_ is None:
            raise RuntimeError("Model not fitted.")
        mu = self.ppca_.mu.to(device=self.device, dtype=self.dtype)  # (F,A)
        W = self.ppca_.loadings.to(device=self.device, dtype=self.dtype)  # (F,A,q)
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
        return float((srmse_spectral(y_hat, y_ref, aggregate_only=True, field_mask=fm) * 1e3).item())

    @torch.no_grad()
    def _kernel_diag(self, k: _GPKernel, *, s: torch.Tensor) -> torch.Tensor:
        rs = k.r_sim[s].to(dtype=self.dtype)
        diag_corr = torch.diagonal(k.corr).to(dtype=self.dtype)[s]
        return (k.sigma_f.to(dtype=self.dtype) ** 2) * (rs * rs) * diag_corr

    @torch.no_grad()
    def _predict_latent_mean_and_var(
        self,
        X: torch.Tensor,
        s: torch.Tensor,
        *,
        include_gp_nugget: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._X_train is None or self._s_train is None:
            raise RuntimeError("Missing training data.")
        Xtr, str_ = self._X_train, self._s_train
        Z_mean = self._predict_latents(X, s)
        if self._shared is None:
            raise RuntimeError("GP not fitted.")

        if self.kernel_mode == "shared":
            if self._L_shared is None:
                raise RuntimeError("Missing shared cholesky factor.")
            K_cross = self._cross(self._shared, Xtr=Xtr, str_=str_, X=X, s=s)
            v = torch.linalg.solve_triangular(self._L_shared, K_cross, upper=False)
            var_f = (self._kernel_diag(self._shared, s=s) - v.square().sum(dim=0)).clamp_min(0.0)
            var = var_f[:, None].expand(-1, int(Z_mean.shape[1]))
        else:
            if self._per_pc is None:
                raise RuntimeError("Per-PC kernels missing.")
            vars_pc: list[torch.Tensor] = []
            for k in self._per_pc:
                L = torch.linalg.cholesky(self._gram(k, X=Xtr, s=str_))
                K_cross = self._cross(k, Xtr=Xtr, str_=str_, X=X, s=s)
                v = torch.linalg.solve_triangular(L, K_cross, upper=False)
                vars_pc.append((self._kernel_diag(k, s=s) - v.square().sum(dim=0)).clamp_min(0.0))
            var = torch.stack(vars_pc, dim=1)

        if include_gp_nugget:
            var = var + self._shared.sigma2_n.to(dtype=var.dtype)
        return Z_mean, var

    # ----------------------------
    # GP fitting / prediction
    # ----------------------------

    def _gram(self, k: _GPKernel, *, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        n = int(X.shape[0])
        K_x = apply_kernel(k.kernel, X, k.ell, k.sigma_f)
        rs = k.r_sim[s]
        K_sim = (rs[:, None] * rs[None, :]) * k.corr[s][:, s]
        eye = torch.eye(n, device=X.device, dtype=X.dtype)
        K = 0.5 * (K_x * K_sim + (K_x * K_sim).T) + eye * (k.sigma2_n + self.jitter)
        return K

    def _cross(self, k: _GPKernel, *, Xtr: torch.Tensor, str_: torch.Tensor, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        K_x = apply_kernel(k.kernel, Xtr, k.ell, k.sigma_f, X2=X)
        rs_tr = k.r_sim[str_][:, None]
        rs_te = k.r_sim[s][None, :]
        return K_x * (rs_tr * rs_te) * k.corr[str_][:, s]

    def _fit_gp_shared(
        self,
        Z: torch.Tensor,
        *,
        gp_steps: int,
        hard_stop_step: int | None,
        lr: float,
        seed: int,
        sh_mask: torch.Tensor,
        ell_init: str | float,
        val_X: torch.Tensor | None,
        val_s: torch.Tensor | None,
        val_Y: torch.Tensor | None,
        val_field_mask: torch.Tensor | None,
        log_every: int,
        patience: int | None,
        min_delta: float,
    ) -> dict[str, Any]:
        if self._X_train is None or self._s_train is None:
            raise RuntimeError("Missing training data.")
        X, s = self._X_train, self._s_train
        _, d = X.shape
        S = int(s.max().item()) + 1
        q = int(Z.shape[1])
        has_val = val_X is not None and val_s is not None and val_Y is not None

        def median_pairwise_dist(X: torch.Tensor, *, max_pairs: int = 20_000) -> float:
            n = int(X.shape[0])
            if n < 2:
                return 1.0
            m = min(int(max_pairs), n * (n - 1) // 2)
            g = torch.Generator().manual_seed(int(seed))
            i = torch.randint(0, n, (m,), generator=g).to(device=X.device)
            j = torch.randint(0, n, (m,), generator=g).to(device=X.device)
            mask = i != j
            d = (X[i[mask]] - X[j[mask]]).square().sum(dim=1).sqrt()
            return 1.0 if d.numel() == 0 else float(d.median().item())

        torch.manual_seed(int(seed))
        ell_init_cfg = ell_init
        if isinstance(ell_init_cfg, (int, float)):
            ell0 = float(ell_init_cfg)
        else:
            mode = str(ell_init_cfg).strip().lower().replace("-", "_")
            if mode == "median_pairwise_dist":
                ell0 = median_pairwise_dist(X)
            elif mode == "sqrt2d":
                ell0 = float(math.sqrt(2.0 * float(d)))
            else:
                raise ValueError(f"Unknown ell_init={ell_init!r} (expected 'median_pairwise_dist', 'sqrt2d', or a float).")
        log_ell = torch.full((d,), math.log(float(ell0)), device=self.device, dtype=self.dtype, requires_grad=True)
        log_sigma_f = torch.zeros((), device=self.device, dtype=self.dtype, requires_grad=True)
        log_sigma2_n = torch.full((), -6.0, device=self.device, dtype=self.dtype, requires_grad=True)
        log_r_sim = torch.zeros((S,), device=self.device, dtype=self.dtype, requires_grad=True)
        tril_corr = torch.eye(S, device=self.device, dtype=self.dtype, requires_grad=True)
        best_metric = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        best_step: int | None = None
        n_evals = 0
        bad_evals = 0
        steps_run = 0
        early_stop = False
        schedule_steps = int(gp_steps)
        total_steps = schedule_steps if hard_stop_step is None else min(schedule_steps, int(hard_stop_step))
        if total_steps < 1:
            raise ValueError(f"hard_stop_step must be >= 1 (got {hard_stop_step}).")

        def fmt_params() -> str:
            ell = torch.exp(log_ell).detach().cpu().tolist()
            sigma_f = float(torch.exp(log_sigma_f).detach().item())
            sigma2_n = float(torch.exp(log_sigma2_n).detach().item())
            r_sim = torch.exp(log_r_sim - log_r_sim.mean()).detach().cpu().tolist()
            ell_str = ",".join(f"{x:.3g}" for x in ell)
            return (
                f"| ell={ell_str} | ell_min={min(ell):.3g} | ell_max={max(ell):.3g} "
                f"| sigma_f={sigma_f:.3g} | sigma2_n={sigma2_n:.3g} "
                f"| r_sim_min={min(r_sim):.3g} | r_sim_max={max(r_sim):.3g}"
            )
        if has_val:
            with torch.no_grad():
                k0 = _GPKernel(
                    kernel=self.kernel,
                    ell=torch.exp(log_ell),
                    sigma_f=torch.exp(log_sigma_f),
                    sigma2_n=torch.exp(log_sigma2_n),
                    r_sim=torch.exp(log_r_sim - log_r_sim.mean()),
                    corr=_corr_from_tril(tril_corr),
                )
                L0 = torch.linalg.cholesky(self._gram(k0, X=X, s=s))
                alpha0 = torch.cholesky_solve(Z, L0)
                quad0 = (Z * alpha0).sum()
                logdet0 = 2.0 * torch.log(torch.diagonal(L0, dim1=-2, dim2=-1)).sum()
                train_obj0 = 0.5 * (quad0 + float(q) * logdet0)
                Z_val0 = self._cross(k0, Xtr=X, str_=s, X=val_X, s=val_s).T @ alpha0
                metric0 = self._eval_norm_srmse_val_1e3(
                    self._predict_from_latents(Z_val0, X=val_X, s=val_s),
                    val_Y,
                    sh_mask=sh_mask,
                    field_mask=val_field_mask,
                )
                best_metric = metric0
                best_step = 0
                best_state = {
                    "log_ell": log_ell.detach().clone(),
                    "log_sigma_f": log_sigma_f.detach().clone(),
                    "log_sigma2_n": log_sigma2_n.detach().clone(),
                    "log_r_sim": log_r_sim.detach().clone(),
                    "tril_corr": tril_corr.detach().clone(),
                }
                n_evals = 1
                print(
                    f"[gp/shared] iter {0:>4}/{int(total_steps)} | train_obj={float(train_obj0.item()):.6e} "
                    f"| val_norm_srmse_1e3={metric0:.6f} | best={best_metric:.6f} "
                    f"| bad={bad_evals}/{patience if patience is not None else '-'} {fmt_params()}"
                )
        opt = torch.optim.Adam([log_ell, log_sigma_f, log_sigma2_n, log_r_sim, tril_corr], lr=float(lr))
        for step_idx in range(int(total_steps)):
            opt.zero_grad(set_to_none=True)
            r_sim = torch.exp(log_r_sim - log_r_sim.mean())
            k = _GPKernel(
                kernel=self.kernel,
                ell=torch.exp(log_ell),
                sigma_f=torch.exp(log_sigma_f),
                sigma2_n=torch.exp(log_sigma2_n),
                r_sim=r_sim,
                corr=_corr_from_tril(tril_corr),
            )
            K = self._gram(k, X=X, s=s)
            L = torch.linalg.cholesky(K)
            alpha = torch.cholesky_solve(Z, L)
            quad = (Z * alpha).sum()
            logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum()
            train_obj = 0.5 * (quad + float(q) * logdet)
            train_obj.backward()
            opt.step()
            steps_run = step_idx + 1

            should_log = steps_run % log_every == 0 or steps_run == int(total_steps)
            if not should_log:
                continue
            param_str = fmt_params()
            msg = f"[gp/shared] iter {steps_run:>4}/{int(total_steps)} | train_obj={float(train_obj.item()):.6e}"
            if has_val:
                n_evals += 1
                with torch.no_grad():
                    k_eval = _GPKernel(
                        kernel=self.kernel,
                        ell=torch.exp(log_ell),
                        sigma_f=torch.exp(log_sigma_f),
                        sigma2_n=torch.exp(log_sigma2_n),
                        r_sim=torch.exp(log_r_sim - log_r_sim.mean()),
                        corr=_corr_from_tril(tril_corr),
                    )
                    L_eval = torch.linalg.cholesky(self._gram(k_eval, X=X, s=s))
                    alpha_eval = torch.cholesky_solve(Z, L_eval)
                    Z_val = self._cross(k_eval, Xtr=X, str_=s, X=val_X, s=val_s).T @ alpha_eval
                    metric = self._eval_norm_srmse_val_1e3(
                        self._predict_from_latents(Z_val, X=val_X, s=val_s),
                        val_Y,
                        sh_mask=sh_mask,
                        field_mask=val_field_mask,
                    )
                    improved = metric < (best_metric - min_delta)
                    if improved:
                        best_metric = metric
                        best_step = int(steps_run)
                        best_state = {
                            "log_ell": log_ell.detach().clone(),
                            "log_sigma_f": log_sigma_f.detach().clone(),
                            "log_sigma2_n": log_sigma2_n.detach().clone(),
                            "log_r_sim": log_r_sim.detach().clone(),
                            "tril_corr": tril_corr.detach().clone(),
                        }
                        bad_evals = 0
                    elif patience is not None:
                        bad_evals += 1
                    msg = (
                        f"{msg} | val_norm_srmse_1e3={metric:.6f} | best={best_metric:.6f} "
                        f"| bad={bad_evals}/{patience if patience is not None else '-'} {param_str}"
                    )
                if patience is not None and bad_evals >= patience:
                    early_stop = True
                    print(f"{msg} | early_stop=1")
                    break
            else:
                msg = f"{msg} {param_str}"
            print(msg)

        if best_state is not None:
            with torch.no_grad():
                log_ell.copy_(best_state["log_ell"])
                log_sigma_f.copy_(best_state["log_sigma_f"])
                log_sigma2_n.copy_(best_state["log_sigma2_n"])
                log_r_sim.copy_(best_state["log_r_sim"])
                tril_corr.copy_(best_state["tril_corr"])
        if has_val:
            print(
                f"[gp/shared] restore_best step={best_step} best_norm_srmse_1e3={best_metric:.6f} {fmt_params()}"
            )

        with torch.no_grad():
            r_sim = torch.exp(log_r_sim - log_r_sim.mean())
            self._shared = _GPKernel(
                kernel=self.kernel,
                ell=torch.exp(log_ell).detach(),
                sigma_f=torch.exp(log_sigma_f).detach(),
                sigma2_n=torch.exp(log_sigma2_n).detach(),
                r_sim=r_sim.detach(),
                corr=_corr_from_tril(tril_corr).detach(),
            )
            K = self._gram(self._shared, X=X, s=s)
            L = torch.linalg.cholesky(K)
            self._L_shared = L
            self._alpha_shared = torch.cholesky_solve(Z, L)
        return {
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
            "log_every": int(log_every),
        }

    def _fit_gp_per_pc(
        self,
        Z: torch.Tensor,
        *,
        gp_steps: int,
        hard_stop_step: int | None,
        lr: float,
        sh_mask: torch.Tensor,
        val_X: torch.Tensor | None,
        val_s: torch.Tensor | None,
        val_Y: torch.Tensor | None,
        val_field_mask: torch.Tensor | None,
        log_every: int,
        patience: int | None,
        min_delta: float,
    ) -> dict[str, Any]:
        if self._shared is None or self._X_train is None or self._s_train is None:
            raise RuntimeError("Shared GP state missing.")
        X, s = self._X_train, self._s_train
        q = int(Z.shape[1])
        sigma2_n = self._shared.sigma2_n.detach()
        has_val = val_X is not None and val_s is not None and val_Y is not None
        best_metric = float("inf")
        best_state: dict[str, list[torch.Tensor]] | None = None
        best_step: int | None = None
        n_evals = 0
        bad_evals = 0
        steps_run = 0
        early_stop = False
        schedule_steps = int(gp_steps)
        total_steps = schedule_steps if hard_stop_step is None else min(schedule_steps, int(hard_stop_step))
        if total_steps < 1:
            raise ValueError(f"hard_stop_step must be >= 1 (got {hard_stop_step}).")

        log_ell_list = [torch.log(self._shared.ell).detach().clone().requires_grad_(True) for _ in range(q)]
        log_sigma_f_list = [torch.log(self._shared.sigma_f).detach().clone().requires_grad_(True) for _ in range(q)]
        opts = [torch.optim.Adam([log_ell_list[i], log_sigma_f_list[i]], lr=float(lr)) for i in range(q)]

        def fmt_params() -> str:
            ell = torch.stack([torch.exp(t.detach()) for t in log_ell_list], dim=0)
            sigma_f = torch.stack([torch.exp(t.detach()) for t in log_sigma_f_list], dim=0)
            s2n = float(sigma2_n.detach().item())
            return (
                f"| ell_min={float(ell.min().item()):.3g} | ell_max={float(ell.max().item()):.3g} "
                f"| sigma_f_min={float(sigma_f.min().item()):.3g} | sigma_f_max={float(sigma_f.max().item()):.3g} "
                f"| sigma2_n={s2n:.3g}"
            )

        def build_state() -> tuple[list[_GPKernel], torch.Tensor]:
            kernels: list[_GPKernel] = []
            alphas: list[torch.Tensor] = []
            for i in range(q):
                z = Z[:, i : i + 1]
                k = _GPKernel(
                    kernel=self.kernel,
                    ell=torch.exp(log_ell_list[i]).detach(),
                    sigma_f=torch.exp(log_sigma_f_list[i]).detach(),
                    sigma2_n=sigma2_n,
                    r_sim=self._shared.r_sim,
                    corr=self._shared.corr,
                )
                L = torch.linalg.cholesky(self._gram(k, X=X, s=s))
                kernels.append(k)
                alphas.append(torch.cholesky_solve(z, L).squeeze(-1))
            return kernels, torch.stack(alphas, dim=0)

        def eval_val(kernels: list[_GPKernel], alphas: torch.Tensor) -> float:
            Z_val = val_X.new_zeros((int(val_X.shape[0]), self.latent_dim))
            for i, k in enumerate(kernels):
                Z_val[:, i] = self._cross(k, Xtr=X, str_=s, X=val_X, s=val_s).T @ alphas[i]
            return self._eval_norm_srmse_val_1e3(
                self._predict_from_latents(Z_val, X=val_X, s=val_s),
                val_Y,
                sh_mask=sh_mask,
                field_mask=val_field_mask,
            )

        if has_val:
            n_evals = 1
            with torch.no_grad():
                kernels_eval, alphas_eval = build_state()
                metric0 = eval_val(kernels_eval, alphas_eval)
                best_metric = metric0
                best_step = 0
                best_state = {
                    "log_ell": [t.detach().clone() for t in log_ell_list],
                    "log_sigma_f": [t.detach().clone() for t in log_sigma_f_list],
                }
                print(
                    f"[gp/per_pc] iter {0:>4}/{int(total_steps)} | val_norm_srmse_1e3={metric0:.6f} "
                    f"| best={best_metric:.6f} | bad={bad_evals}/{patience if patience is not None else '-'} {fmt_params()}"
                )

        for step_idx in range(int(total_steps)):
            train_obj_total = 0.0
            for i in range(q):
                z = Z[:, i : i + 1]
                opts[i].zero_grad(set_to_none=True)
                k = _GPKernel(
                    kernel=self.kernel,
                    ell=torch.exp(log_ell_list[i]),
                    sigma_f=torch.exp(log_sigma_f_list[i]),
                    sigma2_n=sigma2_n,
                    r_sim=self._shared.r_sim,
                    corr=self._shared.corr,
                )
                L = torch.linalg.cholesky(self._gram(k, X=X, s=s))
                alpha = torch.cholesky_solve(z, L)
                quad = (z * alpha).sum()
                logdet = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum()
                train_obj_i = 0.5 * (quad + logdet)
                train_obj_total += float(train_obj_i.item())
                train_obj_i.backward()
                opts[i].step()
            steps_run = step_idx + 1
            should_log = steps_run % log_every == 0 or steps_run == int(total_steps)
            if not should_log:
                continue
            param_str = fmt_params()
            msg = f"[gp/per_pc] iter {steps_run:>4}/{int(total_steps)} | train_obj_sum={train_obj_total:.6e}"
            if has_val:
                n_evals += 1
                with torch.no_grad():
                    kernels_eval, alphas_eval = build_state()
                    metric = eval_val(kernels_eval, alphas_eval)
                    improved = metric < (best_metric - min_delta)
                    if improved:
                        best_metric = metric
                        best_step = int(steps_run)
                        best_state = {
                            "log_ell": [t.detach().clone() for t in log_ell_list],
                            "log_sigma_f": [t.detach().clone() for t in log_sigma_f_list],
                        }
                        bad_evals = 0
                    elif patience is not None:
                        bad_evals += 1
                    msg = (
                        f"{msg} | val_norm_srmse_1e3={metric:.6f} | best={best_metric:.6f} "
                        f"| bad={bad_evals}/{patience if patience is not None else '-'} {param_str}"
                    )
                if patience is not None and bad_evals >= patience:
                    early_stop = True
                    print(f"{msg} | early_stop=1")
                    break
            else:
                msg = f"{msg} {param_str}"
            print(msg)
            if early_stop:
                break

        if best_state is not None:
            with torch.no_grad():
                for i in range(q):
                    log_ell_list[i].copy_(best_state["log_ell"][i])
                    log_sigma_f_list[i].copy_(best_state["log_sigma_f"][i])

        if has_val:
            print(
                f"[gp/per_pc] restore_best step={best_step} best_norm_srmse_1e3={best_metric:.6f} {fmt_params()}"
            )

        kernels, alphas = build_state()
        self._per_pc = kernels
        self._alpha_per_pc = alphas  # (q,n)
        return {
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
            "log_every": int(log_every),
        }

    @torch.no_grad()
    def _predict_latents(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self._X_train is None or self._s_train is None:
            raise RuntimeError("Missing training data.")
        Xtr, str_ = self._X_train, self._s_train

        if self.kernel_mode == "shared":
            if self._shared is None or self._alpha_shared is None:
                raise RuntimeError("Shared GP not fitted.")
            K_cross = self._cross(self._shared, Xtr=Xtr, str_=str_, X=X, s=s)
            return K_cross.T @ self._alpha_shared

        if self._per_pc is None or self._alpha_per_pc is None:
            raise RuntimeError("Per-PC GP not fitted.")
        out = X.new_zeros((int(X.shape[0]), self.latent_dim))
        for i, k in enumerate(self._per_pc):
            out[:, i] = self._cross(k, Xtr=Xtr, str_=str_, X=X, s=s).T @ self._alpha_per_pc[i]
        return out
