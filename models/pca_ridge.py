from __future__ import annotations

import torch

from ._torch_kernels import build_design_matrix, fit_ridge
from ._ppca import PPCAFit, fit_ppca
from ._common import resolve_torch_device


def fit_latent_ridge(H: torch.Tensor, Z: torch.Tensor, *, lambda_reg: float, intercept: bool = True) -> torch.Tensor:
    n = H.shape[0]
    penalty = H.new_full((H.shape[1],), float(lambda_reg))
    if intercept:
        penalty[0] = 0.0
    return torch.linalg.solve(H.T @ H / n + torch.diag(penalty), H.T @ Z / n)


class PCARidge:
    def __init__(
        self,
        *,
        latent_dim: int,
        lambda_reg: float = 1.0e-3,
        design_cfg: dict | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "auto",
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.lambda_reg = float(lambda_reg)
        self.design_cfg = dict(design_cfg or {"intercept": True, "inputs": True, "sim_onehot": True})
        self.dtype = dtype
        self.device = resolve_torch_device(device)

        self.ppca_: PPCAFit | None = None
        self.linear_trend_: dict | None = None
        self.n_sim_types_: int | None = None
        self.B_: torch.Tensor | None = None
        self.fit_stats_: dict | None = None

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
        seed: int = 0,
        n_sim_types: int | None = None,
    ) -> None:
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        Y = Y.to(device=self.device, dtype=self.dtype)
        field_mask = field_mask.to(device=self.device, dtype=torch.bool)
        sh_mask = sh_mask.to(device=self.device, dtype=torch.bool)

        self.n_sim_types_ = int(s.max().item()) + 1 if n_sim_types is None else int(n_sim_types)
        lt_cfg = linear_trend_cfg or {}
        if lt_cfg.get("enabled", False):
            design_cfg = {"intercept": True, "inputs": True, "sim_onehot": False} | dict(lt_cfg.get("design", {}) or {})
            H = build_design_matrix(X, s, n_sim_types=self.n_sim_types_, design_cfg=design_cfg)
            Gamma = fit_ridge(H, Y, lambda_reg=float(lt_cfg.get("lambda", 1.0e-3)), field_mask=field_mask)
            Y = torch.where(field_mask.unsqueeze(1), Y - torch.einsum("np,paf->naf", H, Gamma), Y.new_zeros(()))
            self.linear_trend_ = {"Gamma": Gamma, "design_cfg": design_cfg, "lambda_reg": float(lt_cfg.get("lambda", 1.0e-3))}
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
        H = build_design_matrix(X, s, n_sim_types=self.n_sim_types_, design_cfg=self.design_cfg)
        self.B_ = fit_latent_ridge(H, self.ppca_.Z.detach(), lambda_reg=self.lambda_reg, intercept=bool(self.design_cfg.get("intercept", True)))
        self.fit_stats_ = {"latent_dim": self.latent_dim, "lambda_reg": self.lambda_reg, "ppca_iters": int(ppca_iters), "design": self.design_cfg}

    @torch.no_grad()
    def predict(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.ppca_ is None or self.B_ is None or self.n_sim_types_ is None:
            raise RuntimeError("Model not fitted.")
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        H = build_design_matrix(X, s, n_sim_types=self.n_sim_types_, design_cfg=self.design_cfg)
        return self._predict_from_latents(H @ self.B_, X=X, s=s)

    @torch.no_grad()
    def _predict_from_latents(self, Z_pred: torch.Tensor, *, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.ppca_ is None:
            raise RuntimeError("Model not fitted.")
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
