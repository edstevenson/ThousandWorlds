from __future__ import annotations

"""PCA-GBT: gradient-boosted trees on PPCA latent scores.

This baseline shares the exact representation used by ``pca_ridge`` and
``pca_mlp`` -- PPCA compression of the T21 spectral coefficients, with an
optional shared linear trend removed first -- and only swaps the latent-score
regressor. Where ``pca_ridge`` uses a linear map and ``pca_mlp`` an MLP, this
model fits one gradient-boosted-tree regressor per latent component.

Motivation. Exoplanet climates undergo *regime transitions* (temperate ->
snowball, or temperate -> runaway/moist greenhouse) at sharp thresholds of
stellar flux, CO2, and other parameters. Smooth regressors (ridge, stationary
kernel GPs, MLPs) blur these thresholds. Axis-aligned tree ensembles partition
the parameter space and can place splits at the transition boundaries, which is
exactly the structure the benchmark exposes. Trees are also orders of magnitude
cheaper to fit than the GP baselines.
"""

import numpy as np
import torch

from ._torch_kernels import build_design_matrix, fit_ridge
from ._ppca import PPCAFit, fit_ppca
from ._common import resolve_torch_device


def _fit_one_hgbt(params: dict, Xin: np.ndarray, z_col: np.ndarray, rs: int):
    """Fit a single-component HistGradientBoostingRegressor (single-threaded).

    Defined at module level so joblib can pickle it for process-based parallelism.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits

    reg = HistGradientBoostingRegressor(random_state=rs, **params)
    with threadpool_limits(limits=1):
        reg.fit(Xin, z_col)
    return reg


class PCAGBT:
    def __init__(
        self,
        *,
        latent_dim: int,
        learning_rate: float = 0.05,
        max_iter: int = 600,
        max_leaf_nodes: int = 31,
        min_samples_leaf: int = 15,
        l2_regularization: float = 1.0,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 25,
        n_jobs: int = -1,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "auto",
    ) -> None:
        self.latent_dim = int(latent_dim)
        # gradient boosting (hgbt)
        self.learning_rate = float(learning_rate)
        self.max_iter = int(max_iter)
        self.max_leaf_nodes = int(max_leaf_nodes)
        self.min_samples_leaf = int(min_samples_leaf)
        self.l2_regularization = float(l2_regularization)
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.n_jobs = int(n_jobs)
        self.dtype = dtype
        self.device = resolve_torch_device(device)

        self.ppca_: PPCAFit | None = None
        self.linear_trend_: dict | None = None
        self.n_sim_types_: int | None = None
        self._regressors: list | None = None
        self.field_names_: list[str] | None = None
        self.gbt_fit_stats_: dict | None = None

    # ------------------------------------------------------------------ fit
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
        seed: int = 0,
        n_sim_types: int | None = None,
    ) -> None:
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        Y = Y.to(device=self.device, dtype=self.dtype)
        field_mask = field_mask.to(device=self.device, dtype=torch.bool)
        sh_mask = sh_mask.to(device=self.device, dtype=torch.bool)
        self.field_names_ = None if field_names is None else list(field_names)
        self.n_sim_types_ = int(s.max().item()) + 1 if n_sim_types is None else int(n_sim_types)

        # Optional shared linear trend, removed before PPCA (mirrors pca_mlp).
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
        Z_target = self.ppca_.Z.detach().cpu().numpy()  # (n, q)
        Xin = self._build_inputs_np(X, s)  # (n, d)

        self._fit_hgbt(Xin, Z_target, seed)

    def _fit_hgbt(self, Xin: np.ndarray, Z: np.ndarray, seed: int) -> None:
        from joblib import Parallel, delayed

        n = Xin.shape[0]
        # Internal validation needs a few held-out rows; disable on tiny data.
        early = self.early_stopping and n >= 40
        params = dict(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=early,
            validation_fraction=self.validation_fraction if early else None,
            n_iter_no_change=self.n_iter_no_change,
        )
        # Components are independent -> fit them in parallel (each tree fit single-threaded).
        regs = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(_fit_one_hgbt)(params, Xin, Z[:, j], int(seed) + j) for j in range(Z.shape[1])
        )
        iters_used = [int(getattr(r, "n_iter_", self.max_iter)) for r in regs]
        self._regressors = regs
        self.gbt_fit_stats_ = {
            "tree_backend": "hgbt",
            "latent_dim": int(Z.shape[1]),
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "early_stopping": bool(early),
            "mean_n_iter": float(np.mean(iters_used)),
            "max_n_iter": int(np.max(iters_used)),
        }

    # -------------------------------------------------------------- predict
    @torch.no_grad()
    def predict(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.ppca_ is None or self._regressors is None:
            raise RuntimeError("Model not fitted.")
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        Xin = self._build_inputs_np(X, s)
        Z_pred = np.stack([reg.predict(Xin) for reg in self._regressors], axis=1)
        Z_pred_t = torch.from_numpy(np.ascontiguousarray(Z_pred)).to(device=self.device, dtype=self.dtype)
        return self._predict_from_latents(Z_pred_t, X=X, s=s)

    def _build_inputs_np(self, X: torch.Tensor, s: torch.Tensor) -> np.ndarray:
        if self.n_sim_types_ is None:
            raise RuntimeError("Missing n_sim_types_.")
        X_np = X.detach().cpu().numpy().astype(np.float64, copy=False)
        s_np = s.detach().cpu().numpy().astype(np.int64, copy=False)
        one_hot = np.zeros((s_np.shape[0], self.n_sim_types_), dtype=np.float64)
        one_hot[np.arange(s_np.shape[0]), s_np] = 1.0
        return np.concatenate([X_np, one_hot], axis=1)

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
