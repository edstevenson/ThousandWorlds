from __future__ import annotations

import math
import torch

from ._gplfr_core import GPLFRCore
from ._gplfr_weighting import build_weighting_metadata
from ._common import resolve_torch_device


GPLFR_LINEAR_TREND_CFG = {
    "enabled": True,
    "lambda": 1.0e-3,
    "design": {"intercept": False, "inputs": True, "sim_onehot": True},
}


class GPLFR:
    """Public ThousandWorlds GPLFR baseline."""

    def __init__(
        self,
        *,
        latent_dim: int = 150,
        num_training_steps: int = 2000,
        lr_Z: float = 0.1,
        lr_global: float = 0.3,
        log_every: int = 5,
        inverse_temperature: float = 0.1,
        latent_nugget: float = 0.1,
        variable_weights: str = "learned_per_group",
        output_coregionalization: str = "field_coregionalized",
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "auto",
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.num_training_steps = int(num_training_steps)
        self.lr_Z = float(lr_Z)
        self.lr_global = float(lr_global)
        self.log_every = int(log_every)
        self.inverse_temperature = float(inverse_temperature)
        self.latent_nugget = float(latent_nugget)
        self.variable_weights = str(variable_weights)
        self.output_coregionalization = str(output_coregionalization)
        if self.variable_weights not in {"fixed", "learned_per_group"}:
            raise ValueError("variable_weights must be 'fixed' or 'learned_per_group'.")
        if self.output_coregionalization not in {"none", "field_coregionalized"}:
            raise ValueError("output_coregionalization must be 'none' or 'field_coregionalized'.")
        self.dtype = dtype
        self.device = resolve_torch_device(device)
        self.model_: GPLFRCore | None = None
        self.fit_stats_: dict[str, object] | None = None

    def fit(
        self,
        X: torch.Tensor,
        s: torch.Tensor,
        Y: torch.Tensor,
        *,
        field_mask: torch.Tensor,
        sh_mask: torch.Tensor,
        field_names: list[str],
        seed: int = 0,
        n_sim_types: int | None = None,
        verbose: bool = True,
    ) -> None:
        X = X.to(device=self.device, dtype=self.dtype)
        s = s.to(device=self.device, dtype=torch.long)
        Y = Y.to(device=self.device, dtype=self.dtype)
        field_mask = field_mask.to(device=self.device, dtype=torch.bool)
        sh_mask = sh_mask.to(device=self.device, dtype=torch.bool)
        n_sim_types = int(s.max().item()) + 1 if n_sim_types is None else int(n_sim_types)
        sh_T = int(round(math.sqrt(int(Y.shape[1])) - 1))
        weighting_meta = build_weighting_metadata(fields=list(field_names), sh_T=sh_T, ref=Y)
        model = GPLFRCore(
            latent_dim=self.latent_dim,
            n_sim_types=n_sim_types,
            kernel="matern52",
            ell_mode="shared",
            sigma_f_mode="shared",
            eta_lkj=1.0,
            jitter=1.0e-6,
            inverse_temperature=self.inverse_temperature,
            nugget_noise_by_sim=[self.latent_nugget for _ in range(n_sim_types)],
            sh_T=sh_T,
            dtype=self.dtype,
            decoder_field_coreg_rank=0 if self.output_coregionalization == "none" else 1,
            decoder_field_coreg_sigma_B=0.3,
            decoder_field_coreg_sigma_logpsi=0.5,
            variable_weights=self.variable_weights,
        )
        self.model_ = model.fit(
            X,
            s,
            Y,
            weighting_meta=weighting_meta,
            method="map",
            num_training_steps=self.num_training_steps,
            lr_Z=self.lr_Z,
            lr_global=self.lr_global,
            rng_seed=int(seed),
            verbose=verbose,
            field_mask=field_mask,
            sh_mask=sh_mask,
            linear_trend_cfg=GPLFR_LINEAR_TREND_CFG,
            log_every=self.log_every,
        )
        self.fit_stats_ = dict(self.model_.train_stats_ or {})

    @torch.no_grad()
    def predict(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return self.model_.predict(
            X.to(device=self.device, dtype=self.dtype),
            s.to(device=self.device, dtype=torch.long),
            mean_only=True,
        )

    @torch.no_grad()
    def predict_samples(self, X: torch.Tensor, s: torch.Tensor, *, seed: int = 0, n_samples: int | None = None) -> torch.Tensor:
        if self.model_ is None:
            raise RuntimeError("Model not fitted.")
        return self.model_.predict(
            X.to(device=self.device, dtype=self.dtype),
            s.to(device=self.device, dtype=torch.long),
            n_post_samples=64 if n_samples is None else int(n_samples),
            mean_only=False,
            rng_seed=int(seed),
        )
