from __future__ import annotations

from typing import Optional, Union, Literal, TypedDict, Any
import math
import time
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.infer.autoguide.initialization import init_to_median, init_to_sample
import torch

from jaxtyping import Float, Int
from torch import Tensor
from ._torch_kernels import (
    apply_kernel,
    build_design_matrix,
    compute_sim_type_kernel,
    fit_ridge,
    make_generator,
    sample_randint,
    sample_randn,
    stabilize_kernel,
)
from ._gplfr_weighting import (
    WeightingMetadata,
    build_learned_group_output_weights,
)

PRIOR_CFG = {
    "ell_prior_loc": 0.0,
    "ell_prior_scale": 0.3,
    "sigma_f_prior_scale": 1.0,
    "sigma_prior_scale": 0.5,
    "log_r_sim_uncentred_scale": 0.2,
}

class InferenceSamples(TypedDict, total=False):
    """Typed dictionary for inference samples (both prior and posterior)."""
    U_T: Float[Tensor, "n_post_samples latent_dim n_train_pts"]
    Z_T: Float[Tensor, "n_post_samples latent_dim n_train_pts"]
    tau: Float[Tensor, "n_post_samples latent_dim"]
    sigma: Float[Tensor, "n_post_samples"]
    # NOTE: when ell_mode != "shared" (MAP-only), `ell` may have shape
    #   (n_post_samples, latent_dim, input_dim). For grouped ell, we store `ell_group`.
    ell: Tensor
    ell_group: Tensor
    # NOTE: sigma_f has shape (n_post_samples,) for shared/fixed amplitudes, and
    # (n_post_samples, Q) for sigma_f_mode='per_latent'/'grouped'.
    sigma_f: Tensor
    r_sim: Float[Tensor, "n_post_samples n_sim_types"]
    L_corr: Float[Tensor, "n_post_samples n_sim_types n_sim_types"]
    L_K: Float[Tensor, "n_post_samples n_train_pts n_train_pts"]
    log_alpha_group_uncentred: Float[Tensor, "n_post_samples n_field_groups"]
    B_field: Float[Tensor, "n_post_samples n_fields rank"]
    psi_field: Float[Tensor, "n_post_samples n_fields"]


class GPLFRCore:
    """
    Bayesian Gaussian Process Latent Variable Regressor.
    A GP encoder produces latents from inputs, and a collapsed linear-Gaussian decoder produces outputs from those latents under a MAP objective.

    - Places a coregionalized GP prior over latent trajectories `Z` across
      simulation types.
    - Integrates out decoder weights `W` (no explicit bias term) under Gaussian
      priors with variances governed by decoder-side scales `tau_d`, yielding
      the closed-form marginal likelihood used during inference. Outputs are
      assumed to be centred before fitting.
    - Performs posterior inference over `(Z, ell, sigma_f, L_corr, r_sim, sigma)`
      while treating `sigma_W` and `latent_scales_ratio` (which sets `tau_d`) as
      fixed hyperparameters.
    - Supports predictive queries that draw posterior predictive samples and
      optionally average them to report a mean.
    """

    def __init__(
        self,
        latent_dim: int = 10,
        n_sim_types: int = 1,
        sigma_W: float = 1.0,
        kernel: Literal["rbf", "matern32", "matern52"] = "rbf",
        latent_scales_ratio: float = 1.0,
        ell_mode: Literal["shared", "per_latent"] | int | float = "shared",
        sigma_f_mode: Literal["shared", "per_latent"] | int | float = "shared",
        learn_sigma_f: bool = False,
        eta_lkj: float = 1.0,
        jitter: float = 1e-6,
        nugget_noise: float | None = None,
        nugget_noise_by_sim: list[float] | Tensor | None = None,
        sh_T: int = 21,
        dtype: torch.dtype = torch.float64,
        decoder_field_coreg_rank: int = 1,
        decoder_field_coreg_sigma_B: float = 0.3,
        decoder_field_coreg_sigma_logpsi: float = 0.5,
        variable_weights: str = "learned_per_group",
        ell_prior_loc_override: float | None = None,
        inverse_temperature: float = 1.0,
    ) -> None:
        """
        Initialize the GP model.

        Args:
            latent_dim: Dimension of latent space
            n_sim_types: Number of simulation types
            sigma_W: Overall decoder scale
            kernel: Base GP kernel over inputs; one of `'rbf'`, `'matern32'`, `'matern52'`.
            latent_scales_ratio: Ratio `R = tau_1 / tau_{latent_dim}` controlling
                geometric decay of latent decoder scales.
            ell_mode: Lengthscale mode; `'shared'`, `'per_latent'`, integer
                group count, or float (fixed shared ell value).
            sigma_f_mode: Latent GP amplitude mode; `'shared'`, `'per_latent'`,
                integer group count, or float (fixed shared sigma_f value).
            learn_sigma_f: If True, fixes sigma_f=1.0 (disables learning latent GP amplitudes).
            eta_lkj: LKJ concentration parameter for the simulation-type correlation matrix.
            jitter: Numerical stability jitter for covariance matrices
            nugget_noise: Additive diagonal term for the latent GP covariance (K <- K + nugget_noise I),
                distinct from numerical `jitter`.
            nugget_noise_by_sim: Optional per-sim nugget (length n_sim_types); if set, `nugget_noise`
                is ignored and the per-sim base values are used instead.
            sh_T: Spherical harmonics truncation used to infer degree mapping across coefficients
            dtype: Torch dtype to use for all tensors. Default `torch.float64`; set to
                `torch.float32`.
            decoder_field_coreg_rank: Rank of the low-rank factor B in R_f = corr(BBᵀ + diag(ψ) + εI).
            decoder_field_coreg_sigma_B: Prior stddev for B entries: B_fk ~ Normal(0, sigma_B^2).
            decoder_field_coreg_sigma_logpsi: Prior stddev for log psi: log psi_f ~ Normal(0, sigma_logpsi^2),
                i.e. psi_f ~ LogNormal(0, sigma_logpsi).
            ell_prior_loc_override: Optional override for the LogNormal prior
                location used for GP lengthscales `ell`.
            inverse_temperature: Fixed multiplier on the collapsed decoder log likelihood.
        """
        self.latent_dim = latent_dim
        # The prior model uses whitened latents; fit() switches to direct latents for MAP optimization.
        self._latent_param_mode: Literal["whitened", "direct_z"] = "whitened"
        self.n_sim_types = n_sim_types
        self.kernel: Literal["rbf", "matern32", "matern52"] = kernel
        self.ell_fixed: float | None = None
        self.kernel_groups: Literal["shared", "per_latent"] | int = "shared"
        if isinstance(ell_mode, float):
            self.ell_fixed = float(ell_mode)
            self.kernel_groups = "shared"
        elif isinstance(ell_mode, int):
            self.kernel_groups = int(ell_mode)
        else:
            if str(ell_mode) == "grouped":
                raise ValueError("ell_mode='grouped' requires an integer group count (e.g. ell_mode=4).")
            self.kernel_groups = ell_mode
        if self.ell_fixed is not None and self.ell_fixed <= 0.0:
            raise ValueError("Fixed ell must be > 0.")
        self.ell_prior_loc = float(PRIOR_CFG["ell_prior_loc"] if ell_prior_loc_override is None else ell_prior_loc_override)

        self.sigma_f_fixed: float | None = None
        self._sigma_f_mode_cfg: Literal["shared", "per_latent"] | int = "shared"
        if isinstance(sigma_f_mode, float):
            self.sigma_f_fixed = float(sigma_f_mode)
            self._sigma_f_mode_cfg = "shared"
        elif isinstance(sigma_f_mode, int):
            self._sigma_f_mode_cfg = int(sigma_f_mode)
        else:
            if str(sigma_f_mode) == "grouped":
                raise ValueError("sigma_f_mode='grouped' requires an integer group count (e.g. sigma_f_mode=4).")
            self._sigma_f_mode_cfg = sigma_f_mode
        if self.sigma_f_fixed is not None and self.sigma_f_fixed <= 0.0:
            raise ValueError("Fixed sigma_f must be > 0.")
        self.sigma_f_mode: Literal["shared", "per_latent", "grouped"] = "shared"
        self.learn_sigma_f = bool(learn_sigma_f)
        self.ell_mode: Literal["shared", "per_latent", "grouped"] = "shared"
        self.ell_groups: Optional[list[list[int]]] = None
        self.eta_lkj = eta_lkj
        self.jitter = jitter
        if nugget_noise is None:
            nugget_noise = 0.0
        self.nugget_noise = float(nugget_noise)
        self._nugget_noise_by_sim: list[float] | None = None
        if nugget_noise_by_sim is not None:
            if isinstance(nugget_noise_by_sim, torch.Tensor):
                nugget_noise_by_sim = nugget_noise_by_sim.detach().cpu().tolist()
            self._nugget_noise_by_sim = [float(x) for x in nugget_noise_by_sim]
            if len(self._nugget_noise_by_sim) != int(n_sim_types):
                raise ValueError(
                    f"nugget_noise_by_sim must have length n_sim_types={int(n_sim_types)} "
                    f"(got {len(self._nugget_noise_by_sim)})."
                )
            if any(x < 0.0 for x in self._nugget_noise_by_sim):
                raise ValueError("nugget_noise_by_sim values must be >= 0.")
        self.sh_T = sh_T
        self.dtype = dtype
        self.decoder_field_coreg_rank = decoder_field_coreg_rank
        self.decoder_field_coreg_sigma_B = float(decoder_field_coreg_sigma_B)
        self.decoder_field_coreg_sigma_logpsi = float(decoder_field_coreg_sigma_logpsi)
        self.variable_weights = str(variable_weights)
        if self.variable_weights not in {"fixed", "learned_per_group"}:
            raise ValueError("variable_weights must be 'fixed' or 'learned_per_group'.")
        self.sigma_W = float(sigma_W)
        self.latent_scales_ratio = float(latent_scales_ratio)
        if self.sigma_W <= 0.0:
            raise ValueError("sigma_W must be > 0.")
        if self.latent_scales_ratio <= 0.0:
            raise ValueError("latent_scales_ratio must be > 0.")
        if self.nugget_noise < 0.0:
            raise ValueError("nugget_noise must be >= 0")

        # Passed or fit in fit()
        self.posterior_samples_: Optional[InferenceSamples] = None
        self.X_train_: Optional[Tensor] = None
        self.s_train_: Optional[Tensor] = None
        self.Y_train_: Optional[Tensor] = None
        self.fitted_: bool = False
        self.tau_: Optional[Tensor] = None
        self.weighting_meta_: Optional[WeightingMetadata] = None
        ## observation structure
        self.field_mask_: Optional[Tensor] = None           # (n_train, n_fields)
        self.sh_mask_: Optional[Tensor] = None             # (n_sh_coeffs, n_fields)
        self.example_sh_mask_: Optional[Tensor] = None     # (n_train, n_sh_coeffs)
        self._warned_missing_example_sh_mask_: bool = False

        self.train_stats_: Optional[dict[str, Any]] = None

        # Sigma_f grouping bookkeeping
        self._kernel_group_ids_: Optional[Tensor] = None  # (q,) long, values in [0, Q)
        self._kernel_num_groups_: Optional[int] = None
        self._init_kernel_groups()
        self._init_sigma_f_groups()

        # Grouped ell bookkeeping (only used when ell_mode == "grouped")
        self._ell_group_ids_: Optional[Tensor] = None  # (q,) long, values in [0, Q)
        self._ell_num_groups_: Optional[int] = None
        if self.ell_mode == "grouped":
            self._init_grouped_ell()

        # Optional in-memory cache for expensive batched Cholesky factors (MAP-only ell_mode != "shared")
        self._cached_state_: Optional[dict[int, dict[str, Tensor]]] = None
        self.inverse_temperature = float(inverse_temperature)
        if self.inverse_temperature < 0.0:
            raise ValueError("inverse_temperature must be >= 0.")

    def _nugget_noise_diag(self, s: Tensor, ref: Tensor) -> Tensor:
        if self._nugget_noise_by_sim is None:
            return ref.new_full((int(s.numel()),), float(self.nugget_noise))
        base = ref.new_tensor(self._nugget_noise_by_sim)
        return base.index_select(0, s.to(device=ref.device, dtype=torch.long))

    def _init_grouped_ell(self) -> None:
        q = int(self.latent_dim)
        groups = self.ell_groups
        if not groups:
            raise ValueError("ell_mode='grouped' requires ell_groups (list of disjoint latent-index lists).")
        flat = [int(i) for g in groups for i in g]
        if sorted(flat) != list(range(q)):
            raise ValueError(f"ell_groups must be a partition of [0..{q-1}] (0-based), got flat={sorted(flat)}")
        ids = torch.empty((q,), dtype=torch.long)
        for gi, g in enumerate(groups):
            for idx in g:
                ids[int(idx)] = int(gi)
        self._ell_group_ids_ = ids
        self._ell_num_groups_ = int(len(groups))

    def _init_kernel_groups(self) -> None:
        q = int(self.latent_dim)
        kg = self.kernel_groups
        if isinstance(kg, str):
            if kg == "shared":
                self.ell_mode = "shared"
                return
            if kg == "per_latent":
                self.ell_mode = "per_latent"
                return
            raise ValueError(f"Unknown ell_mode='{kg}' (expected 'shared', 'per_latent', or an integer number of groups).")

        G = int(kg)
        if G < 1 or G > q:
            raise ValueError(f"ell_mode group count must be in [1, latent_dim] (got {kg} for latent_dim={q}).")
        base, rem = divmod(q, G)
        if base == 0:
            raise ValueError(f"ell_mode={G} is too large for latent_dim={q}.")
        if rem:
            print(f"[WARN][GPLFR] ell_mode={G} does not divide latent_dim={q}; putting remainder ({rem}) in final group.")
        groups = [list(range(i * base, (i + 1) * base)) for i in range(G)]
        groups[-1].extend(range(G * base, q))
        self.ell_mode = "grouped"
        self.ell_groups = groups

    def _init_sigma_f_groups(self) -> None:
        q = int(self.latent_dim)
        mode = self._sigma_f_mode_cfg
        if isinstance(mode, str):
            if mode == "shared":
                self.sigma_f_mode = "shared"
                self._kernel_num_groups_ = 1
                self._kernel_group_ids_ = torch.zeros((q,), dtype=torch.long)
                return
            if mode == "per_latent":
                self.sigma_f_mode = "per_latent"
                self._kernel_num_groups_ = q
                self._kernel_group_ids_ = torch.arange(q, dtype=torch.long)
                return
            raise ValueError(f"Unknown sigma_f_mode='{mode}' (expected 'shared', 'per_latent', or an integer number of groups).")

        G = int(mode)
        if G < 1 or G > q:
            raise ValueError(f"sigma_f_mode group count must be in [1, latent_dim] (got {mode} for latent_dim={q}).")
        base, rem = divmod(q, G)
        if base == 0:
            raise ValueError(f"sigma_f_mode={G} is too large for latent_dim={q}.")
        if rem:
            print(f"[WARN][GPLFR] sigma_f_mode={G} does not divide latent_dim={q}; putting remainder ({rem}) in final group.")
        self.sigma_f_mode = "grouped"
        ids = torch.empty((q,), dtype=torch.long)
        for gi in range(G):
            lo = gi * base
            hi = (gi + 1) * base if gi < G - 1 else q
            ids[lo:hi] = int(gi)
        self._kernel_num_groups_ = G
        self._kernel_group_ids_ = ids

    def _sigma_f_latent(self, sigma_f: Tensor, ref: Tensor) -> Tensor:
        """Return per-latent sigma_f of shape (q,) (expand shared scalar if needed)."""
        q = int(self.latent_dim)
        if sigma_f.ndim == 0:
            return sigma_f.to(device=ref.device, dtype=ref.dtype).expand(q)
        if sigma_f.ndim != 1 or int(sigma_f.numel()) != q:
            raise ValueError(f"Expected sigma_f to have shape () or ({q},), got {tuple(sigma_f.shape)}.")
        return sigma_f.to(device=ref.device, dtype=ref.dtype)

    def _build_svi_map(
        self,
        *,
        lr_Z: float,
        lr_global: float,
        init_values: dict[str, Any] | None,
        guide: AutoDelta | None = None,
    ) -> tuple[AutoDelta, SVI]:
        """GPLFR-style MAP SVI builder: init_to_median for globals, init_to_sample for latents, and two-group LRs."""
        def _as_init_values(v: dict[str, Any] | None) -> dict[str, Tensor] | None:
            if v is None:
                return None
            ref = self.X_train_
            if ref is None:
                raise RuntimeError("Expected X_train_ to be set before building SVI.")
            return {k: (val if isinstance(val, Tensor) else ref.new_tensor(val)) for k, val in v.items()}

        if guide is None:
            init_t = _as_init_values(init_values)
            init_med = init_to_median()
            init_samp = init_to_sample()

            def init_loc(site: dict[str, Any]) -> Tensor:
                name = site["name"]
                if init_t is not None and name in init_t:
                    return init_t[name]
                return init_samp(site) if name in {"U_T", "Z_T"} else init_med(site)

            guide = AutoDelta(self.model, init_loc_fn=init_loc)

        lr_Z = float(lr_Z)
        lr_global = float(lr_global)

        def optim_args(param_name: str) -> dict[str, float]:
            name = param_name.rsplit(".", 1)[-1]
            return {"lr": lr_Z} if name in {"Z_T", "U_T"} else {"lr": lr_global}

        optimizer = pyro.optim.ExponentialLR({"optimizer": torch.optim.Adam, "optim_args": optim_args, "gamma": 1.0})
        return guide, SVI(self.model, guide, optimizer, loss=Trace_ELBO())

    def _latent_scales_from_ratio(self, ref: Tensor) -> Tensor:
        """
        Compute decoder prior scales τ with geometric decay controlled by `latent_scales_ratio`.

        This project previously supported a block-structured latent space; that option
        has been removed. τ is always computed for a dense latent vector.
        """
        sigma_W = ref.new_tensor(self.sigma_W)
        if self.latent_dim == 1:
            return sigma_W.view(1)

        log_ratio = torch.log(ref.new_tensor(float(self.latent_scales_ratio)))
        delta = log_ratio / (self.latent_dim - 1)
        idx = torch.arange(self.latent_dim, device=ref.device, dtype=ref.dtype)
        lambda_raw = -idx * delta
        tilde_lambda = lambda_raw - lambda_raw.mean()
        return sigma_W * torch.exp(tilde_lambda)

    def _tensorize_weighting_metadata(self, weighting_meta: WeightingMetadata, ref: Tensor) -> None:
        """
        Convert metadata to tensors.
        """
        # These are already tensors, so use .to() to move to correct device/dtype
        log_lplus1 = weighting_meta["log_lplus1"].to(dtype=ref.dtype, device=ref.device)
        field_log_pressure = weighting_meta["field_log_pressure"].to(dtype=ref.dtype, device=ref.device)
        field_group_ids = weighting_meta["field_group_ids"].to(dtype=torch.long, device=ref.device)

        self.weighting_meta_ = {
            "log_lplus1": log_lplus1,
            "field_log_pressure": field_log_pressure,
            "field_group_ids": field_group_ids,
            "original_group_ids": weighting_meta["original_group_ids"].to(dtype=torch.long, device=ref.device),
            "n_field_groups": weighting_meta["n_field_groups"],
        }

    def _get_alpha_sqrt(self, sample_idx: int, n_sh_coeffs: int, n_fields: int) -> Tensor:
        if self.variable_weights == "fixed":
            ref = next(iter(self.posterior_samples_.values()))
            return ref.new_ones((1, n_sh_coeffs, n_fields))
        log_alpha_group_uncentred = self.posterior_samples_["log_alpha_group_uncentred"][sample_idx]
        alpha = build_learned_group_output_weights(
            log_alpha_group_uncentred=log_alpha_group_uncentred,
            metadata=self.weighting_meta_,
            n_sh_coeffs=n_sh_coeffs,
        )
        return alpha.view(1, n_sh_coeffs, n_fields).sqrt()

    def _gp_kernel(self, X: Float[Tensor, "n d"], s: Int[Tensor, "n"]) -> Float[Tensor, "*batch n n"]:
        """
        Sample GP kernel parameters and compute coregionalized kernel matrix.

        Args:
            X: Input features (n_train_pts, input_dim)
            s: Simulation type indices (n_train_pts,)

        Returns:
            kernel_matrix: Coregionalized kernel (n_train_pts, n_train_pts)
        """
        n_train_pts, input_dim = X.shape

        # Kernel hyperparameters (ell may be fixed/shared/per-latent/grouped)
        if self.ell_fixed is not None:
            ell = X.new_full((input_dim,), float(self.ell_fixed)) if self.ell_mode == "shared" else X.new_full((int(self.latent_dim), input_dim), float(self.ell_fixed))
        else:
            ell_prior_loc = X.new_full((input_dim,), float(self.ell_prior_loc))
            ell_prior_scale = X.new_full((input_dim,), float(PRIOR_CFG["ell_prior_scale"]))
            if self.ell_mode == "shared":
                ell = pyro.sample(
                    "ell",
                    dist.LogNormal(ell_prior_loc, ell_prior_scale).to_event(1),
                )  # (d,)
            elif self.ell_mode == "per_latent":
                q = int(self.latent_dim)
                ell = pyro.sample(
                    "ell",
                    dist.LogNormal(ell_prior_loc, ell_prior_scale)
                    .expand([q, input_dim])
                    .to_event(2),
                )  # (q, d)
            elif self.ell_mode == "grouped":
                if self._ell_group_ids_ is None or self._ell_num_groups_ is None:
                    raise RuntimeError("Grouped ell bookkeeping not initialized.")
                Q = int(self._ell_num_groups_)
                ell_group = pyro.sample(
                    "ell_group",
                    dist.LogNormal(ell_prior_loc, ell_prior_scale)
                    .expand([Q, input_dim])
                    .to_event(2),
                )  # (Q, d)
                ell = ell_group[self._ell_group_ids_.to(device=X.device)]  # (q, d)
            else:
                raise ValueError(f"Unknown ell_mode '{self.ell_mode}' in _gp_kernel().")

        # Kernel amplitude is fixed to 1.0; latent GP amplitudes (shared/per-latent)
        # are represented via `sigma_f` applied directly to Z in model().
        kernel_x_similarities = apply_kernel(self.kernel, X, ell, X.new_tensor(1.0))

        if self.n_sim_types >= 2:
            eta = X.new_tensor(self.eta_lkj)
            L_corr = pyro.sample(
                "L_corr",
                dist.LKJCholesky(dim=self.n_sim_types, concentration=eta),
            )
        else:
            L_corr = X.new_ones((1, 1))
            pyro.deterministic("L_corr", L_corr)

        log_r_sim_uncentred = pyro.sample(
            "log_r_sim_uncentred",
            dist.Normal(
                X.new_zeros(self.n_sim_types),
                X.new_full(
                    (self.n_sim_types,), float(PRIOR_CFG["log_r_sim_uncentred_scale"])
                ),
            ).to_event(1),
        )
        log_r_sim = log_r_sim_uncentred - log_r_sim_uncentred.mean()
        r_sim = torch.exp(log_r_sim)

        kernel_sim_type_correlations = compute_sim_type_kernel(L_corr, r_sim)

        # Combine into coregionalized kernel
        kernel_matrix = kernel_x_similarities * kernel_sim_type_correlations[s.unsqueeze(1), s.unsqueeze(0)]
        eye = torch.eye(kernel_matrix.shape[-1], device=X.device, dtype=X.dtype)
        nugget_diag = self._nugget_noise_diag(s, X)
        kernel_matrix = kernel_matrix + eye * nugget_diag if kernel_matrix.ndim == 2 else kernel_matrix + eye.unsqueeze(0) * nugget_diag
        return stabilize_kernel(kernel_matrix, self.jitter)

    @staticmethod
    def _unsqueeze_inputs(
        X: Union[Float[Tensor, "d"], Float[Tensor, "n d"]],
        s: Union[int, Int[Tensor, ""], Int[Tensor, "n"]],
    ) -> tuple[Float[Tensor, "n d"], Int[Tensor, "n"]]:
        """
        Normalize inputs to batched form (n, d) and (n,).
        """
        # Unsqueeze X if 1D
        if X.dim() == 1:
            X = X.unsqueeze(0)

        # Handle s: int -> tensor, 0D -> 1D
        if isinstance(s, int):
            s = X.new_tensor([s], dtype=torch.long)
        elif s.dim() == 0:
            s = s.unsqueeze(0)

        return X, s

    def _example_sh_mask_or_none(self, *, ref: Tensor) -> Optional[Tensor]:
        """
        Return per-example SH mask on `ref` device/dtype.
        """
        if not hasattr(self, "example_sh_mask_"):
            if not getattr(self, "_warned_missing_example_sh_mask_", False):
                print(
                    "\033[33m[WARN][example_sh_mask] Loaded checkpoint has no 'example_sh_mask_'. "
                    "Falling back to all-True mask (no per-example SH masking).\033[0m"
                )
                self._warned_missing_example_sh_mask_ = True
            return None
        example_sh_mask = self.example_sh_mask_
        return (
            example_sh_mask.to(device=ref.device, dtype=torch.bool)
            if example_sh_mask is not None
            else None
        )

    def _unpack_posterior_state(
        self,
        sample_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor, list[dict[str, Tensor]], None, Tensor, Tensor, Tensor]:
        """
        Build training kernel matrix, decoder posterior, and noise for a posterior sample.

        Returns:
            L_K: Cholesky factor of K_train (N, N)
            Ks: Sim-type kernel matrix (S, S)
            Z_train: Latent trajectories at training points (N, q)
            mu_post: Field-coreg decoder posterior state.
            Sigma_post: Always None for the field-coreg decoder.
            sigma: Noise sigma (scalar or n_outputs vector)
            ell: RBF length scales (input_dim,)
            sigma_f: Latent GP amplitude (scalar or (q,))
        """
        X_train = self.X_train_
        s_train = self.s_train_
        Y_train = self.Y_train_
        _, n_sh_coeffs, n_fields = Y_train.shape

        # Extract/rebuild posterior hyperparameters
        if self.ell_fixed is not None:
            d = int(X_train.shape[1])
            ell = X_train.new_full((d,), float(self.ell_fixed)) if self.ell_mode == "shared" else X_train.new_full((int(self.latent_dim), d), float(self.ell_fixed))
        elif self.ell_mode == "grouped":
            if "ell_group" not in self.posterior_samples_:
                raise KeyError("Expected 'ell_group' in posterior_samples_ for ell_mode='grouped'.")
            if self._ell_group_ids_ is None:
                raise RuntimeError("Grouped ell bookkeeping not initialized.")
            ell_group = self.posterior_samples_["ell_group"][sample_idx]  # (Q, d)
            ell = ell_group[self._ell_group_ids_.to(device=X_train.device)]  # (q, d)
        else:
            ell = self.posterior_samples_["ell"][sample_idx]

        if self.sigma_f_fixed is not None:
            sigma_f = X_train.new_tensor(float(self.sigma_f_fixed))
        elif self.learn_sigma_f:
            sigma_f = X_train.new_tensor(1.0)
        else:
            sigma_f_raw = self.posterior_samples_["sigma_f"][sample_idx]
            if sigma_f_raw.ndim == 0 or int(sigma_f_raw.numel()) == int(self.latent_dim):
                sigma_f = sigma_f_raw
            else:
                ids = self._kernel_group_ids_
                if ids is None:
                    raise RuntimeError("Kernel group ids not initialized.")
                sigma_f = sigma_f_raw[ids.to(device=X_train.device)]
        sigma_f = self._sigma_f_latent(sigma_f, X_train)
        r_sim = self.posterior_samples_["r_sim"][sample_idx]
        if "L_corr" in self.posterior_samples_:
            L_corr = self.posterior_samples_["L_corr"][sample_idx]
        else:
            if self.n_sim_types == 1:
                L_corr = r_sim.new_ones((1, 1))
            else:
                raise KeyError("Expected 'L_corr' in posterior_samples_ for n_sim_types > 1.")
        Ks = compute_sim_type_kernel(L_corr, r_sim)

        # Training kernel Cholesky: stored for shared-ell; computed+cached for per_latent/grouped (MAP-only).
        if self.ell_mode == "shared":
            L_K = self.posterior_samples_["L_K"][sample_idx]
        else:
            if self._cached_state_ is None:
                self._cached_state_ = {}
            cached = self._cached_state_.get(sample_idx)
            if cached is not None and "L_K" in cached:
                L_K = cached["L_K"]
            else:
                Kx = apply_kernel(self.kernel, X_train, ell, X_train.new_tensor(1.0))  # (N,N) or (q,N,N)
                K_train = Kx * Ks[s_train.unsqueeze(1), s_train.unsqueeze(0)]
                eye = torch.eye(K_train.shape[-1], device=X_train.device, dtype=X_train.dtype)
                nugget_diag = self._nugget_noise_diag(s_train, X_train)
                K_train = K_train + eye * nugget_diag if K_train.ndim == 2 else K_train + eye.unsqueeze(0) * nugget_diag
                L_K = torch.linalg.cholesky(stabilize_kernel(K_train, self.jitter))
                self._cached_state_[sample_idx] = {"L_K": L_K}

        # Recover latent Z_train from the stored direct or whitened latent state.
        if "Z_T" in self.posterior_samples_:
            Z_train = self.posterior_samples_["Z_T"][sample_idx].T  # (N, q)
        else:
            U_T = self.posterior_samples_["U_T"][sample_idx]  # (q, N)
            if L_K.dim() == 2:
                Z_train = (U_T @ L_K.T).T  # (N, q)
            else:
                Z_train = torch.bmm(L_K, U_T.unsqueeze(-1)).squeeze(-1).T  # (N, q)
        Z_train = Z_train * sigma_f.unsqueeze(0)

        # Base noise sigma (shared across outputs)
        sigma = self.posterior_samples_["sigma"][sample_idx].to(X_train)

        # Effective training outputs for decoder posterior
        Y_scaled = Y_train * self._get_alpha_sqrt(sample_idx, n_sh_coeffs, n_fields)

        # Compute decoder posterior given (Z_train, Y_scaled)
        tau = self.posterior_samples_["tau"][sample_idx].to(X_train)
        B = self.posterior_samples_["B_field"][sample_idx]
        psi = self.posterior_samples_["psi_field"][sample_idx].clamp_min(0.0)
        mu_post = self._compute_decoder_posterior_field_coreg(
            Z_train * tau.unsqueeze(0), Y_scaled, sigma, B_field=B, psi_field=psi, eps=1e-6
        )
        Sigma_post = None

        return L_K, Ks, Z_train, mu_post, Sigma_post, sigma, ell, sigma_f

    def _field_coreg_prec_terms(
        self,
        B_field: Tensor,   # (F, k)
        psi_field: Tensor, # (F,)
        *,
        eps: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Build a diag-minus-lowrank representation of the field correlation precision.

        We define:
          Σ = B Bᵀ + diag(ψ) + eps I
          R = corr(Σ) = D^{-1/2} Σ D^{-1/2} = diag(r) + C Cᵀ
        where D = diag(Σ), C = B / sqrt(D), and r = (ψ + eps) / D so that diag(R)=1.

        Then:
          R^{-1} = diag(d) - U Uᵀ
        where d = 1/r and U is (F, k).

        Returns:
          d: (F,) diagonal of the base precision term
          U: (F, k) low-rank factor in the precision
          logdet_R: scalar
        """
        F, k = int(B_field.shape[0]), int(B_field.shape[1])
        if psi_field.dim() != 1 or psi_field.shape[0] != F:
            raise ValueError(f"psi_field must have shape (F,), got {tuple(psi_field.shape)}")

        diag_Sigma = (B_field**2).sum(dim=1) + psi_field + eps  # (F,)
        C = B_field / torch.sqrt(diag_Sigma).unsqueeze(1)       # (F, k)
        r = (psi_field + eps) / diag_Sigma                      # (F,)
        d = 1.0 / r                                             # (F,)

        if k == 0:
            logdet_R = torch.sum(torch.log(r))
            return d, B_field, logdet_R  # empty U with shape (F, 0)

        # logdet(R) = logdet(diag(r)) + logdet(I + Cᵀ diag(d) C)
        eye_k = torch.eye(k, device=B_field.device, dtype=B_field.dtype)
        S = eye_k + C.T @ (C * d.unsqueeze(1))  # (k, k)
        L_S = torch.linalg.cholesky(S)
        logdet_S = 2.0 * torch.sum(torch.log(torch.diagonal(L_S)))
        logdet_R = torch.sum(torch.log(r)) + logdet_S

        # R^{-1} = diag(d) - (diag(d) C S^{-1/2}) (diag(d) C S^{-1/2})ᵀ
        DinvC = C * d.unsqueeze(1)  # (F, k)
        U = torch.linalg.solve_triangular(L_S, DinvC.T, upper=False).T  # (F, k)
        return d, U, logdet_R

    def _collapsed_ll_field_coreg(
        self,
        Y: Tensor,   # (N, A, F)
        Z: Tensor,   # (N, q)
        sigma: Union[Tensor, float],
        *,
        tau: Tensor,      # (q,)
        B_field: Tensor,  # (F, k)
        psi_field: Tensor,  # (F,)
        eps: float,
    ) -> Tensor:
        """
        Collapsed log-likelihood for the low-rank + diagonal field-coregionalized decoder.

        Supports:
          - whole-field missingness via `self.field_mask_` (N, F)
          - static coefficient/field masking via `self.sh_mask_` (A, F) (equatorial symmetry/antisymmetry)
          - optional per-example coefficient masking via `self.example_sh_mask_` (N, A)

        Note: data loaders write missing fields as zeros in Y and expose missingness via `field_mask`;
        this routine treats `field_mask=False` and `sh_mask=False` entries as unobserved, so those
        zeros never contribute to the likelihood.
        """
        if isinstance(sigma, (int, float)):
            sigma = Y.new_tensor(sigma)
        N, A, F = Y.shape
        sigma_sq = sigma**2 + self.jitter
        q = int(Z.shape[1])
        V = Z * tau.unsqueeze(0)  # (N, q)

        # Group coefficients by active-field subsets induced by sh_mask_ (symmetry)
        if self.sh_mask_ is None:
            groups = [(torch.arange(A, device=Y.device), torch.arange(F, device=Y.device))]
        else:
            pattern_map: dict[bytes, tuple[Tensor, list[int]]] = {}
            for a in range(A):
                mask_a = self.sh_mask_[a]  # (F,)
                key = mask_a.to(dtype=torch.uint8).cpu().numpy().tobytes()
                if key not in pattern_map:
                    pattern_map[key] = (mask_a, [])
                pattern_map[key][1].append(a)
            groups = []
            for mask_fields, a_list in pattern_map.values():
                if not mask_fields.any():
                    continue
                f_idxs = torch.nonzero(mask_fields, as_tuple=False).flatten()
                a_idxs = Y.new_tensor(a_list, dtype=torch.long)
                groups.append((a_idxs, f_idxs))

        # Whole-field missingness mask (per field) and optional per-example SH mask.
        all_true = torch.ones((N,), dtype=torch.bool, device=Y.device)
        field_mask = self.field_mask_.to(device=Y.device, dtype=torch.bool) if self.field_mask_ is not None else None
        example_sh_mask = self._example_sh_mask_or_none(ref=Y)
        if example_sh_mask is not None:
            expanded_groups: list[tuple[Tensor, Tensor]] = []
            for a_idxs, f_idxs in groups:
                for a in a_idxs.tolist():
                    expanded_groups.append((Y.new_tensor([a], dtype=torch.long), f_idxs))
            groups = expanded_groups

        log_2pi = torch.log(Y.new_tensor(2.0 * torch.pi))
        eye_q = torch.eye(q, device=Y.device, dtype=Y.dtype)
        total_ll = Y.new_tensor(0.0)

        for a_idxs, f_idxs in groups:
            A_g = int(a_idxs.numel())
            F_g = int(f_idxs.numel())
            if A_g == 0 or F_g == 0:
                continue

            B_g = B_field[f_idxs, :]
            psi_g = psi_field[f_idxs]
            d_g, U_g, logdet_R_g = self._field_coreg_prec_terms(B_g, psi_g, eps=eps)
            k = int(U_g.shape[1])

            # Per-field blocks (group-by-missingness pattern to reuse V_Iᵀ V_I)
            b_stack = Y.new_zeros((F_g, q, A_g))
            x0_stack = Y.new_zeros((F_g, q, A_g))
            L_A_stack = Y.new_zeros((F_g, q, q))
            y_sq = Y.new_zeros((A_g,))
            logdet_Lambda0 = Y.new_tensor(0.0)
            M = 0

            pattern_map_fields: dict[bytes, tuple[Tensor, list[int]]] = {}
            a0 = int(a_idxs[0].item()) if example_sh_mask is not None else -1
            for j, f in enumerate(f_idxs.tolist()):
                I_f = field_mask[:, f] if field_mask is not None else all_true
                if example_sh_mask is not None:
                    I_f = I_f & example_sh_mask[:, a0]
                key = I_f.to(dtype=torch.uint8).cpu().numpy().tobytes()
                if key not in pattern_map_fields:
                    pattern_map_fields[key] = (I_f, [])
                pattern_map_fields[key][1].append(j)

            for I_mask, j_list in pattern_map_fields.values():
                n_I = int(I_mask.sum().item())
                if n_I == 0:
                    continue

                V_I = V[I_mask]  # (n_I, q)
                S_I = V_I.T @ V_I  # (q, q)

                j_idx = Y.new_tensor(j_list, dtype=torch.long)
                f_sub = f_idxs.index_select(0, j_idx)
                F_p = int(f_sub.numel())
                M += n_I * F_p

                Y_I = Y[I_mask][:, a_idxs][:, :, f_sub]  # (n_I, A_g, F_p)
                y_sq = y_sq + (Y_I**2).sum(dim=0).sum(dim=1)

                t = (V_I.T @ Y_I.reshape(n_I, A_g * F_p)).view(q, A_g, F_p)
                b = (1.0 / sigma_sq) * t  # (q, A_g, F_p)

                d_p = d_g.index_select(0, j_idx)  # (F_p,)
                A = d_p.view(F_p, 1, 1) * eye_q + (1.0 / sigma_sq) * S_I  # (F_p, q, q)
                L_A = torch.linalg.cholesky(A)
                L_A_stack.index_copy_(0, j_idx, L_A)
                logdet_Lambda0 = logdet_Lambda0 + 2.0 * torch.sum(
                    torch.log(torch.diagonal(L_A, dim1=-2, dim2=-1))
                )

                b_f = b.permute(2, 0, 1)  # (F_p, q, A_g)
                x0 = torch.cholesky_solve(b_f, L_A)  # (F_p, q, A_g)
                b_stack.index_copy_(0, j_idx, b_f)
                x0_stack.index_copy_(0, j_idx, x0)

            bTx0 = torch.einsum("fqa,fqa->a", b_stack, x0_stack)  # (A_g,)

            # Build K = I - U_qᵀ Λ0^{-1} U_q, where U_q = U ⊗ I_q
            if k == 0:
                logdet_Lambda = logdet_Lambda0
                bTx = bTx0
            else:
                kq = k * q
                Ainv_stack = torch.cholesky_solve(eye_q.expand(F_g, -1, -1), L_A_stack)  # (F_g, q, q)
                K_blocks = -torch.einsum("fr,fs,fij->rsij", U_g, U_g, Ainv_stack)  # (k, k, q, q)
                diag_idx = torch.arange(k, device=Y.device)
                K_blocks[diag_idx, diag_idx] = K_blocks[diag_idx, diag_idx] + eye_q
                K = K_blocks.permute(0, 2, 1, 3).reshape(kq, kq)
                K = K + torch.eye(kq, device=Y.device, dtype=Y.dtype) * self.jitter
                L_K = torch.linalg.cholesky(K)
                logdet_K = 2.0 * torch.sum(torch.log(torch.diagonal(L_K)))
                logdet_Lambda = logdet_Lambda0 + logdet_K

                g = torch.einsum("fk,fqa->kqa", U_g, x0_stack).reshape(kq, A_g)
                h = torch.cholesky_solve(g, L_K)
                bTx = bTx0 + torch.sum(g * h, dim=0)

            const_per_coeff = M * (log_2pi + torch.log(sigma_sq)) + logdet_Lambda + q * logdet_R_g
            total_ll = total_ll + (-0.5) * (
                A_g * const_per_coeff + torch.sum(y_sq / sigma_sq - bTx)
            )

        return total_ll

    def model(self, Y: Float[Tensor, "n a f"], X: Float[Tensor, "n d"], s: Int[Tensor, "n"]) -> None:
        """Pyro model defining the latent-GP prior and collapsed decoder likelihood.

        Draws GP hyperparameters, samples latent functions `Z`, and factors in
        the marginal log-likelihood `log p(Y | Z, sigma, sigma_W)` that results
        from integrating out the decoder weights.

        Args:
            Y: Observations `(n_train_pts, n_sh_coeffs, n_fields)`.
            X: Input features `(n_train_pts, input_dim)`.
            s: Simulation type indices `(n_train_pts,)`.
        """
        n_train_pts, _, _ = Y.shape

        # Sample GP latents via whitening or direct GP sampling
        kernel_matrix = self._gp_kernel(X, s)
        L_K = torch.linalg.cholesky(kernel_matrix)
        if self._latent_param_mode == "whitened":
            # Whitened: u ~ N(0, I), z = L_K u
            if L_K.dim() == 2:
                with pyro.plate("latent_dim", self.latent_dim):
                    U_T = pyro.sample(
                        "U_T",
                        dist.Normal(Y.new_tensor(0.0), Y.new_tensor(1.0)).expand([n_train_pts]).to_event(1),
                    )  # (q, n_train_pts)
                Z = (U_T @ L_K.T).T  # (n_train_pts, q)
            else:
                # Batched kernel per latent/group: sample U_T with explicit batch dim
                q = int(self.latent_dim)
                U_T = pyro.sample(
                    "U_T",
                    dist.Normal(Y.new_zeros((q, n_train_pts)), Y.new_ones((q, n_train_pts))).to_event(2),
                )  # (q, n_train_pts)
                Z_T = torch.bmm(L_K, U_T.unsqueeze(-1)).squeeze(-1)  # (q, n_train_pts)
                Z = Z_T.T  # (n_train_pts, q)
        elif self._latent_param_mode == "direct_z":
            # Direct: z ~ N(0, K) per latent dimension
            if L_K.dim() == 2:
                with pyro.plate("latent_dim", self.latent_dim):
                    Z_T = pyro.sample(
                        "Z_T",
                        dist.MultivariateNormal(loc=Y.new_zeros(n_train_pts), scale_tril=L_K),
                    )  # (q, n_train_pts)
                Z = Z_T.T  # (n_train_pts, q)
            else:
                # Batched kernel per latent/group: sample batched MVN directly
                q = int(self.latent_dim)
                Z_T = pyro.sample(
                    "Z_T",
                    dist.MultivariateNormal(loc=Y.new_zeros((q, n_train_pts)), scale_tril=L_K).to_event(1),
                )  # (q, n_train_pts)
                Z = Z_T.T  # (n_train_pts, q)
        else:
            raise ValueError(
                f"Unknown _latent_param_mode '{self._latent_param_mode}' in model(). Expected 'whitened' or 'direct_z'."
            )

        # Latent GP amplitudes: represent the GP variance via a multiplicative scale on Z.
        if self.sigma_f_fixed is not None:
            Z = Z * Y.new_tensor(float(self.sigma_f_fixed))
        elif not self.learn_sigma_f:
            Q = int(self._kernel_num_groups_ or 0)
            if Q <= 0:
                raise RuntimeError("Kernel group bookkeeping not initialized.")
            if Q == 1:
                sigma_f = pyro.sample(
                    "sigma_f",
                    dist.LogNormal(
                        Y.new_zeros(()),
                        Y.new_tensor(float(PRIOR_CFG["sigma_f_prior_scale"])),
                    ),
                )
                Z = Z * sigma_f
            else:
                sigma_f = pyro.sample(
                    "sigma_f",
                    dist.LogNormal(
                        Y.new_zeros((Q,)),
                        Y.new_full((Q,), float(PRIOR_CFG["sigma_f_prior_scale"])),
                    ).to_event(1),
                )  # (Q,)
                ids = self._kernel_group_ids_
                if ids is None:
                    raise RuntimeError("Kernel group ids not initialized.")
                sigma_f_latent = sigma_f[ids.to(device=sigma_f.device)]  # (q,)
                Z = Z * sigma_f_latent.unsqueeze(0)

        # Output noise & learned group weighting
        _, n_sh_coeffs, n_fields = Y.shape
        sigma_scale = Y.new_tensor(float(PRIOR_CFG["sigma_prior_scale"]))
        sigma = pyro.sample("sigma", dist.HalfNormal(sigma_scale))
        if self.variable_weights == "learned_per_group":
            log_alpha_group_uncentred = pyro.sample(
                "log_alpha_group_uncentred",
                dist.Normal(
                    Y.new_zeros(self.weighting_meta_["n_field_groups"]),
                    Y.new_full((self.weighting_meta_["n_field_groups"],), 0.1),
                ).to_event(1),
            )
            alpha = build_learned_group_output_weights(
                log_alpha_group_uncentred=log_alpha_group_uncentred,
                metadata=self.weighting_meta_,
                n_sh_coeffs=n_sh_coeffs,
            )
            Y_eff = Y * alpha.view(1, n_sh_coeffs, n_fields).sqrt()
        else:
            Y_eff = Y
        rank = self.decoder_field_coreg_rank
        B_field = pyro.sample(
            "B_field",
            dist.Normal(
                Y_eff.new_zeros(n_fields, rank),
                Y_eff.new_full((n_fields, rank), float(self.decoder_field_coreg_sigma_B))
            ).to_event(2),
        )
        psi_field = pyro.sample(
            "psi_field",
            dist.LogNormal(
                Y_eff.new_zeros(n_fields),
                Y_eff.new_full((n_fields,), float(self.decoder_field_coreg_sigma_logpsi)),
            ).to_event(1),
        )
        collapsed_ll = self._collapsed_ll_field_coreg(
            Y_eff,
            Z,
            sigma,
            tau=self.tau_,
            B_field=B_field,
            psi_field=psi_field,
            eps=1e-6,
        )
        beta = Y.new_tensor(float(self.inverse_temperature))
        pyro.factor("collapsed_log_likelihood", collapsed_ll * beta)

    def fit(
        self,
        X: Float[Tensor, "n d"],
        s: Int[Tensor, "n"],
        Y: Float[Tensor, "n a f"],
        weighting_meta: WeightingMetadata,
        method: Literal["map"] = "map",
        num_steps: int = 1000,
        lr: float = 1e-2,
        lr_Z: Optional[float] = None,
        lr_global: Optional[float] = None,
        ell_init: str | float | None = None,
        ell_init_manual: float | None = None,
        rng_seed: Optional[int] = None,
        verbose: bool = True,
        field_mask: Optional[Tensor] = None,
        sh_mask: Optional[Tensor] = None,
        example_sh_mask: Optional[Tensor] = None,
        num_training_steps: Optional[int] = None,
        linear_trend_cfg: dict[str, Any] | None = None,
        log_every: int = 100,
    ):
        """
        Fit the promoted MAP GPLFR recipe.
        """
        if rng_seed is not None:
            pyro.set_rng_seed(rng_seed)
        if method != "map":
            raise ValueError("Only method='map' is supported.")

        self._latent_param_mode = "direct_z"
        self._cached_state_ = None
        X = X.to(dtype=self.dtype)
        Y = Y.to(dtype=self.dtype)
        device = X.device
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device)
            print(f"Training on: GPU {device.index} - {gpu_name}")
        elif device.type == "xpu":
            name_fn = getattr(getattr(torch, "xpu", None), "get_device_name", None)
            xpu_name = name_fn(device) if callable(name_fn) else "XPU"
            print(f"Training on: XPU {device.index} - {xpu_name}")
        else:
            print(f"Training on: CPU")

        s = s.to(device=device, dtype=torch.long)

        ## Prepare training data
        self.X_train_ = X
        self.s_train_ = s

        ## Optional linear trend/mean in output space: subtract before weighting
        lt_cfg = linear_trend_cfg or {}
        if lt_cfg.get("enabled", False):
            design_in = lt_cfg.get("design", {}) or {}
            design_cfg = {
                "intercept": design_in.get("intercept", True),
                "inputs": design_in.get("inputs", True),
                "sim_onehot": design_in.get("sim_onehot", False),
            }
            lambda_reg = lt_cfg.get("lambda", 1.0e-3)
            fm = field_mask.to(device=Y.device, dtype=torch.bool) if field_mask is not None else None
            cm = example_sh_mask.to(device=Y.device, dtype=torch.bool) if example_sh_mask is not None else None
            sm = sh_mask.to(device=Y.device, dtype=torch.bool) if sh_mask is not None else None
            H = build_design_matrix(X, s, n_sim_types=self.n_sim_types, design_cfg=design_cfg)
            Gamma = fit_ridge(H, Y, lambda_reg=lambda_reg, field_mask=fm, coeff_mask=cm, sh_mask=sm)
            Y = Y - torch.einsum("np,paf->naf", H, Gamma)
            if fm is not None:
                Y = torch.where(fm.unsqueeze(1), Y, Y.new_zeros(()))

            design_cols = []
            if design_cfg["intercept"]:
                design_cols.append("intercept")
            if design_cfg["inputs"]:
                design_cols += [f"x{i}" for i in range(X.shape[1])]
            if design_cfg["sim_onehot"]:
                start = 1 if design_cfg["intercept"] else 0
                design_cols += [f"sim_{k}" for k in range(start, self.n_sim_types)]

            self.linear_trend_ = {
                "Gamma": Gamma,
                "design_cfg": design_cfg,
                "lambda_reg": lambda_reg,
                "P": H.shape[1],
                "design_cols": design_cols,
            }

            # DEBUG: print fitted linear trend stats
            with torch.no_grad():
                g = Gamma.detach()
                print(f"[linear_trend] enabled: lambda={lambda_reg:.3e} | design={design_cols}")
                print(f"[linear_trend] Gamma shape={tuple(g.shape)} | mean={float(g.mean()):.3e} | std={float(g.std()):.3e} | maxabs={float(g.abs().max()):.3e}")
        else:
            self.linear_trend_ = None

        self._tensorize_weighting_metadata(weighting_meta, ref=Y)
        self.Y_train_ = Y
        self.tau_ = self._latent_scales_from_ratio(ref=X)

        if torch.isnan(self.Y_train_).any():
            raise ValueError(
                "NaNs detected in Y_train_ inside GPLFRCore.fit; "
                "missing outputs must be handled via field_mask/sh_mask "
                "before calling GPLFRCore."
            )

        # Store observation masks on the correct device/dtype
        if field_mask is not None:
            self.field_mask_ = field_mask.to(device=Y.device, dtype=torch.bool)
        else:
            self.field_mask_ = None

        if sh_mask is not None:
            self.sh_mask_ = sh_mask.to(device=Y.device, dtype=torch.bool)
        else:
            self.sh_mask_ = None

        if example_sh_mask is not None:
            esm = example_sh_mask.to(device=Y.device, dtype=torch.bool)
            if esm.shape != (Y.shape[0], Y.shape[1]):
                raise ValueError(
                    f"example_sh_mask must have shape {(Y.shape[0], Y.shape[1])}, got {tuple(esm.shape)}."
                )
            self.example_sh_mask_ = esm
        else:
            self.example_sh_mask_ = None

        t0 = time.time()
        # ``lr`` is only a fallback for direct GPLFRCore callers; the public
        # GPLFR recipe passes ``lr_Z`` and ``lr_global`` explicitly.
        lr_Z_eff = float(lr if lr_Z is None else lr_Z)
        lr_global_eff = float(lr if lr_global is None else lr_global)
        init_values = None
        if ell_init is not None:
            d = int(X.shape[1])
            if isinstance(ell_init, (int, float)):
                ell0 = float(ell_init)
            else:
                mode = str(ell_init).strip().lower().replace("-", "_")
                if mode == "manual":
                    if ell_init_manual is None:
                        raise ValueError("ell_init='manual' requires ell_init_manual to be set.")
                    ell0 = float(ell_init_manual)
                elif mode == "sqrt2d":
                    ell0 = float(math.sqrt(2.0 * float(d)))
                elif mode == "median_pairwise_dist":
                    n = int(X.shape[0])
                    if n < 2:
                        ell0 = 1.0
                    else:
                        m = min(20_000, n * (n - 1) // 2)
                        g = torch.Generator().manual_seed(int(0 if rng_seed is None else rng_seed))
                        i = torch.randint(0, n, (m,), generator=g).to(device=X.device)
                        j = torch.randint(0, n, (m,), generator=g).to(device=X.device)
                        mask = i != j
                        dist = (X[i[mask]] - X[j[mask]]).square().sum(dim=1).sqrt()
                        ell0 = 1.0 if dist.numel() == 0 else float(dist.median().item())
                else:
                    raise ValueError(
                        f"Unknown ell_init={ell_init!r} (expected null, 'median_pairwise_dist', 'sqrt2d', 'manual', or a float)."
                    )
            init_values = {}
            if self.ell_mode == "grouped":
                if self._ell_num_groups_ is None:
                    raise RuntimeError("Grouped ell bookkeeping not initialized.")
                init_values["ell_group"] = X.new_full((int(self._ell_num_groups_), d), ell0)
            elif self.ell_mode == "shared":
                init_values["ell"] = X.new_full((d,), ell0)
            else:
                init_values["ell"] = X.new_full((int(self.latent_dim), d), ell0)

        pyro.clear_param_store()
        loss_history = []
        last_loss = None
        log_every = int(log_every)
        if log_every <= 0:
            raise ValueError(f"log_every must be >= 1 (got {log_every}).")

        total_steps = int(num_steps if num_training_steps is None else num_training_steps)
        if total_steps < 1:
            raise ValueError(f"num_training_steps must be >= 1 (got {num_training_steps}).")

        def step_lr(svi_obj: SVI) -> None:
            stepper = getattr(svi_obj.optim, "step", None)
            if callable(stepper):
                stepper()

        guide, svi = self._build_svi_map(
            lr_Z=lr_Z_eff,
            lr_global=lr_global_eff,
            init_values=init_values,
        )
        for step in range(total_steps):
            loss = float(svi.step(Y, X, s))
            last_loss = loss
            step_lr(svi)
            if step % log_every and step != total_steps - 1:
                continue
            row: dict[str, Any] = {"step": step, "loss": loss}
            loss_history.append(row)
            if verbose:
                print(f"Step {step}/{total_steps}, Loss: {loss:.4f}")

        losses = [entry["loss"] for entry in loss_history]
        train_time = time.time() - t0

        params = guide()
        samples_dict = {
            name: value.unsqueeze(0)
            for name, value in params.items()
            if isinstance(value, torch.Tensor)
        }

        if self.ell_fixed is not None:
            d = int(X.shape[1])
            if self.ell_mode == "shared":
                samples_dict["ell"] = X.new_full((1, d), float(self.ell_fixed))
            elif self.ell_mode == "grouped":
                if self._ell_group_ids_ is None or self._ell_num_groups_ is None:
                    raise RuntimeError("Grouped ell bookkeeping not initialized.")
                ell_group = X.new_full((int(self._ell_num_groups_), d), float(self.ell_fixed))
                samples_dict["ell_group"] = ell_group.unsqueeze(0)
                samples_dict["ell"] = ell_group[self._ell_group_ids_.to(device=X.device)].unsqueeze(0)
            else:
                samples_dict["ell"] = X.new_full((1, int(self.latent_dim), d), float(self.ell_fixed))
        if self.ell_mode == "grouped" and "ell" not in samples_dict:
            if "ell_group" not in samples_dict or self._ell_group_ids_ is None:
                raise RuntimeError("Grouped ell bookkeeping not initialized.")
            ell_group = samples_dict["ell_group"].squeeze(0)
            samples_dict["ell"] = ell_group[self._ell_group_ids_.to(device=X.device)].unsqueeze(0)
        if "log_r_sim_uncentred" in samples_dict:
            centered = samples_dict["log_r_sim_uncentred"] - samples_dict["log_r_sim_uncentred"].mean(dim=-1, keepdim=True)
            samples_dict["r_sim"] = torch.exp(centered)
            del samples_dict["log_r_sim_uncentred"]
        if "L_corr" not in samples_dict:
            ref = next(iter(samples_dict.values()))
            samples_dict["L_corr"] = ref.new_ones((1, 1, 1))
        if self.ell_mode == "shared" and "L_K" not in samples_dict:
            ell = samples_dict["ell"].squeeze(0)
            r_sim = samples_dict["r_sim"].squeeze(0)
            L_corr = samples_dict["L_corr"].squeeze(0)
            kernel_x = apply_kernel(self.kernel, X, ell, X.new_tensor(1.0))
            kernel_sim = compute_sim_type_kernel(L_corr, r_sim)
            kernel_matrix = kernel_x * kernel_sim[s.unsqueeze(1), s.unsqueeze(0)]
            eye = torch.eye(kernel_matrix.shape[-1], device=X.device, dtype=X.dtype)
            nugget_diag = self._nugget_noise_diag(s, X)
            kernel_matrix = kernel_matrix + eye * nugget_diag if kernel_matrix.ndim == 2 else kernel_matrix + eye.unsqueeze(0) * nugget_diag
            samples_dict["L_K"] = torch.linalg.cholesky(stabilize_kernel(kernel_matrix, self.jitter)).unsqueeze(0)
        if self.sigma_f_fixed is not None:
            samples_dict["sigma_f"] = X.new_full((1,), float(self.sigma_f_fixed))
        if "sigma_f" not in samples_dict:
            samples_dict["sigma_f"] = X.new_ones((1,))
        samples_dict["tau"] = self.tau_.view(1, -1)
        self.posterior_samples_ = samples_dict
        self.train_stats_ = {
            "method": "map",
            "num_training_steps": total_steps,
            "lr": lr,
            "lr_Z": lr_Z_eff,
            "lr_global": lr_global_eff,
            "train_time_seconds": train_time,
            "inverse_temperature": float(self.inverse_temperature),
            "nugget_noise_by_sim": None if self._nugget_noise_by_sim is None else [float(x) for x in self._nugget_noise_by_sim],
            "loss_history": loss_history,
            "final_loss": last_loss if last_loss is not None else (losses[-1] if losses else None),
            "min_loss": min(losses) if losses else None,
        }

        self.fitted_ = True
        return self

    # --- Prediction ---
    def predict(
        self,
        X_new: Union[Float[Tensor, "d"], Float[Tensor, "n_test d"]],
        s_new: Union[int, Int[Tensor, "n_test"]],
        n_post_samples: Optional[int] = None,
        mean_only: bool = True,
        rng_seed: Optional[int] = None,
    ) -> Union[Float[Tensor, "n_test n_sh_coeffs n_fields"], Float[Tensor, "n_post_samples n_test n_sh_coeffs n_fields"]]:
        """
        Draw posterior predictive means or samples for the test points.

        Args:
            X_new: New input features `(n_test, input_dim)` or a single vector.
            s_new: New simulation type indices `(n_test,)` or a scalar.
            n_post_samples:
                - Single posterior state + `mean_only=False`: number of predictive draws.
                - Multiple posterior states: number of states to sample without replacement.
                - `mean_only=True`: ignored when only one state is available.
            mean_only: If True, returns the posterior predictive mean with shape `(n_test, n_sh_coeffs, n_fields)`.
                If False, returns the posterior predictive samples with shape `(n_post_samples, n_test, n_sh_coeffs, n_fields)`.
            rng_seed: Optional random seed.

        Returns:
            If `mean_only=True`, returns the posterior predictive mean with shape `(n_test, n_sh_coeffs, n_fields)`.
                With one posterior state this is the exact predictive mean.
                With multiple posterior states this averages their analytic means.
            If `mean_only=False`, returns the posterior predictive samples with shape `(n_post_samples, n_test, n_sh_coeffs, n_fields)`.
        """
        # Setup
        if not self.fitted_:
            raise RuntimeError("Model must be fitted before calling predict_mean(). Call .fit() first.")

        generator = make_generator(self.X_train_, rng_seed)

        # Normalize inputs to batched form
        X_new, s_new = self._unsqueeze_inputs(X_new, s_new)
        X_new = X_new.to(device=self.X_train_.device, dtype=self.X_train_.dtype)
        s_new = s_new.to(device=self.s_train_.device, dtype=self.s_train_.dtype)

        if "U_T" in self.posterior_samples_:
            n_posterior_states = self.posterior_samples_["U_T"].shape[0]
        elif "Z_T" in self.posterior_samples_:
            n_posterior_states = self.posterior_samples_["Z_T"].shape[0]
        else:
            raise KeyError("Expected 'U_T' or 'Z_T' in posterior_samples_.")
        single_state = n_posterior_states == 1

        # Determine which posterior states to use and how many predictive samples to draw.
        if single_state:
            if mean_only:
                posterior_state_indices = X_new.new_zeros(1, dtype=torch.long)
            else:
                if n_post_samples is None:
                    n_predictive_samples = 1
                    print(
                        "WARNING: n_post_samples not provided for mean_only=False with a single posterior state. Using one predictive draw."
                    )
                else:
                    n_predictive_samples = n_post_samples
                posterior_state_indices = None
        else:
            if n_post_samples is None:
                print("WARNING: n_post_samples not provided for multi-state prediction. Using all posterior states.")
                posterior_state_indices = torch.arange(n_posterior_states, device=X_new.device)
            else:
                posterior_state_indices = sample_randint(
                    n_posterior_states,
                    (n_post_samples,),
                    device=X_new.device,
                    generator=generator,
                    replace=False,
                )

        # Prediction computation
        if mean_only:
            # Compute predictive mean by averaging state-conditional analytic means
            mu_y_new_list = []
            for idx in posterior_state_indices:
                mu_y_new = self._sample_state_conditional_posterior_predictive(
                    sample_idx=int(idx),
                    X_new=X_new,
                    s_new=s_new,
                    mean_only=True,  # analytic mean for this posterior state
                    generator=generator,
                )
                mu_y_new_list.append(mu_y_new)

            y_mean = torch.stack(mu_y_new_list, dim=0).mean(dim=0)
            if getattr(self, "linear_trend_", None) is not None:
                lt = self.linear_trend_
                Gamma = lt["Gamma"].to(device=y_mean.device, dtype=y_mean.dtype)
                H = build_design_matrix(X_new, s_new, n_sim_types=self.n_sim_types, design_cfg=lt["design_cfg"])
                y_mean = y_mean + torch.einsum("np,paf->naf", H, Gamma)
            return y_mean

        if single_state:
            sample_idx = 0
            L_K, Ks, Z_train, mu_post, _, sigma, ell, sigma_f = self._unpack_posterior_state(sample_idx)
            z_new_mean, z_new_var = self._latent_posterior_predictive_stats(
                L_K=L_K,
                Ks=Ks,
                Z_train=Z_train,
                ell=ell,
                sigma_f=sigma_f,
                X_new=X_new,
                s_new=s_new,
                mean_only=False,
            )
            _, n_sh_coeffs, n_fields = self.Y_train_.shape
            predictions = torch.stack([
                self._sample_decoder_posterior_predictive_field_coreg(
                    sample_idx=sample_idx,
                    decoder_state=mu_post,  # type: ignore[arg-type]
                    Z_train=Z_train,
                    sigma=sigma,
                    z_new_mean=z_new_mean,
                    z_new_var=z_new_var,
                    n_sh_coeffs=n_sh_coeffs,
                    n_fields=n_fields,
                    mean_only=False,
                    generator=generator,
                )
                for _ in range(n_predictive_samples)
            ], dim=0)
        else:
            predictions = torch.stack([
                self._sample_state_conditional_posterior_predictive(
                    sample_idx=int(sample_idx),
                    X_new=X_new,
                    s_new=s_new,
                    generator=generator,
                    mean_only=False,
                )
                for sample_idx in posterior_state_indices
            ], dim=0)
        if getattr(self, "linear_trend_", None) is not None:
            lt = self.linear_trend_
            Gamma = lt["Gamma"].to(device=predictions.device, dtype=predictions.dtype)
            H = build_design_matrix(X_new, s_new, n_sim_types=self.n_sim_types, design_cfg=lt["design_cfg"])
            predictions = predictions + torch.einsum("np,paf->naf", H, Gamma).unsqueeze(0)
        return predictions

    def _compute_decoder_posterior_field_coreg(
        self,
        V: Tensor,         # (N, q) where V = Z * diag(tau)
        Y_scaled: Tensor,  # (N, A, F)
        sigma: Tensor,     # scalar
        *,
        B_field: Tensor,     # (F, k)
        psi_field: Tensor,   # (F,)
        eps: float,
    ) -> list[dict[str, Tensor]]:
        """
        weight-space decoder posterior for field–field coregionalization with missingness.

        Returns a list of group states keyed by identical `sh_mask_` patterns over fields
        (per SH coefficient index). If `example_sh_mask_` is set, groups are split per coefficient.
        Each group stores the shared precision factors and the
        posterior mean weights μ_w for all coefficients in the group.
        """
        N, A, F = Y_scaled.shape
        q = int(V.shape[1])
        sigma_sq = sigma**2 + self.jitter
        eye_q = torch.eye(q, device=Y_scaled.device, dtype=Y_scaled.dtype)

        # Group coefficients by active-field subsets induced by sh_mask_
        if self.sh_mask_ is None:
            groups = [(torch.arange(A, device=Y_scaled.device), torch.arange(F, device=Y_scaled.device))]
        else:
            pattern_map: dict[bytes, tuple[Tensor, list[int]]] = {}
            for a in range(A):
                mask_a = self.sh_mask_[a]  # (F,)
                key = mask_a.to(dtype=torch.uint8).cpu().numpy().tobytes()
                if key not in pattern_map:
                    pattern_map[key] = (mask_a, [])
                pattern_map[key][1].append(a)
            groups = []
            for mask_fields, a_list in pattern_map.values():
                if not mask_fields.any():
                    continue
                f_idxs = torch.nonzero(mask_fields, as_tuple=False).flatten()
                a_idxs = Y_scaled.new_tensor(a_list, dtype=torch.long)
                groups.append((a_idxs, f_idxs))

        all_true = torch.ones((N,), dtype=torch.bool, device=Y_scaled.device)
        field_mask = self.field_mask_.to(device=Y_scaled.device, dtype=torch.bool) if self.field_mask_ is not None else None
        example_sh_mask = self._example_sh_mask_or_none(ref=Y_scaled)
        if example_sh_mask is not None:
            expanded_groups: list[tuple[Tensor, Tensor]] = []
            for a_idxs, f_idxs in groups:
                for a in a_idxs.tolist():
                    expanded_groups.append((Y_scaled.new_tensor([a], dtype=torch.long), f_idxs))
            groups = expanded_groups

        states: list[dict[str, Tensor]] = []
        for a_idxs, f_idxs in groups:
            A_g = int(a_idxs.numel())
            F_g = int(f_idxs.numel())
            if A_g == 0 or F_g == 0:
                continue

            B_g = B_field[f_idxs, :]
            psi_g = psi_field[f_idxs].clamp_min(0.0)
            d_g, U_g, _ = self._field_coreg_prec_terms(B_g, psi_g, eps=eps)
            k = int(U_g.shape[1])

            A_chol = Y_scaled.new_zeros((F_g, q, q))
            x0_stack = Y_scaled.new_zeros((F_g, q, A_g))

            pattern_map_fields: dict[bytes, tuple[Tensor, list[int]]] = {}
            a0 = int(a_idxs[0].item()) if example_sh_mask is not None else -1
            for j, f in enumerate(f_idxs.tolist()):
                I_f = field_mask[:, f] if field_mask is not None else all_true
                if example_sh_mask is not None:
                    I_f = I_f & example_sh_mask[:, a0]
                key = I_f.to(dtype=torch.uint8).cpu().numpy().tobytes()
                if key not in pattern_map_fields:
                    pattern_map_fields[key] = (I_f, [])
                pattern_map_fields[key][1].append(j)

            for I_mask, j_list in pattern_map_fields.values():
                n_I = int(I_mask.sum().item())
                if n_I == 0:
                    continue

                V_I = V[I_mask]  # (n_I, q)
                S_I = V_I.T @ V_I  # (q, q)

                j_idx = Y_scaled.new_tensor(j_list, dtype=torch.long)
                f_sub = f_idxs.index_select(0, j_idx)
                F_p = int(f_sub.numel())

                Y_I = Y_scaled[I_mask][:, a_idxs][:, :, f_sub]  # (n_I, A_g, F_p)
                t = (V_I.T @ Y_I.reshape(n_I, A_g * F_p)).view(q, A_g, F_p)
                b = (1.0 / sigma_sq) * t  # (q, A_g, F_p)

                d_p = d_g.index_select(0, j_idx)  # (F_p,)
                A = d_p.view(F_p, 1, 1) * eye_q + (1.0 / sigma_sq) * S_I  # (F_p, q, q)
                L_A = torch.linalg.cholesky(A)
                A_chol.index_copy_(0, j_idx, L_A)

                b_f = b.permute(2, 0, 1)  # (F_p, q, A_g)
                x0 = torch.cholesky_solve(b_f, L_A)  # (F_p, q, A_g)
                x0_stack.index_copy_(0, j_idx, x0)

            if k == 0:
                mu_w = x0_stack
                K_chol = Y_scaled.new_zeros((0, 0))
            else:
                kq = k * q
                Ainv_stack = torch.cholesky_solve(eye_q.expand(F_g, -1, -1), A_chol)  # (F_g, q, q)
                K_blocks = -torch.einsum("fr,fs,fij->rsij", U_g, U_g, Ainv_stack)  # (k, k, q, q)
                diag_idx = torch.arange(k, device=Y_scaled.device)
                K_blocks[diag_idx, diag_idx] = K_blocks[diag_idx, diag_idx] + eye_q
                K = K_blocks.permute(0, 2, 1, 3).reshape(kq, kq)
                K = K + torch.eye(kq, device=Y_scaled.device, dtype=Y_scaled.dtype) * self.jitter
                K_chol = torch.linalg.cholesky(K)

                g = torch.einsum("fk,fqa->kqa", U_g, x0_stack).reshape(kq, A_g)
                h = torch.cholesky_solve(g, K_chol).reshape(k, q, A_g)
                m = torch.einsum("fk,kqa->fqa", U_g, h)  # (F_g, q, A_g)
                mu_w = x0_stack + torch.cholesky_solve(m, A_chol)

            states.append(
                {
                    "a_idxs": a_idxs,
                    "f_idxs": f_idxs,
                    "mu_w": mu_w,       # (F_g, q, A_g)
                    "A_chol": A_chol,   # (F_g, q, q)
                    "U": U_g,           # (F_g, k)
                    "K_chol": K_chol,   # (kq, kq) or (0,0)
                }
            )

        return states

    def _field_coreg_predictive_cov(
        self,
        z: Tensor,          # (B, q)
        *,
        A_chol: Tensor,     # (F, q, q)
        U: Tensor,          # (F, k)
        K_chol: Tensor,     # (kq, kq) or (0,0)
        sigma_sq: Tensor,   # scalar
    ) -> Tensor:
        """compute Σ_field(z) = zᵀ Cov(w) z + σ² I for a batch of latent points."""
        B = int(z.shape[0])
        F, q = int(A_chol.shape[0]), int(A_chol.shape[1])
        k = int(U.shape[1])

        g_list: list[Tensor] = []
        var_base = z.new_zeros((B, F))
        z_col = z.unsqueeze(-1)  # (B, q, 1)
        for f in range(F):
            g_f = torch.cholesky_solve(z_col, A_chol[f]).squeeze(-1)  # (B, q)
            g_list.append(g_f)
            var_base[:, f] = torch.sum(z * g_f, dim=1)
        g_stack = torch.stack(g_list, dim=1)  # (B, F, q)

        if k == 0:
            return torch.diag_embed(var_base + sigma_sq)

        kq = k * q
        P = torch.einsum("fk,bfq->bkqf", U, g_stack).reshape(B, kq, F)  # (B, kq, F)
        W = torch.cholesky_solve(P, K_chol)  # (B, kq, F)
        cov_lr = P.transpose(1, 2) @ W  # (B, F, F)
        cov_lr = 0.5 * (cov_lr + cov_lr.transpose(-1, -2))
        return cov_lr + torch.diag_embed(var_base + sigma_sq)

    def _latent_posterior_predictive_stats(
        self,
        L_K: Tensor,
        Ks: Tensor,
        Z_train: Tensor,
        ell: Tensor,
        sigma_f: Tensor,
        X_new: Float[Tensor, "n_test d"],
        s_new: Int[Tensor, "n_test"],
        mean_only: bool = True,
    ) -> tuple[Float[Tensor, "n_test q"], Optional[Tensor]]:
        """
        Derive the mean (and optional variance) of the posterior predictive over latents:
            z_new | Z, X, s, θ ~ N(z_new_mean, z_new_var I_q).

        Note:
            z_new_var has shape (n_test, q) when computed.
        """
        if sigma_f.ndim != 1 or int(sigma_f.numel()) != int(self.latent_dim):
            raise ValueError(f"Expected sigma_f to have shape ({int(self.latent_dim)},), got {tuple(sigma_f.shape)}.")
        # GP Conditional
        ## Cross-kernel
        Kx_new_train = apply_kernel(self.kernel, X_new, ell, X_new.new_tensor(1.0), X2=self.X_train_)
        K_new_train = Kx_new_train * Ks[s_new.unsqueeze(1), self.s_train_.unsqueeze(0)]

        if L_K.dim() == 2:
            ## Solve for mean
            K_inv_k = torch.cholesky_solve(K_new_train.T, L_K).T
            z_new_mean = K_inv_k @ Z_train

            z_new_var = None
            if not mean_only:  # compute pointwise variances (shared across latents)
                k_new_new = Ks[s_new, s_new] + self._nugget_noise_diag(s_new, X_new)
                base_var = (k_new_new - torch.sum(K_new_train * K_inv_k, dim=1)).clamp_min(0.0)
                z_new_var = base_var.unsqueeze(-1) * (sigma_f**2)

            return z_new_mean, z_new_var

        # Batched case: per-latent/grouped kernels
        # K_new_train: (q, n_test, n_train), L_K: (q, n_train, n_train)
        K_rhs = K_new_train.permute(0, 2, 1)  # (q, n_train, n_test)
        K_inv_k = torch.cholesky_solve(K_rhs, L_K).permute(0, 2, 1)  # (q, n_test, n_train)
        z_new_mean = torch.einsum("qtn,qn->tq", K_inv_k, Z_train.T)

        z_new_var_q = None
        if not mean_only:
            k_new_new = Ks[s_new, s_new] + self._nugget_noise_diag(s_new, X_new)  # (n_test,)
            base_var_q = (k_new_new.unsqueeze(0) - torch.sum(K_new_train * K_inv_k, dim=2)).T.clamp_min(0.0)
            z_new_var_q = base_var_q * (sigma_f**2)

        return z_new_mean, z_new_var_q

    def _sample_decoder_posterior_predictive_field_coreg(
        self,
        *,
        sample_idx: int,
        decoder_state: list[dict[str, Tensor]],
        Z_train: Tensor,
        sigma: Tensor,
        z_new_mean: Tensor,
        z_new_var: Optional[Tensor],
        n_sh_coeffs: int,
        n_fields: int,
        mean_only: bool,
        generator: Optional[torch.Generator],
    ) -> Tensor:
        """
        Draw a single state-conditional predictive sample (or analytic mean) under
        field–field coregionalization, using the weight-space decoder posterior.
        """
        n_test = int(z_new_mean.shape[0])

        # Latent points: either sample z_new or use analytic mean
        if mean_only:
            z_new = z_new_mean
        else:
            if z_new_var is None:
                raise ValueError("z_new_var must be provided when mean_only=False.")
            eps_z = sample_randn(Z_train, (n_test, self.latent_dim), generator)
            z_std = torch.sqrt(z_new_var)
            if z_std.dim() == 1:
                z_std = z_std.unsqueeze(-1)
            z_new = z_new_mean + eps_z * z_std

        v_new = z_new * self.tau_.unsqueeze(0)
        y_new = z_new.new_zeros((n_test, n_sh_coeffs, n_fields))
        sigma_sq = sigma**2 + self.jitter

        for st in decoder_state:
            a_idxs = st["a_idxs"].to(device=y_new.device, dtype=torch.long)
            f_idxs = st["f_idxs"].to(device=y_new.device, dtype=torch.long)
            mu_w = st["mu_w"]        # (F_g, q, A_g)
            A_chol = st["A_chol"]    # (F_g, q, q)
            U = st["U"]              # (F_g, k)
            K_chol = st["K_chol"]    # (kq, kq) or (0,0)

            A_g = int(a_idxs.numel())
            F_g = int(f_idxs.numel())
            if A_g == 0 or F_g == 0:
                continue

            if mean_only:
                mu = torch.einsum("nq,fqa->nfa", v_new, mu_w).permute(0, 2, 1)  # (n, A_g, F_g)
                y_sub = mu
            else:
                mu = torch.einsum("nq,fqa->nfa", v_new, mu_w).permute(0, 2, 1)  # (n, A_g, F_g)
                cov = self._field_coreg_predictive_cov(
                    v_new,
                    A_chol=A_chol,
                    U=U,
                    K_chol=K_chol,
                    sigma_sq=sigma_sq,
                )  # (n, F_g, F_g)
                L = torch.linalg.cholesky(cov)
                eps_y = sample_randn(self.Y_train_, (n_test, A_g, F_g), generator)
                y_sub = mu + (eps_y @ L.transpose(-1, -2))

            y_new[:, a_idxs[:, None], f_idxs[None, :]] = y_sub

        # Undo learned weighting to return physical units.
        y_new = y_new / self._get_alpha_sqrt(sample_idx, n_sh_coeffs, n_fields)

        # Enforce static SH mask by zeroing forbidden coefficients.
        if self.sh_mask_ is not None:
            y_new = y_new * self.sh_mask_.to(device=y_new.device, dtype=y_new.dtype).view(1, n_sh_coeffs, n_fields)

        return y_new

    def _sample_state_conditional_posterior_predictive(
        self,
        sample_idx: int,
        X_new: Float[Tensor, "n d"],
        s_new: Int[Tensor, "n"],
        mean_only: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Float[Tensor, "n a f"]:
        """
        Batched posterior predictive for multiple test points conditioned on a
        single posterior state (ie. the state-conditional posterior predictive).

        If `mean_only=True`, returns the analytic mean y_new for that posterior state (no sampling of latent GP or decoder noise).
        If `mean_only=False`, draws a full predictive sample (z_new, y_new).
        """
        _, n_sh_coeffs, n_fields = self.Y_train_.shape

        # Unpack posterior state (once per posterior sample)
        L_K, Ks, Z_train, mu_post, Sigma_post, sigma, ell, sigma_f = self._unpack_posterior_state(sample_idx)

        # Compute latent GP posterior predictive statistics
        z_new_mean, z_new_var = self._latent_posterior_predictive_stats(
            L_K=L_K,
            Ks=Ks,
            Z_train=Z_train,
            ell=ell,
            sigma_f=sigma_f,
            X_new=X_new,
            s_new=s_new,
            mean_only=mean_only,
        )
        return self._sample_decoder_posterior_predictive_field_coreg(
            sample_idx=sample_idx,
            decoder_state=mu_post,  # type: ignore[arg-type]
            Z_train=Z_train,
            sigma=sigma,
            z_new_mean=z_new_mean,
            z_new_var=z_new_var,
            n_sh_coeffs=n_sh_coeffs,
            n_fields=n_fields,
            mean_only=mean_only,
            generator=generator,
        )
