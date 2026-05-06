from __future__ import annotations

import math
from typing import Literal

import torch

KernelName = Literal["rbf", "matern32", "matern52"]


def build_design_matrix(X: torch.Tensor, s: torch.Tensor, *, n_sim_types: int, design_cfg: dict) -> torch.Tensor:
    cols = []
    if design_cfg.get("intercept", True):
        cols.append(X.new_ones((X.shape[0], 1)))
    if design_cfg.get("inputs", True):
        cols.append(X)
    if design_cfg.get("sim_onehot", False):
        oh = torch.nn.functional.one_hot(s.long(), num_classes=n_sim_types).to(dtype=X.dtype, device=X.device)
        if design_cfg.get("intercept", True):
            oh = oh[:, 1:]
        if oh.numel():
            cols.append(oh)
    if not cols:
        raise ValueError("Design matrix has no columns (all design flags are false).")
    return torch.cat(cols, dim=1)


def fit_ridge(
    H: torch.Tensor,
    Y: torch.Tensor,
    *,
    lambda_reg: float,
    field_mask: torch.Tensor | None,
    coeff_mask: torch.Tensor | None = None,
    sh_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    lamI = H.new_tensor(lambda_reg) * torch.eye(H.shape[1], device=H.device, dtype=H.dtype)
    if field_mask is None and coeff_mask is None and sh_mask is None:
        gamma = torch.linalg.solve(H.T @ H + lamI, H.T @ Y.reshape(Y.shape[0], -1))
        return gamma.reshape(H.shape[1], Y.shape[1], Y.shape[2])

    N, A, F = Y.shape
    obs = torch.ones((N, A, F), dtype=torch.bool, device=Y.device)
    if field_mask is not None:
        obs = obs & field_mask.to(device=Y.device, dtype=torch.bool).unsqueeze(1)
    if coeff_mask is not None:
        obs = obs & coeff_mask.to(device=Y.device, dtype=torch.bool).unsqueeze(-1)
    if sh_mask is not None:
        obs = obs & sh_mask.to(device=Y.device, dtype=torch.bool).unsqueeze(0)
    Y_flat, obs_flat = Y.reshape(N, -1), obs.reshape(N, -1)
    Gamma_flat = H.new_zeros((H.shape[1], A * F))
    pattern_map: dict[bytes, tuple[torch.Tensor, list[int]]] = {}
    for j in range(obs_flat.shape[1]):
        m = obs_flat[:, j]
        pattern_map.setdefault(m.to(dtype=torch.uint8).cpu().numpy().tobytes(), (m, []))[1].append(j)
    for m, idxs_list in pattern_map.values():
        if not bool(m.any()):
            continue
        idxs = H.new_tensor(idxs_list, dtype=torch.long)
        Hm, Ym = H[m], Y_flat[m].index_select(1, idxs)
        Gamma_flat.index_copy_(1, idxs, torch.linalg.solve(Hm.T @ Hm + lamI, Hm.T @ Ym))
    return Gamma_flat.reshape(H.shape[1], A, F)


def _kernel_sqdist(X1: torch.Tensor, ell: torch.Tensor, X2: torch.Tensor | None = None) -> torch.Tensor:
    X2 = X1 if X2 is None else X2
    diff = X1[:, None, :] - X2[None, :, :]
    scale = ell if ell.dim() == 1 else ell[:, None, None, :]
    return torch.sum((diff if ell.dim() == 1 else diff.unsqueeze(0)) ** 2 / scale**2, dim=-1)


def apply_kernel(
    kernel: KernelName,
    X1: torch.Tensor,
    ell: torch.Tensor,
    sigma_x: torch.Tensor,
    X2: torch.Tensor | None = None,
) -> torch.Tensor:
    r2 = _kernel_sqdist(X1, ell, X2)
    if kernel == "rbf":
        return sigma_x**2 * torch.exp(-0.5 * r2)
    r = torch.sqrt(r2 + torch.finfo(r2.dtype).eps)
    if kernel == "matern32":
        z = math.sqrt(3.0) * r
        return sigma_x**2 * (1.0 + z) * torch.exp(-z)
    if kernel == "matern52":
        z = math.sqrt(5.0) * r
        return sigma_x**2 * (1.0 + z + (5.0 / 3.0) * r2) * torch.exp(-z)
    raise ValueError(f"Unknown kernel '{kernel}'. Expected 'rbf', 'matern32', or 'matern52'.")


def compute_sim_type_kernel(L_corr: torch.Tensor, r_sim: torch.Tensor) -> torch.Tensor:
    R = L_corr @ L_corr.T
    return r_sim.unsqueeze(1) * r_sim.unsqueeze(0) * R


def stabilize_kernel(K: torch.Tensor, jitter: float) -> torch.Tensor:
    eye = torch.eye(int(K.shape[-1]), device=K.device, dtype=K.dtype)
    K = K + (eye if K.dim() == 2 else eye.unsqueeze(0)) * float(jitter)
    return 0.5 * (K + K.transpose(-1, -2))


def make_generator(ref: torch.Tensor, rng_seed: int | None) -> torch.Generator | None:
    if rng_seed is None:
        return None
    generator = torch.Generator(device=ref.device) if ref.device.type == "cuda" else torch.Generator()
    generator.manual_seed(int(rng_seed))
    return generator


def sample_randn(ref: torch.Tensor, shape: tuple[int, ...], generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        return torch.randn(shape, device=ref.device, dtype=ref.dtype)
    if ref.device.type == "xpu":
        return torch.randn(shape, generator=generator, dtype=ref.dtype).to(device=ref.device)
    return torch.randn(shape, generator=generator, device=ref.device, dtype=ref.dtype)


def sample_randint(
    high: int,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.int64,
    replace: bool = True,
) -> torch.Tensor:
    if replace:
        if generator is None:
            return torch.randint(high, shape, device=device, dtype=dtype)
        if device.type == "xpu":
            return torch.randint(high, shape, generator=generator, dtype=dtype).to(device=device)
        return torch.randint(high, shape, generator=generator, device=device, dtype=dtype)
    total = math.prod(shape)
    if total > high:
        raise ValueError(f"Cannot draw {total} unique integers in [0, {high}) without replacement.")
    if generator is None:
        perm = torch.randperm(high, device=device)
    elif device.type == "xpu":
        perm = torch.randperm(high, generator=generator).to(device=device)
    else:
        perm = torch.randperm(high, device=device, generator=generator)
    return perm.to(dtype=dtype)[:total].view(shape)
