from __future__ import annotations

import torch


def _per_field_mean_with_mask(per_example_per_field: torch.Tensor, field_mask: torch.Tensor | None) -> torch.Tensor:
    if field_mask is None:
        return per_example_per_field.mean(dim=0)
    mask = field_mask.to(device=per_example_per_field.device, dtype=per_example_per_field.dtype)
    count = mask.sum(dim=0)
    total = (per_example_per_field * mask).sum(dim=0)
    return torch.where(count > 0, total / count, total.new_tensor(float("nan")))


def srmse_spectral(
    pred: torch.Tensor,
    target: torch.Tensor,
    aggregate_only: bool = False,
    field_mask: torch.Tensor | None = None,
    coeff_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if pred.dim() < 2:
        raise ValueError("srmse_spectral expects tensors with at least 2 dims [B,...,F]")
    err_sq = (pred - target).square()
    dims = tuple(range(1, pred.dim() - 1))
    if coeff_mask is None:
        mse = err_sq.mean(dim=dims) if dims else err_sq
    else:
        weights = coeff_mask.to(device=err_sq.device, dtype=err_sq.dtype)
        weights = weights if weights.shape == err_sq.shape else weights.expand_as(err_sq)
        mse = ((err_sq * weights).sum(dim=dims) if dims else err_sq * weights) / (
            (weights.sum(dim=dims) if dims else weights).clamp_min(1.0)
        )
    per_field = _per_field_mean_with_mask(torch.sqrt(mse + 1e-12), field_mask)
    agg = per_field.nanmean()
    return agg if aggregate_only else torch.cat((per_field, agg.unsqueeze(0)))
