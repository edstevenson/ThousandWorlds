from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import torch

import thousandworlds as tw
from thousandworlds.field_spec import CANONICAL_FIELD_VARIABLES

_LEVEL_RE = re.compile(r"^(.+?)_(\d+)$")


@dataclass(frozen=True)
class FieldMetadata:
    bases: list[str]
    base_idx: np.ndarray
    level: np.ndarray
    fields_by_base: list[np.ndarray]


def parse_field_name(name: str) -> tuple[str, int]:
    match = _LEVEL_RE.match(name)
    return (match.group(1), int(match.group(2))) if match else (name, -1)


def field_metadata(field_names: list[str]) -> FieldMetadata:
    parsed = [parse_field_name(name) for name in field_names]
    bases = [name for name in CANONICAL_FIELD_VARIABLES if any(base == name for base, _ in parsed)]
    base_to_idx = {name: i for i, name in enumerate(bases)}
    max_level = max([level for _, level in parsed if level >= 0] or [0])
    base_idx = np.asarray([base_to_idx[base] for base, _ in parsed], dtype=np.int64)
    level = np.asarray(
        [-1.0 if k < 0 else (0.0 if max_level == 0 else 2.0 * k / max_level - 1.0) for _, k in parsed],
        dtype=np.float32,
    )
    return FieldMetadata(
        bases=bases,
        base_idx=base_idx,
        level=level,
        fields_by_base=[np.where(base_idx == i)[0].astype(np.int64) for i in range(len(bases))],
    )


def t21_coordinate_features(n_lat: int, n_lon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_mu, lat_w = np.polynomial.legendre.leggauss(int(n_lat))
    lon = np.linspace(-np.pi + np.pi / int(n_lon), np.pi - np.pi / int(n_lon), int(n_lon), dtype=np.float64)
    return (
        lat_mu.astype(np.float32),
        np.sin(lon).astype(np.float32),
        np.cos(lon).astype(np.float32),
        (lat_w / lat_w.sum()).astype(np.float32),
    )


def field_norm_stats(Y: torch.Tensor, field_mask: torch.Tensor, lat_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    means, stds = [], []
    w = lat_weights.to(device=Y.device, dtype=Y.dtype)[None, :, None]
    denom_grid = Y.new_tensor(float(Y.shape[-1]))
    for f in range(Y.shape[1]):
        vals = Y[field_mask[:, f], f]
        finite = torch.isfinite(vals)
        vals = torch.where(finite, vals, vals.new_zeros(()))
        denom = (finite.to(Y.dtype) * w).sum() * denom_grid / float(Y.shape[-1])
        mean = (vals * w).sum() / denom.clamp_min(1.0)
        var = (torch.square(vals - mean) * finite.to(Y.dtype) * w).sum() / denom.clamp_min(1.0)
        means.append(torch.where(denom > 0, mean, Y.new_tensor(0.0)))
        stds.append(torch.sqrt(var).clamp_min(1.0e-6) if denom > 0 else Y.new_tensor(1.0))
    return torch.stack(means), torch.stack(stds)


class BaseVariableSampler:
    def __init__(
        self,
        field_mask: torch.Tensor,
        metadata: FieldMetadata,
        lat_probs: torch.Tensor,
        *,
        device: torch.device,
    ) -> None:
        self.field_mask = field_mask.to(device=device, dtype=torch.bool)
        self.obs_by_field = [torch.nonzero(self.field_mask[:, f], as_tuple=False).flatten() for f in range(self.field_mask.shape[1])]
        self.fields_by_base = [
            torch.as_tensor([int(f) for f in fields if self.obs_by_field[int(f)].numel()], device=device, dtype=torch.long)
            for fields in metadata.fields_by_base
        ]
        self.fields_by_base = [fields for fields in self.fields_by_base if fields.numel()]
        self.lat_probs = lat_probs.to(device=device, dtype=torch.float32)
        self.device = device

    def sample(self, batch_size: int, width: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base = torch.randint(len(self.fields_by_base), (int(batch_size),), generator=gen, device=self.device)
        n = torch.empty(int(batch_size), device=self.device, dtype=torch.long)
        f = torch.empty_like(n)
        for b, fields in enumerate(self.fields_by_base):
            idx = torch.nonzero(base == b, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            f_b = fields[torch.randint(fields.numel(), (idx.numel(),), generator=gen, device=self.device)]
            f[idx] = f_b
            for field in torch.unique(f_b):
                j = idx[f_b == field]
                obs = self.obs_by_field[int(field)]
                n[j] = obs[torch.randint(obs.numel(), (j.numel(),), generator=gen, device=self.device)]
        return (
            n,
            f,
            torch.multinomial(self.lat_probs, int(batch_size), replacement=True, generator=gen),
            torch.randint(int(width), (int(batch_size),), generator=gen, device=self.device),
        )


def area_weighted_equal_base_variable_normalized_rmse_grid(
    pred: np.ndarray,
    target: np.ndarray,
    field_names: list[str],
    train_Y: np.ndarray,
    train_field_mask: np.ndarray,
    field_mask: np.ndarray,
) -> float:
    scale = _field_rmse_scale(train_Y, train_field_mask)
    err = _weighted_mse_per_example_field(
        np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0) / scale[None, :, None, None],
        np.nan_to_num(np.asarray(target, dtype=np.float32), nan=0.0) / scale[None, :, None, None],
    )
    per_field = _per_field_mean(np.sqrt(err + 1.0e-12), field_mask)
    groups = []
    for group in dict.fromkeys(parse_field_name(name)[0] for name in field_names):
        vals = np.asarray([per_field[i] for i, name in enumerate(field_names) if parse_field_name(name)[0] == group], dtype=np.float32)
        if np.any(np.isfinite(vals)):
            groups.append(float(np.nanmean(vals)))
    return float(np.mean(groups)) if groups else float("inf")


def latitude_weights(n_lat: int) -> np.ndarray:
    if int(n_lat) == tw.GRID_SHAPE[0]:
        w = tw.load_latitude_weights().astype(np.float32)
        return w / w.sum()
    return t21_coordinate_features(int(n_lat), 1)[-1]


def _weighted_mse_per_example_field(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    w = latitude_weights(np.asarray(pred).shape[-2])
    return (np.square(pred - target).mean(axis=-1) * w[None, None, :]).sum(axis=-1)


def _per_field_mean(values: np.ndarray, field_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(field_mask, dtype=np.float32)
    count = mask.sum(axis=0)
    total = (np.asarray(values, dtype=np.float32) * mask).sum(axis=0)
    return np.divide(total, count, out=np.full(values.shape[1], np.nan, dtype=np.float32), where=count > 0)


def _field_rmse_scale(Y: np.ndarray, field_mask: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    Y = np.nan_to_num(np.asarray(Y, dtype=np.float32), nan=0.0)
    mask = np.asarray(field_mask, dtype=np.float32)[:, :, None, None]
    count = mask.sum(axis=0)
    center = np.divide((Y * mask).sum(axis=0), count, out=np.zeros_like(Y[0], dtype=np.float32), where=count > 0)
    scale2 = _per_field_mean(_weighted_mse_per_example_field(Y, center[None]), field_mask)
    return np.sqrt(np.maximum(np.nan_to_num(scale2, nan=eps**2), eps**2)).astype(np.float32)
