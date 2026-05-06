from __future__ import annotations

from typing import TypedDict
import math
import re

import numpy as np
import torch

BOTTOM_SQUEEZE_FRACTION = 0.925
P_TOP = 1000.0
SIGMA_LEVELS = (3 / 4) * np.linspace(1, 0, 10) + (7 / 4) * np.linspace(1, 0, 10) ** 3 - (3 / 2) * np.linspace(1, 0, 10) ** 4
SPLIT_VAR_REGEX = re.compile(r"^(.*)_(\d+)$")
OFFSET = P_TOP / (BOTTOM_SQUEEZE_FRACTION * 1e5 - P_TOP)

FIELD_GROUP_NAMES = [
    "temperature",
    "surface_temperature",
    "specific_humidity",
    "wind",
    "surface_pressure",
    "cloud_fraction",
    "olr_asr",
]


class WeightingMetadata(TypedDict):
    log_lplus1: torch.Tensor
    field_log_pressure: torch.Tensor
    field_group_ids: torch.Tensor
    original_group_ids: torch.Tensor
    n_field_groups: int


def build_weighting_metadata(*, fields: list[str], sh_T: int, ref: torch.Tensor) -> WeightingMetadata:
    l_int = torch.arange(sh_T + 1, device=ref.device, dtype=torch.long)
    log_lplus1 = torch.log(torch.repeat_interleave(l_int, 1 + 2 * l_int).to(dtype=ref.dtype) + ref.new_tensor(1.0))
    group_ids, log_pressures = [], []
    for name in fields:
        base, level_idx = _strip_level_suffix(name)
        group_ids.append(retrieve_field_group_index(base))
        sigma_val = 1.0 if level_idx is None else float(SIGMA_LEVELS[int(level_idx)])
        log_pressures.append(math.log(sigma_val + OFFSET))
    raw = torch.tensor(group_ids, dtype=torch.long, device=ref.device)
    original_group_ids, field_group_ids = torch.unique(raw, return_inverse=True)
    return {
        "log_lplus1": log_lplus1,
        "field_group_ids": field_group_ids,
        "field_log_pressure": ref.new_tensor(log_pressures),
        "original_group_ids": original_group_ids,
        "n_field_groups": int(original_group_ids.numel()),
    }


def build_learned_group_output_weights(*, log_alpha_group_uncentred: torch.Tensor, metadata: WeightingMetadata, n_sh_coeffs: int) -> torch.Tensor:
    log_alpha_f = log_alpha_group_uncentred[metadata["field_group_ids"]]
    log_alpha = log_alpha_f.unsqueeze(0).expand(n_sh_coeffs, -1).reshape(-1)
    return torch.exp(log_alpha - torch.mean(log_alpha))


def _strip_level_suffix(field_name: str) -> tuple[str, int | None]:
    match = SPLIT_VAR_REGEX.match(field_name)
    return (match.group(1), int(match.group(2))) if match else (field_name, None)


def retrieve_field_group_index(base_name: str) -> int:
    lower = base_name.lower()
    if lower == "temperature":
        return 0
    if lower == "surface_temperature":
        return 1
    if lower == "specific_humidity":
        return 2
    if lower in ("u", "v", "streamfunction", "velocity_potential"):
        return 3
    if lower in ("surface_pressure", "surface_pressure_frac_dev"):
        return 4
    if lower in ("cloud_fraction", "cloud_propensity"):
        return 5
    if lower in ("olr", "asr", "olr_cloudy", "asr_cloudy"):
        return 6
    raise NotImplementedError(f"Unknown field group for '{base_name}'")
