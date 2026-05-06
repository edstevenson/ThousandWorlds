from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

ASSET_DIR = Path(__file__).resolve().parent / "assets"
GRID_SHAPE = (32, 64)
N_COEFFS = 484


def build_equatorial_symmetry_mask(l_max: int, m_max: int, mode: str) -> np.ndarray:
    if mode not in {"none", "symmetric", "antisymmetric"}:
        raise ValueError(f"Unknown symmetry mode '{mode}'. Expected 'none', 'symmetric', or 'antisymmetric'.")
    n_coeffs = sum(1 + 2 * min(l, m_max - 1) for l in range(l_max + 1))
    if mode == "none":
        return np.ones(n_coeffs, dtype=bool)
    mask = np.zeros(n_coeffs, dtype=bool)
    keep_even = mode == "symmetric"
    idx = 0
    for l in range(l_max + 1):
        for m in range(min(l, m_max - 1) + 1):
            keep = (l + m) % 2 == 0
            keep = keep if keep_even else not keep
            mask[idx] = keep
            idx += 1
            if m:
                mask[idx] = keep
                idx += 1
    return mask


def load_inverse_sht_matrix(path: str | Path | None = None) -> np.ndarray:
    return np.load(ASSET_DIR / "inverse_sht.npy" if path is None else Path(path), allow_pickle=False)


def load_latitude_weights(path: str | Path | None = None) -> np.ndarray:
    return np.load(ASSET_DIR / "latitude_weights.npy" if path is None else Path(path), allow_pickle=False)


def load_symmetry_masks(stats_dir: str | Path, field_names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    stats_dir = Path(stats_dir)
    meta = json.loads((stats_dir / "spectral.meta.json").read_text())
    requested = list(field_names or meta.get("fields", {}).keys())
    with np.load(stats_dir / "spectral.npz", allow_pickle=False) as npz:
        return {
            name: (
                np.asarray(npz[f"{name}__mask"], dtype=bool)
                if f"{name}__mask" in npz
                else np.ones_like(np.asarray(npz[f"{name}__mean"], dtype=np.float32), dtype=bool)
            )
            for name in requested
        }


def apply_symmetry_mask(
    coeffs: np.ndarray,
    field_names: list[str],
    masks: np.ndarray | dict[str, np.ndarray],
) -> np.ndarray:
    arr = np.asarray(coeffs, dtype=np.float32)
    mask_arr = (
        np.stack([masks[name] for name in field_names], axis=0)
        if isinstance(masks, dict)
        else np.asarray(masks, dtype=bool)
    )
    if mask_arr.shape[0] != len(field_names):
        mask_arr = mask_arr.T
    if arr.shape[-2:] != mask_arr.shape:
        raise ValueError(f"Expected coeffs trailing shape {mask_arr.shape}, got {arr.shape[-2:]}.")
    return arr * mask_arr


def to_grid(coeffs: np.ndarray, matrix: np.ndarray | None = None) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float32)
    if coeffs.shape[-1] != N_COEFFS:
        raise ValueError(f"Expected trailing coefficient dimension {N_COEFFS}, got {coeffs.shape[-1]}.")
    matrix = load_inverse_sht_matrix() if matrix is None else np.asarray(matrix, dtype=np.float32)
    return (coeffs @ matrix.T).reshape(*coeffs.shape[:-1], *GRID_SHAPE)
