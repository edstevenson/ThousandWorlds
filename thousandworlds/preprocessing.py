from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np

from .schema import support_path


_LEVEL_RE = re.compile(r"^(.*)_(\d+)$")
ASR_OLR_FIELDS = {"asr_cloudy", "olr_cloudy"}
LOG_Z_SCALING_EPSILON = 1.0e-16


@dataclass
class SpectralFieldStats:
    mean: np.ndarray
    sigma: float
    count: int
    energy_mean: float
    mask: np.ndarray


@dataclass
class Stats:
    stats_dir: Path
    input_names: list[str]
    field_names: list[str]
    per_level: bool
    asr_olr_normalize_by_f_star: bool
    strategies: dict[str, tuple[str, dict]]
    normalize_mean: dict[str, np.ndarray]
    normalize_std: dict[str, np.ndarray]
    spectral: dict[str, SpectralFieldStats]
    spectral_meta: dict
    transforms_meta: dict


@dataclass
class LinearTrend:
    Gamma: np.ndarray
    design_cfg: dict[str, bool]
    lambda_reg: float
    n_gcm: int
    gcm_order: list[int] | None = None

def _base_var(name: str) -> str:
    match = _LEVEL_RE.match(name)
    return match.group(1) if match else name


def _stat_key(name: str, stats: Stats) -> str:
    strategy_name, kwargs = stats.strategies[_base_var(name)]
    key = kwargs["stat_key_pattern"]
    match = _LEVEL_RE.match(name)
    return key.replace(_base_var(name), name, 1) if stats.per_level and match else key


def _strategy_params(name: str, stats: Stats) -> tuple[str, dict]:
    return stats.strategies[_base_var(name)]


def _preprocess_array(x: np.ndarray, strategy_name: str, kwargs: dict) -> np.ndarray:
    if strategy_name == "Z-scaling":
        return x
    if strategy_name == "log_Z-scaling":
        return np.log(np.maximum(x, float(kwargs.get("epsilon", LOG_Z_SCALING_EPSILON))))
    if strategy_name == "arcsinh_Z-scaling":
        return np.arcsinh(x / float(kwargs["s"]))
    if strategy_name == "smoothed_logit_Z-scaling":
        eps = float(kwargs["epsilon"])
        smoothed = (x + eps) / (1.0 + 2.0 * eps)
        return np.log(smoothed / (1.0 - smoothed))
    raise ValueError(f"Unsupported strategy {strategy_name!r}.")


def _inverse_preprocess_array(x: np.ndarray, strategy_name: str, kwargs: dict) -> np.ndarray:
    if strategy_name == "Z-scaling":
        return x
    if strategy_name == "log_Z-scaling":
        return np.exp(x)
    if strategy_name == "arcsinh_Z-scaling":
        return np.sinh(x) * float(kwargs["s"])
    if strategy_name == "smoothed_logit_Z-scaling":
        eps = float(kwargs["epsilon"])
        smoothed = 1.0 / (1.0 + np.exp(-x))
        return np.clip(smoothed * (1.0 + 2.0 * eps) - eps, 0.0, 1.0)
    raise ValueError(f"Unsupported strategy {strategy_name!r}.")


def _field_mask_to_coeff_obs(
    field_mask: np.ndarray | None,
    coeff_mask: np.ndarray | None,
    sh_mask: np.ndarray | None,
    shape: tuple[int, int, int],
) -> np.ndarray:
    n, f, a = shape
    obs = np.ones(shape, dtype=bool)
    if field_mask is not None:
        obs &= np.asarray(field_mask, dtype=bool)[:, :, None]
    if coeff_mask is not None:
        coeff_mask = np.asarray(coeff_mask, dtype=bool)
        obs &= coeff_mask[:, None, :] if coeff_mask.ndim == 2 else coeff_mask
    if sh_mask is not None:
        sh_mask = np.asarray(sh_mask, dtype=bool)
        if sh_mask.shape == (a, f):
            sh_mask = sh_mask.T
        obs &= sh_mask[None, :, :]
    return obs


def load_stats(subset: str, data_dir: str | Path) -> Stats:
    stats_dir = support_path(data_dir, subset, kind="stats_dir")
    transforms_meta = json.loads((stats_dir / "transforms.meta.json").read_text())
    spectral_meta = json.loads((stats_dir / "spectral.meta.json").read_text())
    with np.load(stats_dir / "normalize_mean.npz", allow_pickle=False) as npz:
        normalize_mean = {k: np.asarray(v) for k, v in npz.items()}
    with np.load(stats_dir / "normalize_std.npz", allow_pickle=False) as npz:
        normalize_std = {k: np.asarray(v) for k, v in npz.items()}
    spectral = {}
    with np.load(stats_dir / "spectral.npz", allow_pickle=False) as npz:
        for name in transforms_meta["fields"]:
            spectral[name] = SpectralFieldStats(
                mean=np.asarray(npz[f"{name}__mean"], dtype=np.float32),
                sigma=float(np.asarray(npz[f"{name}__sigma"]).item()),
                count=int(np.asarray(npz[f"{name}__count"]).item()),
                energy_mean=float(np.asarray(npz[f"{name}__energy_mean"]).item()),
                mask=np.asarray(npz[f"{name}__mask"], dtype=bool),
            )
    return Stats(
        stats_dir=stats_dir,
        input_names=list(transforms_meta["inputs"]),
        field_names=list(transforms_meta["fields"]),
        per_level=bool(transforms_meta["per_level"]),
        asr_olr_normalize_by_f_star=bool(transforms_meta["asr_olr_normalize_by_f_star"]),
        strategies={k: (v[0], dict(v[1])) for k, v in transforms_meta["strategies"].items()},
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        spectral=spectral,
        spectral_meta=spectral_meta,
        transforms_meta=transforms_meta,
    )


def transform_inputs(X: np.ndarray, stats: Stats) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    squeeze = X.ndim == 1
    X = X[None, :] if squeeze else X
    cols = []
    for i, name in enumerate(stats.input_names):
        strategy_name, kwargs = _strategy_params(name, stats)
        key = _stat_key(name, stats)
        processed = _preprocess_array(X[:, i], strategy_name, kwargs)
        cols.append((processed - stats.normalize_mean[key].item()) / stats.normalize_std[key].item())
    out = np.stack(cols, axis=1).astype(np.float32)
    return out[0] if squeeze else out


def inverse_transform_inputs(X_t: np.ndarray, stats: Stats) -> np.ndarray:
    X_t = np.asarray(X_t, dtype=np.float32)
    squeeze = X_t.ndim == 1
    X_t = X_t[None, :] if squeeze else X_t
    cols = []
    for i, name in enumerate(stats.input_names):
        strategy_name, kwargs = _strategy_params(name, stats)
        key = _stat_key(name, stats)
        preprocessed = X_t[:, i] * stats.normalize_std[key].item() + stats.normalize_mean[key].item()
        cols.append(_inverse_preprocess_array(preprocessed, strategy_name, kwargs))
    out = np.stack(cols, axis=1).astype(np.float32)
    return out[0] if squeeze else out


def preprocess_outputs_grid(Y: np.ndarray, field_names: list[str], stats: Stats, *, X: np.ndarray | None = None) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    squeeze = Y.ndim == 3
    Y = Y[None, ...] if squeeze else Y
    X = None if X is None else np.asarray(X, dtype=np.float32)
    X = X[None, :] if X is not None and X.ndim == 1 else X
    if stats.asr_olr_normalize_by_f_star and any(_base_var(name) in ASR_OLR_FIELDS for name in field_names) and X is None:
        raise ValueError("X is required for ASR/OLR preprocessing when asr_olr_normalize_by_f_star is enabled.")
    f_star = None if X is None else X[:, stats.input_names.index("F_star")][:, None, None]
    out = Y.copy()
    for j, name in enumerate(field_names):
        strategy_name, kwargs = _strategy_params(name, stats)
        channel = out[:, j]
        if stats.asr_olr_normalize_by_f_star and _base_var(name) in ASR_OLR_FIELDS:
            channel = channel / f_star
        out[:, j] = _preprocess_array(channel, strategy_name, kwargs)
    return out[0] if squeeze else out


def inverse_preprocess_outputs_grid(Y_pp: np.ndarray, field_names: list[str], stats: Stats, *, X: np.ndarray | None = None) -> np.ndarray:
    Y_pp = np.asarray(Y_pp, dtype=np.float32)
    squeeze = Y_pp.ndim == 3
    Y_pp = Y_pp[None, ...] if squeeze else Y_pp
    X = None if X is None else np.asarray(X, dtype=np.float32)
    X = X[None, :] if X is not None and X.ndim == 1 else X
    if stats.asr_olr_normalize_by_f_star and any(_base_var(name) in ASR_OLR_FIELDS for name in field_names) and X is None:
        raise ValueError("X is required for ASR/OLR inverse preprocessing when asr_olr_normalize_by_f_star is enabled.")
    f_star = None if X is None else X[:, stats.input_names.index("F_star")][:, None, None]
    out = Y_pp.copy()
    for j, name in enumerate(field_names):
        strategy_name, kwargs = _strategy_params(name, stats)
        channel = _inverse_preprocess_array(out[:, j], strategy_name, kwargs)
        if stats.asr_olr_normalize_by_f_star and _base_var(name) in ASR_OLR_FIELDS:
            channel = channel * f_star
        out[:, j] = channel
    return out[0] if squeeze else out


def normalise_spectral(coeffs: np.ndarray, field_names: list[str], stats: Stats) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float32)
    squeeze = coeffs.ndim == 2
    coeffs = coeffs[None, ...] if squeeze else coeffs
    out = np.zeros_like(coeffs)
    for j, name in enumerate(field_names):
        record = stats.spectral[name]
        out[:, j, record.mask] = (coeffs[:, j, record.mask] - record.mean) / record.sigma
    return out[0] if squeeze else out


def unnormalise_spectral(coeffs_n: np.ndarray, field_names: list[str], stats: Stats) -> np.ndarray:
    coeffs_n = np.asarray(coeffs_n, dtype=np.float32)
    squeeze = coeffs_n.ndim == 2
    coeffs_n = coeffs_n[None, ...] if squeeze else coeffs_n
    out = np.zeros_like(coeffs_n)
    for j, name in enumerate(field_names):
        record = stats.spectral[name]
        out[:, j, record.mask] = coeffs_n[:, j, record.mask] * record.sigma + record.mean
    return out[0] if squeeze else out


def build_design_matrix(X_std: np.ndarray, gcm_idx: np.ndarray, *, n_gcm: int, design_cfg: dict) -> np.ndarray:
    X_std = np.asarray(X_std, dtype=np.float64)
    gcm_idx = np.asarray(gcm_idx, dtype=np.int64)
    cols = []
    if design_cfg.get("intercept", True):
        cols.append(np.ones((X_std.shape[0], 1), dtype=np.float64))
    if design_cfg.get("inputs", True):
        cols.append(X_std)
    if design_cfg.get("sim_onehot", False):
        oh = np.eye(int(n_gcm), dtype=np.float64)[gcm_idx]
        if design_cfg.get("intercept", True):
            oh = oh[:, 1:]
        if oh.size:
            cols.append(oh)
    if not cols:
        raise ValueError("Design matrix has no columns.")
    return np.concatenate(cols, axis=1)


def fit_linear_trend(
    X_std: np.ndarray,
    gcm_idx: np.ndarray,
    coeffs_norm: np.ndarray,
    *,
    lambda_reg: float = 1.0e-3,
    field_mask: np.ndarray | None = None,
    coeff_mask: np.ndarray | None = None,
    sh_mask: np.ndarray | None = None,
    design_cfg: dict | None = None,
    gcm_order: list[int] | None = None,
) -> LinearTrend:
    coeffs_norm = np.asarray(coeffs_norm, dtype=np.float64)
    n, f, a = coeffs_norm.shape
    design_cfg = {"intercept": False, "inputs": True, "sim_onehot": True} if design_cfg is None else dict(design_cfg)
    n_gcm = int(np.max(gcm_idx)) + 1
    H = build_design_matrix(X_std, gcm_idx, n_gcm=n_gcm, design_cfg=design_cfg)
    lamI = float(lambda_reg) * np.eye(H.shape[1], dtype=np.float64)
    if field_mask is None and coeff_mask is None and sh_mask is None:
        gamma = np.linalg.solve(H.T @ H + lamI, H.T @ coeffs_norm.reshape(n, -1))
        return LinearTrend(gamma.reshape(H.shape[1], f, a).astype(np.float32), design_cfg, float(lambda_reg), n_gcm, gcm_order)

    obs = _field_mask_to_coeff_obs(field_mask, coeff_mask, sh_mask, coeffs_norm.shape)
    Y_flat = coeffs_norm.reshape(n, -1)
    obs_flat = obs.reshape(n, -1)
    Gamma_flat = np.zeros((H.shape[1], f * a), dtype=np.float64)
    pattern_map: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    for j in range(obs_flat.shape[1]):
        m = obs_flat[:, j]
        key = m.astype(np.uint8).tobytes()
        if key not in pattern_map:
            pattern_map[key] = (m, [])
        pattern_map[key][1].append(j)
    for m, idxs_list in pattern_map.values():
        if not np.any(m):
            continue
        idxs = np.asarray(idxs_list, dtype=np.int64)
        Hm = H[m]
        Ym = Y_flat[m][:, idxs]
        gamma = np.linalg.solve(Hm.T @ Hm + lamI, Hm.T @ Ym)
        Gamma_flat[:, idxs] = gamma
    return LinearTrend(Gamma_flat.reshape(H.shape[1], f, a).astype(np.float32), design_cfg, float(lambda_reg), n_gcm, gcm_order)


def apply_linear_trend(X_std: np.ndarray, gcm_idx: np.ndarray, trend: LinearTrend) -> np.ndarray:
    H = build_design_matrix(X_std, gcm_idx, n_gcm=trend.n_gcm, design_cfg=trend.design_cfg)
    return np.einsum("np,pfa->nfa", H.astype(np.float32), trend.Gamma.astype(np.float32))


def remove_linear_trend(
    X_std: np.ndarray,
    gcm_idx: np.ndarray,
    coeffs_norm: np.ndarray,
    trend: LinearTrend,
    *,
    field_mask: np.ndarray | None = None,
) -> np.ndarray:
    residual = np.asarray(coeffs_norm, dtype=np.float32) - apply_linear_trend(X_std, gcm_idx, trend)
    return np.where(np.asarray(field_mask, dtype=bool)[:, :, None], residual, 0.0) if field_mask is not None else residual
