from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

from .data import load as load_bundle
from .field_spec import public_name
from .spectral import load_latitude_weights

SPLIT_VAR_REGEX = re.compile(r"^(.+?)_(\d+)$")
VARIABLE_ORDER = (
    "surface_temperature",
    "temperature",
    "specific_humidity_dex",
    "cloud_fraction",
    "u",
    "v",
    "asr",
    "olr",
)
VARIABLE_LABELS = {
    "surface_temperature": "surface temperature / K",
    "temperature": "temperature mean / K",
    "specific_humidity_dex": "specific humidity mean / dex",
    "cloud_fraction": "cloud fraction mean / 1",
    "u": "east-west wind mean / m s^-1",
    "v": "north-south wind mean / m s^-1",
    "asr": "absorbed shortwave radiation / W m^-2",
    "olr": "outgoing longwave radiation / W m^-2",
}


def _lat_weights() -> np.ndarray:
    w = load_latitude_weights().astype(np.float32)
    return w / w.sum()


def _sanitize_grid(arr: np.ndarray, field_mask: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if field_mask is None:
        return arr
    mask = np.asarray(field_mask, dtype=bool)[None, ...] if arr.ndim == 5 else np.asarray(field_mask, dtype=bool)
    while mask.ndim < arr.ndim:
        mask = mask[..., None]
    return np.where(mask, np.nan_to_num(arr, nan=0.0), 0.0)


def _per_field_mean_with_mask(per_example_per_field: np.ndarray, field_mask: np.ndarray | None) -> np.ndarray:
    if field_mask is None:
        return per_example_per_field.mean(axis=0)
    mask = np.asarray(field_mask, dtype=np.float32)
    count = mask.sum(axis=0)
    total = (per_example_per_field * mask).sum(axis=0)
    out = np.full(total.shape, np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _base_var(name: str) -> str:
    match = SPLIT_VAR_REGEX.match(name)
    return match.group(1) if match else name


def _variable_key(name: str) -> str:
    base = _base_var(name)
    if base == "specific_humidity":
        return "specific_humidity_dex"
    return public_name(base)


def _field_metric_name(name: str, *, dex_specific_humidity: bool) -> str:
    name = public_name(name)
    return f"{name}_dex" if dex_specific_humidity and name.startswith("specific_humidity") else name


def _variable_means(per_field: np.ndarray, field_names: list[str]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i, name in enumerate(field_names):
        value = float(per_field[i])
        if value != value:
            continue
        key = _variable_key(name)
        sums[key] = sums.get(key, 0.0) + value
        counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / counts[key] for key in VARIABLE_ORDER if counts.get(key, 0) > 0}


def _masked_row_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0)
    mask = np.asarray(mask, dtype=np.float32)
    count = mask.sum(axis=1)
    total = (values * mask).sum(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=np.float32)
    np.divide(total, count, out=out, where=count > 0)
    return out


def _relative_variable_means(
    num: np.ndarray,
    den: np.ndarray,
    valid: np.ndarray,
    field_names: list[str],
    *,
    eps: float = 1.0e-6,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in VARIABLE_ORDER:
        idx = [i for i, name in enumerate(field_names) if _variable_key(name) == key]
        if not idx:
            continue
        mask = np.asarray(valid[:, idx], dtype=bool)
        num_g = _masked_row_mean(num[:, idx], mask)
        den_g = _masked_row_mean(den[:, idx], mask)
        ratio = np.full_like(num_g, np.nan)
        ok = np.isfinite(num_g) & np.isfinite(den_g) & (den_g > float(eps))
        np.divide(num_g, den_g, out=ratio, where=ok)
        if np.any(ok):
            out[key] = float(np.nanmean(ratio))
    return out


def _metric_space(arr: np.ndarray, field_names: list[str], *, humidity: str | None) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32).copy()
    if humidity is None:
        return out
    for i, name in enumerate(field_names):
        if not name.startswith("specific_humidity"):
            continue
        if humidity == "ln":
            out[..., i, :, :] = np.log(np.clip(out[..., i, :, :], 1.0e-12, None))
        elif humidity == "dex":
            out[..., i, :, :] = np.log10(np.clip(out[..., i, :, :], 1.0e-12, None))
        else:
            raise ValueError(f"Unknown humidity metric space: {humidity}")
    return out


def _weighted_mse_per_example_field(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    error = np.square(pred - target)
    lon_mean = error.mean(axis=-1)
    return (lon_mean * _lat_weights()[None, None, :]).sum(axis=-1)


def _acc_per_example_field(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    w = (_lat_weights()[None, None, :, None] / pred.shape[-1]).astype(np.float32)
    mu_pred = (pred * w).sum(axis=(-2, -1), keepdims=True)
    mu_target = (target * w).sum(axis=(-2, -1), keepdims=True)
    pred0 = pred - mu_pred
    target0 = target - mu_target
    dot = (w * pred0 * target0).sum(axis=(-2, -1))
    norm_pred = np.sqrt((w * np.square(pred0)).sum(axis=(-2, -1)) + 1.0e-12)
    norm_target = np.sqrt((w * np.square(target0)).sum(axis=(-2, -1)) + 1.0e-12)
    return dot / (norm_pred * norm_target + 1.0e-12)


def _energy_score_per_example_field(samples: np.ndarray, target: np.ndarray, *, n_pairs_mc: int | None = None, seed: int = 0) -> np.ndarray:
    M, N, F, H, W = samples.shape
    sqrt_w = np.sqrt(_lat_weights()[None, None, None, :, None] / W).astype(np.float32)
    out = np.empty((N, F), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for f in range(F):
        samples_flat = (samples[:, :, f] * sqrt_w[0, 0, 0]).reshape(M, N, -1)
        target_flat = (target[:, f] * sqrt_w[0, 0, 0]).reshape(N, -1)
        diffs = samples_flat - target_flat[None]
        term1 = np.sqrt(np.square(diffs).sum(axis=-1) + 1.0e-12).mean(axis=0)
        if M == 1:
            out[:, f] = term1
            continue

        X = np.transpose(samples_flat, (1, 0, 2))
        if n_pairs_mc is None or n_pairs_mc >= (M * (M - 1) // 2):
            sq_norm = np.square(X).sum(axis=-1)
            gram = np.einsum("bmd,bnd->bmn", X, X)
            dmat = np.sqrt(np.clip(sq_norm[:, :, None] + sq_norm[:, None, :] - 2.0 * gram, 0.0, None) + 1.0e-12)
            offdiag_sum = dmat.sum(axis=(1, 2)) - np.trace(dmat, axis1=1, axis2=2)
            term2 = 0.5 * (offdiag_sum / (M * (M - 1)))
        else:
            i_idx = rng.integers(0, M, size=(N, n_pairs_mc))
            j_raw = rng.integers(0, M - 1, size=(N, n_pairs_mc))
            j_idx = j_raw + (j_raw >= i_idx)
            Xi = X[np.arange(N)[:, None], i_idx]
            Xj = X[np.arange(N)[:, None], j_idx]
            term2 = 0.5 * np.sqrt(np.square(Xi - Xj).sum(axis=-1) + 1.0e-12).mean(axis=-1)
        out[:, f] = term1 - term2
    return out


def _spread_skill_ratio_per_field(samples: np.ndarray, target: np.ndarray, field_mask: np.ndarray | None = None) -> np.ndarray:
    M, _, _, _, W = samples.shape
    if M < 2:
        raise ValueError("spread_skill_ratio requires at least 2 samples.")
    sqrt_w = np.sqrt(_lat_weights()[None, None, None, :, None] / W).astype(np.float32)
    samples_flat = (samples * sqrt_w).reshape(M, *samples.shape[1:3], -1)
    target_flat = (target * sqrt_w[0]).reshape(*target.shape[:2], -1)
    mean_flat = samples_flat.mean(axis=0)
    spread2 = np.square(samples_flat - mean_flat[None]).sum(axis=(0, -1)) / (M - 1)
    mse = np.square(mean_flat - target_flat).sum(axis=-1)
    if field_mask is None:
        return np.sqrt(spread2.sum(axis=0) / np.maximum(mse.sum(axis=0), 1.0e-12))
    mask = np.asarray(field_mask, dtype=np.float32)
    count = mask.sum(axis=0)
    num = (spread2 * mask).sum(axis=0)
    den = np.maximum((mse * mask).sum(axis=0), 1.0e-12)
    out = np.full(num.shape, np.nan, dtype=np.float32)
    np.sqrt(num / den, out=out, where=count > 0)
    return out


def _result(per_field: np.ndarray, field_names: list[str] | None, *, dex_specific_humidity: bool = False) -> dict:
    per_variable = _variable_means(per_field, field_names) if field_names is not None else {}
    return {
        "per_field": (
            dict(zip([_field_metric_name(name, dex_specific_humidity=dex_specific_humidity) for name in field_names], per_field.tolist()))
            if field_names is not None
            else per_field
        ),
        "per_variable": per_variable,
    }


def _per_field_vector(result: dict, field_names: list[str], *, dex_specific_humidity: bool = False) -> np.ndarray:
    return np.asarray([result["per_field"][_field_metric_name(name, dex_specific_humidity=dex_specific_humidity)] for name in field_names], dtype=np.float32)


def _relative_result(
    num: np.ndarray,
    den: np.ndarray,
    valid: np.ndarray,
    field_names: list[str] | None,
    *,
    dex_specific_humidity: bool = False,
) -> dict:
    per_field = _per_field_mean_with_mask(np.divide(num, den, out=np.zeros_like(num, dtype=np.float32), where=valid), valid)
    return {
        "per_field": (
            dict(zip([_field_metric_name(name, dex_specific_humidity=dex_specific_humidity) for name in field_names], per_field.tolist()))
            if field_names is not None
            else per_field
        ),
        "per_variable": _relative_variable_means(num, den, valid, field_names) if field_names is not None else {},
    }


def _shared_planets_group_result(
    num_exo: np.ndarray,
    num_um: np.ndarray,
    den: np.ndarray,
    pair_mask: np.ndarray,
    field_names: list[str],
    *,
    dex_specific_humidity: bool = False,
) -> dict:
    _, valid = _relative_ratio(num_exo, den, pair_mask)
    ratio_exo = np.divide(num_exo, den, out=np.zeros_like(num_exo, dtype=np.float32), where=valid)
    ratio_um = np.divide(num_um, den, out=np.zeros_like(num_um, dtype=np.float32), where=valid)
    per_field = _per_field_mean_with_mask(0.5 * (ratio_exo + ratio_um), valid)
    per_variable: dict[str, float] = {}
    for key in VARIABLE_ORDER:
        idx = [i for i, name in enumerate(field_names) if _variable_key(name) == key]
        if not idx:
            continue
        mask = np.asarray(pair_mask[:, idx], dtype=bool)
        num_group = 0.5 * (
            _masked_row_mean(num_exo[:, idx], mask)
            + _masked_row_mean(num_um[:, idx], mask)
        )
        den_group = _masked_row_mean(den[:, idx], mask)
        ratio = np.full_like(num_group, np.nan)
        ok = np.isfinite(num_group) & np.isfinite(den_group) & (den_group > 1.0e-6)
        np.divide(num_group, den_group, out=ratio, where=ok)
        if np.any(ok):
            per_variable[key] = float(np.nanmean(ratio))
    return {
        "per_field": dict(zip([_field_metric_name(name, dex_specific_humidity=dex_specific_humidity) for name in field_names], per_field.tolist())),
        "per_variable": per_variable,
    }


def _relative_ratio(
    num: np.ndarray,
    den: np.ndarray,
    field_mask: np.ndarray | None,
    *,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    valid = den > float(eps)
    if field_mask is not None:
        valid &= np.asarray(field_mask, dtype=bool)
    ratio = np.zeros_like(num, dtype=np.float32)
    np.divide(num, den, out=ratio, where=valid)
    return ratio, valid


def rmse(
    pred: np.ndarray,
    target: np.ndarray,
    field_mask: np.ndarray | None = None,
    field_names: list[str] | None = None,
    *,
    dex_specific_humidity: bool = False,
) -> dict:
    pred = _sanitize_grid(pred, field_mask)
    target = _sanitize_grid(target, field_mask)
    per_field = _per_field_mean_with_mask(np.sqrt(_weighted_mse_per_example_field(pred, target) + 1.0e-12), field_mask)
    return _result(per_field, field_names, dex_specific_humidity=dex_specific_humidity)


def acc(
    pred: np.ndarray,
    target: np.ndarray,
    field_mask: np.ndarray | None = None,
    field_names: list[str] | None = None,
    *,
    dex_specific_humidity: bool = False,
) -> dict:
    pred = _sanitize_grid(pred, field_mask)
    target = _sanitize_grid(target, field_mask)
    per_field = _per_field_mean_with_mask(_acc_per_example_field(pred, target), field_mask)
    return _result(per_field, field_names, dex_specific_humidity=dex_specific_humidity)


def energy_score(
    samples: np.ndarray,
    target: np.ndarray,
    field_mask: np.ndarray | None = None,
    field_names: list[str] | None = None,
    *,
    n_pairs_mc: int | None = None,
    seed: int = 0,
) -> dict:
    samples = _sanitize_grid(samples, field_mask)
    target = _sanitize_grid(target, field_mask)
    per_field = _per_field_mean_with_mask(
        _energy_score_per_example_field(samples, target, n_pairs_mc=n_pairs_mc, seed=seed),
        field_mask,
    )
    return _result(per_field, field_names)


def spread_skill_ratio(
    samples: np.ndarray,
    target: np.ndarray,
    field_mask: np.ndarray | None = None,
    field_names: list[str] | None = None,
) -> dict:
    samples = _sanitize_grid(samples, field_mask)
    target = _sanitize_grid(target, field_mask)
    return _result(_spread_skill_ratio_per_field(samples, target, field_mask), field_names)


def relative_rmse(
    pred: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    field_mask: np.ndarray | None = None,
    field_names: list[str] | None = None,
) -> dict:
    pred = _sanitize_grid(pred, field_mask)
    target_a = _sanitize_grid(target_a, field_mask)
    target_b = _sanitize_grid(target_b, field_mask)
    num = np.sqrt(_weighted_mse_per_example_field(pred, target_a))
    den = np.sqrt(_weighted_mse_per_example_field(target_a, target_b))
    _, valid = _relative_ratio(num, den, field_mask)
    return _relative_result(num, den, valid, field_names, dex_specific_humidity=True)


def relative_acc(
    pred: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    field_mask: np.ndarray | None = None,
    field_names: list[str] | None = None,
) -> dict:
    pred = _sanitize_grid(pred, field_mask)
    target_a = _sanitize_grid(target_a, field_mask)
    target_b = _sanitize_grid(target_b, field_mask)
    num = _acc_per_example_field(pred, target_a)
    den = _acc_per_example_field(target_a, target_b)
    _, valid = _relative_ratio(num, den, field_mask)
    return _relative_result(num, den, valid, field_names, dex_specific_humidity=True)


def relative_energy_score(
    samples: np.ndarray,
    target_a: np.ndarray,
    target_b: np.ndarray,
    field_mask: np.ndarray | None = None,
    field_names: list[str] | None = None,
    *,
    n_pairs_mc: int | None = None,
    seed: int = 0,
) -> dict:
    samples = _sanitize_grid(samples, field_mask)
    target_a = _sanitize_grid(target_a, field_mask)
    target_b = _sanitize_grid(target_b, field_mask)
    num = _energy_score_per_example_field(samples, target_a, n_pairs_mc=n_pairs_mc, seed=seed)
    den = np.sqrt(_weighted_mse_per_example_field(target_a, target_b))
    _, valid = _relative_ratio(num, den, field_mask)
    return _relative_result(num, den, valid, field_names)


def _shared_planet_pair_indices(meta_test) -> list[tuple[int, int]]:
    pairs = []
    for _, group in meta_test.groupby("planet_id", sort=True):
        if len(group) != 2:
            raise ValueError("shared_planets test set must contain exactly two rows per planet_id.")
        idxs = group.index.to_list()
        gcms = group["gcm_label"].tolist()
        if set(gcms) != {"exocam", "um"}:
            raise ValueError(f"Expected exocam/um pair, got {gcms}.")
        i_exo = idxs[gcms.index("exocam")]
        i_um = idxs[gcms.index("um")]
        pairs.append((i_exo, i_um))
    return pairs


def score(
    predictions_path: str | Path,
    data_dir: str | Path,
    subset: str,
    protocol: str = "standard",
    *,
    include_probabilistic: bool = True,
    point_predictions_path: str | Path | None = None,
) -> dict:
    """Score a submission NPZ file against the benchmark test set.

    Submission NPZ format
    ---------------------
    The file must contain exactly three arrays:

    ``predictions`` : float32, shape ``(M, N, F, 32, 64)``
        M – ensemble members (use M=1 for point/deterministic predictions).
        N – test planets, ordered by ``simulation_id``.
        F – output fields, in canonical order (see ``field_names`` below).
        Spatial axes are (latitude, longitude) on the standard 32×64 grid.

    ``simulation_id`` : int32, shape ``(N,)``
        Simulation IDs for each test planet.  Must exactly match
        ``tw.load(subset, protocol, data_dir=...).test_ids``.

    ``field_names`` : array of strings, shape ``(F,)``
        Canonical field names for the subset, as returned by
        ``tw.canonical_field_names(subset)``.  Both public names
        (e.g. ``"specific_humidity_dex_1"``) and raw names
        (e.g. ``"specific_humidity_1"``) are accepted.

    Parameters
    ----------
    predictions_path:
        Path to the submission ``.npz`` file.
    data_dir:
        Path to the downloaded benchmark dataset (passed to ``tw.load``).
    subset:
        One of ``tw.BENCHMARK_SUBSETS``.
    protocol:
        ``"standard"`` (default) or ``"shared_planets"``.
    include_probabilistic:
        If true, compute probabilistic metrics for ensemble submissions.

    Returns
    -------
    dict with keys ``subset``, ``protocol``, ``n_test``, ``n_samples``,
    ``rmse``, and (if M > 1) ``energy_score`` and ``spread_skill_ratio``.
    For ``shared_planets`` protocol, also includes ``relative_rmse`` and (if M > 1)
    ``relative_energy_score``.  Each metric is a dict with ``per_field``
    and ``per_variable`` sub-dicts.
    """
    bundle = load_bundle(subset, protocol, data_dir=data_dir, space="grid")
    with np.load(Path(predictions_path), allow_pickle=False) as npz:
        predictions = np.asarray(npz["predictions"], dtype=np.float32)
        simulation_id = np.asarray(npz["simulation_id"], dtype=np.int32)
        field_names = np.asarray(npz["field_names"]).tolist()
    if predictions.ndim != 5:
        raise ValueError("predictions must have shape (M, N, F, 32, 64).")
    if simulation_id.tolist() != bundle.test_ids.tolist():
        raise ValueError("submission simulation_id does not match test split order.")
    if field_names == bundle.raw_field_names:
        field_names = list(bundle.field_names)
    if field_names != bundle.field_names:
        raise ValueError("submission field_names do not match benchmark canonical order.")
    return score_predictions(
        predictions,
        bundle,
        include_probabilistic=include_probabilistic,
        point_predictions=None
        if point_predictions_path is None
        else _load_point_predictions(
            point_predictions_path,
            expected_ids=bundle.test_ids,
            expected_field_names=bundle.field_names,
            raw_field_names=bundle.raw_field_names,
        ),
    )


def score_predictions(
    predictions: np.ndarray,
    bundle,
    *,
    include_probabilistic: bool = True,
    point_predictions: np.ndarray | None = None,
) -> dict:
    """Score in-memory predictions against a loaded ``tw.DataBundle``."""
    predictions = np.asarray(predictions, dtype=np.float32)
    predictions = predictions[None] if predictions.ndim == 4 else predictions
    if predictions.ndim != 5:
        raise ValueError("predictions must have shape (M, N, F, 32, 64).")
    if predictions.shape[1:] != (len(bundle.test_ids), len(bundle.field_names), 32, 64):
        raise ValueError(f"Unexpected predictions shape {predictions.shape}.")
    if point_predictions is not None:
        point_predictions = np.asarray(point_predictions, dtype=np.float32)
        point_predictions = point_predictions[None] if point_predictions.ndim == 4 else point_predictions
        if point_predictions.ndim != 5:
            raise ValueError("point predictions must have shape (M, N, F, 32, 64).")
        if point_predictions.shape[1:] != (len(bundle.test_ids), len(bundle.field_names), 32, 64):
            raise ValueError(f"Unexpected point predictions shape {point_predictions.shape}.")

    pred_prob_native = _metric_space(predictions, bundle.field_names, humidity="dex")
    target_point_dex = _metric_space(bundle.Y_test, bundle.field_names, humidity="dex")
    if point_predictions is None:
        pred_point_dex = pred_prob_native.mean(axis=0)
    else:
        pred_point_dex = _metric_space(point_predictions.mean(axis=0), bundle.field_names, humidity="dex")
    target_prob_native = None
    out = {
        "subset": getattr(bundle, "subset", None),
        "protocol": getattr(bundle, "protocol", None),
        "n_test": int(len(bundle.test_ids)),
        "n_samples": int(predictions.shape[0]),
        "rmse": rmse(pred_point_dex, target_point_dex, bundle.field_mask_test, bundle.field_names, dex_specific_humidity=True),
        "acc": acc(pred_point_dex, target_point_dex, bundle.field_mask_test, bundle.field_names, dex_specific_humidity=True),
    }
    if include_probabilistic and predictions.shape[0] > 1:
        target_prob_native = _metric_space(bundle.Y_test, bundle.field_names, humidity="dex")
        out["energy_score"] = energy_score(pred_prob_native, target_prob_native, bundle.field_mask_test, bundle.field_names)
        out["spread_skill_ratio"] = spread_skill_ratio(pred_prob_native, target_prob_native, bundle.field_mask_test, bundle.field_names)
    if getattr(bundle, "protocol", None) == "shared_planets":
        pairs = _shared_planet_pair_indices(bundle.meta_test)
        exo = np.asarray([i for i, _ in pairs], dtype=np.int32)
        um = np.asarray([j for _, j in pairs], dtype=np.int32)
        pair_mask = bundle.field_mask_test[exo] & bundle.field_mask_test[um]
        rr_num_exo = np.sqrt(_weighted_mse_per_example_field(_sanitize_grid(pred_point_dex[exo], pair_mask), _sanitize_grid(target_point_dex[exo], pair_mask)))
        rr_num_um = np.sqrt(_weighted_mse_per_example_field(_sanitize_grid(pred_point_dex[um], pair_mask), _sanitize_grid(target_point_dex[um], pair_mask)))
        rr_den = np.sqrt(_weighted_mse_per_example_field(_sanitize_grid(target_point_dex[exo], pair_mask), _sanitize_grid(target_point_dex[um], pair_mask)))
        ra_num_exo = _acc_per_example_field(_sanitize_grid(pred_point_dex[exo], pair_mask), _sanitize_grid(target_point_dex[exo], pair_mask))
        ra_num_um = _acc_per_example_field(_sanitize_grid(pred_point_dex[um], pair_mask), _sanitize_grid(target_point_dex[um], pair_mask))
        ra_den = _acc_per_example_field(_sanitize_grid(target_point_dex[exo], pair_mask), _sanitize_grid(target_point_dex[um], pair_mask))
        out["relative_rmse"] = _shared_planets_group_result(
            rr_num_exo,
            rr_num_um,
            rr_den,
            pair_mask,
            bundle.field_names,
            dex_specific_humidity=True,
        )
        out["relative_acc"] = _shared_planets_group_result(
            ra_num_exo,
            ra_num_um,
            ra_den,
            pair_mask,
            bundle.field_names,
            dex_specific_humidity=True,
        )
        if include_probabilistic and predictions.shape[0] > 1:
            out["relative_energy_score"] = _shared_planets_group_result(
                _energy_score_per_example_field(_sanitize_grid(pred_prob_native[:, exo], pair_mask), _sanitize_grid(target_prob_native[exo], pair_mask)),
                _energy_score_per_example_field(_sanitize_grid(pred_prob_native[:, um], pair_mask), _sanitize_grid(target_prob_native[um], pair_mask)),
                rr_den.copy(),
                pair_mask,
                bundle.field_names,
            )
    return out


def _load_point_predictions(
    path: str | Path,
    *,
    expected_ids: np.ndarray,
    expected_field_names: list[str],
    raw_field_names: list[str],
) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as npz:
        predictions = np.asarray(npz["predictions"], dtype=np.float32)
        simulation_id = np.asarray(npz["simulation_id"], dtype=np.int32)
        field_names = np.asarray(npz["field_names"]).tolist()
    if predictions.ndim == 4:
        predictions = predictions[None]
    if predictions.ndim != 5:
        raise ValueError("point predictions must have shape (M, N, F, 32, 64).")
    if simulation_id.tolist() != np.asarray(expected_ids, dtype=np.int32).tolist():
        raise ValueError("point prediction simulation_id does not match test split order.")
    if field_names == raw_field_names:
        field_names = list(expected_field_names)
    if field_names != expected_field_names:
        raise ValueError("point prediction field_names do not match benchmark canonical order.")
    return predictions


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m thousandworlds.evaluate", description="Score a ThousandWorlds prediction NPZ.")
    parser.add_argument("predictions_path", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--protocol", default="standard", choices=("standard", "shared_planets"))
    parser.add_argument("--include-probabilistic", action="store_true", help="Also compute energy score and spread-skill ratio for ensembles.")
    parser.add_argument("--point-predictions-path", type=Path, default=None, help="Optional deterministic point prediction NPZ for RMSE/ACC.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path. Defaults to stdout.")
    args = parser.parse_args(argv)
    result = score(
        args.predictions_path,
        data_dir=args.data_dir,
        subset=args.subset,
        protocol=args.protocol,
        include_probabilistic=args.include_probabilistic,
        point_predictions_path=args.point_predictions_path,
    )
    text = json.dumps(result, indent=2) + "\n"
    if args.out is None:
        sys.stdout.write(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
