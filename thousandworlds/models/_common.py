from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re

import numpy as np
import torch
import thousandworlds as tw

LEVEL_RE = re.compile(r"^(.+?)_(\d+)$")
REPO_ROOT = Path(__file__).resolve().parents[2]
TW_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = TW_ROOT / "dataset"
DEFAULT_RESULTS_DIR = TW_ROOT / "results" / "models"


def resolve_torch_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    spec = str(device).lower()
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
            return torch.device("xpu")
        return torch.device("cpu")
    return torch.device(spec)


@dataclass
class PreparedTWData:
    grid_bundle: tw.DataBundle
    spectral_bundle: tw.DataBundle
    stats: tw.Stats
    X_train_std: np.ndarray
    X_test_std: np.ndarray
    s_train: np.ndarray
    s_test: np.ndarray
    sh_mask: np.ndarray  # (A, F)
    inverse_sht: np.ndarray
    gcm_labels: list[str]


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    return DEFAULT_DATA_DIR if data_dir is None else Path(data_dir)


def default_output_dir(method: str, subset: str) -> Path:
    return DEFAULT_RESULTS_DIR / subset / method


def load_stats_dir(stats_dir: str | Path) -> tw.Stats:
    stats_dir = Path(stats_dir)
    transforms_meta = json.loads((stats_dir / "transforms.meta.json").read_text())
    spectral_meta = json.loads((stats_dir / "spectral.meta.json").read_text())
    with np.load(stats_dir / "normalize_mean.npz", allow_pickle=False) as npz:
        normalize_mean = {k: np.asarray(v) for k, v in npz.items()}
    with np.load(stats_dir / "normalize_std.npz", allow_pickle=False) as npz:
        normalize_std = {k: np.asarray(v) for k, v in npz.items()}
    with np.load(stats_dir / "spectral.npz", allow_pickle=False) as npz:
        spectral = {
            name: tw.preprocessing.SpectralFieldStats(
                mean=np.asarray(npz[f"{name}__mean"], dtype=np.float32),
                sigma=float(np.asarray(npz[f"{name}__sigma"]).item()),
                count=int(np.asarray(npz[f"{name}__count"]).item()),
                energy_mean=float(np.asarray(npz[f"{name}__energy_mean"]).item()),
                mask=np.asarray(npz[f"{name}__mask"], dtype=bool),
            )
            for name in transforms_meta["fields"]
        }
    return tw.Stats(
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


def prepare_tw_data(
    subset: str,
    protocol: str = "standard",
    *,
    data_dir: str | Path | None = None,
    stats_dir: str | Path | None = None,
) -> PreparedTWData:
    data_dir = resolve_data_dir(data_dir)
    grid_bundle = tw.load(subset=subset, protocol=protocol, data_dir=data_dir, space="grid")
    spectral_bundle = tw.load(subset=subset, protocol=protocol, data_dir=data_dir, space="spectral")
    stats = tw.load_stats(subset, data_dir) if stats_dir is None else load_stats_dir(stats_dir)
    X_train_std = tw.transform_inputs(grid_bundle.X_train, stats)
    X_test_std = tw.transform_inputs(grid_bundle.X_test, stats)
    gcm_map_path = Path(stats.stats_dir) / "gcm_to_idx.json"
    gcm_to_idx = json.loads(gcm_map_path.read_text()) if gcm_map_path.exists() else {
        name: i for i, name in enumerate(sorted(set(grid_bundle.meta_train["gcm_label"]).union(grid_bundle.meta_test["gcm_label"])))
    }
    gcm_labels = [name for name, _ in sorted(gcm_to_idx.items(), key=lambda item: item[1])]
    s_train = grid_bundle.meta_train["gcm_label"].map(gcm_to_idx).to_numpy(dtype=np.int64)
    s_test = grid_bundle.meta_test["gcm_label"].map(gcm_to_idx).to_numpy(dtype=np.int64)
    sh_mask = np.stack([stats.spectral[name].mask for name in spectral_bundle.raw_field_names], axis=1)
    return PreparedTWData(
        grid_bundle=grid_bundle,
        spectral_bundle=spectral_bundle,
        stats=stats,
        X_train_std=X_train_std.astype(np.float32),
        X_test_std=X_test_std.astype(np.float32),
        s_train=s_train,
        s_test=s_test,
        sh_mask=sh_mask.astype(bool),
        inverse_sht=tw.load_inverse_sht_matrix(),
        gcm_labels=gcm_labels,
    )


def average_space_grid(Y: np.ndarray, field_names: list[str], stats: tw.Stats, *, X: np.ndarray) -> np.ndarray:
    return tw.preprocess_outputs_grid(Y, field_names, stats, X=X).astype(np.float32)


def inverse_average_space_grid(Y: np.ndarray, field_names: list[str], stats: tw.Stats, *, X: np.ndarray) -> np.ndarray:
    return tw.inverse_preprocess_outputs_grid(Y, field_names, stats, X=X).astype(np.float32)


def enforce_equatorial_symmetry_grid(Y: np.ndarray, field_names: list[str]) -> np.ndarray:
    out = np.asarray(Y, dtype=np.float32).copy()
    for i, name in enumerate(field_names):
        flipped = np.flip(out[..., i, :, :], axis=-2)
        out[..., i, :, :] = 0.5 * (out[..., i, :, :] - flipped) if _field_group(name) == "v" else 0.5 * (out[..., i, :, :] + flipped)
    return out


def normalised_spectral(coeffs: np.ndarray, field_names: list[str], stats: tw.Stats) -> np.ndarray:
    return tw.normalise_spectral(coeffs, field_names, stats).astype(np.float32)


def decode_spectral_predictions(
    coeffs_norm: np.ndarray,
    field_names: list[str],
    stats: tw.Stats,
    *,
    sh_mask: np.ndarray,
    inverse_sht: np.ndarray | None = None,
) -> np.ndarray:
    inverse_sht = tw.load_inverse_sht_matrix() if inverse_sht is None else inverse_sht

    def _decode(arr: np.ndarray) -> np.ndarray:
        coeffs = tw.unnormalise_spectral(arr, field_names, stats)
        coeffs = tw.apply_symmetry_mask(coeffs, field_names, sh_mask.T)
        return tw.to_grid(coeffs, inverse_sht).astype(np.float32)

    coeffs_norm = np.asarray(coeffs_norm, dtype=np.float32)
    return np.stack([_decode(sample) for sample in coeffs_norm], axis=0) if coeffs_norm.ndim == 4 else _decode(coeffs_norm)


def masked_mean_grid(Y: np.ndarray, field_mask: np.ndarray | None = None) -> np.ndarray:
    Y = np.asarray(Y, dtype=np.float32)
    valid = np.isfinite(Y)
    if field_mask is not None:
        valid &= np.asarray(field_mask, dtype=bool)[:, :, None, None]
    count = valid.sum(axis=0)
    total = np.where(valid, Y, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total, dtype=np.float32), where=count > 0)
    return mean.astype(np.float32)


def masked_mse_grid(pred: np.ndarray, target: np.ndarray, field_mask: np.ndarray | None = None) -> float:
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0)
    target = np.nan_to_num(np.asarray(target, dtype=np.float32), nan=0.0)
    err = np.square(pred - target)
    if field_mask is None:
        return float(err.mean())
    mask = np.asarray(field_mask, dtype=np.float32)[:, :, None, None]
    denom = float(mask.sum() * err.shape[-1] * err.shape[-2])
    return float((err * mask).sum() / max(denom, 1.0))


def _weighted_mse_per_example_field(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    w = tw.load_latitude_weights().astype(np.float32)
    w = w / w.sum()
    return (np.square(pred - target).mean(axis=-1) * w[None, None, :]).sum(axis=-1)


def _per_field_mean(values: np.ndarray, field_mask: np.ndarray | None = None) -> np.ndarray:
    if field_mask is None:
        return values.mean(axis=0)
    mask = np.asarray(field_mask, dtype=np.float32)
    count = mask.sum(axis=0)
    total = (values * mask).sum(axis=0)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float32), where=count > 0)


def _field_group(name: str) -> str:
    match = LEVEL_RE.match(name)
    return match.group(1) if match else name


def field_rmse_scale_grid(Y: np.ndarray, field_mask: np.ndarray | None = None, eps: float = 1.0e-6) -> np.ndarray:
    Y = np.nan_to_num(np.asarray(Y, dtype=np.float32), nan=0.0)
    center = masked_mean_grid(Y, field_mask)
    scale = np.sqrt(np.maximum(
        _per_field_mean(_weighted_mse_per_example_field(Y, center[None]), field_mask),
        eps**2,
    ))
    return np.nan_to_num(scale, nan=eps, posinf=eps, neginf=eps).astype(np.float32)


def equal_group_normalized_rmse_grid(
    pred: np.ndarray,
    target: np.ndarray,
    field_names: list[str],
    field_scale: np.ndarray,
    field_mask: np.ndarray | None = None,
) -> float:
    pred = np.nan_to_num(np.asarray(pred, dtype=np.float32), nan=0.0)
    target = np.nan_to_num(np.asarray(target, dtype=np.float32), nan=0.0)
    scale = np.asarray(field_scale, dtype=np.float32)[None, :, None, None]
    per_field = _per_field_mean(
        np.sqrt(_weighted_mse_per_example_field(pred / scale, target / scale) + 1.0e-12),
        field_mask,
    )
    group_scores = []
    for group in dict.fromkeys(_field_group(name) for name in field_names):
        values = np.asarray([per_field[i] for i, name in enumerate(field_names) if _field_group(name) == group])
        if np.any(np.isfinite(values)):
            group_scores.append(float(np.nanmean(values)))
    return float(np.mean(group_scores)) if group_scores else float("inf")


def save_submission(path: str | Path, predictions: np.ndarray, *, simulation_id: np.ndarray, field_names: list[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = np.asarray(predictions, dtype=np.float32)
    preds = preds[None, ...] if preds.ndim == 4 else preds
    np.savez_compressed(
        path,
        predictions=preds,
        simulation_id=np.asarray(simulation_id, dtype=np.int32),
        field_names=np.asarray(field_names),
    )
    return path


def subset_submission(predictions: np.ndarray, simulation_id: np.ndarray, *, wanted_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = dict(zip(np.asarray(simulation_id, dtype=np.int32).tolist(), range(len(simulation_id))))
    idx = np.asarray([lookup[int(sim_id)] for sim_id in np.asarray(wanted_ids, dtype=np.int32)], dtype=np.int32)
    preds = np.asarray(predictions, dtype=np.float32)
    preds = preds[None, ...] if preds.ndim == 4 else preds
    return preds[:, idx], np.asarray(simulation_id, dtype=np.int32)[idx]


def score_saved_submission(path: str | Path, *, data_dir: str | Path, subset: str, protocol: str, point_predictions_path: str | Path | None = None) -> dict:
    return tw.evaluate.score(path, data_dir=data_dir, subset=subset, protocol=protocol, point_predictions_path=point_predictions_path)


def _to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def save_json(path: str | Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2) + "\n", encoding="utf-8")
    return path
