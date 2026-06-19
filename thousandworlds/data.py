from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from .field_spec import CANONICAL_INPUT_NAMES, public_field_names
from .schema import (
    DEFAULT_DATA_ROOT,
    PROJECT_ROOT,
    SUBSET_TO_FIELDS,
    resolve_data_root,
    subset_path,
    support_path,
)

CSV_TO_INPUT = {
    "T_star": "stellar_temperature",
    "F_star": "stellar_flux",
    "radius": "radius",
    "gravity": "gravity",
    "P_rot": "rotation_period",
    "P0": "surface_pressure",
    "CO2": "co2",
    "CH4": "ch4",
}
DATASET_PAGE_URL = "https://doi.org/10.57967/hf/8695"
HF_REPO_ID = "es833/ThousandWorlds"
HF_REVISION = "v1.0.0"
DATASET_ARCHIVE = "dataset.tar.gz"
BASELINES_RESULTS_ARCHIVES = (
    "results-baselines-multi-partial-deterministic.tar.gz",
    "results-baselines-multi-complete-deterministic.tar.gz",
    "results-baselines-single-complete-deterministic.tar.gz",
    "results-baselines-multi-partial-gplfr.tar.gz",
    "results-baselines-multi-partial-ppca_icm.tar.gz",
    "results-baselines-multi-complete-gplfr.tar.gz",
    "results-baselines-multi-complete-ppca_icm.tar.gz",
    "results-baselines-single-complete-gplfr.tar.gz",
    "results-baselines-single-complete-ppca_icm.tar.gz",
)


@dataclass
class DataBundle:
    X_train: np.ndarray
    X_test: np.ndarray
    Y_train: np.ndarray
    Y_test: np.ndarray
    field_mask_train: np.ndarray
    field_mask_test: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    field_names: list[str]
    input_names: list[str]
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    space: str
    subset: str
    protocol: str
    raw_field_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.raw_field_names = list(self.field_names) if self.raw_field_names is None else list(self.raw_field_names)
        self.field_names = list(self.field_names)

def _load_split_ids(path: Path) -> np.ndarray:
    return pd.read_csv(path)["simulation_id"].to_numpy(dtype=np.int32)


def _load_archive(data_root: Path, subset: str, space: str) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray | None]:
    path = support_path(data_root, subset, kind="archive", space=space)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing benchmark archive {path}. "
            f"Place the extracted dataset under {data_root} or call thousandworlds.download_dataset(...)."
        )
    key = "fields" if space == "grid" else "coefficients"
    with np.load(path, allow_pickle=False) as npz:
        ids = np.asarray(npz["simulation_id"], dtype=np.int32)
        field_names = np.asarray(npz["field_names"]).tolist()
        values = np.asarray(npz[key], dtype=np.float32)
        field_mask = None if space == "grid" else np.asarray(npz["field_mask"], dtype=bool)
    return ids, field_names, values, field_mask


def _field_mask_from_grid(fields: np.ndarray) -> np.ndarray:
    nan_any = np.isnan(fields).any(axis=(-1, -2))
    nan_all = np.isnan(fields).all(axis=(-1, -2))
    if np.any(nan_any != nan_all):
        raise ValueError("Grid archives must use only whole-field missingness (all-NaN), not partial NaNs.")
    return ~nan_all


def _slice_archive(values: np.ndarray, ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    idx = np.asarray([dict(zip(ids.tolist(), range(len(ids))))[int(sim_id)] for sim_id in wanted], dtype=np.int32)
    return values[idx]


def _slice_archive_with_lookup(values: np.ndarray, ids: np.ndarray, wanted: np.ndarray) -> np.ndarray:
    lookup = dict(zip(ids.tolist(), range(len(ids))))
    return values[np.asarray([lookup[int(sim_id)] for sim_id in wanted], dtype=np.int32)]


def _build_X(meta: pd.DataFrame) -> np.ndarray:
    return np.stack([meta[CSV_TO_INPUT[name]].to_numpy(dtype=np.float32) for name in CANONICAL_INPUT_NAMES], axis=1)


def load(
    subset: str,
    protocol: str = "standard",
    *,
    data_dir: str | Path,
    space: str = "grid",
) -> DataBundle:
    data_root = resolve_data_root(data_dir)
    train_ids = _load_split_ids(subset_path(data_root, subset, split="train"))
    test_ids = _load_split_ids(subset_path(data_root, subset, split="test", protocol=protocol))

    inputs = pd.read_csv(data_root / "inputs.csv").set_index("simulation_id", drop=False)
    archive_ids, raw_field_names, values, field_mask = _load_archive(data_root, subset, space)
    expected = SUBSET_TO_FIELDS[subset]
    if raw_field_names != expected:
        raise ValueError(f"Unexpected field order for {subset}: {raw_field_names[:5]} ...")

    Y_train = _slice_archive_with_lookup(values, archive_ids, train_ids)
    Y_test = _slice_archive_with_lookup(values, archive_ids, test_ids)
    if space == "grid":
        fm_train = np.ones(Y_train.shape[:2], dtype=bool) if subset != "multi-partial" else _field_mask_from_grid(Y_train)
        fm_test = np.ones(Y_test.shape[:2], dtype=bool) if subset != "multi-partial" else _field_mask_from_grid(Y_test)
    else:
        assert field_mask is not None
        fm_train = _slice_archive_with_lookup(field_mask, archive_ids, train_ids)
        fm_test = _slice_archive_with_lookup(field_mask, archive_ids, test_ids)

    meta_train = inputs.loc[train_ids].reset_index(drop=True)
    meta_test = inputs.loc[test_ids].reset_index(drop=True)
    field_names = public_field_names(raw_field_names)
    return DataBundle(
        X_train=_build_X(meta_train),
        X_test=_build_X(meta_test),
        Y_train=Y_train,
        Y_test=Y_test,
        field_mask_train=fm_train,
        field_mask_test=fm_test,
        train_ids=train_ids,
        test_ids=test_ids,
        field_names=field_names,
        raw_field_names=raw_field_names,
        input_names=list(CANONICAL_INPUT_NAMES),
        meta_train=meta_train,
        meta_test=meta_test,
        space=space,
        subset=subset,
        protocol=protocol,
    )


def _checked_tar_members(tar: tarfile.TarFile, dest_dir: Path) -> list[tarfile.TarInfo]:
    dest_root = dest_dir.resolve()
    members = tar.getmembers()
    for member in members:
        target = (dest_dir / member.name).resolve()
        if target != dest_root and dest_root not in target.parents:
            raise ValueError(f"Archive member would extract outside destination: {member.name}")
    return members


def _existing_protected_members(
    members: list[tarfile.TarInfo],
    dest_dir: Path,
    protected_names: frozenset[str],
) -> list[str]:
    existing = []
    for member in members:
        if not member.isfile() or Path(member.name).name not in protected_names:
            continue
        if (dest_dir / member.name).exists():
            existing.append(member.name)
    return existing


def _extract_tar(
    archive_path: Path,
    dest_dir: Path,
    *,
    force: bool = False,
    protected_existing_names: frozenset[str] = frozenset(),
    skip_existing_members: tuple[str, ...] = (),
) -> None:
    with tarfile.open(archive_path) as tar:
        members = _checked_tar_members(tar, dest_dir)
        existing = [] if force else _existing_protected_members(members, dest_dir, protected_existing_names)
        if existing:
            preview = ", ".join(existing[:5])
            suffix = "" if len(existing) <= 5 else f", ... ({len(existing)} total)"
            raise FileExistsError(
                f"Refusing to overwrite existing protected archive member(s): {preview}{suffix}. "
                f"Pass force=True to replace existing baseline result files."
            )
        members = [
            member
            for member in members
            if not any(member.name == path or member.name.startswith(f"{path}/") for path in skip_existing_members)
            or not (dest_dir / member.name).exists()
        ]
        tar.extractall(dest_dir, members=members)


def _download_archive(
    filename: str,
    dest_dir: str | Path,
    *,
    force: bool = False,
    protected_existing_names: frozenset[str] = frozenset(),
    skip_existing_members: tuple[str, ...] = (),
) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        hf_hub_download(repo_id=HF_REPO_ID, repo_type="dataset", revision=HF_REVISION, filename=f"archives/{filename}")
    )
    print(f"Extracting {filename}...", flush=True)
    _extract_tar(
        archive_path,
        dest_dir,
        force=force,
        protected_existing_names=protected_existing_names,
        skip_existing_members=skip_existing_members,
    )
    return archive_path


def download_dataset(dest_dir: str | Path | None = None, *, force: bool = False) -> Path:
    """Fetch the benchmark dataset, defaulting to ``<package>/dataset`` regardless of cwd."""
    data_root = DEFAULT_DATA_ROOT if dest_dir is None else Path(dest_dir) / "dataset"
    present = all((data_root / "fields" / name).is_file() for name in ("all-obs.npz", "complete-obs-only.npz"))
    if present and not force:
        print(f"Dataset already present at `{data_root.resolve()}/`, skipping download.", flush=True)
        return data_root
    print(f"Downloading dataset to `{data_root.resolve()}/`...", flush=True)
    _download_archive(DATASET_ARCHIVE, data_root.parent, force=force)
    return data_root


def download_baselines(
    dest_dir: str | Path | None = None,
    *,
    archives: tuple[str, ...] | list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    """Fetch baseline results, defaulting beside the package (``<package>/results``) regardless of cwd."""
    dest_dir = PROJECT_ROOT if dest_dir is None else dest_dir
    return [
        _download_archive(
            name,
            dest_dir,
            force=force,
            protected_existing_names=frozenset({"predictions.npz"}),
            skip_existing_members=("results/README.md", "results/scores.csv", "results/tables"),
        )
        for name in (archives or BASELINES_RESULTS_ARCHIVES)
    ]
