from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import hashlib
import tarfile
import time

import numpy as np
import pandas as pd
import requests

from .field_spec import CANONICAL_INPUT_NAMES, public_field_names
from .schema import (
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
DATASET_PAGE_URL = "https://huggingface.co/datasets/es833/ThousandWorlds"
DATA_URL_ENVVAR = "THOUSANDWORLDS_DATA_URL"
HF_ARCHIVE_ROOT = "https://huggingface.co/datasets/es833/ThousandWorlds/resolve/v1.0.0/archives"
DATA_URL = f"{HF_ARCHIVE_ROOT}/dataset.tar.gz"
BASELINES_URLS_ENVVAR = "THOUSANDWORLDS_BASELINES_URLS"
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
BASELINES_URLS = (
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-partial-deterministic.tar.gz",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-complete-deterministic.tar.gz",
    f"{HF_ARCHIVE_ROOT}/results-baselines-single-complete-deterministic.tar.gz",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-partial-gplfr.tar.gz",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-partial-ppca_icm.tar.gz",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-complete-gplfr.tar.gz",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-complete-ppca_icm.tar.gz",
    f"{HF_ARCHIVE_ROOT}/results-baselines-single-complete-gplfr.tar.gz",
    f"{HF_ARCHIVE_ROOT}/results-baselines-single-complete-ppca_icm.tar.gz",
)
KNOWN_SHA256 = {
    f"{HF_ARCHIVE_ROOT}/dataset.tar.gz": "356c6cc14f6d23f6ffaef2155bfb668f6e365e07bb5b2f83736afa5343dae8b3",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-partial-deterministic.tar.gz": "59f9979ed6c31fba1f43163485ba821e8cc486e1ca1893148528568b3a00b5ab",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-complete-deterministic.tar.gz": "204c492fd2af1b921946b1a9de07e39e8581c4f18cb01490b123bebf74cbc789",
    f"{HF_ARCHIVE_ROOT}/results-baselines-single-complete-deterministic.tar.gz": "9725c008210b7c6bd5094a7707da0c305629fc36e9d533876ea85edde492ff9d",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-partial-gplfr.tar.gz": "add9bcd8ba3761200ac89d013e259ebafeeb221b71043d062609ab3708ae2613",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-partial-ppca_icm.tar.gz": "5d65fabd3662ce5da4aa0f3d1480285e633e6e1819e8c1a3e0f128b0017ae30a",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-complete-gplfr.tar.gz": "099018b9bb4c98fbc479889bd3a977fa249e0602fa8d7cd57f7a3ac78d42d356",
    f"{HF_ARCHIVE_ROOT}/results-baselines-multi-complete-ppca_icm.tar.gz": "c3578059971aae6d3323b52dab27ad6f8133dcff7fefb8d79074a737200c21e3",
    f"{HF_ARCHIVE_ROOT}/results-baselines-single-complete-gplfr.tar.gz": "cf5a54f053836e0cd6758fa24c64d9d51ee1130c39831522dea12e8a59529a15",
    f"{HF_ARCHIVE_ROOT}/results-baselines-single-complete-ppca_icm.tar.gz": "380331557807129ce722cc0884ec2518b7b334733fefc88e5e5797a1361860b5",
}


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
            f"Place the extracted dataset under {data_root}, call thousandworlds.download_dataset(...), "
            f"or set {DATA_URL_ENVVAR} to another dataset archive URL."
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


def _maybe_sha256(url: str) -> str | None:
    for candidate in (f"{url}.sha256", f"{url}.sha256sum"):
        resp = requests.get(candidate, timeout=30)
        if resp.ok and resp.text.strip():
            return resp.text.strip().split()[0]
    return None


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GB"


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


def download(
    url: str,
    dest_dir: str | Path,
    *,
    force: bool = False,
    archive_name: str | None = None,
    protected_existing_names: frozenset[str] = frozenset(),
    skip_existing_members: tuple[str, ...] = (),
) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dest_dir / (archive_name or Path(url).name)
    if archive_path.exists() and not force:
        return archive_path

    checksum = KNOWN_SHA256.get(url) or _maybe_sha256(url)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        print(f"Downloading {archive_path.name}" + (f" ({_format_bytes(total)})" if total else "") + "...", flush=True)
        digest = hashlib.sha256()
        downloaded = 0
        last_update = 0.0
        with archive_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if not chunk:
                    continue
                fh.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_update >= 1.0:
                    if total:
                        print(f"  {downloaded / total:5.1%} ({_format_bytes(downloaded)} / {_format_bytes(total)})", flush=True)
                    else:
                        print(f"  {_format_bytes(downloaded)}", flush=True)
                    last_update = now
        print(f"Downloaded {archive_path.name} ({_format_bytes(downloaded)}).", flush=True)
    if checksum and digest.hexdigest() != checksum:
        raise ValueError(f"SHA256 mismatch for {archive_path.name}.")
    if tarfile.is_tarfile(archive_path):
        print(f"Extracting {archive_path.name}...", flush=True)
        _extract_tar(
            archive_path,
            dest_dir,
            force=force,
            protected_existing_names=protected_existing_names,
            skip_existing_members=skip_existing_members,
        )
        print(f"Extracted {archive_path.name}.", flush=True)
    return archive_path


def download_dataset(dest_dir: str | Path, *, url: str | None = None, force: bool = False) -> Path:
    dest_dir = Path(dest_dir)
    data_root = dest_dir / "dataset"
    if all((data_root / "fields" / name).is_file() for name in ("all-obs.npz", "complete-obs-only.npz")) and not force:
        return data_root
    url = url or os.environ.get(DATA_URL_ENVVAR) or DATA_URL
    if url is None:
        raise RuntimeError(
            f"No default ThousandWorlds dataset archive URL is configured yet. "
            f"Set {DATA_URL_ENVVAR}, pass url=..., or download manually from {DATASET_PAGE_URL}."
        )
    return download(url, dest_dir, force=force, archive_name="dataset.tar.gz" if url == DATA_URL else None)


def _split_urls(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.replace("\n", ",").split(",") if part.strip())


def download_baselines(
    dest_dir: str | Path,
    *,
    url: str | None = None,
    urls: tuple[str, ...] | list[str] | None = None,
    force: bool = False,
) -> list[Path]:
    baseline_urls = tuple(urls or ())
    if url is not None:
        baseline_urls = (url,)
    if not baseline_urls:
        env_urls = os.environ.get(BASELINES_URLS_ENVVAR)
        baseline_urls = _split_urls(env_urls) if env_urls else tuple(BASELINES_URLS or ())
    if not baseline_urls:
        raise RuntimeError(
            f"No default ThousandWorlds baseline-results archive URLs are configured yet. "
            f"Set {BASELINES_URLS_ENVVAR}, pass urls=..., or download manually from {DATASET_PAGE_URL}."
        )
    archive_names = BASELINES_RESULTS_ARCHIVES if baseline_urls == BASELINES_URLS else (None,) * len(baseline_urls)
    return [
        download(
            baseline_url,
            dest_dir,
            force=force,
            archive_name=archive_name,
            protected_existing_names=frozenset({"predictions.npz"}),
            skip_existing_members=("results/README.md", "results/scores.csv", "results/tables"),
        )
        for baseline_url, archive_name in zip(baseline_urls, archive_names)
    ]
