from __future__ import annotations

from pathlib import Path

from .field_spec import FIELDS_ALL_OBS, FIELDS_COMPLETE_OBS_ONLY

GRID_SHAPE = (32, 64)
T = 21
N_COEFFS = 484
TARGET_GCMS = frozenset({"exocam", "um"})
TARGET_PHYSICAL_DOMAIN = {
    "P0": (5.0e4, 5.0e5),
    "radius": (0.7 * 6371e3, 1.4 * 6371e3),
    "gravity": (6.0, 16.0),
    "F_star": (500.0, 1500.0),
    "T_star": (2500.0, 5800.0),
    "P_rot": (0.1, 1000.0),
    "CH4": (0.0, 0.05),
}

BENCHMARK_SUBSETS = (
    "multi-partial",
    "multi-complete",
    "single-complete",
)
BENCHMARK_SPLITS = ("train", "test")
BENCHMARK_PROTOCOLS = ("standard", "shared_planets")
BENCHMARK_SPACES = ("grid", "spectral")
SPACE_TO_ARCHIVE_DIR = {"grid": "fields", "spectral": "coefficients"}
SUBSET_TO_ARCHIVE = {
    "multi-partial": "all-obs",
    "multi-complete": "complete-obs-only",
    "single-complete": "complete-obs-only",
}
SUBSET_TO_FIELDS = {
    "multi-partial": FIELDS_ALL_OBS,
    "multi-complete": FIELDS_COMPLETE_OBS_ONLY,
    "single-complete": FIELDS_COMPLETE_OBS_ONLY,
}
PROTOCOL_TO_TEST_FILE = {"standard": "test.csv", "shared_planets": "test_shared_planets_only.csv"}


def _looks_like_data_root(path: Path) -> bool:
    return (path / "inputs.csv").is_file() and (path / "subsets").is_dir() and (path / "fields").is_dir()


def resolve_data_root(data_dir: str | Path) -> Path:
    path = Path(data_dir)
    pkg_root = Path(__file__).resolve().parents[1]
    for candidate in (path, path / "dataset", pkg_root / path, pkg_root / path / "dataset"):
        if _looks_like_data_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find ThousandWorlds dataset root from {path!s}. "
        f"Tried: {path}, {path / 'dataset'}, {pkg_root / path}, {pkg_root / path / 'dataset'}. "
        f"Download the dataset first or call thousandworlds.download_dataset(...)."
    )


def _require_subset(subset: str) -> None:
    if subset not in SUBSET_TO_ARCHIVE:
        raise ValueError(f"Unknown subset {subset!r}.")


def _require_protocol(subset: str, protocol: str) -> None:
    if protocol not in PROTOCOL_TO_TEST_FILE:
        raise ValueError(f"Unknown protocol {protocol!r}.")
    if not supports_protocol(subset, protocol):
        raise ValueError(f"{protocol} protocol is not defined for {subset!r}.")


def _require_space(space: str) -> None:
    if space not in SPACE_TO_ARCHIVE_DIR:
        raise ValueError(f"Unknown space {space!r}.")


def subset_path(data_dir: str | Path, subset: str, *, split: str = "train", protocol: str = "standard") -> Path:
    _require_subset(subset)
    if split not in BENCHMARK_SPLITS:
        raise ValueError(f"Unknown split {split!r}.")
    data_root = resolve_data_root(data_dir)
    if split == "test":
        _require_protocol(subset, protocol)
    return (
        data_root / "subsets" / subset / "train.csv"
        if split == "train"
        else data_root / "subsets" / subset / PROTOCOL_TO_TEST_FILE[protocol]
    )


def support_path(
    data_dir: str | Path,
    subset: str,
    *,
    kind: str,
    protocol: str = "standard",
    space: str = "grid",
) -> Path:
    data_root = resolve_data_root(data_dir)
    _require_subset(subset)
    if kind == "subset_dir":
        return data_root / "subsets" / subset
    if kind == "stats_dir":
        return data_root / "norm_stats" / subset
    if kind == "archive":
        _require_space(space)
        return data_root / SPACE_TO_ARCHIVE_DIR[space] / f"{SUBSET_TO_ARCHIVE[subset]}.npz"
    if kind == "test_file":
        _require_protocol(subset, protocol)
        return subset_path(data_root, subset, split="test", protocol=protocol)
    raise ValueError(f"Unknown support kind {kind!r}.")


def canonical_field_names(subset: str) -> list[str]:
    return list(SUBSET_TO_FIELDS[subset])


def supports_protocol(subset: str, protocol: str) -> bool:
    return protocol == "standard" or (protocol == "shared_planets" and subset.startswith("multi-"))
