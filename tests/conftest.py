from __future__ import annotations

from pathlib import Path

import pytest


DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"
_HAS_DATASET = (DATA_DIR / "fields" / "all-obs.npz").is_file()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_dataset: requires the full ThousandWorlds NPZ dataset (call tw.download_dataset first)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _HAS_DATASET:
        return
    skip = pytest.mark.skip(reason="benchmark dataset not present — run tw.download_dataset(...) first")
    for item in items:
        if "requires_dataset" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR
