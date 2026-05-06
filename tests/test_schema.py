from pathlib import Path

import thousandworlds as tw


def test_public_schema_surface_is_stable(data_dir):
    assert tw.BENCHMARK_SUBSETS == (
        "multi-partial",
        "multi-complete",
        "single-complete",
    )
    assert tw.PROTOCOL_TO_TEST_FILE["shared_planets"] == "test_shared_planets_only.csv"
    assert tw.SPACE_TO_ARCHIVE_DIR == {"grid": "fields", "spectral": "coefficients"}
    assert tw.supports_protocol("single-complete", "standard")
    assert not tw.supports_protocol("single-complete", "shared_planets")
    assert tw.canonical_field_names("multi-complete") == tw.FIELDS_COMPLETE_OBS_ONLY
    assert tw.resolve_data_root(data_dir.parent) == data_dir
    assert tw.support_path(data_dir.parent, "multi-partial", kind="archive").name == "all-obs.npz"
    assert tw.support_path(data_dir, "multi-complete", kind="stats_dir") == data_dir / "norm_stats" / "multi-complete"
    assert tw.support_path(data_dir, "multi-partial", kind="test_file", protocol="shared_planets").name == "test_shared_planets_only.csv"


def test_resolve_data_root_accepts_relative_dataset_from_parent_of_source_tree(monkeypatch, data_dir):
    monkeypatch.chdir(data_dir.parents[1])
    assert tw.resolve_data_root(Path("dataset")) == data_dir
