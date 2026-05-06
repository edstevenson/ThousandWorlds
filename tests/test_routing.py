import pytest

import thousandworlds as tw


def test_subset_path_routes_train_and_protocol_specific_test_files(data_dir):
    assert tw.BENCHMARK_SPLITS == ("train", "test")
    assert tw.subset_path(data_dir.parent, "multi-partial").name == "train.csv"
    assert tw.subset_path(data_dir, "multi-partial", split="test").name == "test.csv"
    assert tw.subset_path(data_dir, "multi-partial", split="test", protocol="shared_planets").name == "test_shared_planets_only.csv"


def test_subset_path_rejects_unknown_or_unsupported_routes(data_dir):
    with pytest.raises(ValueError, match="Unknown split"):
        tw.subset_path(data_dir, "multi-partial", split="validation")
    with pytest.raises(ValueError, match="shared_planets"):
        tw.subset_path(data_dir, "single-complete", split="test", protocol="shared_planets")
