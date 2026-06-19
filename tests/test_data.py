from pathlib import Path
import tarfile

import numpy as np
import pytest

import thousandworlds as tw


@pytest.mark.requires_dataset
def test_load_all_obs_standard(data_dir):
    bundle = tw.load("multi-partial", "standard", data_dir=data_dir, space="grid")
    assert bundle.X_train.shape == (1626, 8)
    assert bundle.X_test.shape == (100, 8)
    assert bundle.Y_train.shape == (1626, 53, 32, 64)
    assert bundle.Y_test.shape == (100, 53, 32, 64)
    assert bundle.field_mask_train.shape == (1626, 53)
    assert bundle.field_mask_test.shape == (100, 53)
    assert np.array_equal(bundle.meta_test["simulation_id"].to_numpy(dtype=np.int32), bundle.test_ids)


@pytest.mark.requires_dataset
def test_load_complete_obs_spectral(data_dir):
    bundle = tw.load("multi-complete", "standard", data_dir=data_dir, space="spectral")
    assert bundle.Y_train.shape == (1538, 48, 484)
    assert bundle.Y_test.shape == (90, 48, 484)
    assert bundle.field_mask_train.all()
    assert bundle.field_mask_test.all()


def test_download_dataset_dispatches_dataset_archive(monkeypatch, tmp_path):
    calls = {}

    def fake_archive(name: str, dest_dir: str | Path, *, force: bool = False, **kwargs) -> Path:
        calls.update(name=name, dest_dir=Path(dest_dir), force=force)
        return Path(dest_dir) / name

    monkeypatch.setattr(tw.data, "_download_archive", fake_archive)
    out = tw.download_dataset(tmp_path, force=True)

    assert out == tmp_path / "dataset"
    assert calls == {"name": "dataset.tar.gz", "dest_dir": tmp_path, "force": True}


def test_download_dataset_defaults_to_package_data_root(monkeypatch):
    calls = {}

    def fake_archive(name: str, dest_dir: str | Path, *, force: bool = False, **kwargs) -> Path:
        calls.update(name=name, dest_dir=Path(dest_dir))
        return Path(dest_dir) / "dataset"

    monkeypatch.setattr(tw.data, "_download_archive", fake_archive)
    out = tw.download_dataset(force=True)  # force bypasses the present-dataset skip

    assert out == tw.schema.DEFAULT_DATA_ROOT
    assert calls == {"name": "dataset.tar.gz", "dest_dir": tw.schema.DEFAULT_DATA_ROOT.parent}


def test_download_dataset_downloads_when_only_one_field_archive_exists(monkeypatch, tmp_path):
    calls = {}
    (tmp_path / "dataset" / "fields").mkdir(parents=True)
    (tmp_path / "dataset" / "fields" / "complete-obs-only.npz").write_bytes(b"npz")

    def fake_archive(name: str, dest_dir: str | Path, *, force: bool = False, **kwargs) -> Path:
        calls.update(name=name)
        return Path(dest_dir) / name

    monkeypatch.setattr(tw.data, "_download_archive", fake_archive)
    assert tw.download_dataset(tmp_path) == tmp_path / "dataset"
    assert calls["name"] == "dataset.tar.gz"


def test_download_dataset_skips_existing_extracted_archive(monkeypatch, tmp_path):
    data_root = tmp_path / "dataset"
    (data_root / "fields").mkdir(parents=True)
    (data_root / "fields" / "all-obs.npz").write_bytes(b"npz")
    (data_root / "fields" / "complete-obs-only.npz").write_bytes(b"npz")

    def fail_archive(*args, **kwargs):
        raise AssertionError("_download_archive should not be called")

    monkeypatch.setattr(tw.data, "_download_archive", fail_archive)
    assert tw.download_dataset(tmp_path) == data_root


def test_download_baselines_dispatches_each_archive(monkeypatch, tmp_path):
    calls = []

    def fake_archive(
        name: str,
        dest_dir: str | Path,
        *,
        force: bool = False,
        protected_existing_names: frozenset[str] = frozenset(),
        skip_existing_members: tuple[str, ...] = (),
    ) -> Path:
        calls.append((name, Path(dest_dir), force, protected_existing_names, skip_existing_members))
        return Path(dest_dir) / name

    monkeypatch.setattr(tw.data, "_download_archive", fake_archive)
    out = tw.download_baselines(tmp_path, force=True)

    assert out == [tmp_path / name for name in tw.data.BASELINES_RESULTS_ARCHIVES]
    assert calls == [
        (
            name,
            tmp_path,
            True,
            frozenset({"predictions.npz"}),
            ("results/README.md", "results/scores.csv", "results/tables"),
        )
        for name in tw.data.BASELINES_RESULTS_ARCHIVES
    ]


def test_download_baselines_accepts_archive_subset(monkeypatch, tmp_path):
    names = ("results-baselines-single-complete-deterministic.tar.gz",)
    monkeypatch.setattr(tw.data, "_download_archive", lambda name, dest_dir, **kwargs: Path(dest_dir) / name)
    assert tw.download_baselines(tmp_path, archives=names) == [tmp_path / names[0]]


def test_download_baselines_defaults_to_package_root(monkeypatch):
    dests = []
    monkeypatch.setattr(
        tw.data, "_download_archive", lambda name, dest_dir, **kwargs: dests.append(Path(dest_dir)) or Path(dest_dir) / name
    )
    tw.download_baselines(archives=("results-baselines-single-complete-deterministic.tar.gz",))
    assert dests == [tw.schema.PROJECT_ROOT]


def test_download_archive_fetches_from_hub_and_extracts(monkeypatch, tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"npz")
    archive = tmp_path / "dataset.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(payload, arcname="dataset/fields/all-obs.npz")

    seen = {}

    def fake_hub(*, repo_id: str, repo_type: str, revision: str, filename: str) -> str:
        seen.update(repo_id=repo_id, repo_type=repo_type, revision=revision, filename=filename)
        return str(archive)

    monkeypatch.setattr(tw.data, "hf_hub_download", fake_hub)
    dest = tmp_path / "out"
    tw.data._download_archive("dataset.tar.gz", dest)

    assert (dest / "dataset" / "fields" / "all-obs.npz").read_bytes() == b"npz"
    assert seen == {
        "repo_id": tw.data.HF_REPO_ID,
        "repo_type": "dataset",
        "revision": tw.data.HF_REVISION,
        "filename": "archives/dataset.tar.gz",
    }


def test_baseline_extract_refuses_existing_predictions_without_force(tmp_path):
    archive_path = tmp_path / tw.data.BASELINES_RESULTS_ARCHIVES[0]
    source = tmp_path / "source_predictions.npz"
    source.write_bytes(b"new")
    member_name = "results/models/single-complete/train_mean/predictions.npz"
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source, arcname=member_name)

    existing = tmp_path / member_name
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")

    with pytest.raises(FileExistsError, match="predictions.npz"):
        tw.data._extract_tar(
            archive_path,
            tmp_path,
            protected_existing_names=frozenset({"predictions.npz"}),
        )

    tw.data._extract_tar(
        archive_path,
        tmp_path,
        force=True,
        protected_existing_names=frozenset({"predictions.npz"}),
    )
    assert existing.read_bytes() == b"new"


def test_baseline_extract_preserves_existing_results_docs(tmp_path):
    archive_path = tmp_path / tw.data.BASELINES_RESULTS_ARCHIVES[0]
    readme = tmp_path / "new_readme.md"
    scores = tmp_path / "new_scores.csv"
    table = tmp_path / "new_rmse.md"
    prediction = tmp_path / "predictions.npz"
    readme.write_text("new readme\n", encoding="utf-8")
    scores.write_text("new scores\n", encoding="utf-8")
    table.write_text("new table\n", encoding="utf-8")
    prediction.write_bytes(b"new prediction")
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(readme, arcname="results/README.md")
        tar.add(scores, arcname="results/scores.csv")
        tar.add(table, arcname="results/tables/single-complete/standard/rmse.md")
        tar.add(prediction, arcname="results/models/single-complete/train_mean/predictions.npz")

    existing_readme = tmp_path / "results/README.md"
    existing_scores = tmp_path / "results/scores.csv"
    existing_table = tmp_path / "results/tables/single-complete/standard/rmse.md"
    existing_pred = tmp_path / "results/models/single-complete/train_mean/predictions.npz"
    for path, text in (
        (existing_readme, "old readme\n"),
        (existing_scores, "old scores\n"),
        (existing_table, "old table\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    tw.data._extract_tar(
        archive_path,
        tmp_path,
        force=True,
        skip_existing_members=("results/README.md", "results/scores.csv", "results/tables"),
    )

    assert existing_readme.read_text(encoding="utf-8") == "old readme\n"
    assert existing_scores.read_text(encoding="utf-8") == "old scores\n"
    assert existing_table.read_text(encoding="utf-8") == "old table\n"
    assert existing_pred.read_bytes() == b"new prediction"


def test_load_missing_archive_raises_helpful_message(tmp_path):
    data_root = tmp_path / "dataset"
    (data_root / "subsets" / "single-complete").mkdir(parents=True)
    (data_root / "subsets" / "single-complete" / "train.csv").write_text("simulation_id\n1\n", encoding="utf-8")
    (data_root / "subsets" / "single-complete" / "test.csv").write_text("simulation_id\n1\n", encoding="utf-8")
    (data_root / "inputs.csv").write_text(
        ",".join(["simulation_id", "stellar_temperature", "stellar_flux", "radius", "gravity", "rotation_period", "surface_pressure", "co2", "ch4"]) + "\n"
        "1,3000,900,1,10,1,1,0.1,0.1\n",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="download_dataset"):
        tw.load("single-complete", data_dir=data_root)
