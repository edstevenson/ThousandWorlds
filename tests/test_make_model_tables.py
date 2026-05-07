from pathlib import Path

import pytest

pytest.importorskip("torch")

from thousandworlds.make_model_tables import _clear_generated_files


def test_clear_generated_files_preserves_custom_table_collateral(tmp_path: Path) -> None:
    protocol_dir = tmp_path / "multi-partial" / "standard"
    custom_dir = protocol_dir / "custom_notes"
    per_level_dir = protocol_dir / "per_level"
    custom_dir.mkdir(parents=True)
    per_level_dir.mkdir()

    (protocol_dir / "rmse.csv").write_text("old\n", encoding="utf-8")
    (protocol_dir / "README.md").write_text("keep\n", encoding="utf-8")
    (custom_dir / "rmse.csv").write_text("keep\n", encoding="utf-8")
    (per_level_dir / "rmse.md").write_text("old\n", encoding="utf-8")
    (per_level_dir / "README.md").write_text("keep\n", encoding="utf-8")

    _clear_generated_files(tmp_path, {("multi-partial", "standard")})

    assert not (protocol_dir / "rmse.csv").exists()
    assert not (per_level_dir / "rmse.md").exists()
    assert (protocol_dir / "README.md").read_text(encoding="utf-8") == "keep\n"
    assert (custom_dir / "rmse.csv").read_text(encoding="utf-8") == "keep\n"
    assert (per_level_dir / "README.md").read_text(encoding="utf-8") == "keep\n"
