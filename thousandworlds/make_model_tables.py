from __future__ import annotations
"""Build benchmark summary tables from per-model metrics under results/models."""

import argparse
from pathlib import Path

import json
import pandas as pd

from thousandworlds.models._common import DEFAULT_RESULTS_DIR

PUBLIC_METHODS = {"coord_deeponet", "coord_mlp", "gplfr", "knn", "ppca_icm", "pca_mlp", "pca_ridge", "train_mean"}
GENERATED_METRICS = ("rmse", "acc", "energy_score", "spread_skill_ratio", "relative_rmse", "relative_acc", "relative_energy_score")


def _markdown_table(df: pd.DataFrame) -> str:
    md_df = df.copy()
    for col in md_df.columns:
        if pd.api.types.is_numeric_dtype(md_df[col]):
            md_df[col] = md_df[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.3g}")
    try:
        return md_df.to_markdown(index=False)
    except ImportError:
        cols = list(md_df.columns)
        rows = [cols, ["---"] * len(cols)]
        rows += [[str(row[col]) for col in cols] for _, row in md_df.iterrows()]
        return "\n".join("| " + " | ".join(row) + " |" for row in rows)


def _metric_groups(metrics: dict, key: str) -> dict[str, float]:
    node = metrics.get(key)
    return {} if node is None else {variable: value for variable, value in (node.get("per_variable") or {}).items()}


def _metric_fields(metrics: dict, key: str) -> dict[str, float]:
    node = metrics.get(key)
    return {} if node is None else {field: value for field, value in (node.get("per_field") or {}).items()}


def _tables(results_root: Path) -> tuple[
    dict[tuple[str, str, str], list[dict[str, float | str]]],
    dict[tuple[str, str, str], list[dict[str, float | str]]],
]:
    tables: dict[tuple[str, str, str], list[dict[str, float | str]]] = {}
    per_level_tables: dict[tuple[str, str, str], list[dict[str, float | str]]] = {}
    for path in sorted(results_root.glob("*/*/metrics_*.json")):
        subset, method = path.parts[-3:-1]
        if method not in PUBLIC_METHODS:
            continue
        protocol = path.stem.removeprefix("metrics_")
        metrics = json.loads(path.read_text())
        for metric in GENERATED_METRICS:
            row = {"method": method, **_metric_groups(metrics, metric)}
            if len(row) > 1:
                tables.setdefault((subset, protocol, metric), []).append(row)
            per_level_row = {"method": method, **_metric_fields(metrics, metric)}
            if len(per_level_row) > 1:
                per_level_tables.setdefault((subset, protocol, metric), []).append(per_level_row)
    return tables, per_level_tables


def _write_tables(out_dir: Path, tables: dict[tuple[str, str, str], list[dict[str, float | str]]], *, per_level: bool = False) -> list[Path]:
    written: list[Path] = []
    for subset, protocol, metric in sorted(tables):
        df = pd.DataFrame(tables[(subset, protocol, metric)]).sort_values("method").reset_index(drop=True)
        target_dir = out_dir / subset / protocol / "per_level" if per_level else out_dir / subset / protocol
        target_dir.mkdir(parents=True, exist_ok=True)
        csv_path = target_dir / f"{metric}.csv"
        md_path = target_dir / f"{metric}.md"
        df.to_csv(csv_path, index=False)
        md_path.write_text(_markdown_table(df) + "\n", encoding="utf-8")
        written += [csv_path, md_path]
    return written


def _clear_generated_files(out_dir: Path, keys: set[tuple[str, str]]) -> None:
    for subset, protocol in sorted(keys):
        for target_dir in (out_dir / subset / protocol, out_dir / subset / protocol / "per_level"):
            for metric in GENERATED_METRICS:
                for suffix in (".csv", ".md"):
                    path = target_dir / f"{metric}{suffix}"
                    if path.exists():
                        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m thousandworlds.make_model_tables", description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "tables")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables, per_level_tables = _tables(args.results_root)
    _clear_generated_files(args.out_dir, {(subset, protocol) for subset, protocol, _ in set(tables) | set(per_level_tables)})
    print(*_write_tables(args.out_dir, tables), *_write_tables(args.out_dir, per_level_tables, per_level=True), sep="\n")


if __name__ == "__main__":
    main()
