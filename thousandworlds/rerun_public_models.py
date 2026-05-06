from __future__ import annotations
"""Batch wrapper around ``python -m thousandworlds.run_model`` for the public non-GPLFR matrix."""

import argparse
from pathlib import Path
import subprocess
import sys

from thousandworlds.run_model import COORD_DEEPONET_PRESETS, COORD_MLP_PRESETS

SUBSETS = ["multi-partial", "multi-complete", "single-complete"]
METHODS = ("train_mean", "knn", "pca_ridge", "pca_mlp", "ppca_icm", "gplfr", "coord_mlp", "coord_deeponet")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m thousandworlds.rerun_public_models", description="Rerun the public non-GPLFR ThousandWorlds model matrix.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="*", choices=METHODS, default=list(METHODS))
    parser.add_argument("--subsets", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Additional flags forwarded to python -m thousandworlds.run_model")
    args = parser.parse_args()

    selected = [subset for subset in SUBSETS if not args.subsets or subset in args.subsets]
    if not selected:
        raise SystemExit("No subsets selected.")
    extra = [arg for arg in args.extra if arg != "--"]
    matrix_pairs = [
        (method, subset)
        for subset in selected
        for method in args.methods
        if method != "coord_mlp" or subset in COORD_MLP_PRESETS
        if method != "coord_deeponet" or subset in COORD_DEEPONET_PRESETS
    ]
    commands = [
        [sys.executable, "-m", "thousandworlds.run_model", method, subset, "--data-dir", str(args.data_dir), "--seed", str(args.seed), *extra]
        for method, subset in matrix_pairs
    ]
    for cmd in commands:
        print("+", " ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
