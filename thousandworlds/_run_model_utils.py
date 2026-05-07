from __future__ import annotations

import argparse

import numpy as np


def _parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


def _parse_float_list(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _csv(values) -> str:
    if isinstance(values, str):
        return values
    return ",".join(str(x) for x in values)


def _mark_explicit_args(args: argparse.Namespace, parser: argparse.ArgumentParser, argv: list[str]) -> argparse.Namespace:
    explicit = set()
    for action in parser._actions:
        if any(opt in argv or any(item.startswith(f"{opt}=") for item in argv) for opt in action.option_strings):
            explicit.add(action.dest)
    args._explicit_args = explicit
    return args


def _kfold_indices(n: int, n_folds: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_folds = max(2, min(int(n_folds), n))
    fold_sizes = np.full(n_folds, n // n_folds, dtype=int)
    fold_sizes[: n % n_folds] += 1
    splits = []
    start = 0
    for size in fold_sizes:
        stop = start + size
        splits.append((np.concatenate([order[:start], order[stop:]]), order[start:stop]))
        start = stop
    return splits


def _append_gcm_block(X: np.ndarray, s: np.ndarray, n_gcm: int, penalty: float) -> np.ndarray:
    if n_gcm <= 1:
        return X
    oh = np.eye(n_gcm, dtype=np.float32)[np.asarray(s, dtype=np.int64)] * float(penalty)
    return np.concatenate([np.asarray(X, dtype=np.float32), oh], axis=1)


def _torch():
    import torch

    return torch
