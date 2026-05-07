from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import thousandworlds.models as models
from thousandworlds import run_model
from thousandworlds.models.coord_mlp import CoordMLP, parse_field_name


def test_coord_mlp_export_parse_fit_predict_and_missingness():
    assert models.CoordMLP is not None
    assert parse_field_name("surface_temperature") == ("surface_temperature", -1)
    assert parse_field_name("temperature_3") == ("temperature", 3)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3)).astype(np.float32)
    s = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    Y = rng.normal(size=(5, 3, 4, 6)).astype(np.float32)
    field_mask = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool)
    Y[~field_mask] = np.nan

    model = CoordMLP(hidden_width=8, num_layers=1, batch_size=16, dtype=torch.float32, device="cpu")
    model.fit(
        X,
        s,
        Y,
        field_mask=field_mask,
        field_names=["surface_temperature", "temperature_0", "v_0"],
        n_sim_types=2,
        num_steps=2,
        log_every=1,
        val_X=X[:2],
        val_s=s[:2],
        val_Y=Y[:2],
        val_field_mask=field_mask[:2],
        val_max_points=64,
    )
    pred = model.predict(X[:2], s[:2], chunk_size=17)

    assert pred.shape == (2, 3, 4, 6)
    assert torch.isfinite(pred).all()
    assert model.fit_stats_["steps_run"] == 2
    assert model._field_mean.shape == (3,)


def test_run_model_help_and_coord_config_roundtrip(tmp_path):
    result = subprocess.run([sys.executable, "-m", "thousandworlds.run_model", "--help"], capture_output=True, text=True, check=True)
    assert "coord_mlp" in result.stdout
    assert "--coord-hidden-width" in result.stdout

    args = argparse.Namespace(
        method="coord_mlp",
        subset="multi-partial",
        data_dir=Path("dataset"),
        out_dir=tmp_path,
        seed=7,
        dtype="float32",
        device="cpu",
        coord_hidden_width=16,
        coord_num_layers=2,
        coord_activation="relu",
        coord_batch_size=32,
        coord_predict_chunk_size=64,
        coord_num_steps=5,
        coord_lr=3.0e-4,
        coord_weight_decay=1.0e-4,
        coord_hard_stop_step=4,
    )
    cfg = run_model._resolved_config(args, out_dir=tmp_path, data_dir=Path("dataset"))
    assert cfg["coord_mlp"] == {
        "hidden_width": 16,
        "num_layers": 2,
        "activation": "relu",
        "batch_size": 32,
        "predict_chunk_size": 64,
        "num_steps": 5,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "hard_stop_step": 4,
        "preset": None,
    }


def test_coord_mlp_preset_accepts_sweep_summary_bare_keys(monkeypatch):
    monkeypatch.setitem(
        run_model.COORD_MLP_PRESETS,
        "multi-partial",
        {"hidden_width": 9, "num_layers": 1, "activation": "tanh", "batch_size": 7, "predict_chunk_size": 11, "num_steps": 13, "lr": 1.0e-4, "weight_decay": 0.0, "hard_stop_step": 5},
    )
    args = argparse.Namespace(subset="multi-partial", **run_model.COORD_MLP_ARG_DEFAULTS)

    hparams = run_model._coord_mlp_hparams(args)

    assert hparams["coord_hidden_width"] == 9
    assert hparams["coord_hard_stop_step"] == 5
