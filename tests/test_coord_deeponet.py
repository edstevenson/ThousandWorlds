from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

import thousandworlds.models as models
from thousandworlds import run_model
from thousandworlds.models.coord_deeponet import CoordDeepONet
from thousandworlds.models._coordinate import BaseVariableSampler, field_metadata, parse_field_name, t21_coordinate_features


def test_coord_deeponet_export_parse_fit_predict_and_missingness():
    assert models.CoordDeepONet is not None
    assert parse_field_name("surface_temperature") == ("surface_temperature", -1)
    assert parse_field_name("temperature_3") == ("temperature", 3)

    lat_mu, lon_sin, lon_cos, lat_w = t21_coordinate_features(4, 6)
    assert np.isfinite(lat_mu).all()
    assert np.isfinite(lon_sin).all()
    assert np.isfinite(lon_cos).all()
    assert np.isclose(lat_w.sum(), 1.0)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3)).astype(np.float32)
    s = np.array([0, 1, 0, 1, 0], dtype=np.int64)
    Y = rng.normal(size=(5, 3, 4, 6)).astype(np.float32)
    field_mask = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=bool)
    Y[~field_mask] = np.nan
    field_names = ["surface_temperature", "temperature_0", "v_0"]

    sampler = BaseVariableSampler(
        torch.as_tensor(field_mask),
        field_metadata(field_names),
        torch.as_tensor(lat_w),
        device=torch.device("cpu"),
    )
    n, f, _, _ = sampler.sample(128, 6, torch.Generator().manual_seed(0))
    assert field_mask[n.numpy(), f.numpy()].all()

    model = CoordDeepONet(rank=4, branch_hidden_width=8, trunk_hidden_width=8, branch_num_layers=1, trunk_num_layers=1, batch_size=16, dtype=torch.float32, device="cpu")
    model.fit(
        X,
        s,
        Y,
        field_mask=field_mask,
        field_names=field_names,
        n_sim_types=2,
        num_steps=2,
        log_every=1,
        val_X=X[:2],
        val_s=s[:2],
        val_Y=Y[:2],
        val_field_mask=field_mask[:2],
        val_max_points=20,
    )
    pred = model.predict(X[:2], s[:2], chunk_size=17)

    assert pred.shape == (2, 3, 4, 6)
    assert torch.isfinite(pred).all()
    assert model.fit_stats_["steps_run"] == 2
    assert model.fit_stats_["best_val_sampled_normalized_rmse"] is not None
    assert model.fit_stats_["best_val_equal_base_normalized_rmse_grid"] is None
    assert model._field_mean.shape == (3,)


def test_run_model_help_and_coord_deeponet_config_roundtrip(tmp_path):
    result = subprocess.run([sys.executable, "-m", "thousandworlds.run_model", "--help"], capture_output=True, text=True, check=True)
    assert "coord_deeponet" in result.stdout
    assert "--deeponet-rank" in result.stdout

    args = argparse.Namespace(
        method="coord_deeponet",
        subset="multi-partial",
        data_dir=Path("dataset"),
        out_dir=tmp_path,
        seed=7,
        dtype="float32",
        device="cpu",
        deeponet_rank=16,
        deeponet_branch_hidden_width=32,
        deeponet_trunk_hidden_width=24,
        deeponet_branch_num_layers=2,
        deeponet_trunk_num_layers=1,
        deeponet_activation="relu",
        deeponet_batch_size=64,
        deeponet_predict_chunk_size=128,
        deeponet_num_steps=5,
        deeponet_lr=3.0e-4,
        deeponet_weight_decay=1.0e-4,
        deeponet_hard_stop_step=4,
    )
    cfg = run_model._resolved_config(args, out_dir=tmp_path, data_dir=Path("dataset"))
    assert cfg["coord_deeponet"] == {
        "rank": 16,
        "branch_hidden_width": 32,
        "trunk_hidden_width": 24,
        "branch_num_layers": 2,
        "trunk_num_layers": 1,
        "activation": "relu",
        "batch_size": 64,
        "predict_chunk_size": 128,
        "num_steps": 5,
        "lr": 3.0e-4,
        "weight_decay": 1.0e-4,
        "hard_stop_step": 4,
        "optimizer": "AdamW",
        "target_normalization": "per_field_training_grid_latitude_weighted",
        "coordinate_encoding": "t21_lat_mu_lon_sincos",
        "sampling": "base_variable_first_area_weighted_latitude",
        "cv_objective": "area_weighted_equal_base_variable_normalized_rmse_grid",
        "equatorial_symmetry": True,
        "preset": None,
    }
