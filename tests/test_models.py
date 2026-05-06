import importlib
import inspect
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

import thousandworlds.models as models
import thousandworlds.run_model as run_model
import thousandworlds.rerun_public_models as rerun_public_models
from thousandworlds.models._torch_kernels import build_design_matrix
from thousandworlds.models._gplfr_core import GPLFRCore
from thousandworlds.models._gplfr_weighting import retrieve_field_group_index
from thousandworlds.models._common import enforce_equatorial_symmetry_grid, equal_group_normalized_rmse_grid, masked_mean_grid
from thousandworlds.models.pca_mlp import PCAMLP, _equal_group_mean
from thousandworlds.models.pca_ridge import PCARidge, fit_latent_ridge


def test_models_surface_smoke_and_knn_path():
    assert models.KNN is not None
    assert models.TrainMean is not None
    assert models.GPLFR is not None
    assert models.PPCAICM is not None
    assert models.PCAMLP is not None
    assert models.PCARidge is not None
    assert models.CoordDeepONet is not None
    assert models.CoordMLP is not None

    model = models.KNN()
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    Y_train = np.array([
        [[[10.0]], [[100.0]]],
        [[[20.0]], [[200.0]]],
        [[[30.0]], [[300.0]]],
    ], dtype=np.float32)
    field_mask = np.array([[True, False], [True, True], [True, True]])

    model.fit(X_train, Y_train, k_candidates=[2], field_mask=field_mask)
    pred = model.predict(np.array([[0.05, 0.05], [1.95, 2.05]], dtype=np.float32), k=2)

    assert pred.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(pred[:, :, 0, 0], [[15.0, 200.0], [25.0, 250.0]])


def test_knn_ignores_missing_neighbor_fields_and_falls_back_to_field_mean():
    model = models.KNN()
    X_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    Y_train = np.array([
        [[[10.0]], [[np.nan]]],
        [[[20.0]], [[np.nan]]],
        [[[30.0]], [[300.0]]],
    ], dtype=np.float32)
    field_mask = np.array([[True, False], [True, False], [True, True]])

    model.fit(X_train, Y_train, k_candidates=[2], field_mask=field_mask)
    pred = model.predict(np.array([[0.1]], dtype=np.float32), k=2)

    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred[0, :, 0, 0], [15.0, 300.0])


def test_knn_cv_objective_equal_weights_normalized_variable_groups():
    target = np.zeros((1, 3, 32, 64), dtype=np.float32)
    pred = np.zeros_like(target)
    pred[:, 0] = 2.0
    pred[:, 1] = 20.0
    pred[:, 2] = 4.0

    score = equal_group_normalized_rmse_grid(
        pred,
        target,
        ["temperature_0", "temperature_1", "u_0"],
        np.array([1.0, 10.0, 1.0], dtype=np.float32),
        np.ones((1, 3), dtype=bool),
    )

    np.testing.assert_allclose(score, 3.0, atol=1.0e-6)


def test_equatorial_symmetry_grid_enforces_scalar_symmetry_and_v_antisymmetry():
    Y = np.array([[
        [[1.0], [3.0], [5.0], [7.0]],
        [[2.0], [4.0], [6.0], [8.0]],
    ]], dtype=np.float32)

    out = enforce_equatorial_symmetry_grid(Y, ["temperature_0", "v_0"])

    np.testing.assert_allclose(out[0, 0, :, 0], [4.0, 4.0, 4.0, 4.0])
    np.testing.assert_allclose(out[0, 1, :, 0], [-3.0, -1.0, 1.0, 3.0])


def test_train_mean_masked_grid_mean_excludes_nans():
    Y_train = np.array([
        [[[10.0]], [[np.nan]]],
        [[[20.0]], [[np.nan]]],
        [[[30.0]], [[300.0]]],
    ], dtype=np.float32)
    field_mask = np.array([[True, False], [True, False], [True, True]])

    mean = masked_mean_grid(Y_train, field_mask)

    assert np.isfinite(mean).all()
    np.testing.assert_allclose(mean[:, 0, 0], [20.0, 300.0])


def test_run_model_help_lists_supported_methods():
    result = subprocess.run(
        [sys.executable, "-m", "thousandworlds.run_model", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "{train_mean,knn,pca_ridge,pca_mlp,ppca_icm,gplfr,coord_mlp,coord_deeponet}" in result.stdout
    assert "1,2,3,5,10" in result.stdout
    assert "0.0,0.3,1.0,3.0" in result.stdout
    assert "--lambda-reg" in result.stdout
    assert "--cv-latent-dim" in result.stdout
    assert "--cv-lambda" in result.stdout


def test_gplfr_resolver_defaults_and_config_replay():
    args = run_model.argparse.Namespace(
        latent_dim=60,
        gplfr_num_training_steps=3000,
        gplfr_inverse_temperature=0.25,
        gplfr_latent_nugget=0.03,
        _explicit_args=set(),
    )
    hparams = run_model._gplfr_hparams(args)
    assert hparams["latent_dim"] == 150
    assert hparams["num_training_steps"] == 2000
    assert hparams["inverse_temperature"] == 0.1
    assert hparams["latent_nugget"] == 0.1
    assert hparams["lr_Z"] == 0.1
    assert hparams["lr_global"] == 0.3
    assert hparams["variable_weights"] == "learned_per_group"
    assert hparams["output_coregionalization"] == "field_coregionalized"

    args._explicit_args = {"latent_dim", "gplfr_num_training_steps", "gplfr_inverse_temperature", "gplfr_latent_nugget"}
    hparams = run_model._gplfr_hparams(args)
    assert hparams["latent_dim"] == 60
    assert hparams["num_training_steps"] == 3000
    assert hparams["inverse_temperature"] == 0.25
    assert hparams["latent_nugget"] == 0.03

    replay = run_model._gplfr_hparams(args, {"gplfr": {"latent_dim": 4, "num_training_steps": 29, "inverse_temperature": 0.5, "latent_nugget": 0.01, "variable_weights": "fixed", "output_coregionalization": "none", "optimizer": {"lr_Z": 0.2, "lr_global": 0.4}}})
    assert replay["latent_dim"] == 4
    assert replay["num_training_steps"] == 29
    assert replay["inverse_temperature"] == 0.5
    assert replay["latent_nugget"] == 0.01
    assert replay["variable_weights"] == "fixed"
    assert replay["output_coregionalization"] == "none"
    assert replay["lr_Z"] == 0.2
    assert replay["lr_global"] == 0.4


def test_gplfr_weighting_accepts_public_radiation_field_names():
    assert retrieve_field_group_index("asr") == retrieve_field_group_index("asr_cloudy")
    assert retrieve_field_group_index("olr") == retrieve_field_group_index("olr_cloudy")


def test_ppca_icm_tuned_preset_allows_explicit_cli_overrides():
    args = run_model.argparse.Namespace(
        subset="multi-partial",
        ppca_icm_preset="tuned",
        latent_dim=120,
        kernel="matern52",
        kernel_mode="shared",
        ppca_iters=50,
        gp_steps=4000,
        gp_lr=1.0e-3,
        n_samples=64,
        hard_stop_step=1300,
        ell_init="4.0",
        _explicit_args={"latent_dim", "gp_lr", "hard_stop_step", "ell_init"},
    )

    hparams = run_model._ppca_icm_hparams(args)

    assert hparams["latent_dim"] == 120
    assert hparams["gp_lr"] == 1.0e-3
    assert hparams["hard_stop_step"] == 1300
    assert hparams["ell_init"] == 4.0
    assert hparams["gp_steps"] == 4000


def test_gplfr_public_surface_uses_core_module():
    from thousandworlds.models.gplfr import GPLFR

    assert models.GPLFR.__name__ == GPLFR.__name__ == "GPLFR"
    assert models.GPLFR.__module__.endswith("models.gplfr")
    assert GPLFRCore.__name__ == "GPLFRCore"


def test_rerun_public_models_dry_run_exposes_gplfr():
    assert "gplfr" in rerun_public_models.METHODS
    result = subprocess.run(
        [sys.executable, "-m", "thousandworlds.rerun_public_models", "--dry-run", "--methods", "gplfr", "--subsets", "multi-partial"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "-m thousandworlds.run_model gplfr multi-partial" in result.stdout


def test_gplfr_reference_module_was_deleted():
    assert not Path(models.__file__).with_name("_gplfr_reference.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("thousandworlds.models._gplfr_reference")


def test_gplfr_core_has_no_forbidden_reference_strings():
    source = Path(inspect.getsourcefile(GPLFRCore)).read_text()
    for token in ("MCMC", "NUTS", "wandb", "pca_init", "frozen_Z", "log_pred_density", "equicorr", "intel_extension_for_pytorch"):
        assert token not in source


def test_public_gplfr_run_does_not_fit_on_test_targets():
    source = inspect.getsource(run_model._run_gplfr)
    assert "test_Y=" not in source
    assert "test_field_mask=" not in source


def test_pca_ridge_latent_ridge_does_not_penalize_intercept():
    H = torch.ones((4, 1), dtype=torch.float64)
    Z = torch.full((4, 1), 3.0, dtype=torch.float64)
    B = fit_latent_ridge(H, Z, lambda_reg=1.0e6, intercept=True)
    torch.testing.assert_close(B, torch.tensor([[3.0]], dtype=torch.float64))

    B = fit_latent_ridge(H, torch.ones_like(Z), lambda_reg=1.0, intercept=False)
    torch.testing.assert_close(B, torch.tensor([[0.5]], dtype=torch.float64))


def test_pca_ridge_design_drops_reference_gcm_with_intercept():
    X = torch.zeros((3, 2), dtype=torch.float64)
    s = torch.tensor([0, 1, 2])
    H = build_design_matrix(X, s, n_sim_types=3, design_cfg={"intercept": True, "inputs": True, "sim_onehot": True})
    assert H.shape == (3, 5)
    torch.testing.assert_close(H[:, 3:], torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))


def test_pca_ridge_resolver_defaults_and_config_replay():
    args = run_model.argparse.Namespace(
        latent_dim=60,
        lambda_reg=1.0e-3,
        ppca_iters=50,
        n_folds=5,
        cv_latent_dim="20,50,100,150",
        cv_lambda="1.0e-6,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0",
        best_latent_dim=None,
        best_lambda_reg=None,
        _explicit_args=set(),
    )
    hparams = run_model._pca_ridge_hparams(args)
    assert hparams["latent_dim"] == 50
    assert hparams["n_folds"] == 3

    cfg = {
        "pca_ridge": {"latent_dim": 100, "lambda_reg": 0.1, "ppca_iters": 7},
        "CV_sweep": {"n_folds": 2, "latent_dim": [50, 100], "lambda_reg": [0.1, 1.0], "scores": [[1.0, 0.5], [0.4, 0.6]]},
        "best": {"latent_dim": 100, "lambda_reg": 0.1, "cv_equal_group_normalized_rmse": 0.4},
    }
    replay = run_model._pca_ridge_hparams(args, cfg)
    assert replay["best_latent_dim"] == 100
    assert replay["best_lambda_reg"] == 0.1
    assert replay["cv_sweep_scores"] == [[1.0, 0.5], [0.4, 0.6]]

    args._explicit_args = {"latent_dim"}
    args.latent_dim = 50
    override = run_model._pca_ridge_hparams(args, cfg)
    assert override["best_latent_dim"] is None
    assert override["latent_dim"] == 50


def test_pca_ridge_fit_predict_smoke():
    model = PCARidge(latent_dim=1, lambda_reg=1.0e-3, dtype=torch.float64, device="cpu")
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    s = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    Y = torch.randn((4, 3, 2), dtype=torch.float64)
    field_mask = torch.ones((4, 2), dtype=torch.bool)
    sh_mask = torch.ones((3, 2), dtype=torch.bool)
    model.fit(X, s, Y, field_mask=field_mask, sh_mask=sh_mask, ppca_iters=2, seed=0, n_sim_types=2)
    pred = model.predict(X[:2], s[:2])
    assert pred.shape == (2, 3, 2)
    assert torch.isfinite(pred).all()


def test_pca_mlp_equal_group_mean_weights_groups_not_field_counts():
    per_field = torch.tensor([10.0, 2.0, 4.0], dtype=torch.float32)
    field_names = ["surface_temperature", "temperature_0", "temperature_1"]
    out = _equal_group_mean(per_field, field_names)
    assert torch.isclose(out, torch.tensor(6.5))


def test_pca_mlp_restores_best_validation_step(monkeypatch):
    saved = {"call": 0, "best_state": None}
    metric_seq = [5.0, 4.0, 3.0, 4.0, 5.0]

    def fake_eval(self, *args, **kwargs):
        value = metric_seq[saved["call"]]
        if saved["call"] == 2:
            saved["best_state"] = deepcopy(self._net.state_dict())
        saved["call"] += 1
        return value

    monkeypatch.setattr(PCAMLP, "_eval_norm_srmse_val_1e3", fake_eval)

    model = PCAMLP(latent_dim=1, hidden_width=4, dtype=torch.float64, device="cpu")
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [1.5, 1.5]], dtype=torch.float64)
    s = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    Y = torch.randn((4, 2, 1), dtype=torch.float64)
    field_mask = torch.ones((4, 1), dtype=torch.bool)
    sh_mask = torch.ones((2, 1), dtype=torch.bool)

    model.fit(
        X[:2],
        s[:2],
        Y[:2],
        field_mask=field_mask[:2],
        sh_mask=sh_mask,
        field_names=["surface_temperature"],
        num_steps=4,
        ppca_iters=2,
        lr=1.0e-2,
        weight_decay=0.0,
        seed=0,
        val_X=X[2:],
        val_s=s[2:],
        val_Y=Y[2:],
        val_field_mask=field_mask[2:],
        log_every=1,
        early_stop_patience_evals=2,
    )

    assert model.mlp_fit_stats_["best_step"] == 2
    assert model.mlp_fit_stats_["steps_run"] == 4
    assert model.mlp_fit_stats_["early_stop_triggered"] is True
    for key, value in model._net.state_dict().items():
        torch.testing.assert_close(value, saved["best_state"][key])
