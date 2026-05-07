from __future__ import annotations
"""Run one public ThousandWorlds model on one subset."""

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
import thousandworlds as tw

from thousandworlds.models._common import (
    average_space_grid,
    decode_spectral_predictions,
    default_output_dir,
    enforce_equatorial_symmetry_grid,
    equal_group_normalized_rmse_grid,
    field_rmse_scale_grid,
    inverse_average_space_grid,
    prepare_tw_data,
    save_json,
    save_submission,
    score_saved_submission,
    subset_submission,
)
from ._run_model_config import (
    COORD_DEEPONET_ARG_DEFAULTS,
    COORD_DEEPONET_PRESETS,
    COORD_MLP_ARG_DEFAULTS,
    COORD_MLP_PRESETS,
    PCA_MLP_ARG_DEFAULTS,
    PCA_MLP_LINEAR_TREND_CFG,
    PCA_RIDGE_DESIGN_CFG,
    PCA_RIDGE_LINEAR_TREND_CFG,
    PPCA_ICM_LINEAR_TREND_CFG,
    TW_ROOT,
    _coord_deeponet_hparams,
    _coord_mlp_hparams,
    _gplfr_hparams,
    _merge_config_args,
    _pca_mlp_hparams,
    _pca_ridge_hparams,
    _ppca_icm_hparams,
    _resolved_config,
)
from ._run_model_utils import (
    _append_gcm_block,
    _kfold_indices,
    _mark_explicit_args,
    _parse_float_list,
    _parse_int_list,
    _torch,
)


def run(
    method: str,
    subset: str,
    *,
    data_dir: str | Path = Path("dataset"),
    seed: int = 0,
    **overrides,
) -> dict:
    """Run a baseline in memory and return predictions, metrics, and metadata."""
    args = argparse.Namespace(
        method=method,
        subset=subset,
        data_dir=Path(data_dir),
        out_dir=None,
        seed=seed,
        config=None,
        n_folds=5,
        k="1,2,3,5,10",
        gcm_penalty="0.0,0.3,1.0,3.0,10.0",
        best_k=None,
        best_gcm_penalty=None,
        latent_dim=60,
        lambda_reg=1.0e-3,
        cv_latent_dim="20,50,100,150",
        cv_lambda="1.0e-6,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0",
        best_latent_dim=None,
        best_lambda_reg=None,
        hidden_width=128,
        activation="silu",
        ppca_iters=50,
        num_steps=3000,
        lr=1.0e-3,
        weight_decay=1.0e-3,
        hard_stop_step=None,
        coord_hidden_width=256,
        coord_num_layers=6,
        coord_activation="tanh",
        coord_batch_size=128,
        coord_predict_chunk_size=262_144,
        coord_num_steps=3000,
        coord_lr=1.0e-4,
        coord_weight_decay=0.0,
        coord_hard_stop_step=None,
        deeponet_rank=128,
        deeponet_branch_hidden_width=256,
        deeponet_trunk_hidden_width=256,
        deeponet_branch_num_layers=3,
        deeponet_trunk_num_layers=3,
        deeponet_activation="silu",
        deeponet_batch_size=32_768,
        deeponet_predict_chunk_size=262_144,
        deeponet_num_steps=3000,
        deeponet_lr=3.0e-4,
        deeponet_weight_decay=1.0e-4,
        deeponet_hard_stop_step=None,
        kernel="matern52",
        kernel_mode="shared",
        gp_steps=50,
        gp_lr=5.0e-2,
        n_samples=64,
        ell_init="median_pairwise_dist",
        ppca_icm_preset="tuned",
        gplfr_num_training_steps=2000,
        gplfr_inverse_temperature=0.1,
        gplfr_latent_nugget=0.1,
        gplfr_variable_weights="learned_per_group",
        gplfr_output_coregionalization="field_coregionalized",
        gplfr_log_every=5,
        dtype="float64",
        device="auto",
        _config=None,
        _explicit_args=set(overrides),
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    result = _run_method(args)
    data = result["data"]
    return {
        "predictions": result["predictions"],
        "point_predictions": result.get("point_predictions", result["predictions"][:1]),
        "metrics": tw.score_predictions(
            result["predictions"],
            data.grid_bundle,
            point_predictions=result.get("point_predictions"),
        ),
        "meta": result.get("meta", {}),
        "data": data,
    }


def _gplfr_stats_dir(subset: str) -> Path:
    return TW_ROOT / "dataset" / "norm_stats" / subset


def _gplfr_inverse_preprocess_grid(Y: np.ndarray, field_names: list[str], stats: tw.Stats, field_idxs: list[int]) -> np.ndarray:
    out = np.asarray(Y, dtype=np.float32).copy()
    for j in field_idxs:
        strategy_name, kwargs = tw.preprocessing._strategy_params(field_names[j], stats)
        out[:, j] = tw.preprocessing._inverse_preprocess_array(out[:, j], strategy_name, kwargs)
    return out


def _gplfr_decode_public_predictions(coeffs: np.ndarray, data) -> np.ndarray:
    field_names = data.spectral_bundle.raw_field_names
    coeffs_norm = np.transpose(coeffs, (0, 1, 3, 2)).astype(np.float32, copy=False)
    scales = np.ones(coeffs_norm.shape[1:3], dtype=np.float32)
    if data.stats.asr_olr_normalize_by_f_star:
        f_star = data.grid_bundle.X_test[:, data.stats.input_names.index("F_star")].astype(np.float32)
        for j, name in enumerate(field_names):
            if tw.preprocessing._base_var(name) in tw.preprocessing.ASR_OLR_FIELDS:
                scales[:, j] = f_star
    cloud_idxs = [i for i, name in enumerate(field_names) if name.startswith("cloud_fraction")]
    export_idxs = [i for i in range(len(field_names)) if i not in cloud_idxs]
    predictions = []
    for sample in coeffs_norm:
        sample_coeffs = tw.unnormalise_spectral(sample, field_names, data.stats) * scales[:, :, None]
        sample_coeffs = tw.apply_symmetry_mask(sample_coeffs, field_names, data.sh_mask.T)
        grid = tw.to_grid(sample_coeffs, data.inverse_sht).astype(np.float32)
        grid = _gplfr_inverse_preprocess_grid(grid, field_names, data.stats, cloud_idxs)
        if cloud_idxs:
            grid[:, cloud_idxs] = np.clip(grid[:, cloud_idxs], 0.0, 1.0)
        predictions.append(_gplfr_inverse_preprocess_grid(grid, field_names, data.stats, export_idxs))
    out = np.stack(predictions, axis=0)
    if cloud_idxs:
        out[:, :, cloud_idxs] = np.clip(out[:, :, cloud_idxs], 0.0, 1.0)
    return out.astype(np.float32)


def _run_train_mean(args: argparse.Namespace) -> dict:
    from thousandworlds.models.train_mean import TrainMean

    data = prepare_tw_data(args.subset, data_dir=args.data_dir)
    model = TrainMean().fit(
        data.grid_bundle.Y_train,
        data.grid_bundle.raw_field_names,
        data.stats,
        X_train=data.grid_bundle.X_train,
        field_mask=data.grid_bundle.field_mask_train,
    )
    return {"data": data, "predictions": model.predict(data.grid_bundle.X_test)[None], "meta": {"method": "train_mean"}}


def _run_knn(args: argparse.Namespace) -> dict:
    from thousandworlds.models.knn import KNN

    data = prepare_tw_data(args.subset, data_dir=args.data_dir)
    Y_train_avg = average_space_grid(data.grid_bundle.Y_train, data.grid_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_train)
    k_grid = _parse_int_list(args.k)
    penalty_grid = _parse_float_list(args.gcm_penalty)
    if args.best_k is not None and args.best_gcm_penalty is not None:
        best = {"metric": getattr(args, "best_cv_equal_group_normalized_rmse", None), "k": int(args.best_k), "penalty": float(args.best_gcm_penalty)}
        sweep = getattr(args, "cv_sweep_scores", None) or {}
    else:
        best = {"metric": float("inf"), "k": None, "penalty": None}
        folds = [
            (
                train_idx,
                val_idx,
                field_rmse_scale_grid(Y_train_avg[train_idx], data.grid_bundle.field_mask_train[train_idx]),
            )
            for train_idx, val_idx in _kfold_indices(len(data.X_train_std), n_folds=args.n_folds, seed=args.seed)
        ]
        sweep: dict[str, dict[int, float]] = {}
        for penalty in penalty_grid:
            X_train = _append_gcm_block(data.X_train_std, data.s_train, len(data.gcm_labels), penalty)
            losses = {}
            for k in k_grid:
                fold_losses = []
                for train_idx, val_idx, field_scale in folds:
                    model = KNN()
                    model.fit(
                        X_train[train_idx],
                        Y_train_avg[train_idx],
                        k_candidates=[k],
                        field_mask=data.grid_bundle.field_mask_train[train_idx],
                    )
                    fold_losses.append(
                        equal_group_normalized_rmse_grid(
                            enforce_equatorial_symmetry_grid(model.predict(X_train[val_idx], k), data.grid_bundle.raw_field_names),
                            Y_train_avg[val_idx],
                            data.grid_bundle.raw_field_names,
                            field_scale,
                            data.grid_bundle.field_mask_train[val_idx],
                        )
                    )
                mean_loss = float(np.mean(fold_losses))
                losses[int(k)] = float("inf") if not np.isfinite(mean_loss) else mean_loss
                if losses[int(k)] < best["metric"]:
                    best = {"metric": losses[int(k)], "k": int(k), "penalty": float(penalty)}
            sweep[str(penalty)] = losses
    if best["k"] is None or best["penalty"] is None:
        best = {"metric": float("inf"), "k": int(k_grid[0]), "penalty": float(penalty_grid[0])}
    X_train = _append_gcm_block(data.X_train_std, data.s_train, len(data.gcm_labels), best["penalty"])
    X_test = _append_gcm_block(data.X_test_std, data.s_test, len(data.gcm_labels), best["penalty"])
    model = KNN()
    model.fit(X_train, Y_train_avg, k_candidates=k_grid, field_mask=data.grid_bundle.field_mask_train)
    pred_avg = enforce_equatorial_symmetry_grid(model.predict(X_test, int(best["k"])), data.grid_bundle.raw_field_names)
    predictions = tw.inverse_preprocess_outputs_grid(pred_avg, data.grid_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_test)[None]
    metric = None if best["metric"] is None else float(best["metric"])
    return {"data": data, "predictions": predictions, "meta": {"method": "knn", "best_k": int(best["k"]), "best_gcm_penalty": float(best["penalty"]), "equatorial_symmetry": True, "spectral_truncation": False, "cv_equal_group_normalized_rmse": metric, "cv_sweep": sweep}}


def _run_pca_ridge(args: argparse.Namespace) -> dict:
    from thousandworlds.models._torch_kernels import build_design_matrix
    from thousandworlds.models.pca_ridge import PCARidge, fit_latent_ridge

    torch = _torch()
    data = prepare_tw_data(args.subset, data_dir=args.data_dir)
    hparams = _pca_ridge_hparams(args)
    latent_grid = _parse_int_list(str(hparams["cv_latent_dim"]))
    lambda_grid = _parse_float_list(str(hparams["cv_lambda"]))
    Y_train = np.transpose(tw.normalise_spectral(data.spectral_bundle.Y_train, data.spectral_bundle.raw_field_names, data.stats), (0, 2, 1))
    Y_train_avg = average_space_grid(data.grid_bundle.Y_train, data.grid_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_train)

    if hparams["best_latent_dim"] is not None and hparams["best_lambda_reg"] is not None:
        best = {
            "latent_dim": int(hparams["best_latent_dim"]),
            "lambda_reg": float(hparams["best_lambda_reg"]),
            "metric": hparams["best_cv_equal_group_normalized_rmse"],
        }
        scores = hparams["cv_sweep_scores"]
    else:
        best = {"latent_dim": None, "lambda_reg": None, "metric": float("inf")}
        scores = np.full((len(latent_grid), len(lambda_grid)), np.inf, dtype=np.float64)
        folds = [
            (
                train_idx,
                val_idx,
                field_rmse_scale_grid(Y_train_avg[train_idx], data.grid_bundle.field_mask_train[train_idx]),
            )
            for train_idx, val_idx in _kfold_indices(len(data.X_train_std), n_folds=int(hparams["n_folds"]), seed=args.seed)
        ]
        for i, latent_dim in enumerate(latent_grid):
            fold_models = []
            for train_idx, val_idx, field_scale in folds:
                model = PCARidge(
                    latent_dim=int(latent_dim),
                    lambda_reg=float(lambda_grid[0]),
                    design_cfg=PCA_RIDGE_DESIGN_CFG,
                    dtype=getattr(torch, args.dtype),
                    device=args.device,
                )
                model.fit(
                    torch.from_numpy(data.X_train_std[train_idx]),
                    torch.from_numpy(data.s_train[train_idx]),
                    torch.from_numpy(Y_train[train_idx]),
                    field_mask=torch.from_numpy(data.spectral_bundle.field_mask_train[train_idx]),
                    sh_mask=torch.from_numpy(data.sh_mask),
                    linear_trend_cfg=PCA_RIDGE_LINEAR_TREND_CFG,
                    ppca_iters=int(hparams["ppca_iters"]),
                    seed=args.seed,
                    n_sim_types=len(data.gcm_labels),
                )
                H_train = build_design_matrix(
                    torch.as_tensor(data.X_train_std[train_idx], device=model.device, dtype=model.dtype),
                    torch.as_tensor(data.s_train[train_idx], device=model.device, dtype=torch.long),
                    n_sim_types=model.n_sim_types_,
                    design_cfg=PCA_RIDGE_DESIGN_CFG,
                )
                fold_models.append((model, H_train, train_idx, val_idx, field_scale))
            for j, lambda_reg in enumerate(lambda_grid):
                fold_losses = []
                for model, H_train, train_idx, val_idx, field_scale in fold_models:
                    model.B_ = fit_latent_ridge(
                        H_train,
                        model.ppca_.Z.detach(),
                        lambda_reg=float(lambda_reg),
                        intercept=bool(PCA_RIDGE_DESIGN_CFG["intercept"]),
                    )
                    pred_coeffs = model.predict(
                        torch.from_numpy(data.X_train_std[val_idx]),
                        torch.from_numpy(data.s_train[val_idx].copy()),
                    ).detach().cpu().numpy().transpose(0, 2, 1)
                    pred_avg = decode_spectral_predictions(pred_coeffs, data.spectral_bundle.raw_field_names, data.stats, sh_mask=data.sh_mask, inverse_sht=data.inverse_sht)
                    fold_losses.append(
                        equal_group_normalized_rmse_grid(
                            pred_avg,
                            Y_train_avg[val_idx],
                            data.grid_bundle.raw_field_names,
                            field_scale,
                            data.grid_bundle.field_mask_train[val_idx],
                        )
                    )
                mean_loss = float(np.mean(fold_losses))
                scores[i, j] = float("inf") if not np.isfinite(mean_loss) else mean_loss
                if scores[i, j] < best["metric"]:
                    best = {"latent_dim": int(latent_dim), "lambda_reg": float(lambda_reg), "metric": float(scores[i, j])}
        scores = scores.tolist()

    model = PCARidge(
        latent_dim=int(best["latent_dim"]),
        lambda_reg=float(best["lambda_reg"]),
        design_cfg=PCA_RIDGE_DESIGN_CFG,
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )
    model.fit(
        torch.from_numpy(data.X_train_std),
        torch.from_numpy(data.s_train.copy()),
        torch.from_numpy(Y_train),
        field_mask=torch.from_numpy(data.spectral_bundle.field_mask_train),
        sh_mask=torch.from_numpy(data.sh_mask),
        linear_trend_cfg=PCA_RIDGE_LINEAR_TREND_CFG,
        ppca_iters=int(hparams["ppca_iters"]),
        seed=args.seed,
        n_sim_types=len(data.gcm_labels),
    )
    pred_coeffs = model.predict(torch.from_numpy(data.X_test_std), torch.from_numpy(data.s_test.copy())).detach().cpu().numpy().transpose(0, 2, 1)
    pred_avg = decode_spectral_predictions(pred_coeffs, data.spectral_bundle.raw_field_names, data.stats, sh_mask=data.sh_mask, inverse_sht=data.inverse_sht)
    predictions = inverse_average_space_grid(pred_avg, data.spectral_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_test)[None]
    return {
        "data": data,
        "predictions": predictions,
        "meta": {
            "method": "pca_ridge",
            "best_latent_dim": int(best["latent_dim"]),
            "best_lambda_reg": float(best["lambda_reg"]),
            "cv_equal_group_normalized_rmse": None if best["metric"] is None else float(best["metric"]),
            "cv_sweep_scores": scores,
            "fit": model.fit_stats_,
        },
    }


def _run_pca_mlp(args: argparse.Namespace) -> dict:
    from thousandworlds.models.pca_mlp import PCAMLP

    torch = _torch()
    data = prepare_tw_data(args.subset, data_dir=args.data_dir)
    hparams = _pca_mlp_hparams(args)
    Y_train = np.transpose(tw.normalise_spectral(data.spectral_bundle.Y_train, data.spectral_bundle.raw_field_names, data.stats), (0, 2, 1))
    model = PCAMLP(latent_dim=int(hparams["latent_dim"]), hidden_width=int(hparams["hidden_width"]), activation=args.activation, dtype=getattr(torch, args.dtype), device=args.device)
    model.fit(
        torch.from_numpy(data.X_train_std),
        torch.from_numpy(data.s_train.copy()),
        torch.from_numpy(Y_train),
        field_mask=torch.from_numpy(data.spectral_bundle.field_mask_train),
        sh_mask=torch.from_numpy(data.sh_mask),
        linear_trend_cfg=PCA_MLP_LINEAR_TREND_CFG,
        field_names=data.spectral_bundle.raw_field_names,
        ppca_iters=args.ppca_iters,
        num_steps=int(hparams["num_steps"]),
        lr=float(hparams["lr"]),
        weight_decay=float(hparams["weight_decay"]),
        seed=args.seed,
        hard_stop_step=hparams["hard_stop_step"],
    )
    pred_coeffs = model.predict(torch.from_numpy(data.X_test_std), torch.from_numpy(data.s_test.copy())).detach().cpu().numpy().transpose(0, 2, 1)
    pred_avg = decode_spectral_predictions(pred_coeffs, data.spectral_bundle.raw_field_names, data.stats, sh_mask=data.sh_mask, inverse_sht=data.inverse_sht)
    predictions = inverse_average_space_grid(pred_avg, data.spectral_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_test)[None]
    return {"data": data, "predictions": predictions, "meta": {"method": "pca_mlp", "mlp_fit": model.mlp_fit_stats_}}


def _run_coord_mlp(args: argparse.Namespace) -> dict:
    from thousandworlds.models.coord_mlp import CoordMLP

    torch = _torch()
    data = prepare_tw_data(args.subset, data_dir=args.data_dir)
    hparams = _coord_mlp_hparams(args)
    Y_train_avg = average_space_grid(data.grid_bundle.Y_train, data.grid_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_train)
    model = CoordMLP(
        hidden_width=int(hparams["coord_hidden_width"]),
        num_layers=int(hparams["coord_num_layers"]),
        activation=str(hparams["coord_activation"]),
        batch_size=int(hparams["coord_batch_size"]),
        predict_chunk_size=int(hparams["coord_predict_chunk_size"]),
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )
    model.fit(
        data.X_train_std,
        data.s_train,
        Y_train_avg,
        field_mask=data.grid_bundle.field_mask_train,
        field_names=data.grid_bundle.raw_field_names,
        n_sim_types=len(data.gcm_labels),
        num_steps=int(hparams["coord_num_steps"]),
        lr=float(hparams["coord_lr"]),
        weight_decay=float(hparams["coord_weight_decay"]),
        seed=args.seed,
        hard_stop_step=hparams["coord_hard_stop_step"],
    )
    pred_avg = enforce_equatorial_symmetry_grid(
        model.predict(data.X_test_std, data.s_test).detach().cpu().numpy(),
        data.grid_bundle.raw_field_names,
    )
    predictions = inverse_average_space_grid(pred_avg, data.grid_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_test)[None]
    return {"data": data, "predictions": predictions, "meta": {"method": "coord_mlp", "fit": model.fit_stats_, "equatorial_symmetry": True}}


def _run_coord_deeponet(args: argparse.Namespace) -> dict:
    from thousandworlds.models.coord_deeponet import CoordDeepONet

    torch = _torch()
    data = prepare_tw_data(args.subset, data_dir=args.data_dir)
    hparams = _coord_deeponet_hparams(args)
    Y_train_avg = average_space_grid(data.grid_bundle.Y_train, data.grid_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_train)
    model = CoordDeepONet(
        rank=int(hparams["deeponet_rank"]),
        branch_hidden_width=int(hparams["deeponet_branch_hidden_width"]),
        trunk_hidden_width=int(hparams["deeponet_trunk_hidden_width"]),
        branch_num_layers=int(hparams["deeponet_branch_num_layers"]),
        trunk_num_layers=int(hparams["deeponet_trunk_num_layers"]),
        activation=str(hparams["deeponet_activation"]),
        batch_size=int(hparams["deeponet_batch_size"]),
        predict_chunk_size=int(hparams["deeponet_predict_chunk_size"]),
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )
    model.fit(
        data.X_train_std,
        data.s_train,
        Y_train_avg,
        field_mask=data.grid_bundle.field_mask_train,
        field_names=data.grid_bundle.raw_field_names,
        n_sim_types=len(data.gcm_labels),
        num_steps=int(hparams["deeponet_num_steps"]),
        lr=float(hparams["deeponet_lr"]),
        weight_decay=float(hparams["deeponet_weight_decay"]),
        seed=args.seed,
        hard_stop_step=hparams["deeponet_hard_stop_step"],
    )
    pred_avg = enforce_equatorial_symmetry_grid(
        model.predict(data.X_test_std, data.s_test).detach().cpu().numpy(),
        data.grid_bundle.raw_field_names,
    )
    predictions = inverse_average_space_grid(pred_avg, data.grid_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_test)[None]
    return {"data": data, "predictions": predictions, "meta": {"method": "coord_deeponet", "fit": model.fit_stats_, "equatorial_symmetry": True}}


def _run_ppca_icm(args: argparse.Namespace) -> dict:
    from thousandworlds.models.ppca_icm import PPCAICM

    torch = _torch()
    data = prepare_tw_data(args.subset, data_dir=args.data_dir)
    hparams = _ppca_icm_hparams(args)
    Y_train = np.transpose(tw.normalise_spectral(data.spectral_bundle.Y_train, data.spectral_bundle.raw_field_names, data.stats), (0, 2, 1))
    model = PPCAICM(
        latent_dim=int(hparams["latent_dim"]),
        kernel=str(hparams["kernel"]),
        kernel_mode=str(hparams["kernel_mode"]),
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )
    model.fit(
        torch.from_numpy(data.X_train_std),
        torch.from_numpy(data.s_train),
        torch.from_numpy(Y_train),
        field_mask=torch.from_numpy(data.spectral_bundle.field_mask_train),
        sh_mask=torch.from_numpy(data.sh_mask),
        linear_trend_cfg=PPCA_ICM_LINEAR_TREND_CFG,
        ppca_iters=int(hparams["ppca_iters"]),
        gp_steps=int(hparams["gp_steps"]),
        gp_lr=float(hparams["gp_lr"]),
        seed=args.seed,
        ell_init=hparams["ell_init"],
        hard_stop_step=hparams["hard_stop_step"],
    )
    point_coeffs = model.predict(torch.from_numpy(data.X_test_std), torch.from_numpy(data.s_test)).detach().cpu().numpy()[None]
    coeffs = model.predict_samples(
        torch.from_numpy(data.X_test_std),
        torch.from_numpy(data.s_test),
        n_post_samples=int(hparams["n_samples"]),
        seed=args.seed,
        include_gp_nugget=True,
        include_ppca_noise=True,
    ).detach().cpu().numpy()
    point_decoded = decode_spectral_predictions(
        np.transpose(point_coeffs, (0, 1, 3, 2)),
        data.spectral_bundle.raw_field_names,
        data.stats,
        sh_mask=data.sh_mask,
        inverse_sht=data.inverse_sht,
    )
    point_predictions = np.stack(
        [inverse_average_space_grid(point_decoded[m], data.spectral_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_test) for m in range(point_decoded.shape[0])],
        axis=0,
    )
    decoded = decode_spectral_predictions(np.transpose(coeffs, (0, 1, 3, 2)), data.spectral_bundle.raw_field_names, data.stats, sh_mask=data.sh_mask, inverse_sht=data.inverse_sht)
    predictions = np.stack(
        [inverse_average_space_grid(decoded[m], data.spectral_bundle.raw_field_names, data.stats, X=data.grid_bundle.X_test) for m in range(decoded.shape[0])],
        axis=0,
    )
    return {
        "data": data,
        "predictions": predictions,
        "point_predictions": point_predictions,
        "meta": {
            "method": "ppca_icm",
            "gp_fit": model.gp_fit_stats_,
            "runtime": {
                "n_samples": int(hparams["n_samples"]),
                "latent_dim": int(hparams["latent_dim"]),
                "gp_steps": int(hparams["gp_steps"]),
                "gp_lr": float(hparams["gp_lr"]),
                "hard_stop_step": None if hparams["hard_stop_step"] is None else int(hparams["hard_stop_step"]),
                "ell_init": hparams["ell_init"],
            },
        },
    }


def _run_gplfr(args: argparse.Namespace) -> dict:
    from thousandworlds.models.gplfr import GPLFR

    torch = _torch()
    hparams = _gplfr_hparams(args, getattr(args, "_config", None))
    stats_dir = _gplfr_stats_dir(args.subset)
    data = prepare_tw_data(args.subset, data_dir=args.data_dir, stats_dir=stats_dir)
    Y_train = np.transpose(tw.normalise_spectral(data.spectral_bundle.Y_train, data.spectral_bundle.raw_field_names, data.stats), (0, 2, 1))
    model = GPLFR(
        latent_dim=int(hparams["latent_dim"]),
        num_training_steps=int(hparams["num_training_steps"]),
        inverse_temperature=float(hparams["inverse_temperature"]),
        latent_nugget=float(hparams["latent_nugget"]),
        lr_Z=float(hparams["lr_Z"]),
        lr_global=float(hparams["lr_global"]),
        variable_weights=str(hparams["variable_weights"]),
        output_coregionalization=str(hparams["output_coregionalization"]),
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )
    model.fit(
        torch.from_numpy(data.X_train_std),
        torch.from_numpy(data.s_train.copy()),
        torch.from_numpy(Y_train),
        field_mask=torch.from_numpy(data.spectral_bundle.field_mask_train),
        sh_mask=torch.from_numpy(data.sh_mask),
        field_names=data.spectral_bundle.raw_field_names,
        seed=args.seed,
        n_sim_types=len(data.gcm_labels),
    )
    point_coeffs = model.predict(
        torch.from_numpy(data.X_test_std),
        torch.from_numpy(data.s_test.copy()),
    ).detach().cpu().numpy()[None]
    coeffs = model.predict_samples(
        torch.from_numpy(data.X_test_std),
        torch.from_numpy(data.s_test.copy()),
        seed=args.seed,
    ).detach().cpu().numpy()
    point_predictions = _gplfr_decode_public_predictions(point_coeffs, data)
    predictions = _gplfr_decode_public_predictions(coeffs, data)
    cloud_idxs = [i for i, name in enumerate(data.spectral_bundle.raw_field_names) if name.startswith("cloud_fraction")]
    if cloud_idxs:
        predictions[:, :, cloud_idxs] = np.clip(predictions[:, :, cloud_idxs], 0.0, 1.0)
    return {
        "data": data,
        "predictions": predictions,
        "point_predictions": point_predictions,
        "meta": {
            "method": "gplfr",
            "fit": model.fit_stats_,
        },
    }


def _run_method(args: argparse.Namespace) -> dict:
    return {
        "train_mean": _run_train_mean,
        "knn": _run_knn,
        "pca_ridge": _run_pca_ridge,
        "pca_mlp": _run_pca_mlp,
        "coord_mlp": _run_coord_mlp,
        "coord_deeponet": _run_coord_deeponet,
        "ppca_icm": _run_ppca_icm,
        "gplfr": _run_gplfr,
    }[args.method](args)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m thousandworlds.run_model", description="Run extracted ThousandWorlds models.")
    parser.add_argument("method", nargs="?", choices=["train_mean", "knn", "pca_ridge", "pca_mlp", "ppca_icm", "gplfr", "coord_mlp", "coord_deeponet"])
    parser.add_argument("subset", nargs="?")
    # ---- Shared / general arguments ----
    parser.add_argument("--config", type=Path, default=None, help="Resolved config.json written by a previous run")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)

    # ---- KNN arguments ----
    parser.add_argument("--n-folds", type=int, default=5, help="KNN: Number of cross-validation folds")
    parser.add_argument("--k", default="1,2,3,5,10", help="KNN: List of k neighbors to try (default: 1,2,3,5,10)")
    parser.add_argument("--gcm-penalty", default="0.0,0.3,1.0,3.0,10.0", help="KNN: List of GCM penalty values to try (default: 0.0,0.3,1.0,3.0,10.0)")
    parser.add_argument("--best-k", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--best-gcm-penalty", type=float, default=None, help=argparse.SUPPRESS)

    # ---- Latent-model arguments ----
    parser.add_argument("--latent-dim", type=int, default=60, help="PCA-MLP / PPCA-ICM / PCA-Ridge / GPLFR: Latent dimension")
    parser.add_argument("--lambda-reg", type=float, default=1.0e-3, help="PCA-Ridge: Ridge penalty")
    parser.add_argument("--cv-latent-dim", default="20,50,100,150", help="PCA-Ridge: CV latent dimension grid")
    parser.add_argument("--cv-lambda", default="1.0e-6,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0", help="PCA-Ridge: CV ridge penalty grid")
    parser.add_argument("--best-latent-dim", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--best-lambda-reg", type=float, default=None, help=argparse.SUPPRESS)

    # ---- PCA-MLP arguments ----
    parser.add_argument("--hidden-width", type=int, default=128, help="PCA-MLP: Hidden width of MLP")
    parser.add_argument("--activation", choices=["silu", "relu"], default="silu", help="PCA-MLP: Activation function")
    parser.add_argument("--ppca-iters", type=int, default=50, help="PCA-MLP / PCA-Ridge / PPCA-ICM: Number of PPCA iterations")
    parser.add_argument("--num-steps", type=int, default=3000, help="PCA-MLP: Number of training steps")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="PCA-MLP: Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1.0e-3, help="PCA-MLP: Weight decay for MLP")
    parser.add_argument("--hard-stop-step", type=int, default=None, help="PCA-MLP: Optional hard stop step")

    # ---- Coord-MLP arguments ----
    parser.add_argument("--coord-hidden-width", type=int, default=256, help="Coord-MLP: Hidden width")
    parser.add_argument("--coord-num-layers", type=int, default=6, help="Coord-MLP: Number of hidden layers")
    parser.add_argument("--coord-activation", choices=["tanh", "silu", "relu"], default="tanh", help="Coord-MLP: Activation function")
    parser.add_argument("--coord-batch-size", type=int, default=128, help="Coord-MLP: Point batch size")
    parser.add_argument("--coord-predict-chunk-size", type=int, default=262_144, help="Coord-MLP: Point prediction chunk size")
    parser.add_argument("--coord-num-steps", type=int, default=3000, help="Coord-MLP: Number of training steps")
    parser.add_argument("--coord-lr", type=float, default=1.0e-4, help="Coord-MLP: Learning rate")
    parser.add_argument("--coord-weight-decay", type=float, default=0.0, help="Coord-MLP: Weight decay")
    parser.add_argument("--coord-hard-stop-step", type=int, default=None, help="Coord-MLP: Optional hard stop step")

    # ---- Coord-DeepONet arguments ----
    parser.add_argument("--deeponet-rank", type=int, default=128, help="Coord-DeepONet: Branch/trunk rank")
    parser.add_argument("--deeponet-branch-hidden-width", type=int, default=256, help="Coord-DeepONet: Branch hidden width")
    parser.add_argument("--deeponet-trunk-hidden-width", type=int, default=256, help="Coord-DeepONet: Trunk hidden width")
    parser.add_argument("--deeponet-branch-num-layers", type=int, default=3, help="Coord-DeepONet: Number of branch hidden layers")
    parser.add_argument("--deeponet-trunk-num-layers", type=int, default=3, help="Coord-DeepONet: Number of trunk hidden layers")
    parser.add_argument("--deeponet-activation", choices=["silu", "relu", "tanh"], default="silu", help="Coord-DeepONet: Activation function")
    parser.add_argument("--deeponet-batch-size", type=int, default=32_768, help="Coord-DeepONet: Point batch size")
    parser.add_argument("--deeponet-predict-chunk-size", type=int, default=262_144, help="Coord-DeepONet: Point prediction chunk size")
    parser.add_argument("--deeponet-num-steps", type=int, default=3000, help="Coord-DeepONet: Number of training steps")
    parser.add_argument("--deeponet-lr", type=float, default=3.0e-4, help="Coord-DeepONet: Learning rate")
    parser.add_argument("--deeponet-weight-decay", type=float, default=1.0e-4, help="Coord-DeepONet: Weight decay")
    parser.add_argument("--deeponet-hard-stop-step", type=int, default=None, help="Coord-DeepONet: Optional hard stop step")

    # ---- PPCA-ICM arguments ----
    parser.add_argument("--kernel", choices=["rbf", "matern32", "matern52"], default="matern52", help="PPCA-ICM: GP kernel type")
    parser.add_argument("--kernel-mode", choices=["shared", "per_pc"], default="shared", help="PPCA-ICM: GP kernel sharing mode")
    parser.add_argument("--gp-steps", type=int, default=50, help="PPCA-ICM: Number of GP optimization steps")
    parser.add_argument("--gp-lr", type=float, default=5.0e-2, help="PPCA-ICM: Learning rate for GP")
    parser.add_argument("--n-samples", type=int, default=8, help="PPCA-ICM: Number of posterior samples")
    parser.add_argument("--ell-init", default="median_pairwise_dist", help="PPCA-ICM: Initial lengthscale or strategy")
    parser.add_argument("--ppca-icm-preset", choices=["default", "tuned"], default="tuned", help="PPCA-ICM: Hyperparameter preset to use")
    parser.add_argument("--gplfr-num-training-steps", type=int, default=2000, help=argparse.SUPPRESS)
    parser.add_argument("--gplfr-inverse-temperature", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--gplfr-latent-nugget", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--gplfr-variable-weights", choices=["fixed", "learned_per_group"], default="learned_per_group", help=argparse.SUPPRESS)
    parser.add_argument("--gplfr-output-coregionalization", choices=["none", "field_coregionalized"], default="field_coregionalized", help=argparse.SUPPRESS)
    parser.add_argument("--gplfr-log-every", type=int, default=5, help=argparse.SUPPRESS)

    # ---- Torch related ----
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64", help="Torch dtype for neural/GP models")
    parser.add_argument("--device", default="auto", help="Torch device for neural/GP models")

    args = _mark_explicit_args(parser.parse_args(), parser, sys.argv[1:])
    args, cfg = _merge_config_args(args, parser)
    args._config = cfg
    if args.method is None or args.subset is None:
        parser.error("method and subset are required unless provided via --config")

    result = _run_method(args)
    data = result["data"]
    out_dir = default_output_dir(args.method, args.subset) if args.out_dir is None else args.out_dir
    pred_path = save_submission(out_dir / "predictions.npz", result["predictions"], simulation_id=data.grid_bundle.test_ids, field_names=data.grid_bundle.field_names)
    point_pred_path = save_submission(out_dir / "predictions_mean.npz", result.get("point_predictions", result["predictions"][:1]), simulation_id=data.grid_bundle.test_ids, field_names=data.grid_bundle.field_names)
    save_json(out_dir / "config.json", _resolved_config(args, out_dir=out_dir, data_dir=args.data_dir, meta=result.get("meta")))
    save_json(out_dir / "metrics_standard.json", score_saved_submission(pred_path, data_dir=args.data_dir, subset=args.subset, protocol="standard", point_predictions_path=point_pred_path))
    if tw.supports_protocol(args.subset, "shared_planets"):
        shared = tw.load(subset=args.subset, protocol="shared_planets", data_dir=args.data_dir, space="grid")
        shared_predictions, shared_ids = subset_submission(result["predictions"], data.grid_bundle.test_ids, wanted_ids=shared.test_ids)
        shared_point_predictions, _ = subset_submission(result.get("point_predictions", result["predictions"][:1]), data.grid_bundle.test_ids, wanted_ids=shared.test_ids)
        with tempfile.TemporaryDirectory() as tmpdir:
            shared_path = save_submission(Path(tmpdir) / "predictions_shared_planets.npz", shared_predictions, simulation_id=shared_ids, field_names=data.grid_bundle.field_names)
            shared_point_path = save_submission(Path(tmpdir) / "predictions_mean_shared_planets.npz", shared_point_predictions, simulation_id=shared_ids, field_names=data.grid_bundle.field_names)
            save_json(out_dir / "metrics_shared_planets.json", score_saved_submission(shared_path, data_dir=args.data_dir, subset=args.subset, protocol="shared_planets", point_predictions_path=shared_point_path))
    print(pred_path)


if __name__ == "__main__":
    main()
