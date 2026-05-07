from __future__ import annotations

import argparse
import json
from pathlib import Path

from ._run_model_utils import _csv, _parse_float_list, _parse_int_list

TW_ROOT = Path(__file__).resolve().parents[1]

PCA_MLP_ARG_DEFAULTS = {
    "latent_dim": 60,
    "hidden_width": 128,
    "num_steps": 3000,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-3,
    "hard_stop_step": None,
}
PCA_MLP_PRESETS = {
    "multi-partial": {"latent_dim": 50, "hidden_width": 1024, "num_steps": 4000, "lr": 3.0e-4, "weight_decay": 1.0e-4, "hard_stop_step": 860},
    "multi-complete": {"latent_dim": 50, "hidden_width": 1024, "num_steps": 4000, "lr": 3.0e-4, "weight_decay": 1.0e-4, "hard_stop_step": 630},
    "single-complete": {"latent_dim": 150, "hidden_width": 512, "num_steps": 4000, "lr": 3.0e-4, "weight_decay": 1.0e-4, "hard_stop_step": 330},
}
PCA_MLP_LINEAR_TREND_CFG = {"enabled": True, "lambda": 1.0e-3, "design": {"intercept": True, "inputs": True, "sim_onehot": False}}

PCA_RIDGE_ARG_DEFAULTS = {
    "latent_dim": 50,
    "lambda_reg": 1.0e-3,
    "ppca_iters": 50,
    "n_folds": 3,
    "cv_latent_dim": "20,50,100,150",
    "cv_lambda": "1.0e-6,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0",
}
PCA_RIDGE_DESIGN_CFG = {"intercept": True, "inputs": True, "sim_onehot": True}
PCA_RIDGE_LINEAR_TREND_CFG = {"enabled": True, "lambda": 1.0e-3, "design": {"intercept": True, "inputs": True, "sim_onehot": False}}

COORD_MLP_ARG_DEFAULTS = {
    "coord_hidden_width": 256,
    "coord_num_layers": 6,
    "coord_activation": "tanh",
    "coord_batch_size": 128,
    "coord_predict_chunk_size": 262_144,
    "coord_num_steps": 3000,
    "coord_lr": 1.0e-4,
    "coord_weight_decay": 0.0,
    "coord_hard_stop_step": None,
}
COORD_MLP_PRESETS: dict[str, dict] = {
    "multi-partial": {"hidden_width": 1024, "num_layers": 4, "activation": "tanh", "batch_size": 128, "predict_chunk_size": 262_144, "num_steps": 8000, "lr": 3.0e-4, "weight_decay": 0.0, "hard_stop_step": 7800},
    "multi-complete": {"hidden_width": 1024, "num_layers": 4, "activation": "tanh", "batch_size": 128, "predict_chunk_size": 262_144, "num_steps": 8000, "lr": 3.0e-4, "weight_decay": 0.0, "hard_stop_step": 7700},
    "single-complete": {"hidden_width": 512, "num_layers": 6, "activation": "tanh", "batch_size": 128, "predict_chunk_size": 262_144, "num_steps": 8000, "lr": 3.0e-4, "weight_decay": 0.0, "hard_stop_step": 7800},
}

COORD_DEEPONET_ARG_DEFAULTS = {
    "deeponet_rank": 128,
    "deeponet_branch_hidden_width": 256,
    "deeponet_trunk_hidden_width": 256,
    "deeponet_branch_num_layers": 3,
    "deeponet_trunk_num_layers": 3,
    "deeponet_activation": "silu",
    "deeponet_batch_size": 32_768,
    "deeponet_predict_chunk_size": 262_144,
    "deeponet_num_steps": 3000,
    "deeponet_lr": 3.0e-4,
    "deeponet_weight_decay": 1.0e-4,
    "deeponet_hard_stop_step": None,
}
COORD_DEEPONET_PRESETS: dict[str, dict] = {}

PPCA_ICM_ARG_DEFAULTS = {
    "latent_dim": 60,
    "kernel": "matern52",
    "kernel_mode": "shared",
    "ppca_iters": 50,
    "gp_steps": 50,
    "gp_lr": 5.0e-2,
    "n_samples": 8,
    "hard_stop_step": None,
    "ell_init": "median_pairwise_dist",
}
PPCA_ICM_PRESETS = {
    "multi-partial": {
        "latent_dim": 150,
        "kernel": "matern52",
        "kernel_mode": "shared",
        "ppca_iters": 50,
        "gp_steps": 4000,
        "gp_lr": 1.0e-3,
        "n_samples": 64,
        "hard_stop_step": 1340,
        "ell_init": 4.0,
    },
    "multi-complete": {
        "latent_dim": 150,
        "kernel": "matern52",
        "kernel_mode": "shared",
        "ppca_iters": 50,
        "gp_steps": 4000,
        "gp_lr": 1.0e-3,
        "n_samples": 64,
        "hard_stop_step": 1340,
        "ell_init": 4.0,
    },
    "single-complete": {
        "latent_dim": 150,
        "kernel": "matern52",
        "kernel_mode": "shared",
        "ppca_iters": 50,
        "gp_steps": 4000,
        "gp_lr": 1.0e-3,
        "n_samples": 64,
        "hard_stop_step": 1940,
        "ell_init": 4.0,
    },
}
PPCA_ICM_LINEAR_TREND_CFG = {"enabled": True, "lambda": 1.0e-3, "design": {"intercept": True, "inputs": True, "sim_onehot": False}}

GPLFR_ARG_DEFAULTS = {
    "latent_dim": 150,
    "num_training_steps": 2000,
    "inverse_temperature": 0.1,
    "latent_nugget": 0.1,
    "lr_Z": 0.1,
    "lr_global": 0.3,
    "variable_weights": "learned_per_group",
    "output_coregionalization": "field_coregionalized",
}


def _json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_root(path: Path) -> Path:
    parts = path.resolve().parts
    for i in range(len(parts) - 1):
        if parts[i : i + 2] == ("results", "models"):
            return Path(*parts[:i])
    return TW_ROOT


def _rooted_path(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _config_path_value(path: Path, root: Path = TW_ROOT) -> str:
    path = _rooted_path(path, root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _coerce_ell_init(value: int | float | str | None) -> int | float | str | None:
    if not isinstance(value, str):
        return value
    try:
        return float(value)
    except ValueError:
        return value


def _pca_mlp_hparams(args: argparse.Namespace, cfg: dict | None = None) -> dict[str, int | float | None]:
    if cfg is not None and "pca_mlp" in cfg:
        return {**PCA_MLP_ARG_DEFAULTS, **cfg["pca_mlp"]}
    values = {key: getattr(args, key) for key in PCA_MLP_ARG_DEFAULTS}
    if values == PCA_MLP_ARG_DEFAULTS and args.subset in PCA_MLP_PRESETS:
        return dict(PCA_MLP_PRESETS[args.subset])
    return values


def _pca_ridge_hparams(args: argparse.Namespace, cfg: dict | None = None) -> dict:
    explicit = set(getattr(args, "_explicit_args", set()))
    cfg = cfg or {}
    method_cfg = cfg.get("pca_ridge") or {}
    cv_cfg = cfg.get("CV_sweep") or {}
    best_cfg = cfg.get("best") or {}
    values = dict(PCA_RIDGE_ARG_DEFAULTS)
    values.update(
        {
            "latent_dim": int(method_cfg.get("latent_dim", values["latent_dim"])),
            "lambda_reg": float(method_cfg.get("lambda_reg", values["lambda_reg"])),
            "ppca_iters": int(method_cfg.get("ppca_iters", values["ppca_iters"])),
            "n_folds": int(cv_cfg.get("n_folds", values["n_folds"])),
            "cv_latent_dim": _csv(cv_cfg.get("latent_dim", values["cv_latent_dim"])),
            "cv_lambda": _csv(cv_cfg.get("lambda_reg", values["cv_lambda"])),
        }
    )
    for key in PCA_RIDGE_ARG_DEFAULTS:
        if key in explicit:
            values[key] = getattr(args, key)
    if not cfg:
        if "latent_dim" in explicit or getattr(args, "latent_dim") != 60:
            values["latent_dim"] = getattr(args, "latent_dim")
        if "ppca_iters" in explicit or getattr(args, "ppca_iters") != 50:
            values["ppca_iters"] = getattr(args, "ppca_iters")
        if "n_folds" in explicit or getattr(args, "n_folds") != 5:
            values["n_folds"] = getattr(args, "n_folds")
        for key in ("lambda_reg", "cv_latent_dim", "cv_lambda"):
            if key in explicit or getattr(args, key) != PCA_RIDGE_ARG_DEFAULTS[key]:
                values[key] = getattr(args, key)

    sweep_override = bool(explicit & {"latent_dim", "lambda_reg", "n_folds", "cv_latent_dim", "cv_lambda"})
    hidden_best = getattr(args, "best_latent_dim", None) is not None and getattr(args, "best_lambda_reg", None) is not None
    use_best = bool(best_cfg) and (not sweep_override or hidden_best)
    if hidden_best:
        values["best_latent_dim"] = int(args.best_latent_dim)
        values["best_lambda_reg"] = float(args.best_lambda_reg)
        values["best_cv_equal_group_normalized_rmse"] = getattr(args, "best_cv_equal_group_normalized_rmse", None)
        values["cv_sweep_scores"] = getattr(args, "cv_sweep_scores", None)
    elif use_best:
        values["best_latent_dim"] = int(best_cfg["latent_dim"])
        values["best_lambda_reg"] = float(best_cfg["lambda_reg"])
        values["best_cv_equal_group_normalized_rmse"] = best_cfg.get("cv_equal_group_normalized_rmse")
        values["cv_sweep_scores"] = cv_cfg.get("scores")
    else:
        values["best_latent_dim"] = None
        values["best_lambda_reg"] = None
        values["best_cv_equal_group_normalized_rmse"] = None
        values["cv_sweep_scores"] = None
    return values


def _ppca_icm_hparams(args: argparse.Namespace, cfg: dict | None = None) -> dict[str, int | float | str | None]:
    if cfg is not None and "ppca_icm" in cfg:
        values = {**PPCA_ICM_ARG_DEFAULTS, **cfg["ppca_icm"]}
        values["ell_init"] = _coerce_ell_init(values.get("ell_init"))
        return values
    use_preset = args.ppca_icm_preset == "tuned" and args.subset in PPCA_ICM_PRESETS
    values = dict(PPCA_ICM_PRESETS[args.subset]) if use_preset else {key: getattr(args, key) for key in PPCA_ICM_ARG_DEFAULTS}
    if use_preset:
        for key in set(getattr(args, "_explicit_args", set())) & set(PPCA_ICM_ARG_DEFAULTS):
            values[key] = getattr(args, key)
        values["ell_init"] = _coerce_ell_init(values.get("ell_init"))
        return values
    values["ell_init"] = _coerce_ell_init(values.get("ell_init"))
    return values


def _gplfr_hparams(args: argparse.Namespace, cfg: dict | None = None) -> dict[str, int | float | str | None]:
    if cfg is not None and "gplfr" in cfg:
        values = dict(GPLFR_ARG_DEFAULTS)
        values.update({k: cfg["gplfr"][k] for k in ("latent_dim", "num_training_steps", "inverse_temperature", "latent_nugget", "variable_weights", "output_coregionalization") if k in cfg["gplfr"]})
        values.update((cfg["gplfr"].get("optimizer") or {}))
        return values
    values = dict(GPLFR_ARG_DEFAULTS)
    explicit = set(getattr(args, "_explicit_args", set()))
    for key in ("latent_dim", "gplfr_num_training_steps", "gplfr_inverse_temperature", "gplfr_latent_nugget", "gplfr_variable_weights", "gplfr_output_coregionalization"):
        if key in explicit:
            values[key[6:] if key.startswith("gplfr_") else key] = getattr(args, key)
    return values


def _coord_mlp_hparams(args: argparse.Namespace, cfg: dict | None = None) -> dict[str, int | float | str | None]:
    if cfg is not None and "coord_mlp" in cfg:
        values = dict(COORD_MLP_ARG_DEFAULTS)
        values.update(
            {
                f"coord_{k}" if not str(k).startswith("coord_") else str(k): v
                for k, v in cfg["coord_mlp"].items()
                if k != "preset"
            }
        )
        return values
    values = {key: getattr(args, key) for key in COORD_MLP_ARG_DEFAULTS}
    if values == COORD_MLP_ARG_DEFAULTS and args.subset in COORD_MLP_PRESETS:
        return _coord_mlp_hparams(args, {"coord_mlp": COORD_MLP_PRESETS[args.subset]})
    return values


def _coord_deeponet_hparams(args: argparse.Namespace, cfg: dict | None = None) -> dict[str, int | float | str | None]:
    if cfg is not None and "coord_deeponet" in cfg:
        values = dict(COORD_DEEPONET_ARG_DEFAULTS)
        values.update(
            {
                f"deeponet_{k}" if not str(k).startswith("deeponet_") else str(k): v
                for k, v in cfg["coord_deeponet"].items()
                if k != "preset"
            }
        )
        return values
    values = {key: getattr(args, key) for key in COORD_DEEPONET_ARG_DEFAULTS}
    if values == COORD_DEEPONET_ARG_DEFAULTS and args.subset in COORD_DEEPONET_PRESETS:
        return _coord_deeponet_hparams(args, {"coord_deeponet": COORD_DEEPONET_PRESETS[args.subset]})
    return values


def _merge_config_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[argparse.Namespace, dict | None]:
    if args.config is None:
        return args, None
    cfg = _json_load(args.config)
    config_root = _artifact_root(args.config.parent)
    defaults = {a.dest: a.default for a in parser._actions if a.dest not in ("help", "config")}
    for key, default in defaults.items():
        if hasattr(args, key) and getattr(args, key) == default and key in cfg:
            value = cfg[key]
            if key in {"data_dir", "out_dir"} and value is not None:
                value = _rooted_path(value, config_root)
            setattr(args, key, value)
    method_cfg = cfg.get(str(args.method)) if args.method is not None else None
    if isinstance(method_cfg, dict) and args.method not in {"coord_mlp", "coord_deeponet", "pca_ridge"}:
        for key, default in defaults.items():
            if hasattr(args, key) and getattr(args, key) == default and key in method_cfg:
                setattr(args, key, method_cfg[key])
    if args.method == "coord_mlp" and isinstance(method_cfg, dict):
        for key in COORD_MLP_ARG_DEFAULTS:
            bare = key.removeprefix("coord_")
            if getattr(args, key) == defaults[key] and bare in method_cfg:
                setattr(args, key, method_cfg[bare])
    if args.method == "coord_deeponet" and isinstance(method_cfg, dict):
        for key in COORD_DEEPONET_ARG_DEFAULTS:
            bare = key.removeprefix("deeponet_")
            if getattr(args, key) == defaults[key] and bare in method_cfg:
                setattr(args, key, method_cfg[bare])
    if args.method == "pca_ridge":
        cv_sweep = cfg.get("CV_sweep") or {}
        explicit = set(getattr(args, "_explicit_args", set()))
        if isinstance(method_cfg, dict):
            for key in ("latent_dim", "lambda_reg", "ppca_iters"):
                if key not in explicit and key in method_cfg:
                    setattr(args, key, method_cfg[key])
        if "n_folds" not in explicit and "n_folds" in cv_sweep:
            args.n_folds = int(cv_sweep["n_folds"])
        if "cv_latent_dim" not in explicit and "latent_dim" in cv_sweep:
            args.cv_latent_dim = _csv(cv_sweep["latent_dim"])
        if "cv_lambda" not in explicit and "lambda_reg" in cv_sweep:
            args.cv_lambda = _csv(cv_sweep["lambda_reg"])
        best = cfg.get("best") or {}
        sweep_override = bool(explicit & {"latent_dim", "lambda_reg", "n_folds", "cv_latent_dim", "cv_lambda"})
        hidden_best = getattr(args, "best_latent_dim", None) is not None and getattr(args, "best_lambda_reg", None) is not None
        if best and (not sweep_override or hidden_best):
            if getattr(args, "best_latent_dim", None) is None and "latent_dim" in best:
                args.best_latent_dim = int(best["latent_dim"])
            if getattr(args, "best_lambda_reg", None) is None and "lambda_reg" in best:
                args.best_lambda_reg = float(best["lambda_reg"])
            args.best_cv_equal_group_normalized_rmse = best.get("cv_equal_group_normalized_rmse")
            args.cv_sweep_scores = cv_sweep.get("scores")
    if args.method == "knn":
        cv_sweep = cfg.get("CV_sweep") or {}
        if getattr(args, "n_folds") == defaults["n_folds"] and "n_folds" in cv_sweep:
            args.n_folds = int(cv_sweep["n_folds"])
        if getattr(args, "k") == defaults["k"] and "k" in cv_sweep:
            args.k = _csv(cv_sweep["k"])
        if getattr(args, "gcm_penalty") == defaults["gcm_penalty"] and "gcm_penalty" in cv_sweep:
            args.gcm_penalty = _csv(cv_sweep["gcm_penalty"])
        best = cfg.get("best") or {}
        if getattr(args, "best_k", None) is None and "k" in best:
            args.best_k = int(best["k"])
        if getattr(args, "best_gcm_penalty", None) is None and "gcm_penalty" in best:
            args.best_gcm_penalty = float(best["gcm_penalty"])
        args.best_cv_equal_group_normalized_rmse = best.get("cv_equal_group_normalized_rmse")
        args.cv_sweep_scores = cv_sweep.get("scores")
    return args, cfg


def _resolved_config(args: argparse.Namespace, *, out_dir: Path, data_dir: Path, meta: dict | None = None) -> dict:
    artifact_root = _artifact_root(out_dir)
    cfg = {
        "method": args.method,
        "subset": args.subset,
        "data_dir": _config_path_value(data_dir, artifact_root),
        "out_dir": _config_path_value(out_dir, artifact_root),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "device": str(args.device),
    }
    if args.method == "knn":
        cv_sweep = {
            "n_folds": int(args.n_folds),
            "k": _parse_int_list(str(args.k)),
            "gcm_penalty": _parse_float_list(str(args.gcm_penalty)),
            "objective": "equal_group_normalized_rmse",
        }
        sweep_scores = (meta or {}).get("cv_sweep") or getattr(args, "cv_sweep_scores", None)
        if sweep_scores:
            cv_sweep["scores"] = sweep_scores
        cfg["CV_sweep"] = cv_sweep
        best_k = (meta or {}).get("best_k")
        best_penalty = (meta or {}).get("best_gcm_penalty")
        cfg["best"] = {
            "k": int(args.best_k if best_k is None else best_k),
            "gcm_penalty": float(args.best_gcm_penalty if best_penalty is None else best_penalty),
            "cv_equal_group_normalized_rmse": (meta or {}).get(
                "cv_equal_group_normalized_rmse",
                getattr(args, "best_cv_equal_group_normalized_rmse", None),
            ),
        }
    elif args.method == "pca_mlp":
        hparams = _pca_mlp_hparams(args)
        cfg["pca_mlp"] = {
            "latent_dim": int(hparams["latent_dim"]),
            "hidden_width": int(hparams["hidden_width"]),
            "activation": str(args.activation),
            "ppca_iters": int(args.ppca_iters),
            "num_steps": int(hparams["num_steps"]),
            "lr": float(hparams["lr"]),
            "weight_decay": float(hparams["weight_decay"]),
            "hard_stop_step": None if hparams["hard_stop_step"] is None else int(hparams["hard_stop_step"]),
            "linear_trend_cfg": PCA_MLP_LINEAR_TREND_CFG,
            "preset": args.subset if {k: getattr(args, k) for k in PCA_MLP_ARG_DEFAULTS} == PCA_MLP_ARG_DEFAULTS and args.subset in PCA_MLP_PRESETS else None,
        }
    elif args.method == "pca_ridge":
        hparams = _pca_ridge_hparams(args)
        cv_scores = (meta or {}).get("cv_sweep_scores", hparams.get("cv_sweep_scores"))
        cv_metric = (meta or {}).get("cv_equal_group_normalized_rmse", hparams.get("best_cv_equal_group_normalized_rmse"))
        latent_dim = int((meta or {}).get("best_latent_dim", hparams.get("best_latent_dim") or hparams["latent_dim"]))
        lambda_reg = float((meta or {}).get("best_lambda_reg", hparams.get("best_lambda_reg") or hparams["lambda_reg"]))
        cfg["pca_ridge"] = {
            "latent_dim": latent_dim,
            "lambda_reg": lambda_reg,
            "ppca_iters": int(hparams["ppca_iters"]),
            "design": PCA_RIDGE_DESIGN_CFG,
            "linear_trend_cfg": PCA_RIDGE_LINEAR_TREND_CFG,
        }
        cfg["CV_sweep"] = {
            "n_folds": int(hparams["n_folds"]),
            "latent_dim": _parse_int_list(str(hparams["cv_latent_dim"])),
            "lambda_reg": _parse_float_list(str(hparams["cv_lambda"])),
            "objective": "equal_group_normalized_rmse",
            "scores": cv_scores,
        }
        cfg["best"] = {
            "latent_dim": latent_dim,
            "lambda_reg": lambda_reg,
            "cv_equal_group_normalized_rmse": cv_metric,
        }
    elif args.method == "coord_mlp":
        hparams = _coord_mlp_hparams(args)
        cfg["coord_mlp"] = {
            "hidden_width": int(hparams["coord_hidden_width"]),
            "num_layers": int(hparams["coord_num_layers"]),
            "activation": str(hparams["coord_activation"]),
            "batch_size": int(hparams["coord_batch_size"]),
            "predict_chunk_size": int(hparams["coord_predict_chunk_size"]),
            "num_steps": int(hparams["coord_num_steps"]),
            "lr": float(hparams["coord_lr"]),
            "weight_decay": float(hparams["coord_weight_decay"]),
            "hard_stop_step": None if hparams["coord_hard_stop_step"] is None else int(hparams["coord_hard_stop_step"]),
            "preset": args.subset if {k: getattr(args, k) for k in COORD_MLP_ARG_DEFAULTS} == COORD_MLP_ARG_DEFAULTS and args.subset in COORD_MLP_PRESETS else None,
        }
    elif args.method == "coord_deeponet":
        hparams = _coord_deeponet_hparams(args)
        cfg["coord_deeponet"] = {
            "rank": int(hparams["deeponet_rank"]),
            "branch_hidden_width": int(hparams["deeponet_branch_hidden_width"]),
            "trunk_hidden_width": int(hparams["deeponet_trunk_hidden_width"]),
            "branch_num_layers": int(hparams["deeponet_branch_num_layers"]),
            "trunk_num_layers": int(hparams["deeponet_trunk_num_layers"]),
            "activation": str(hparams["deeponet_activation"]),
            "batch_size": int(hparams["deeponet_batch_size"]),
            "predict_chunk_size": int(hparams["deeponet_predict_chunk_size"]),
            "num_steps": int(hparams["deeponet_num_steps"]),
            "lr": float(hparams["deeponet_lr"]),
            "weight_decay": float(hparams["deeponet_weight_decay"]),
            "hard_stop_step": None if hparams["deeponet_hard_stop_step"] is None else int(hparams["deeponet_hard_stop_step"]),
            "optimizer": "AdamW",
            "target_normalization": "per_field_training_grid_latitude_weighted",
            "coordinate_encoding": "t21_lat_mu_lon_sincos",
            "sampling": "base_variable_first_area_weighted_latitude",
            "cv_objective": "area_weighted_equal_base_variable_normalized_rmse_grid",
            "equatorial_symmetry": True,
            "preset": args.subset if {k: getattr(args, k) for k in COORD_DEEPONET_ARG_DEFAULTS} == COORD_DEEPONET_ARG_DEFAULTS and args.subset in COORD_DEEPONET_PRESETS else None,
        }
    elif args.method == "ppca_icm":
        hparams = _ppca_icm_hparams(args)
        cfg["ppca_icm"] = {
            "latent_dim": int(hparams["latent_dim"]),
            "kernel": str(hparams["kernel"]),
            "kernel_mode": str(hparams["kernel_mode"]),
            "ppca_iters": int(hparams["ppca_iters"]),
            "gp_steps": int(hparams["gp_steps"]),
            "gp_lr": float(hparams["gp_lr"]),
            "n_samples": int(hparams["n_samples"]),
            "hard_stop_step": None if hparams["hard_stop_step"] is None else int(hparams["hard_stop_step"]),
            "ell_init": hparams["ell_init"],
            "linear_trend_cfg": PPCA_ICM_LINEAR_TREND_CFG,
        }
    elif args.method == "gplfr":
        hparams = _gplfr_hparams(args, getattr(args, "_config", None))
        cfg["gplfr"] = {
            "latent_dim": int(hparams["latent_dim"]),
            "num_training_steps": int(hparams["num_training_steps"]),
            "inverse_temperature": float(hparams["inverse_temperature"]),
            "latent_nugget": float(hparams["latent_nugget"]),
            "variable_weights": str(hparams["variable_weights"]),
            "output_coregionalization": str(hparams["output_coregionalization"]),
            "optimizer": {
                "lr_Z": float(hparams["lr_Z"]),
                "lr_global": float(hparams["lr_global"]),
            },
        }
    return cfg
