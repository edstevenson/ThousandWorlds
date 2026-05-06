from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np

import thousandworlds.evaluate as evaluate
from thousandworlds.evaluate import _metric_space, acc, energy_score, relative_acc, relative_rmse, rmse, score, spread_skill_ratio


def test_rmse_zero_on_identity():
    target = np.ones((3, 5, 32, 64), dtype=np.float32)
    field_mask = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=bool)
    result = rmse(target, target, field_mask=field_mask, field_names=[f"temperature_{i}" for i in range(5)])
    assert "aggregate" not in result
    np.testing.assert_allclose(result["per_variable"]["temperature"], 0.0, atol=1.0e-6)


def test_acc_identifies_matching_and_opposite_patterns():
    lat_pattern = np.linspace(-1.0, 1.0, 32, dtype=np.float32)[None, None, :, None]
    target = np.broadcast_to(lat_pattern, (1, 2, 32, 64)).copy()
    pred = target.copy()
    pred[:, 1] *= -1.0
    result = acc(pred, target, field_mask=np.ones((1, 2), dtype=bool), field_names=["temperature_0", "temperature_1"])
    np.testing.assert_allclose(result["per_field"]["temperature_0"], 1.0, atol=1.0e-6)
    np.testing.assert_allclose(result["per_field"]["temperature_1"], -1.0, atol=1.0e-6)
    np.testing.assert_allclose(result["per_variable"]["temperature"], 0.0, atol=1.0e-6)


def test_energy_score_zero_for_identical_single_sample():
    target = np.ones((3, 5, 32, 64), dtype=np.float32)
    samples = target[None]
    result = energy_score(samples, target, field_mask=np.ones((3, 5), dtype=bool))
    np.testing.assert_allclose(np.asarray(result["per_field"]), 0.0, atol=1.0e-6)


def test_spread_skill_ratio_averages_field_ssrs_into_variables():
    target = np.ones((1, 2, 32, 64), dtype=np.float32)
    samples = np.stack([-target, target], axis=0)
    samples[:, :, 1] *= 2.0
    result = spread_skill_ratio(samples, target, field_mask=np.ones((1, 2), dtype=bool), field_names=["temperature_0", "temperature_1"])
    np.testing.assert_allclose(result["per_field"]["temperature_0"], np.sqrt(2.0), rtol=1.0e-5)
    np.testing.assert_allclose(result["per_field"]["temperature_1"], np.sqrt(8.0), rtol=1.0e-5)
    np.testing.assert_allclose(result["per_variable"]["temperature"], (np.sqrt(2.0) + np.sqrt(8.0)) / 2.0, rtol=1.0e-5)


def test_relative_rmse_ignores_zero_denominator_fields():
    pred = np.zeros((1, 2, 32, 64), dtype=np.float32)
    target_a = np.zeros_like(pred)
    target_b = np.zeros_like(pred)
    target_a[0, 0] = 1.0
    target_b[0, 0] = 3.0
    field_mask = np.array([[True, True]])
    result = relative_rmse(pred, target_a, target_b, field_mask=field_mask, field_names=["a", "b"])
    assert "aggregate" not in result
    assert np.isfinite(result["per_field"]["a"])
    assert np.isnan(result["per_field"]["b"])


def test_relative_acc_uses_acc_ratio():
    target_a = np.broadcast_to(np.linspace(-1.0, 1.0, 32, dtype=np.float32)[None, None, :, None], (1, 1, 32, 64)).copy()
    target_b = 2.0 * target_a
    pred = -target_a
    result = relative_acc(pred, target_a, target_b, field_mask=np.ones((1, 1), dtype=bool), field_names=["temperature_0"])
    np.testing.assert_allclose(result["per_field"]["temperature_0"], -1.0, atol=1.0e-6)
    np.testing.assert_allclose(result["per_variable"]["temperature"], -1.0, atol=1.0e-6)


def test_relative_rmse_per_variable_uses_ratio_of_grouped_means():
    pred = np.zeros((1, 2, 32, 64), dtype=np.float32)
    target_a = np.zeros_like(pred)
    target_b = np.zeros_like(pred)
    target_a[0, 0] = 1.0
    target_b[0, 0] = 3.0
    target_a[0, 1] = 1.0
    target_b[0, 1] = 2.0
    result = relative_rmse(pred, target_a, target_b, field_mask=np.ones((1, 2), dtype=bool), field_names=["temperature_0", "temperature_1"])
    np.testing.assert_allclose(result["per_field"]["temperature_0"], 0.5, atol=1.0e-6)
    np.testing.assert_allclose(result["per_field"]["temperature_1"], 1.0, atol=1.0e-6)
    np.testing.assert_allclose(result["per_variable"]["temperature"], 2.0 / 3.0, atol=1.0e-6)


def test_score_uses_dex_for_specific_humidity_energy_score(tmp_path, monkeypatch):
    field_names = ["specific_humidity_0"]
    target = np.full((1, 1, 32, 64), 1.0e-4, dtype=np.float32)
    predictions = np.stack(
        [
            np.full((1, 1, 32, 64), 1.0e-5, dtype=np.float32),
            np.full((1, 1, 32, 64), 1.0e-3, dtype=np.float32),
        ],
        axis=0,
    )
    bundle = SimpleNamespace(
        test_ids=np.array([7], dtype=np.int32),
        field_names=field_names,
        raw_field_names=field_names,
        Y_test=target,
        field_mask_test=np.ones((1, 1), dtype=bool),
        meta_test=None,
    )
    monkeypatch.setattr(evaluate, "load_bundle", lambda *args, **kwargs: bundle)
    pred_path = Path(tmp_path) / "predictions.npz"
    np.savez_compressed(pred_path, predictions=predictions, simulation_id=bundle.test_ids, field_names=np.asarray(field_names))
    result = score(pred_path, data_dir=tmp_path, subset="dummy", protocol="standard")
    det_result = score(pred_path, data_dir=tmp_path, subset="dummy", protocol="standard", include_probabilistic=False)
    expected = energy_score(
        _metric_space(predictions, field_names, humidity="dex"),
        _metric_space(target[0], field_names, humidity="dex"),
        bundle.field_mask_test,
        field_names,
    )
    np.testing.assert_allclose(
        result["energy_score"]["per_variable"]["specific_humidity_dex"],
        expected["per_variable"]["specific_humidity_dex"],
        atol=1.0e-6,
    )
    assert "acc" in result
    assert "spread_skill_ratio" in result
    assert "energy_score" not in det_result
    assert "spread_skill_ratio" not in det_result


def test_evaluate_cli_include_probabilistic_flag(tmp_path, monkeypatch):
    seen = {}

    def fake_score(predictions_path, *, data_dir, subset, protocol, include_probabilistic, point_predictions_path=None):
        seen["include_probabilistic"] = include_probabilistic
        seen["point_predictions_path"] = point_predictions_path
        return {"predictions_path": str(predictions_path), "data_dir": str(data_dir), "subset": subset, "protocol": protocol}

    monkeypatch.setattr(evaluate, "score", fake_score)
    out = tmp_path / "metrics.json"
    evaluate.main(["predictions.npz", "--data-dir", str(tmp_path), "--subset", "dummy", "--include-probabilistic", "--out", str(out)])
    assert seen["include_probabilistic"] is True
    assert seen["point_predictions_path"] is None
    assert json.loads(out.read_text())["subset"] == "dummy"


def test_public_metric_aliases_hide_cloudy_suffix():
    target = np.ones((1, 2, 32, 64), dtype=np.float32)
    result = rmse(target, target, field_names=["asr_cloudy", "olr_cloudy"])
    assert "asr" in result["per_field"]
    assert "olr" in result["per_field"]
    assert "asr" in result["per_variable"]
    assert "olr" in result["per_variable"]
