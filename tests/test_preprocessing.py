import numpy as np
import pytest

from thousandworlds import (
    inverse_preprocess_outputs_grid,
    inverse_transform_inputs,
    load_stats,
    normalise_spectral,
    preprocess_outputs_grid,
    transform_inputs,
    unnormalise_spectral,
)


pytestmark = pytest.mark.requires_dataset


def test_round_trip_inputs(data_dir):
    stats = load_stats("multi-complete", data_dir)
    X = np.array([[3000.0, 900.0, 7.1e6, 9.8, 10.0, 1.0e5, 1.0e-4, 1.0e-7]], dtype=np.float32)
    X_t = transform_inputs(X, stats)
    np.testing.assert_allclose(inverse_transform_inputs(X_t, stats), X, rtol=1.0e-6, atol=1.0e-6)


def test_round_trip_outputs(data_dir):
    stats = load_stats("multi-complete", data_dir)
    field_names = ["specific_humidity_0", "cloud_fraction_0", "asr_cloudy", "surface_temperature"]
    Y = np.stack(
        [
            np.full((32, 64), 1.0e-5, dtype=np.float32),
            np.full((32, 64), 0.2, dtype=np.float32),
            np.full((32, 64), 200.0, dtype=np.float32),
            np.full((32, 64), 280.0, dtype=np.float32),
        ],
        axis=0,
    )[None]
    X = np.array([[3000.0, 900.0, 7.1e6, 9.8, 10.0, 1.0e5, 1.0e-4, 1.0e-7]], dtype=np.float32)
    Y_pp = preprocess_outputs_grid(Y, field_names, stats, X=X)
    np.testing.assert_allclose(inverse_preprocess_outputs_grid(Y_pp, field_names, stats, X=X), Y, rtol=1.0e-6, atol=1.0e-6)


def test_log_preprocess_clips_zero_specific_humidity_to_finite(data_dir):
    stats = load_stats("multi-partial", data_dir)
    Y = np.zeros((1, 1, 32, 64), dtype=np.float32)
    Y_pp = preprocess_outputs_grid(Y, ["specific_humidity_0"], stats)
    assert np.isfinite(Y_pp).all()
    assert (inverse_preprocess_outputs_grid(Y_pp, ["specific_humidity_0"], stats) >= 0.0).all()


def test_spectral_round_trip(data_dir):
    stats = load_stats("multi-complete", data_dir)
    coeffs = np.zeros((2, 3, 484), dtype=np.float32)
    for j, name in enumerate(stats.field_names[:3]):
        mask = stats.spectral[name].mask
        coeffs[:, j, mask] = np.random.default_rng(0).normal(size=(2, mask.sum())).astype(np.float32)
    coeffs_n = normalise_spectral(coeffs, stats.field_names[:3], stats)
    coeffs_rt = unnormalise_spectral(coeffs_n, stats.field_names[:3], stats)
    np.testing.assert_allclose(coeffs_rt, coeffs, rtol=1.0e-4, atol=1.0e-4)
