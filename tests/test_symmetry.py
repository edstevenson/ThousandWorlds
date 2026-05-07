import numpy as np
import pytest

pytest.importorskip("torch")

from thousandworlds.models._common import enforce_equatorial_symmetry_grid


def test_equatorial_symmetry_grid_preserves_all_nan_missing_fields():
    Y = np.array(
        [[
            [[np.nan], [np.nan], [np.nan], [np.nan]],
            [[1.0], [2.0], [3.0], [4.0]],
        ]],
        dtype=np.float32,
    )

    out = enforce_equatorial_symmetry_grid(Y, ["temperature_0", "v_0"])

    assert np.isnan(out[0, 0]).all()
    np.testing.assert_allclose(out[0, 1, :, 0], [-1.5, -0.5, 0.5, 1.5])
