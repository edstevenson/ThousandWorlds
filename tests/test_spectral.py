import numpy as np
import pytest

from thousandworlds import build_equatorial_symmetry_mask, load_symmetry_masks, to_grid


def test_to_grid_shape():
    coeffs = np.random.default_rng(0).normal(size=(7, 484)).astype(np.float32)
    out = to_grid(coeffs)
    assert out.shape == (7, 32, 64)
    assert out.dtype == np.float32


@pytest.mark.requires_dataset
def test_symmetry_masks_have_expected_counts(data_dir):
    masks = load_symmetry_masks(data_dir / "norm_stats" / "multi-partial", ["surface_temperature", "v_0"])
    assert int(masks["surface_temperature"].sum()) == 253
    assert int(masks["v_0"].sum()) == 231


def test_build_equatorial_symmetry_mask_matches_expected_parity():
    mask = build_equatorial_symmetry_mask(l_max=2, m_max=3, mode="symmetric")
    assert mask.dtype == bool
    assert mask.tolist() == [True, False, True, True, True, False, False, True, True]
    assert np.array_equal(mask, ~build_equatorial_symmetry_mask(l_max=2, m_max=3, mode="antisymmetric"))
    assert build_equatorial_symmetry_mask(l_max=2, m_max=3, mode="none").all()
