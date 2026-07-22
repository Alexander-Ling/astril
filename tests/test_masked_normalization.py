import numpy as np

from astril.preprocessing_utils import _normalize_masked_frame


def test_robust_normalization_transforms_background_and_limits_outlier_consistently():
    data = np.array([0.0] + [100.0] * 998 + [10_000.0], dtype=np.float32)
    mask = np.zeros(data.shape, dtype=bool)
    mask[1:] = True

    out = _normalize_masked_frame(data, mask, percentiles=(0.1, 99.9), z_clip=5.0)

    # The foreground-derived percentile bounds apply to background too.
    lo, hi = np.percentile(data[mask], (0.1, 99.9))
    foreground = np.clip(data[mask], lo, hi)
    location, scale = foreground.mean(), foreground.std()
    expected_background = np.clip((np.clip(0.0, lo, hi) - location) / scale, -5.0, 5.0)
    np.testing.assert_allclose(out[0], expected_background, rtol=1e-6)
    assert np.max(np.abs(out[mask])) <= 5.0


def test_robust_normalization_accepts_empty_constant_and_nonfinite_masks():
    empty = _normalize_masked_frame(
        np.array([0.0, np.nan, np.inf], dtype=np.float32),
        np.array([False, True, True]),
    )
    np.testing.assert_array_equal(empty, np.zeros(3, dtype=np.float32))

    constant = _normalize_masked_frame(
        np.array([0.0, 7.0, 7.0], dtype=np.float32),
        np.array([False, True, True]),
    )
    np.testing.assert_array_equal(constant, np.zeros(3, dtype=np.float32))

    near_constant = _normalize_masked_frame(
        np.array([0.0, 7.0, 7.0 + 1e-7], dtype=np.float32),
        np.array([False, True, True]),
    )
    np.testing.assert_allclose(near_constant, np.zeros(3, dtype=np.float32))
