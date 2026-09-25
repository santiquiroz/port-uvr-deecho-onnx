"""Input contract of DeEchoDriver.separate: [2, N] / [1, N] / [N] floating point.

soundfile.read returns [N, C], so the channels-last mistake is the likeliest
integration bug; before this check it silently produced a (2, 0) output.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.pipeline import DeEchoDriver, as_stereo_float32

N = 30000


def fake_graph(window: np.ndarray) -> np.ndarray:
    return np.clip(window * 3.0, 0.0, 1.0).astype(np.float32)


def noise(shape, dtype=np.float32) -> np.ndarray:
    return (np.random.default_rng(7).standard_normal(shape) * 0.2).astype(dtype)


def separate(mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return DeEchoDriver(fake_graph).separate(mix)


def test_channels_last_raises_with_transpose_hint():
    with pytest.raises(ValueError, match=r"mix\.T"):
        separate(noise((44100, 2)))


def test_mono_row_matches_flat_mono():
    flat = noise(N)
    row_primary, row_secondary = separate(flat[None, :])
    flat_primary, flat_secondary = separate(flat)
    assert np.array_equal(row_primary, flat_primary)
    assert np.array_equal(row_secondary, flat_secondary)


def test_more_than_two_channels_raises():
    with pytest.raises(ValueError, match="channels"):
        separate(noise((6, N)))


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.uint8])
def test_integer_dtype_raises(dtype):
    with pytest.raises(ValueError, match="float"):
        separate(np.zeros((2, N), dtype=dtype))


@pytest.mark.parametrize("ndim_shape", [(), (2, 2, N)])
def test_wrong_rank_raises(ndim_shape):
    with pytest.raises(ValueError, match=r"1-D \[N\] or 2-D"):
        separate(np.zeros(ndim_shape, dtype=np.float32))


def test_stereo_float32_passes_through_untouched():
    mix = noise((2, N))
    assert as_stereo_float32(mix) is mix


def test_float64_is_cast_to_float32():
    mix = noise((2, N), dtype=np.float64)
    out = as_stereo_float32(mix)
    assert out.dtype == np.float32 and np.array_equal(out, mix.astype(np.float32))


def test_mono_is_duplicated_to_stereo():
    mono = noise(N)
    out = as_stereo_float32(mono)
    assert out.shape == (2, N)
    assert np.array_equal(out[0], mono) and np.array_equal(out[1], mono)


def test_input_not_mutated():
    mix = noise((1, N), dtype=np.float64)
    original = mix.copy()
    separate(mix)
    assert np.array_equal(mix, original)
