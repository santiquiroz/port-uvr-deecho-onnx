"""separate() must stay numerically equal to the frozen pre/post path (frozen_prepost.py)
while dsp/multiband/pipeline are reworked for speed and memory."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import frozen_prepost
from driver import dsp
from driver.pipeline import DeEchoDriver

TOLERANCE = 1e-5
LENGTHS = [480, 44100, 10 * 44100 + 123]
BAND_CONFIGS = [(640, 80), (320, 80), (512, 160), (960, 480)]


def sigmoid_graph(window: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(2.0 - 20.0 * window))).astype(np.float32)


def noise(n: int) -> np.ndarray:
    return (np.random.default_rng(n).standard_normal((2, n)) * 0.3).astype(np.float32)


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    assert a.shape == b.shape
    return float(np.max(np.abs(a - b), initial=0.0))


@pytest.mark.parametrize("n_fft,hop", BAND_CONFIGS)
@pytest.mark.parametrize("n_frames", [1, 2, 7, 500])
def test_istft_matches_frozen_loop(n_fft, hop, n_frames):
    rng = np.random.default_rng(n_frames)
    shape = (n_fft // 2 + 1, n_frames)
    spec = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
    assert max_abs_diff(dsp.istft(spec, hop), frozen_prepost.istft(spec, hop)) < TOLERANCE


@pytest.mark.parametrize("is_non_accom_stem", [False, True])
@pytest.mark.parametrize("aggression", [0.0, 5.0])
@pytest.mark.parametrize("n", LENGTHS)
def test_separate_matches_frozen(n, aggression, is_non_accom_stem):
    mix = noise(n)
    driver = DeEchoDriver(sigmoid_graph, aggression=aggression, is_non_accom_stem=is_non_accom_stem)
    primary, secondary = driver.separate(mix)
    ref_primary, ref_secondary = frozen_prepost.separate(mix, sigmoid_graph, aggression, is_non_accom_stem)
    assert max_abs_diff(primary, ref_primary) < TOLERANCE
    assert max_abs_diff(secondary, ref_secondary) < TOLERANCE


@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_separate_silence_matches_frozen():
    mix = np.zeros((2, 44100), dtype=np.float32)
    primary, secondary = DeEchoDriver(sigmoid_graph).separate(mix)
    ref_primary, ref_secondary = frozen_prepost.separate(mix, sigmoid_graph)
    assert np.array_equal(primary, ref_primary)
    assert np.array_equal(secondary, ref_secondary)
