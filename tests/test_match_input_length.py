"""Output length of DeEchoDriver.separate.

By default the multiband iSTFT returns 480 * floor(N / 480) samples (the
reference's length, so parity holds); match_input_length=True returns exactly N.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.pipeline import DeEchoDriver

HOP = 480
LENGTHS = [1, 479, 480, 44100, 441000, 441479]


def fake_graph(window: np.ndarray) -> np.ndarray:
    return np.clip(window * 3.0, 0.0, 1.0).astype(np.float32)


def noise(n: int) -> np.ndarray:
    return (np.random.default_rng(n).standard_normal((2, n)) * 0.2).astype(np.float32)


def separate(mix: np.ndarray, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    return DeEchoDriver(fake_graph).separate(mix, **kwargs)


@pytest.mark.parametrize("n", LENGTHS)
def test_match_input_length_returns_n_samples(n):
    primary, secondary = separate(noise(n), match_input_length=True)
    assert primary.shape == (2, n)
    assert secondary.shape == (2, n)
    assert primary.dtype == np.float32 and secondary.dtype == np.float32


@pytest.mark.parametrize("n", LENGTHS)
def test_default_keeps_reference_length(n):
    primary, secondary = separate(noise(n))
    assert primary.shape == (2, HOP * (n // HOP))
    assert secondary.shape == (2, HOP * (n // HOP))


def test_default_is_explicit_false():
    mix = noise(44100)
    default = separate(mix)
    explicit = separate(mix, match_input_length=False)
    assert all(np.array_equal(a, b) for a, b in zip(default, explicit))


def test_match_input_length_mono_input():
    primary, _ = separate(noise(44100)[0], match_input_length=True)
    assert primary.shape == (2, 44100)


def test_match_input_length_agrees_with_default_away_from_tail():
    n = 441000
    mix = noise(n)
    default, _ = separate(mix)
    matched, _ = separate(mix, match_input_length=True)
    head = default.shape[1] - 4 * HOP
    assert np.max(np.abs(matched[:, :head] - default[:, :head])) < 1e-4
