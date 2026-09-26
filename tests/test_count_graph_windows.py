"""count_graph_windows(N) predicts how many run_graph calls separate() makes,
so a caller can report progress without re-deriving the driver's geometry."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver import multiband
from driver.pipeline import DeEchoDriver, count_graph_windows

LENGTHS = [1, 480, 44100, 441000, 441479, 44100 * 130]


class CountingGraph:
    def __init__(self):
        self.calls = 0

    def __call__(self, window: np.ndarray) -> np.ndarray:
        self.calls += 1
        return np.full_like(window, 0.5)


def noise(n: int) -> np.ndarray:
    return (np.random.default_rng(n).standard_normal((2, n)) * 0.2).astype(np.float32)


def graph_calls(n: int, **kwargs) -> int:
    graph = CountingGraph()
    DeEchoDriver(graph).separate(noise(n), **kwargs)
    return graph.calls


@pytest.mark.parametrize("n", LENGTHS)
def test_count_matches_separate_calls(n):
    assert count_graph_windows(n) == graph_calls(n)


def test_count_is_a_plain_int():
    assert type(count_graph_windows(44100)) is int


@pytest.mark.parametrize("n", [1, 441479])
def test_count_matches_match_input_length_calls(n):
    assert count_graph_windows(n, match_input_length=True) == graph_calls(n, match_input_length=True)


@pytest.mark.parametrize("n", [1, 2, 3, 159, 160, 479, 480, 481, 959, 960, 1439, 1440, 147455, 147456])
def test_combined_frame_count_matches_analysis(n):
    assert multiband.combined_frame_count(n) == multiband.wave_to_combined_spec(noise(n)).shape[2]


@pytest.mark.parametrize("n", [382 * 480, 383 * 480, 766 * 480, 767 * 480])
def test_count_matches_calls_at_window_boundaries(n):
    assert count_graph_windows(n) == graph_calls(n)
