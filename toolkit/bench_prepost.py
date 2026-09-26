"""CPU bench of the numpy pre/post chain around the graph (no ONNX, no GPU).

Usage: python toolkit/bench_prepost.py [--seconds 60] [--runs 5] [--driver-root DIR]

DeEchoDriver.separate() on seeded stereo noise with a cheap fake graph, so the
wall time is the driver's own analysis + masking + synthesis. Reports the min of
--runs timed runs (after one warmup) minus the time spent inside the fake graph,
and the tracemalloc peak of one extra run. --driver-root points at another
checkout's driver/ parent (e.g. `git archive a57dc91 driver | tar -x -C base`)
to compare revisions with the same script.
"""
from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

SR = 44100
MIB = 1024 * 1024


class TimedFakeGraph:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self, window: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        mask = np.clip(window * 3.0, 0.0, 1.0).astype(np.float32)
        self.seconds += time.perf_counter() - t0
        return mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--driver-root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def load_driver_class(driver_root: Path):
    sys.path.insert(0, str(driver_root.resolve()))
    from driver import pipeline

    print(f"driver: {pipeline.__file__}")
    return pipeline.DeEchoDriver


def noise_mix(seconds: float) -> np.ndarray:
    n_samples = int(seconds * SR)
    return (np.random.default_rng(0).standard_normal((2, n_samples)) * 0.3).astype(np.float32)


def prepost_seconds(driver_class, mix: np.ndarray) -> float:
    graph = TimedFakeGraph()
    t0 = time.perf_counter()
    driver_class(graph).separate(mix)
    return time.perf_counter() - t0 - graph.seconds


def peak_mib(driver_class, mix: np.ndarray) -> float:
    tracemalloc.start()
    driver_class(TimedFakeGraph()).separate(mix)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / MIB


def main() -> None:
    args = parse_args()
    driver_class = load_driver_class(args.driver_root)
    mix = noise_mix(args.seconds)
    prepost_seconds(driver_class, mix)
    times = [prepost_seconds(driver_class, mix) for _ in range(args.runs)]
    print(f"pre/post: min {min(times):.3f} s over {args.runs} runs ({args.seconds:g} s stereo noise)")
    print(f"tracemalloc peak: {peak_mib(driver_class, mix):.1f} MiB")


if __name__ == "__main__":
    main()
