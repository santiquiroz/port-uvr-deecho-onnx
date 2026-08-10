"""Throughput bench: CPU-EP vs DirectML for the De-Echo/De-Reverb graphs.

Usage: .venv/Scripts/python.exe toolkit/bench_dml.py [model-name ...]

Two measurements per model/provider:
  window   ms per [1, 2, 673, 512] network call (the unit of work). Each call
           advances 384 ROI frames x 480-sample hop @44.1 kHz = 4.18 s of audio,
           so realtime factor = 4.18 / (ms/1000).
  e2e      wall seconds for DeEchoDriver.separate() on the 10 s fixture
           (includes the numpy pre/post chain, 3 network calls).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.pipeline import MODEL_SPECS, DeEchoDriver
from driver.vr_params import OFFSET, WINDOW_SIZE

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
FIXTURE = REPO / "refs" / "inputs" / "fixture_mix.wav"

WARMUP = 3
RUNS = 12
AUDIO_S_PER_WINDOW = (WINDOW_SIZE - 2 * OFFSET) * 480 / 44100


def make_session(name: str, provider: str) -> ort.InferenceSession:
    providers = [("DmlExecutionProvider", {"device_id": 0})] if provider == "dml" else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(ARTIFACTS / f"{name}.onnx"), providers=providers)


def bench_window(sess: ort.InferenceSession) -> tuple[float, float]:
    rng = np.random.default_rng(7)
    window = (rng.random((1, 2, 673, 512), dtype=np.float32) * 0.8).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {"mag": window})
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"mag": window})
        times.append(time.perf_counter() - t0)
    mean_ms = float(np.mean(times) * 1000)
    return mean_ms, AUDIO_S_PER_WINDOW / (mean_ms / 1000)


def bench_e2e(sess: ort.InferenceSession, mix: np.ndarray, name: str) -> float:
    driver = DeEchoDriver(
        lambda w: sess.run(None, {"mag": w})[0],
        is_non_accom_stem=MODEL_SPECS[name]["is_non_accom_stem"],
    )
    t0 = time.perf_counter()
    driver.separate(mix)
    return time.perf_counter() - t0


def main() -> None:
    names = sys.argv[1:] or list(MODEL_SPECS)
    mix, _ = sf.read(FIXTURE, dtype="float32")
    mix = mix.T
    audio_s = mix.shape[1] / 44100
    print(f"fixture: {audio_s:.1f} s stereo | window unit = {AUDIO_S_PER_WINDOW:.2f} s audio")
    for name in names:
        print(name)
        results = {}
        for provider in ("cpu", "dml"):
            sess = make_session(name, provider)
            mean_ms, rtf = bench_window(sess)
            e2e_s = bench_e2e(sess, mix, name)
            results[provider] = mean_ms
            print(f"  [{provider}] window {mean_ms:.1f} ms ({rtf:.1f}x realtime) | e2e {e2e_s:.2f} s ({audio_s / e2e_s:.1f}x realtime)")
            del sess
        print(f"  DML speedup (window): {results['cpu'] / results['dml']:.1f}x")


if __name__ == "__main__":
    main()
