"""Deterministic synthetic test fixture: a copyright-free 10 s stereo 44.1 kHz
"song" (plucked pentatonic melody + noise percussion + pad chirp) with a
feedback-delay echo and a synthetic decaying-noise reverb tail baked in.

Usage: .venv/Scripts/python.exe toolkit/make_fixture.py

Writes refs/inputs/fixture_mix.wav (the model input) and refs/inputs/fixture_dry.wav
(the clean signal, for listening comparison only -- the parity golden is
audio-separator's own output, not this dry track).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "refs" / "inputs"

SR = 44100
DURATION_S = 10.0
PENTATONIC_HZ = (220.0, 261.63, 293.66, 329.63, 392.0, 440.0, 523.25, 587.33)


def pluck(freq: float, length_s: float, rng: np.random.Generator) -> np.ndarray:
    n = int(length_s * SR)
    t = np.arange(n) / SR
    env = np.exp(-t * 6.0)
    tone = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t) + 0.15 * np.sin(2 * np.pi * 3 * freq * t)
    return (tone * env * (0.5 + 0.3 * rng.random())).astype(np.float32)


def noise_hit(length_s: float, rng: np.random.Generator) -> np.ndarray:
    n = int(length_s * SR)
    env = np.exp(-np.arange(n) / SR * 40.0)
    return (rng.standard_normal(n) * env * 0.25).astype(np.float32)


def chirp_pad(length_s: float) -> np.ndarray:
    n = int(length_s * SR)
    t = np.arange(n) / SR
    freq = 110.0 + 30.0 * np.sin(2 * np.pi * 0.25 * t)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return (0.12 * np.sin(phase) * np.hanning(n)).astype(np.float32)


def place(canvas: np.ndarray, clip: np.ndarray, start_s: float, gain: float = 1.0) -> None:
    start = int(start_s * SR)
    end = min(start + len(clip), len(canvas))
    canvas[start:end] += clip[: end - start] * gain


def build_dry(rng: np.random.Generator) -> np.ndarray:
    n = int(DURATION_S * SR)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for beat in range(16):
        t0 = beat * 0.55 + 0.2
        note = pluck(PENTATONIC_HZ[int(rng.integers(len(PENTATONIC_HZ)))], 0.9, rng)
        pan = 0.35 + 0.3 * rng.random()
        place(left, note, t0, 1.0 - pan)
        place(right, note, t0, pan)
        if beat % 2 == 0:
            hit = noise_hit(0.25, rng)
            place(left, hit, t0 + 0.28, 0.6)
            place(right, hit, t0 + 0.28, 0.6)
    pad = chirp_pad(DURATION_S - 1.0)
    place(left, pad, 0.5, 0.8)
    place(right, pad, 0.5, 1.0)
    return np.stack([left, right])


def feedback_delay(dry: np.ndarray, delay_s: float, feedback: float, taps: int) -> np.ndarray:
    wet = np.zeros_like(dry)
    delay = int(delay_s * SR)
    gain = feedback
    for tap in range(1, taps + 1):
        shift = tap * delay
        if shift >= dry.shape[1]:
            break
        wet[:, shift:] += dry[:, : dry.shape[1] - shift] * gain
        gain *= feedback
    return wet


def noise_reverb(dry: np.ndarray, decay_s: float, level: float, rng: np.random.Generator) -> np.ndarray:
    ir_len = int(decay_s * SR)
    t = np.arange(ir_len) / SR
    wet = np.zeros_like(dry)
    for ch in range(2):
        ir = rng.standard_normal(ir_len).astype(np.float32) * np.exp(-t * 3.5) * level
        wet[ch] = np.convolve(dry[ch], ir)[: dry.shape[1]]
    return wet


def main() -> None:
    rng = np.random.default_rng(20260809)
    dry = build_dry(rng)
    wet = feedback_delay(dry, 0.22, 0.45, 4) + noise_reverb(dry, 0.4, 0.05, rng)
    mix = dry + wet
    mix *= 0.6 / np.max(np.abs(mix))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(OUT_DIR / "fixture_mix.wav", mix.T, SR, subtype="PCM_16")
    sf.write(OUT_DIR / "fixture_dry.wav", (dry * 0.6 / np.max(np.abs(mix))).T, SR, subtype="PCM_16")
    print(f"wrote {OUT_DIR / 'fixture_mix.wav'} peak={np.max(np.abs(mix)):.3f}")


if __name__ == "__main__":
    main()
