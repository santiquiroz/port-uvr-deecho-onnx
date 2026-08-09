"""End-to-end parity gate: numpy/scipy driver + ONNX graphs vs the audio-separator
golden dumps (refs/golden/, from toolkit/capture_baseline.py).

Usage: .venv/Scripts/python.exe toolkit/validate_ort.py [model-name ...] [--cpu-only]

Gates per model/provider:
  mask     max_abs(driver mask_raw - golden mask_raw) < 1e-4   (network + pre-chain)
  stems    SI-SDR(driver stem, golden stem) > 40 dB            (full wav -> wav)

Also reported (informational, no gate):
  pre-chain   max_abs on the combined complex spectrogram (driver STFT/resample vs librosa)
  synth-only  SI-SDR of driver's synthesis chain fed the GOLDEN y/v specs -- isolates
              the one deliberate divergence (polyphase upsampling vs libsamplerate
              sinc_fastest) from any mask error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver import multiband
from driver.pipeline import MODEL_SPECS, DeEchoDriver

REPO = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO / "artifacts"
FIXTURE = REPO / "refs" / "inputs" / "fixture_mix.wav"
GOLDEN_DIR = REPO / "refs" / "golden"

MASK_GATE = 1e-4
SISDR_GATE_DB = 40.0


def si_sdr_db(est: np.ndarray, ref: np.ndarray) -> float:
    n = min(est.shape[-1], ref.shape[-1])
    est, ref = est[..., :n].astype(np.float64), ref[..., :n].astype(np.float64)
    scores = []
    for ch in range(est.shape[0]):
        alpha = np.dot(est[ch], ref[ch]) / np.dot(ref[ch], ref[ch])
        target = alpha * ref[ch]
        noise = est[ch] - target
        scores.append(10 * np.log10(np.sum(target**2) / np.sum(noise**2)))
    return float(np.mean(scores))


def load_fixture() -> np.ndarray:
    wave, sr = sf.read(FIXTURE, dtype="float32")
    assert sr == 44100, sr
    return wave.T


def make_session(name: str, provider: str) -> ort.InferenceSession:
    providers = [("DmlExecutionProvider", {"device_id": 0})] if provider == "dml" else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(ARTIFACTS / f"{name}.onnx"), providers=providers)


def validate(name: str, provider: str, mix: np.ndarray) -> list[str]:
    golden = GOLDEN_DIR / name
    sess = make_session(name, provider)
    driver = DeEchoDriver(lambda window: sess.run(None, {"mag": window})[0])
    failures = []

    spec = multiband.wave_to_combined_spec(mix)
    g_spec = np.load(golden / "combined_spec.npy")
    pre_max_abs = float(np.max(np.abs(spec - g_spec)))

    mask = driver.infer_mask(spec)
    g_mask = np.load(golden / "mask_raw.npy")
    mask_max_abs = float(np.max(np.abs(mask - g_mask)))
    mask_ok = mask_max_abs < MASK_GATE
    if not mask_ok:
        failures.append(f"mask {mask_max_abs:.2e} >= {MASK_GATE}")

    primary, secondary = driver.separate(mix)
    g_primary = np.load(golden / "primary.npy")
    g_secondary = np.load(golden / "secondary.npy")
    sdr_primary = si_sdr_db(primary, g_primary)
    sdr_secondary = si_sdr_db(secondary, g_secondary)
    stems_ok = sdr_primary > SISDR_GATE_DB and sdr_secondary > SISDR_GATE_DB
    if not stems_ok:
        failures.append(f"stems SI-SDR {sdr_primary:.1f}/{sdr_secondary:.1f} dB <= {SISDR_GATE_DB}")

    synth_primary = multiband.combined_spec_to_wave(np.load(golden / "y_spec.npy"))
    synth_sdr = si_sdr_db(synth_primary, g_primary)

    print(f"  [{provider}] pre-chain max_abs={pre_max_abs:.2e} (info)")
    print(f"  [{provider}] mask max_abs={mask_max_abs:.2e} [{'OK' if mask_ok else 'FAIL'}] (gate {MASK_GATE})")
    print(
        f"  [{provider}] stems SI-SDR primary={sdr_primary:.1f} dB secondary={sdr_secondary:.1f} dB "
        f"[{'OK' if stems_ok else 'FAIL'}] (gate >{SISDR_GATE_DB:.0f} dB)"
    )
    print(f"  [{provider}] synth-only (golden y_spec -> wav) SI-SDR={synth_sdr:.1f} dB (info)")
    return failures


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    providers = ["cpu"] if "--cpu-only" in sys.argv else ["cpu", "dml"]
    names = args or list(MODEL_SPECS)
    mix = load_fixture()
    all_failures = []
    for name in names:
        print(name)
        for provider in providers:
            all_failures += [f"{name}/{provider}: {f}" for f in validate(name, provider, mix)]
    if all_failures:
        print("\nFAILED gates:")
        for failure in all_failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("\nall gates OK")


if __name__ == "__main__":
    main()
