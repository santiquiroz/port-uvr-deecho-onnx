"""Golden reference capture: run the 3 De-Echo/De-Reverb models through
python-audio-separator's own VR code path and dump every intermediate needed to
gate the ONNX driver.

Usage: .venv/Scripts/python.exe toolkit/capture_baseline.py [model-name ...]

This mirrors VRSeparator.separate() (audio_separator 0.44.5) exactly -- same
functions, same order, same defaults for these models: window_size=512,
batch_size=1, aggression=5, no TTA, no post-process, no high_end_process. The
only thing skipped is CommonSeparator.final_process's output peak-normalize
(normalization_threshold=0.9), which only rescales when a stem peaks above 0.9;
the committed fixture peaks well below that, so golden == what the CLI writes.

Dumps to refs/golden/<model>/:
  combined_spec.npy  complex64 [2, 673, F]  (input to mag/phase split)
  mask_raw.npy       float32   [2, 673, F]  (network output, pre-aggression)
  mask_aggr.npy      float32   [2, 673, F]  (post adjust_aggr, the applied mask)
  y_spec.npy / v_spec.npy      [2, 673, F]  (masked complex specs)
  primary.npy / secondary.npy  [2, N]       (final 44.1 kHz stems)
  primary.wav / secondary.wav               (same, for listening)
  meta.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from audio_separator.separator.uvr_lib_v5 import spec_utils
from audio_separator.separator.uvr_lib_v5.vr_network import nets_new
from audio_separator.separator.uvr_lib_v5.vr_network.model_param_init import ModelParameters

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models"
FIXTURE = REPO / "refs" / "inputs" / "fixture_mix.wav"
GOLDEN_DIR = REPO / "refs" / "golden"

WINDOW_SIZE = 512
BATCH_SIZE = 1
AGGRESSION = 5

MODELS = {
    "UVR-De-Echo-Normal": {"nout": 48, "primary_stem": "No Echo"},
    "UVR-De-Echo-Aggressive": {"nout": 48, "primary_stem": "No Echo"},
    "UVR-DeEcho-DeReverb": {"nout": 64, "primary_stem": "No Reverb"},
}


def model_params() -> ModelParameters:
    import audio_separator.separator as sep_pkg

    params_path = Path(sep_pkg.__file__).parent / "uvr_lib_v5" / "vr_network" / "modelparams" / "4band_v3.json"
    return ModelParameters(str(params_path))


def nn_arch_size_for(pth: Path) -> int:
    nn_arch_sizes = [31191, 33966, 56817, 123821, 123812, 129605, 218409, 537238, 537227]
    model_size = math.ceil(pth.stat().st_size / 1024)
    return min(nn_arch_sizes, key=lambda x: abs(x - model_size))


def load_network(name: str, mp: ModelParameters) -> nets_new.CascadedNet:
    pth = MODELS_DIR / f"{name}.pth"
    net = nets_new.CascadedNet(mp.param["bins"] * 2, nn_arch_size_for(pth), nout=MODELS[name]["nout"], nout_lstm=128)
    net.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))
    net.eval()
    return net


def loading_mix(mp: ModelParameters) -> np.ndarray:
    """VRSeparator.loading_mix, verbatim flow (is_v51_model=True, no high_end_process)."""
    x_wave, x_spec_s = {}, {}
    bands_n = len(mp.param["band"])
    for d in range(bands_n, 0, -1):
        bp = mp.param["band"][d]
        if d == bands_n:
            x_wave[d], _ = librosa.load(str(FIXTURE), sr=bp["sr"], mono=False, dtype=np.float32, res_type=bp["res_type"])
            if x_wave[d].ndim == 1:
                x_wave[d] = np.asarray([x_wave[d], x_wave[d]])
        else:
            x_wave[d] = librosa.resample(
                x_wave[d + 1], orig_sr=mp.param["band"][d + 1]["sr"], target_sr=bp["sr"], res_type=bp["res_type"]
            )
        x_spec_s[d] = spec_utils.wave_to_spectrogram(x_wave[d], bp["hl"], bp["n_fft"], mp, band=d, is_v51_model=True)
    return spec_utils.combine_spectrograms(x_spec_s, mp, is_v51_model=True)


def predict_raw_mask(net: nets_new.CascadedNet, x_mag: np.ndarray) -> np.ndarray:
    """VRSeparator.inference_vr._execute, verbatim flow (batch_size=1, no TTA)."""
    n_frame = x_mag.shape[2]
    pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, WINDOW_SIZE, net.offset)
    x_mag_pad = np.pad(x_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
    x_mag_pad /= x_mag_pad.max()
    patches = (x_mag_pad.shape[2] - 2 * net.offset) // roi_size
    dataset = np.asarray([x_mag_pad[:, :, i * roi_size : i * roi_size + WINDOW_SIZE] for i in range(patches)])
    with torch.no_grad():
        chunks = []
        for i in range(0, patches, BATCH_SIZE):
            batch = torch.from_numpy(dataset[i : i + BATCH_SIZE])
            pred = net.predict_mask(batch).detach().cpu().numpy()
            chunks.append(np.concatenate(pred, axis=2))
        mask = np.concatenate(chunks, axis=2)
    return mask[:, :, :n_frame]


def capture(name: str, mp: ModelParameters) -> None:
    print(name)
    out_dir = GOLDEN_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    net = load_network(name, mp)

    combined = loading_mix(mp)
    x_mag, x_phase = spec_utils.preprocess(combined)
    mask_raw = predict_raw_mask(net, x_mag)

    aggressiveness = {
        "value": AGGRESSION / 100,
        "split_bin": mp.param["band"][1]["crop_stop"],
        "aggr_correction": mp.param.get("aggr_correction"),
    }
    # "No Echo"/"No Reverb" are not in NON_ACCOM_STEMS -> is_non_accom_stem=False
    mask_aggr = spec_utils.adjust_aggr(mask_raw.copy(), False, aggressiveness)

    y_spec = np.nan_to_num(mask_aggr * x_mag * np.exp(1.0j * x_phase), nan=0.0, posinf=0.0, neginf=0.0)
    v_spec = np.nan_to_num((1 - mask_aggr) * x_mag * np.exp(1.0j * x_phase), nan=0.0, posinf=0.0, neginf=0.0)

    primary = spec_utils.cmb_spectrogram_to_wave(y_spec, mp, is_v51_model=True)
    secondary = spec_utils.cmb_spectrogram_to_wave(v_spec, mp, is_v51_model=True)

    np.save(out_dir / "combined_spec.npy", combined.astype(np.complex64))
    np.save(out_dir / "mask_raw.npy", mask_raw.astype(np.float32))
    np.save(out_dir / "mask_aggr.npy", mask_aggr.astype(np.float32))
    np.save(out_dir / "y_spec.npy", y_spec.astype(np.complex64))
    np.save(out_dir / "v_spec.npy", v_spec.astype(np.complex64))
    np.save(out_dir / "primary.npy", primary.astype(np.float32))
    np.save(out_dir / "secondary.npy", secondary.astype(np.float32))
    sf.write(out_dir / "primary.wav", primary.T, mp.param["sr"], subtype="FLOAT")
    sf.write(out_dir / "secondary.wav", secondary.T, mp.param["sr"], subtype="FLOAT")
    meta = {
        "fixture": str(FIXTURE.relative_to(REPO)),
        "primary_stem": MODELS[name]["primary_stem"],
        "aggression": AGGRESSION,
        "window_size": WINDOW_SIZE,
        "batch_size": BATCH_SIZE,
        "librosa": librosa.__version__,
        "audio_separator": __import__("importlib.metadata", fromlist=["version"]).version("audio-separator"),
        "torch": torch.__version__,
        "combined_spec_shape": list(combined.shape),
        "stem_samples": int(primary.shape[1]),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  spec {combined.shape} mask [{mask_raw.min():.4f}, {mask_raw.max():.4f}] stems {primary.shape}")


def main() -> None:
    if not FIXTURE.exists():
        raise SystemExit("fixture missing -- run toolkit/make_fixture.py first")
    mp = model_params()
    for name in sys.argv[1:] or list(MODELS):
        capture(name, mp)


if __name__ == "__main__":
    main()
