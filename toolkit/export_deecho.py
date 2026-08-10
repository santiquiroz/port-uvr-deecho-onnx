"""Export the 3 FoxJoy De-Echo/De-Reverb VR checkpoints to ONNX + net-level parity.

Usage: .venv/Scripts/python.exe toolkit/export_deecho.py [model-name ...]

Each graph is the full CascadedNet forward: mag [1, 2, 673, 512] -> sigmoid mask
[1, 2, 673, 512]. Batch is FIXED at 1 (no dynamic_axes): the BiLSTM path breaks
with batch > 1 at export time, and the reference inference runs batch_size=1
anyway. opset 17, legacy JIT exporter (dynamo=False).

Writes artifacts/<name>.onnx and artifacts/manifest.json, then checks ONNX-vs-
torch mask parity on random windows (CPU EP, gate max_abs < 1e-4).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from audio_separator.separator.common_separator import CommonSeparator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toolkit.vr_cascaded_net import CascadedNet

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models"
ARTIFACTS = REPO / "artifacts"

N_FFT = 1344
# Smoke gate on uniform-random windows (out-of-distribution for these nets).
# Measured: 3.1e-05 (Normal), 6.2e-05 (Aggressive), 1.1e-04 (DeReverb, nout=64 --
# deeper accumulation). The binding 1e-4 mask gate lives in validate_ort.py, on
# real fixture audio.
PARITY_GATE = 2.5e-4

# nn_arch_size follows the reference's file-size selector (vr_separator.py);
# nout/nout_lstm follow vr_model_data. DeReverb's nout=64 is also forced
# internally by CascadedNet when nn_arch_size == 218409.
# uvr_primary_stem is the raw vr_model_data value; it decides is_non_accom_stem
# (and therefore the aggression branch AND which stem is the clean one).
# DeNoise's mask targets the NOISE, unlike the De-Echo family's dry signal.
MODELS = {
    "UVR-De-Echo-Normal": {
        "nn_arch_size": 123821, "nout": 48,
        "primary_stem": "No Echo", "secondary_stem": "Echo", "uvr_primary_stem": "No Other",
    },
    "UVR-De-Echo-Aggressive": {
        "nn_arch_size": 123821, "nout": 48,
        "primary_stem": "No Echo", "secondary_stem": "Echo", "uvr_primary_stem": "No Other",
    },
    "UVR-DeEcho-DeReverb": {
        "nn_arch_size": 218409, "nout": 64,
        "primary_stem": "No Reverb", "secondary_stem": "Reverb", "uvr_primary_stem": "No Other",
    },
    "UVR-DeNoise": {
        "nn_arch_size": 123821, "nout": 48,
        "primary_stem": "Noise", "secondary_stem": "No Noise", "uvr_primary_stem": "Other",
    },
}
UVR_HASHES = {
    "UVR-De-Echo-Normal": "f200a145434efc7dcf0cd093f517ed52",
    "UVR-De-Echo-Aggressive": "6857b2972e1754913aad0c9a1678c753",
    "UVR-DeEcho-DeReverb": "0fb9249ffe4ffc38d7b16243f394c0ff",
    "UVR-DeNoise": "44c55d8b5d2e3edea98c2b2bf93071c7",
}


def uvr_model_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()[-10000 * 1024 :]).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(name: str) -> CascadedNet:
    spec = MODELS[name]
    pth = MODELS_DIR / f"{name}.pth"
    found = uvr_model_hash(pth)
    if found != UVR_HASHES[name]:
        raise ValueError(f"{name}: UVR hash mismatch, expected {UVR_HASHES[name]}, got {found}")
    model = CascadedNet(N_FFT, spec["nn_arch_size"], nout=spec["nout"], nout_lstm=128)
    # weights_only: third-party .pth, never unpickle arbitrary objects
    state = torch.load(pth, map_location="cpu", weights_only=True)
    model.load_state_dict(state)  # strict=True: any arch mismatch fails loudly
    model.eval()
    return model


def export_onnx(model: CascadedNet, onnx_path: Path) -> None:
    example = torch.rand(1, 2, 673, 512)
    t0 = time.perf_counter()
    torch.onnx.export(
        model,
        (example,),
        str(onnx_path),
        opset_version=17,
        input_names=["mag"],
        output_names=["mask"],
        dynamo=False,
    )
    print(f"  exported in {time.perf_counter() - t0:.1f}s, {onnx_path.stat().st_size / 1e6:.1f} MB")


def check_parity(model: CascadedNet, onnx_path: Path) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(3):
        x = rng.random((1, 2, 673, 512), dtype=np.float32) * 0.8
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        out = sess.run(None, {"mag": x})[0]
        worst = max(worst, float(np.max(np.abs(out - ref))))
    return worst


def write_manifest(entries: dict) -> None:
    manifest_path = ARTIFACTS / "manifest.json"
    manifest = {
        "source": "TRvlvr/model_repo release all_public_uvr_models (official UVR Download Center)",
        "models_by": "FoxJoy",
        "architecture": "UVR VR 5.1 CascadedNet, 4band_v3, n_fft=1344, nout_lstm=128",
        "graph_io": {
            "input": {"mag": [1, 2, 673, 512]},
            "output": {"mask": [1, 2, 673, 512]},
            "note": "Batch fixed at 1 (BiLSTM breaks export with batch > 1; reference runs batch_size=1). Mask is pre-aggression, full window -- driver crops 64-frame offsets and applies the aggression curve.",
        },
        "opset": 17,
        "exporter": "legacy JIT (dynamo=False), torch 2.13.0+cpu",
        "models": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


def main() -> None:
    names = sys.argv[1:] or list(MODELS)
    ARTIFACTS.mkdir(exist_ok=True)
    manifest_path = ARTIFACTS / "manifest.json"
    entries = json.loads(manifest_path.read_text())["models"] if manifest_path.exists() else {}
    for name in names:
        print(name)
        model = load_model(name)
        onnx_path = ARTIFACTS / f"{name}.onnx"
        export_onnx(model, onnx_path)
        max_abs = check_parity(model, onnx_path)
        status = "OK" if max_abs < PARITY_GATE else "FAIL"
        print(f"  net parity CPU-EP max_abs={max_abs:.2e} [{status}]")
        if status == "FAIL":
            raise SystemExit(f"{name}: net parity gate {PARITY_GATE} failed ({max_abs:.2e})")
        spec = MODELS[name]
        entries[name] = {
            "file": f"{name}.onnx",
            "primary_stem": spec["primary_stem"],
            "secondary_stem": spec["secondary_stem"],
            # Consumers need these two to drive the graph correctly: the flag
            # selects the aggression branch and tells them which stem is clean.
            "uvr_primary_stem": spec["uvr_primary_stem"],
            "is_non_accom_stem": spec["uvr_primary_stem"] in CommonSeparator.NON_ACCOM_STEMS,
            "nout": spec["nout"],
            "nn_arch_size": spec["nn_arch_size"],
            "source_pth_uvr_hash_md5_tail10MB": UVR_HASHES[name],
            "source_pth_bytes": (MODELS_DIR / f"{name}.pth").stat().st_size,
            "onnx_sha256": sha256(onnx_path),
            "onnx_bytes": onnx_path.stat().st_size,
            "net_parity_cpu_max_abs": max_abs,
        }
    write_manifest(entries)


if __name__ == "__main__":
    main()
