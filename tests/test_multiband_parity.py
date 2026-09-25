"""Multiband analysis/synthesis and ROI windowing, against python-audio-separator's
own spec_utils on in-memory audio -- no refs/golden, no artifacts/*.onnx.

toolkit/validate_ort.py gates the same path end to end, but it needs the golden
and the exported graphs, both gitignored; these tests keep the DSP around the
network pinned in a clean checkout. The reference parameters come from the
4band_v3.json shipped with audio-separator, not from driver/vr_params.py, so a
mistyped band constant fails here.
"""
import sys
from pathlib import Path

import librosa
import numpy as np
import pytest
import audio_separator.separator as sep_pkg
from audio_separator.separator.uvr_lib_v5 import spec_utils
from audio_separator.separator.uvr_lib_v5.vr_network.model_param_init import ModelParameters

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver import multiband
from driver.pipeline import DeEchoDriver, make_padding
from driver.vr_params import OFFSET, WINDOW_SIZE

PARAMS_JSON = Path(sep_pkg.__file__).parent / "uvr_lib_v5" / "vr_network" / "modelparams" / "4band_v3.json"
N_SAMPLES = 2 * 44100
FRAME_COUNTS = [1, 7, 383, 384, 385, 767, 768, 1000, 1537]


@pytest.fixture(scope="module")
def mp() -> ModelParameters:
    return ModelParameters(str(PARAMS_JSON))


def noise_mix(n: int = N_SAMPLES) -> np.ndarray:
    return np.random.default_rng(4).standard_normal((2, n)).astype(np.float32) * 0.3


def random_mask(shape: tuple[int, ...]) -> np.ndarray:
    return np.random.default_rng(5).random(shape).astype(np.float32)


def reference_band_waves(mix: np.ndarray, mp: ModelParameters) -> dict[int, np.ndarray]:
    bands = mp.param["band"]
    waves = {len(bands): mix}
    for d in range(len(bands) - 1, 0, -1):
        waves[d] = librosa.resample(
            waves[d + 1], orig_sr=bands[d + 1]["sr"], target_sr=bands[d]["sr"], res_type=bands[d]["res_type"]
        )
    return waves


def reference_combined_spec(mix: np.ndarray, mp: ModelParameters) -> np.ndarray:
    """VRSeparator.loading_mix from an in-memory mix (is_v51_model=True)."""
    specs = {
        d: spec_utils.wave_to_spectrogram(wave, mp.param["band"][d]["hl"], mp.param["band"][d]["n_fft"], mp, band=d, is_v51_model=True)
        for d, wave in reference_band_waves(mix, mp).items()
    }
    return spec_utils.combine_spectrograms(specs, mp, is_v51_model=True)


def reference_infer_mask(mag: np.ndarray, run_graph) -> np.ndarray:
    """VRSeparator.inference_vr._execute (batch_size=1), with net.predict_mask's edge crop."""
    n_frame = mag.shape[2]
    pad_l, pad_r, roi_size = spec_utils.make_padding(n_frame, WINDOW_SIZE, OFFSET)
    mag_pad = np.pad(mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")
    mag_pad /= mag_pad.max()
    patches = (mag_pad.shape[2] - 2 * OFFSET) // roi_size
    dataset = np.asarray([mag_pad[:, :, i * roi_size : i * roi_size + WINDOW_SIZE] for i in range(patches)])
    chunks = []
    for i in range(patches):
        pred = run_graph(dataset[i : i + 1])[:, :, :, OFFSET:-OFFSET]
        chunks.append(np.concatenate(pred, axis=2))
    return np.concatenate(chunks, axis=2)[:, :, :n_frame]


def identity_graph(window: np.ndarray) -> np.ndarray:
    assert window.shape == (1, 2, 673, WINDOW_SIZE)
    return window.copy()


def frame_index_spec(n_frames: int) -> np.ndarray:
    frames = np.arange(1, n_frames + 1, dtype=np.float32)
    return np.broadcast_to(frames, (2, 673, n_frames)).astype(np.complex64)


def test_analysis_matches_reference(mp):
    mix = noise_mix()
    ref = reference_combined_spec(mix, mp)
    mine = multiband.wave_to_combined_spec(mix)
    assert mine.shape == ref.shape
    assert np.max(np.abs(ref - mine)) < 1e-5


def test_synthesis_matches_reference(mp, monkeypatch):
    # The reference upsamples with libsamplerate sinc_fastest, the one documented
    # divergence; with polyphase on both sides the chains must agree.
    monkeypatch.setattr(spec_utils, "wav_resolution", "polyphase")
    combined = multiband.wave_to_combined_spec(noise_mix())
    masked = combined * random_mask(combined.shape)
    ref = spec_utils.cmb_spectrogram_to_wave(masked, mp, is_v51_model=True)
    mine = multiband.combined_spec_to_wave(masked)
    assert mine.shape == ref.shape
    assert np.max(np.abs(ref - mine)) < 1e-5


@pytest.mark.parametrize("n_frames", FRAME_COUNTS)
def test_make_padding_matches_reference(n_frames):
    assert make_padding(n_frames) == spec_utils.make_padding(n_frames, WINDOW_SIZE, OFFSET)


@pytest.mark.parametrize("n_frames", FRAME_COUNTS)
def test_windowed_mask_matches_reference_loop(n_frames):
    spec = frame_index_spec(n_frames)
    mine = DeEchoDriver(identity_graph).infer_mask(spec)
    ref = reference_infer_mask(np.abs(spec), identity_graph)
    assert mine.shape == ref.shape == (2, 673, n_frames)
    assert np.array_equal(mine, ref)


@pytest.mark.parametrize("n_frames", FRAME_COUNTS)
def test_windowed_mask_stitches_every_frame_once_in_order(n_frames):
    """With an identity graph, frame t must come back as (t + 1) / n_frames."""
    mask = DeEchoDriver(identity_graph).infer_mask(frame_index_spec(n_frames))
    expected = np.arange(1, n_frames + 1, dtype=np.float32) / np.float32(n_frames)
    assert np.array_equal(mask[0, 0], expected)
