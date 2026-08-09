import sys
from pathlib import Path

import librosa
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver import dsp

BAND_CONFIGS = [(640, 80), (320, 80), (512, 160), (960, 480)]


def random_wave(n: int = 44100) -> np.ndarray:
    return np.random.default_rng(0).standard_normal(n).astype(np.float32) * 0.3


@pytest.mark.parametrize("n_fft,hop", BAND_CONFIGS)
def test_stft_matches_librosa(n_fft, hop):
    wave = random_wave()
    ref = librosa.stft(wave, n_fft=n_fft, hop_length=hop)
    mine = dsp.stft(wave, n_fft, hop)
    assert mine.shape == ref.shape
    assert np.max(np.abs(ref - mine)) < 1e-5


@pytest.mark.parametrize("n_fft,hop", BAND_CONFIGS)
def test_istft_matches_librosa(n_fft, hop):
    spec = librosa.stft(random_wave(), n_fft=n_fft, hop_length=hop)
    ref = librosa.istft(spec, hop_length=hop)
    mine = dsp.istft(spec, hop)
    assert mine.shape == ref.shape
    assert np.max(np.abs(ref - mine)) < 1e-5


@pytest.mark.parametrize("orig_sr,target_sr", [(44100, 14700), (14700, 7350), (7350, 14700), (14700, 44100)])
def test_resample_matches_librosa_polyphase(orig_sr, target_sr):
    wave = np.random.default_rng(1).standard_normal((2, orig_sr)).astype(np.float32) * 0.3
    ref = librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr, res_type="polyphase")
    mine = dsp.resample(wave, orig_sr, target_sr)
    assert mine.shape == ref.shape
    assert np.max(np.abs(ref - mine)) < 1e-7


def test_resample_identity_returns_input():
    wave = random_wave()
    assert dsp.resample(wave, 7350, 7350) is wave
