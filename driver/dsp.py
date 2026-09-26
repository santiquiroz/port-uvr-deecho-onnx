"""librosa-compatible STFT/iSTFT and polyphase resampling in pure numpy/scipy.

stft/istft reproduce librosa.stft/librosa.istft defaults (hann periodic window,
win_length = n_fft, center=True, pad_mode="constant") bit-for-bit in float32.
resample matches librosa's res_type="polyphase" (scipy.signal.resample_poly with
sr // gcd factors), which is what the reference uses for every downsample.
The reference's upsamples use libsamplerate "sinc_fastest"; this driver uses the
same polyphase kernel instead -- divergence is measured, not hidden (see README).
"""
from __future__ import annotations

import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample_poly


def hann_periodic(n_fft: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)


def stft(wave: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    pad = n_fft // 2
    y = np.pad(wave.astype(np.float32, copy=False), pad, mode="constant")
    frames = sliding_window_view(y, n_fft)[::hop]
    spec = np.fft.rfft(frames * hann_periodic(n_fft), axis=-1)
    return spec.astype(np.complex64, copy=False).T


def stft_frame_count(n_samples: int, n_fft: int, hop: int) -> int:
    pad = n_fft // 2
    return 1 + (n_samples + 2 * pad - n_fft) // hop


def overlap_add(frames: np.ndarray, hop: int) -> np.ndarray:
    n_frames, n_fft = frames.shape
    n_chunks = -(-n_fft // hop)
    blocks = np.zeros((n_frames + n_chunks - 1, hop), dtype=np.float32)
    # Last chunk first: every output sample then sums its frames in ascending
    # order, bit-identical to a per-frame loop.
    for chunk in reversed(range(n_chunks)):
        start = chunk * hop
        width = min(hop, n_fft - start)
        blocks[chunk : chunk + n_frames, :width] += frames[:, start : start + width]
    return blocks.reshape(-1)[: n_fft + hop * (n_frames - 1)]


def istft(spec: np.ndarray, hop: int) -> np.ndarray:
    n_bins, n_frames = spec.shape
    n_fft = 2 * (n_bins - 1)
    window = hann_periodic(n_fft)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=-1).astype(np.float32, copy=False)
    frames *= window
    wave = overlap_add(frames, hop)
    win_sq_sum = overlap_add(np.broadcast_to(window * window, frames.shape), hop)
    np.divide(wave, win_sq_sum, out=wave, where=win_sq_sum > np.finfo(np.float32).tiny)
    pad = n_fft // 2
    return wave[pad : wave.shape[0] - pad]


def stft_stereo(wave: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    return np.stack([stft(wave[0], n_fft, hop), stft(wave[1], n_fft, hop)])


def istft_stereo(spec: np.ndarray, hop: int) -> np.ndarray:
    return np.stack([istft(spec[0], hop), istft(spec[1], hop)])


def resample(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wave
    gcd = np.gcd(orig_sr, target_sr)
    out = resample_poly(wave, target_sr // gcd, orig_sr // gcd, axis=-1)
    return out.astype(np.float32, copy=False)


def resampled_length(n_samples: int, orig_sr: int, target_sr: int) -> int:
    gcd = math.gcd(orig_sr, target_sr)
    up, down = target_sr // gcd, orig_sr // gcd
    return -(-n_samples * up // down)


def lp_filter_mask(n_bins: int, bin_start: int, bin_stop: int) -> np.ndarray:
    return np.concatenate(
        [np.ones((bin_start - 1, 1)), np.linspace(1, 0, bin_stop - bin_start + 1)[:, None], np.zeros((n_bins - bin_stop, 1))]
    )


def hp_filter_mask(n_bins: int, bin_start: int, bin_stop: int) -> np.ndarray:
    return np.concatenate(
        [np.zeros((bin_stop + 1, 1)), np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None], np.ones((n_bins - bin_start - 2, 1))]
    )
