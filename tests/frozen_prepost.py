"""Frozen copy of the pre/post path of DeEchoDriver.separate as of 6f538c0.

The reference for test_prepost_equivalence.py: any speed/memory rework of
driver/dsp.py, driver/multiband.py or driver/pipeline.py must keep separate()
numerically equal to this. Do not edit it to follow the driver.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from driver import dsp
from driver.pipeline import as_stereo_float32, make_padding, predict_mask
from driver.vr_params import AGGR_SPLIT_BIN, BANDS, BINS, PRE_FILTER_START, PRE_FILTER_STOP


def stft(wave, n_fft, hop):
    pad = n_fft // 2
    y = np.pad(wave.astype(np.float32, copy=False), pad, mode="constant")
    frames = sliding_window_view(y, n_fft)[::hop]
    spec = np.fft.rfft(frames * dsp.hann_periodic(n_fft), axis=-1)
    return spec.astype(np.complex64).T


def istft(spec, hop):
    n_bins, n_frames = spec.shape
    n_fft = 2 * (n_bins - 1)
    window = dsp.hann_periodic(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=0).real.astype(np.float32) * window[:, None]
    total = n_fft + hop * (n_frames - 1)
    wave = np.zeros(total, dtype=np.float32)
    win_sq_sum = np.zeros(total, dtype=np.float32)
    win_sq = window * window
    for t in range(n_frames):
        start = t * hop
        wave[start : start + n_fft] += frames[:, t]
        win_sq_sum[start : start + n_fft] += win_sq
    nonzero = win_sq_sum > np.finfo(np.float32).tiny
    wave[nonzero] /= win_sq_sum[nonzero]
    pad = n_fft // 2
    return wave[pad : total - pad]


def band_waves(mix):
    waves = [None] * len(BANDS)
    waves[-1] = mix.astype(np.float32, copy=False)
    for idx in range(len(BANDS) - 2, -1, -1):
        waves[idx] = dsp.resample(waves[idx + 1], BANDS[idx + 1]["sr"], BANDS[idx]["sr"])
    return waves


def wave_to_combined_spec(mix):
    specs = [
        np.stack([stft(wave[0], band["n_fft"], band["hl"]), stft(wave[1], band["n_fft"], band["hl"])])
        for band, wave in zip(BANDS, band_waves(mix))
    ]
    n_frames = min(spec.shape[2] for spec in specs)
    combined = np.zeros((2, BINS + 1, n_frames), dtype=np.complex64)
    row = 0
    for band, spec in zip(BANDS, specs):
        height = band["crop_stop"] - band["crop_start"]
        combined[:, row : row + height] = spec[:, band["crop_start"] : band["crop_stop"], :n_frames]
        row += height
    combined *= dsp.lp_filter_mask(BINS + 1, PRE_FILTER_START, PRE_FILTER_STOP)
    return combined


def band_full_spec(combined, band_idx, row):
    band = BANDS[band_idx]
    n_bins = band["n_fft"] // 2 + 1
    spec = np.zeros((2, n_bins, combined.shape[2]), dtype=np.complex64)
    height = band["crop_stop"] - band["crop_start"]
    spec[:, band["crop_start"] : band["crop_stop"]] = combined[:, row : row + height]
    if band_idx > 0:
        spec *= dsp.hp_filter_mask(n_bins, band["hpf_start"], band["hpf_stop"] - 1)
    if band_idx < len(BANDS) - 1:
        spec *= dsp.lp_filter_mask(n_bins, band["lpf_start"], band["lpf_stop"])
    return spec


def combined_spec_to_wave(combined):
    wave = None
    row = 0
    for band_idx, band in enumerate(BANDS):
        full = band_full_spec(combined, band_idx, row)
        band_wave = np.stack([istft(full[0], band["hl"]), istft(full[1], band["hl"])])
        row += band["crop_stop"] - band["crop_start"]
        wave = band_wave if wave is None else wave + band_wave
        if band_idx < len(BANDS) - 1:
            wave = dsp.resample(wave, band["sr"], BANDS[band_idx + 1]["sr"])
    return wave


def adjust_aggression(mask, aggression, is_non_accom_stem):
    aggr = (aggression / 100.0) * 2.0
    if aggr == 0:
        return mask
    if is_non_accom_stem:
        aggr = 1 - aggr
    adjusted = mask.copy()
    adjusted[:, :AGGR_SPLIT_BIN] = np.power(adjusted[:, :AGGR_SPLIT_BIN], 1 + aggr / 3)
    adjusted[:, AGGR_SPLIT_BIN:] = np.power(adjusted[:, AGGR_SPLIT_BIN:], 1 + aggr)
    return adjusted


def infer_mask(spec, run_graph):
    mag = np.abs(spec)
    n_frames = mag.shape[2]
    pad_left, pad_right, roi_size = make_padding(n_frames)
    mag_padded = np.pad(mag, ((0, 0), (0, 0), (pad_left, pad_right)))
    mag_padded /= mag_padded.max()
    mask = predict_mask(mag_padded.astype(np.float32), roi_size, run_graph)
    return mask[:, :, :n_frames]


def separate(mix, run_graph, aggression=5.0, is_non_accom_stem=False):
    spec = wave_to_combined_spec(as_stereo_float32(mix))
    mask = adjust_aggression(infer_mask(spec, run_graph), aggression, is_non_accom_stem)
    mag, phase = np.abs(spec), np.angle(spec)
    primary_spec = mask * mag * np.exp(1.0j * phase)
    secondary_spec = (1 - mask) * mag * np.exp(1.0j * phase)
    primary = combined_spec_to_wave(np.nan_to_num(primary_spec, nan=0.0, posinf=0.0, neginf=0.0))
    secondary = combined_spec_to_wave(np.nan_to_num(secondary_spec, nan=0.0, posinf=0.0, neginf=0.0))
    return primary, secondary
