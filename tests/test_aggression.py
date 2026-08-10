"""The aggression curve, against python-audio-separator's own spec_utils.adjust_aggr.

The `is_non_accom_stem` branch is the one thing a caller cannot guess from the
graph: it flips the exponent to `1 - aggr` and is selected by the model's UVR
primary_stem. Getting it wrong produces a plausible-looking but wrong mask, so
both the formula and the per-model flag are pinned here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from audio_separator.separator.common_separator import CommonSeparator
from audio_separator.separator.uvr_lib_v5 import spec_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from driver.pipeline import MODEL_SPECS, adjust_aggression
from driver.vr_params import AGGR_SPLIT_BIN

AGGRESSION = 5.0


def random_mask(bins: int = 673, frames: int = 40) -> np.ndarray:
    return np.random.default_rng(3).random((2, bins, frames)).astype(np.float32)


def reference(mask: np.ndarray, is_non_accom_stem: bool, aggression: float = AGGRESSION) -> np.ndarray:
    return spec_utils.adjust_aggr(
        mask.copy(),
        is_non_accom_stem,
        {"value": aggression / 100, "split_bin": AGGR_SPLIT_BIN, "aggr_correction": None},
    )


@pytest.mark.parametrize("is_non_accom_stem", [False, True])
def test_matches_reference(is_non_accom_stem):
    mask = random_mask()
    mine = adjust_aggression(mask, AGGRESSION, is_non_accom_stem)
    assert np.max(np.abs(mine - reference(mask, is_non_accom_stem))) < 1e-6


def test_branches_actually_differ():
    """Guards against the flag being accepted and then ignored."""
    mask = random_mask()
    accom = adjust_aggression(mask, AGGRESSION, False)
    non_accom = adjust_aggression(mask, AGGRESSION, True)
    assert np.max(np.abs(accom - non_accom)) > 0.01


@pytest.mark.parametrize("is_non_accom_stem", [False, True])
def test_zero_aggression_is_identity(is_non_accom_stem):
    """aggr==0 short-circuits BEFORE the 1-aggr flip, as in the reference."""
    mask = random_mask()
    assert np.array_equal(adjust_aggression(mask, 0.0, is_non_accom_stem), mask)


def test_input_mask_not_mutated():
    mask = random_mask()
    original = mask.copy()
    adjust_aggression(mask, AGGRESSION, True)
    assert np.array_equal(mask, original)


@pytest.mark.parametrize("name", list(MODEL_SPECS))
def test_model_flag_matches_reference_stem_table(name):
    """is_non_accom_stem must equal `uvr_primary_stem in NON_ACCOM_STEMS`."""
    spec = MODEL_SPECS[name]
    expected = spec["uvr_primary_stem"] in CommonSeparator.NON_ACCOM_STEMS
    assert spec["is_non_accom_stem"] == expected
