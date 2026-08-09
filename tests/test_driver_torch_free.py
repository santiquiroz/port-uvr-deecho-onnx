import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NORMAL_ONNX = REPO / "artifacts" / "UVR-De-Echo-Normal.onnx"

# Subprocess isolation: sibling test modules legitimately import librosa/torch in
# this process, so sys.modules here can't prove anything about driver/ itself.
IMPORT_CHECK = """
import sys
sys.path.insert(0, {repo!r})
import driver.pipeline
assert "torch" not in sys.modules, "driver imports torch"
assert "librosa" not in sys.modules, "driver imports librosa"
print("clean")
"""

SMOKE = """
import sys
sys.path.insert(0, {repo!r})
import numpy as np
import onnxruntime as ort
from driver.pipeline import DeEchoDriver
sess = ort.InferenceSession({onnx!r}, providers=["CPUExecutionProvider"])
driver = DeEchoDriver(lambda w: sess.run(None, {{"mag": w}})[0])
mix = np.random.default_rng(2).standard_normal((2, 44100)).astype(np.float32) * 0.2
dry, wet = driver.separate(mix)
assert dry.shape == wet.shape and dry.shape[0] == 2 and dry.shape[1] > 40000
assert np.isfinite(dry).all() and np.isfinite(wet).all()
assert "torch" not in sys.modules and "librosa" not in sys.modules
print("clean")
"""


def run_isolated(code: str) -> None:
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_driver_imports_no_heavy_deps():
    run_isolated(IMPORT_CHECK.format(repo=str(REPO)))


@pytest.mark.skipif(not NORMAL_ONNX.exists(), reason="run toolkit/export_deecho.py first")
def test_separate_smoke_torch_free():
    run_isolated(SMOKE.format(repo=str(REPO), onnx=str(NORMAL_ONNX)))
