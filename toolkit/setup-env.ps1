# Crea el venv CPU-only del toolkit (py3.11): torch CPU + audio-separator (golden) +
# onnxruntime-directml (validacion/bench). Uso: pwsh -File toolkit/setup-env.ps1
$ErrorActionPreference = 'Continue'
$repo = Split-Path $PSScriptRoot -Parent
$venv = Join-Path $repo '.venv'

if (-not (Test-Path $venv)) {
    py -3.11 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}
$python = Join-Path $venv 'Scripts\python.exe'

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }

& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
if ($LASTEXITCODE -ne 0) { throw "requirements install failed" }

# audio-separator pulls plain onnxruntime; swap it for the DirectML build so the
# same venv can run the DML validation/bench. Same package namespace, so the plain
# wheel must go first.
& $python -m pip uninstall -y onnxruntime
& $python -m pip install onnxruntime-directml==1.24.4
if ($LASTEXITCODE -ne 0) { throw "onnxruntime-directml install failed" }

& $python -c "import torch; assert not torch.cuda.is_available(); print('torch', torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw "torch import failed (or unexpectedly CUDA-enabled)" }

& $python -c "import onnxruntime as ort; assert 'DmlExecutionProvider' in ort.get_available_providers(), ort.get_available_providers(); print('ort', ort.__version__, ort.get_available_providers())"
if ($LASTEXITCODE -ne 0) { throw "onnxruntime DML check failed" }

& $python -c "import audio_separator, librosa, samplerate, soundfile, scipy, numpy; print('golden deps OK')"
if ($LASTEXITCODE -ne 0) { throw "golden dependency import failed" }

Write-Host 'Environment ready.'
