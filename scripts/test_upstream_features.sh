#!/usr/bin/env bash
# Verifies upstream-only capabilities before GH200 patches are applied.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "$REPO_ROOT"

log() { printf "[upstream-test] %s\n" "$*"; }

if ! $PYTHON_BIN - <<'PY' >/dev/null 2>&1; then
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('torch') else 1)
PY
  log "torch not available; skipping upstream feature test"
  exit 0
fi

log "1/3 Attention backend smoke test"
$PYTHON_BIN - <<'PY'
from types import SimpleNamespace
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
cfg = SimpleNamespace(attention_mechanism="flash_attn_3")
AttentionBackendController.apply(cfg, AttentionPhase.TRAIN)
AttentionBackendController.restore_default()
print("attention backend apply/restore succeeded")
PY

log "2/3 Manual validation API presence check"
$PYTHON_BIN - <<'PY'
from inspect import signature
from simpletuner.helpers.training import trainer
required = ["register_manual_validation_trigger", "_consume_manual_validation_request"]
missing = [name for name in required if not hasattr(trainer.Trainer, name)]
if missing:
    raise SystemExit(f"Missing manual validation hooks: {missing}")
print("manual validation hooks detected")
PY

log "3/3 Audio loader sanity"
$PYTHON_BIN - <<'PY'
import io, wave, numpy as np
from simpletuner.helpers.audio import load_audio
sr = 16000
data = (0.1 * np.sin(2 * np.pi * np.arange(sr) * 440 / sr)).astype(np.float32)
byte_buf = io.BytesIO()
with wave.open(byte_buf, 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sr)
    wav.writeframes((data * 32767).astype('<i2').tobytes())
byte_buf.seek(0)
waveform, sample_rate = load_audio(byte_buf)
assert waveform.shape[1] == sr, waveform.shape
assert sample_rate == sr
print("audio loader ok")
PY

log "All upstream feature checks passed"
