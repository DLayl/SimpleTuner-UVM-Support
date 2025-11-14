#!/usr/bin/env bash
# Exercises combined upstream + GH200 behaviours (attention + manual validation hooks + audio batch hints).
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.verify_env" ]]; then
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.verify_env"
fi
AUDIO_INTEGRATION=${AUDIO_INTEGRATION:-defer}

log() { printf "[combined-test] %s\n" "$*"; }

if ! $PYTHON_BIN - <<'PY' >/dev/null 2>&1; then
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('torch') else 1)
PY
  log "torch not available; skipping combined tests"
  exit 0
fi

log "1/3 Attention backend + GH200 env"
"$PYTHON_BIN" - <<'PY'
import os
from types import SimpleNamespace
from simpletuner.gh200 import set_uvm_hint_override
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
set_uvm_hint_override(True)
cfg = SimpleNamespace(attention_mechanism="flash_attn_3")
AttentionBackendController.apply(cfg, AttentionPhase.TRAIN)
AttentionBackendController.restore_default()
set_uvm_hint_override(None)
print("attention+gh200 ok")
PY

log "2/3 Manual validation + GA ramp plumbing present"
"$PYTHON_BIN" - <<'PY'
from simpletuner.helpers.training import trainer
assert hasattr(trainer.Trainer, '_configure_gh200_runtime')
assert hasattr(trainer.Trainer, '_apply_dynamic_gradient_accumulation')
print("trainer GH200 hooks detected")
PY

if [[ $AUDIO_INTEGRATION == pre-integrate ]]; then
  log "3/3 Batch optimization w/ audio tensors"
else
  log "3/3 (skipped) Audio tensors not integrated yet"
fi
if [[ $AUDIO_INTEGRATION == pre-integrate ]]; then
SIMPLETUNER_GH200_ENABLE_UVM_HINTS=1 "$PYTHON_BIN" - <<'PY'
import torch
from simpletuner.gh200 import optimize_batch_for_gh200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch = {
    'latents': torch.randn(2, 4, 64, 64, device=device),
    'encoder_hidden_states': torch.randn(2, 77, 1024, device=device),
    'audio_latents': torch.randn(2, 16000, device=device)
}
optimize_batch_for_gh200(batch)
print("optimize_batch_for_gh200 handled audio tensors")
PY
fi

log "Combined tests completed"
