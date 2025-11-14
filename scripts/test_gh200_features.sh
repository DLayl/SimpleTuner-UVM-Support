#!/usr/bin/env bash
# Validates GH200-specific helpers after patches are applied.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "$REPO_ROOT"

if [[ -f "$REPO_ROOT/.verify_env" ]]; then
  # shellcheck disable=SC1090
  source "$REPO_ROOT/.verify_env"
fi
IMPORT_PATH=${IMPORT_PATH:-gh200}
DIAG_SKIP_FLAG=${DIAGNOSTIC_SKIP_FLAG:-}
DIAG_OUTPUT_FLAG=${DIAGNOSTIC_OUTPUT_FLAG:-}

log() { printf "[gh200-test] %s\n" "$*"; }

if ! $PYTHON_BIN - <<'PY' >/dev/null 2>&1; then
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('torch') else 1)
PY
  log "torch not available; skipping GH200 feature test"
  exit 0
fi

log "1/4 Running gh200_diagnostic"
DIAG_CMD=($PYTHON_BIN gh200_diagnostic.py)
[[ -n "$DIAG_SKIP_FLAG" ]] && DIAG_CMD+=("$DIAG_SKIP_FLAG")
if [[ -n "$DIAG_OUTPUT_FLAG" ]]; then
  DIAG_CMD+=("$DIAG_OUTPUT_FLAG" "gh200_diagnostic_report.json")
fi
"${DIAG_CMD[@]}" >/tmp/gh200_diag.log
if [[ -n "$DIAG_OUTPUT_FLAG" ]]; then
  log "   Report written to gh200_diagnostic_report.json"
else
  log "   Diagnostic completed (no output flag available)"
fi

log "2/4 Validating UVM helper imports"
"$PYTHON_BIN" - <<'PY'
from simpletuner import gh200
assert hasattr(gh200, 'optimize_batch_for_gh200')
print("gh200 module exports verified")
PY

log "3/4 In-memory backend instantiation (dry run)"
"$PYTHON_BIN" - "$IMPORT_PATH" <<'PY'
import importlib
import sys
from pathlib import Path
from simpletuner.helpers.data_backend.base import BaseDataBackend
from tempfile import TemporaryDirectory
from PIL import Image
import numpy as np
module = importlib.import_module(sys.argv[1] + '.in_memory_backend')
Backend = getattr(module, 'GH200InMemoryBackend')
with TemporaryDirectory() as tmp:
    path = Path(tmp)
    img = Image.fromarray((np.ones((8, 8, 3)) * 127).astype('uint8'))
    img.save(path / 'sample.png')
    backend = Backend(accelerator=None, id='test', instance_data_dir=tmp)
    assert backend.list_files(['png'])
print("in-memory backend smoke test ok")
PY

log "4/4 UVM placement helpers"
"$PYTHON_BIN" - <<'PY'
import torch
from simpletuner.gh200 import set_uvm_hint_override, gh200_uvm_enabled
set_uvm_hint_override(True)
assert gh200_uvm_enabled() is True
set_uvm_hint_override(None)
print("uvm hint override toggles ok")
PY

log "GH200 feature tests completed"
