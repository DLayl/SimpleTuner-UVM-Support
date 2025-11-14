#!/usr/bin/env bash
# Runs complete GH200 + upstream validation suite
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
cd "$REPO_ROOT"

log() { printf "[run-all-tests] %s\n" "$*"; }

# Ensure individual scripts are executable
for script in test_upstream_features.sh test_gh200_features.sh test_combined_features.sh; do
  if [[ ! -x "scripts/$script" ]]; then
    log "ERROR: scripts/$script is not executable"
    log "Run: chmod +x scripts/$script"
    exit 1
  fi
done

log "Using Python: $PYTHON_BIN"
log "=== Running GH200 Validation Suite ==="
log ""

log "[1/3] Upstream features..."
PYTHON_BIN=$PYTHON_BIN ./scripts/test_upstream_features.sh || {
  log "ERROR: Upstream tests failed"
  exit 1
}

log ""
log "[2/3] GH200 features..."
PYTHON_BIN=$PYTHON_BIN ./scripts/test_gh200_features.sh || {
  log "ERROR: GH200 tests failed"
  exit 1
}

log ""
log "[3/3] Combined integration..."
PYTHON_BIN=$PYTHON_BIN ./scripts/test_combined_features.sh || {
  log "ERROR: Combined tests failed"
  exit 1
}

log ""
log "✅ All validation tests passed"
