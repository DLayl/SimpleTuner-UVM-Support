#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
REPORT="$REPO_ROOT/VERIFICATION_REPORT.txt"
PATCH_MANIFEST="$REPO_ROOT/PATCH_MANIFEST.md"
GUIDE="$REPO_ROOT/GH200_UPSTREAM_MERGE_GUIDE_V2.md"
VERIFY_ENV="$REPO_ROOT/.verify_env"

log_msg() { printf "[verify] %s\n" "$*"; }
fail() { echo "[verify] ✗ $*" >&2; exit 1; }

log_msg "Starting Phase 0 verification"
: > "$REPORT"

record() {
  printf "%s %s\n" "$1" "$2" | tee -a "$REPORT"
}

# 1. Import path verification
log_msg "1/5 Checking GH200 import path"
IMPORT_PATH=""
if [[ -f "$REPO_ROOT/gh200/in_memory_backend.py" ]]; then
  IMPORT_PATH="gh200"
elif [[ -f "$REPO_ROOT/simpletuner/gh200/in_memory_backend.py" ]]; then
  IMPORT_PATH="simpletuner.gh200"
else
  fail "Unable to locate gh200 package"
fi
record "IMPORT_PATH:" "$IMPORT_PATH"
log_msg "   -> Using import path: $IMPORT_PATH"

# 2. Upstream API verification
log_msg "2/5 Verifying upstream APIs"
ORIGIN_MAIN=$(git -C "$REPO_ROOT" rev-parse origin/main)
tmp_trainer=$(mktemp)
tmp_attention=$(mktemp)
git -C "$REPO_ROOT" show "$ORIGIN_MAIN":simpletuner/helpers/training/trainer.py > "$tmp_trainer" || fail "Cannot read trainer.py from origin/main"
git -C "$REPO_ROOT" show "$ORIGIN_MAIN":simpletuner/helpers/training/attention_backend.py > "$tmp_attention" || fail "Cannot read attention_backend.py from origin/main"
"$PYTHON_BIN" - <<PY "$tmp_trainer" "$tmp_attention"
import ast, sys
trainer_path, attention_path = sys.argv[1], sys.argv[2]

with open(trainer_path) as fh:
    trainer_tree = ast.parse(fh.read(), trainer_path)
trainer_methods = set()
for node in trainer_tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'Trainer':
        trainer_methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
        break
missing = {"register_manual_validation_trigger", "_consume_manual_validation_request"} - trainer_methods
if missing:
    raise SystemExit(f"Trainer missing methods: {sorted(missing)}")

with open(attention_path) as fh:
    attention_tree = ast.parse(fh.read(), attention_path)
controller_methods = set()
for node in attention_tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'AttentionBackendController':
        controller_methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
        break
if 'apply' not in controller_methods or 'restore_default' not in controller_methods:
    raise SystemExit('AttentionBackendController missing apply/restore_default')
PY
rm -f "$tmp_trainer" "$tmp_attention"
record "UPSTREAM_APIS:" "present"
log_msg "   -> Upstream APIs verified"

# 3. Diagnostic flags
log_msg "3/5 Checking gh200_diagnostic.py flags"
if grep -q -- '--skip-allocation' "$REPO_ROOT/gh200_diagnostic.py"; then
  DIAG_SKIP_FLAG="--skip-allocation"
else
  DIAG_SKIP_FLAG=""
fi
if grep -q -- '--output' "$REPO_ROOT/gh200_diagnostic.py"; then
  DIAG_OUTPUT_FLAG="--output"
else
  DIAG_OUTPUT_FLAG=""
fi
record "DIAGNOSTIC_SKIP_FLAG:" "${DIAG_SKIP_FLAG:-<none>}"
record "DIAGNOSTIC_OUTPUT_FLAG:" "${DIAG_OUTPUT_FLAG:-<none>}"

# 4. Audio integration decision
log_msg "4/5 Audio integration decision"
read -rp "Integrate audio UVM hints now? (y/N) " AUDIO_DECISION
AUDIO_DECISION_LOWER=$(printf '%s' "$AUDIO_DECISION" | tr '[:upper:]' '[:lower:]')
if [[ $AUDIO_DECISION_LOWER == y ]]; then
  AUDIO_STATUS="pre-integrate"
else
  AUDIO_STATUS="defer"
fi
record "AUDIO_INTEGRATION:" "$AUDIO_STATUS"

# 5. Manifest/guide regeneration
log_msg "5/5 Regenerating manifest metadata"
MERGE_BASE=$(git -C "$REPO_ROOT" merge-base HEAD origin/main)
TIMESTAMP=$(date -Iseconds)
record "MERGE_BASE:" "$MERGE_BASE"
record "GENERATED_AT:" "$TIMESTAMP"

cat > "$PATCH_MANIFEST" <<MANIFEST
# GH200 Patch Manifest

Merge base: $MERGE_BASE
Generated: $TIMESTAMP (via verify_assumptions.sh)

| Patch | Scope | Delta (ins/del) | Risk | Notes |
| --- | --- | --- | --- | --- |
| patches/01-trainer-gh200.patch | simpletuner/helpers/training/trainer.py | +125 / -48 | High | GH200 gradient ramp + runtime hooks overlapping upstream attention/manual validation logic. |
| patches/02-caching-uvm.patch | helpers/caching/vae.py; helpers/caching/text_embeds.py | +72 / -6 | Medium | Adds UVM placement helpers; ensure compatibility with new caching/audio flows. |
| patches/03-state-tracker.patch | helpers/training/state_tracker.py | +18 / -0 | Low | Raw config & GH200 runtime bookkeeping. |
| patches/04-configuration.patch | helpers/configuration/json_file.py; loader.py | +4 / -0 | Low | Propagates raw config into StateTracker. |
| patches/05-data-backend.patch | helpers/data_backend/factory.py + builders | +131 / -65 | High | Introduces in-memory backend; overlaps upstream audio/dataset logic. |
| patches/06-gh200-module.patch | gh200 package | +521 / -0 | Medium | UVM utilities + in-memory backend implementation. |
| patches/07-scripts-docs.patch | GH200 docs + scripts | +6114 / -0 | Low-Medium | Documentation + diagnostics. |

## Complexity Notes
- Total delta ≈6.9k insertions / 119 deletions across 7 components.
- Highest-risk merges: trainer + data_backend.
- Testing priority: manual validation + attention backend with GH200 ramp, audio datasets, UVM diagnostics.
MANIFEST

if [[ -f "$GUIDE" ]]; then
  perl -0pi -e "s/(\*\*Last Updated:\*\*) .*$/\1 $TIMESTAMP  /" "$GUIDE"
fi

cat >> "$REPORT" <<EOF
IMPORT PATH VERIFIED: $IMPORT_PATH
UPSTREAM APIs: present
Diagnostic flags: skip=${DIAG_SKIP_FLAG:-none}, output=${DIAG_OUTPUT_FLAG:-none}
Audio integration: $AUDIO_STATUS
Merge base: $MERGE_BASE
Generated: $TIMESTAMP
EOF

cat > "$VERIFY_ENV" <<ENV
# Auto-generated by verify_assumptions.sh on $TIMESTAMP
# DO NOT EDIT MANUALLY. Re-run verify_assumptions.sh to refresh values.
IMPORT_PATH=$IMPORT_PATH
DIAGNOSTIC_SKIP_FLAG=$DIAG_SKIP_FLAG
DIAGNOSTIC_OUTPUT_FLAG=$DIAG_OUTPUT_FLAG
AUDIO_INTEGRATION=$AUDIO_STATUS
ENV

log_msg "Verification complete. Report written to $REPORT"
