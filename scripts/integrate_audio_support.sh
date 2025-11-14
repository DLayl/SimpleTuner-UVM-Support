#!/usr/bin/env bash
# Applies the audio support snippet into gh200/uvm.py if not already present.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TARGET="$REPO_ROOT/simpletuner/gh200/uvm.py"
SNIPPET="$REPO_ROOT/simpletuner/gh200/audio_support_snippet.py"

if grep -q "audio_latents" "$TARGET"; then
  echo "[audio-integrate] Audio keys already handled in $TARGET"
  exit 0
fi

if [[ ! -f "$SNIPPET" ]]; then
  echo "[audio-integrate] Snippet $SNIPPET not found" >&2
  exit 1
fi

echo "[audio-integrate] Integrating audio support into gh200/uvm.py"
python - "$TARGET" "$SNIPPET" <<'PY'
import re
import sys
from pathlib import Path

uvm_path = Path(sys.argv[1])
snippet_path = Path(sys.argv[2])
text = uvm_path.read_text()
if 'apply_audio_uvm_hints' in text:
    sys.exit(0)

helper = snippet_path.read_text().strip()
text = text.rstrip() + '\n\n' + helper + '\n'

pattern = r"(def optimize_batch_for_gh200\(batch:[^\n]+\) -> None:\n(?:    .+\n)+?    if not gh200_uvm_enabled\(\):\n\s+return\n)"
replacement = r"\1\n    apply_audio_uvm_hints(batch)\n"
new_text, count = re.subn(pattern, replacement, text, count=1)
if count == 0:
    raise SystemExit('Failed to insert audio hint call into optimize_batch_for_gh200; please update manually.')

uvm_path.write_text(new_text)
PY

echo "[audio-integrate] Done"
