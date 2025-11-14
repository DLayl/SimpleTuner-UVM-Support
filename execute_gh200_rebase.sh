#!/usr/bin/env bash
# Orchestrates GH200 rebase workflow described in GH200_UPSTREAM_MERGE_GUIDE_V2.md
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PATCH_DIR="$REPO_ROOT/patches"
CHECKLIST="$REPO_ROOT/GH200_REBASE_CHECKLIST.md"
UPSTREAM_BRANCH="origin/main"
REBASE_BRANCH="gh200-reintegration"
BACKUP_BRANCH="gh200-backup-$(date +%Y%m%d_%H%M%S)"

usage() {
  cat <<USAGE
Usage: $0 [--rollback] [--skip-tests]

Default flow:
  1. Pre-flight checks
  2. Backup current branch
  3. Create $REBASE_BRANCH from $UPSTREAM_BRANCH
  4. Sequentially attempt to apply component patches (01..07)
  5. Run validation scripts unless --skip-tests specified

Options:
  --rollback    Reset working tree to last backup branch listed in MERGE_LOG.txt
  --skip-tests  Skip scripted validation runs (manual testing still recommended)
USAGE
}

rollback() {
  if [[ ! -f MERGE_LOG.txt ]]; then
    echo "MERGE_LOG.txt not found; cannot rollback" >&2
    exit 1
  fi
  target=$(awk -F: '/Backup branch/ {print $2}' MERGE_LOG.txt | tail -n1 | xargs)
  if [[ -z "$target" ]]; then
    echo "No backup branch recorded" >&2
    exit 1
  fi
  read -rp "Reset to $target? (y/N) " ans
  [[ $ans == y || $ans == Y ]] || exit 0
  git reset --hard "$target"
  echo "Rolled back to $target"
  exit 0
}

run_tests=1
for arg in "$@"; do
  case "$arg" in
    --rollback)
      rollback
      ;;
    --skip-tests)
      run_tests=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
  esac
  shift || true
done

cd "$REPO_ROOT"

preflight() {
  echo "[1/7] Pre-flight checks"
  git status -sb
  for patch in "$PATCH_DIR"/*.patch; do
    [[ -f "$patch" ]] || { echo "Patch $patch missing" >&2; exit 1; }
  done
  if [[ -x "$REPO_ROOT/verify_assumptions.sh" ]]; then
    echo "Running verify_assumptions.sh..."
    "$REPO_ROOT/verify_assumptions.sh"
  else
    echo "verify_assumptions.sh not found; please run Phase 0 verification manually." >&2
    exit 1
  fi
  echo "All patches present and verification complete."
}

create_backup_branch() {
  echo "[2/7] Creating backup branch $BACKUP_BRANCH"
  current_branch=$(git rev-parse --abbrev-ref HEAD)
  git branch "$BACKUP_BRANCH"
  echo "Backup branch: $BACKUP_BRANCH" >> MERGE_LOG.txt
  echo "Starting branch: $current_branch" >> MERGE_LOG.txt
}

checkout_upstream_branch() {
  echo "[3/7] Resetting to upstream baseline ($UPSTREAM_BRANCH)"
  git fetch origin main
  git checkout -B "$REBASE_BRANCH" "$UPSTREAM_BRANCH"
}

apply_patch_sequence() {
  echo "[4/7] Applying component patches"
  declare -a PATCHES=(
    "01-trainer-gh200.patch"
    "02-caching-uvm.patch"
    "03-state-tracker.patch"
    "04-configuration.patch"
    "05-data-backend.patch"
    "06-gh200-module.patch"
    "07-scripts-docs.patch"
  )
  mkdir -p logs
  for file in "${PATCHES[@]}"; do
    echo "  -> Applying $file"
    if git apply --3way "$PATCH_DIR/$file" >"logs/$file.log" 2>&1; then
      echo "     Applied cleanly"
    else
      echo "     ⚠️  git apply reported conflicts. See logs/$file.log"
      echo "     Please open the patch and manually port changes before continuing."
      read -rp "Mark $file as manually merged? (y/N) " ans
      [[ $ans == y || $ans == Y ]] || { echo "Aborting."; exit 1; }
      if ! git diff --check; then
        echo "     Unresolved conflict markers remain; fix before continuing." >&2
        git diff --check
        exit 1
      fi
    fi
    py_targets=$(grep '^\+\+\+ ' "$PATCH_DIR/$file" | awk '{print $2}' | sed 's|b/||' | grep -E '\.py$' || true)
    if [[ -n "$py_targets" ]]; then
      while IFS= read -r pyf; do
        [[ -z "$pyf" ]] && continue
        if [[ -f "$pyf" ]]; then
          $PYTHON_BIN -m py_compile "$pyf" || { echo "Python syntax error in $pyf after applying $file" >&2; exit 1; }
        fi
      done <<< "$py_targets"
    fi
  done
}

run_validation() {
  echo "[5/7] Running validation scripts"
  scripts/test_upstream_features.sh || { echo "Upstream test failed"; exit 1; }
  scripts/test_gh200_features.sh || { echo "GH200 test failed"; exit 1; }
  scripts/test_combined_features.sh || { echo "Combined test failed"; exit 1; }
}

finalize() {
  echo "[7/7] Updating checklist"
  if [[ -f "$CHECKLIST" ]]; then
    printf '\n- %s Completed automated rebase steps.\n' "$(date -Iseconds)" >> "$CHECKLIST"
  fi
  git status -sb
  echo "Done. Review results, run extended tests, and commit when satisfied."
}

preflight
create_backup_branch
checkout_upstream_branch
apply_patch_sequence
if [[ $run_tests -eq 1 ]]; then
  run_validation
else
  echo "Tests skipped (per flag)."
fi
finalize
