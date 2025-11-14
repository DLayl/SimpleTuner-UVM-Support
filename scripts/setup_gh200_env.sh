#!/usr/bin/env bash
# GH200 environment bootstrap for SimpleTuner.
#
# Source this script from your shell before launching training on the
# Grace Hopper system:
#   source scripts/setup_gh200_env.sh
#
# The defaults assume you are using the patched PyTorch build that ships
# with unified virtual memory support. Adjust values as needed for your
# workload.

set -euo pipefail

# Bail out quickly when CUDA is not present (useful if sourced by mistake on macOS).
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup_gh200_env] nvidia-smi not found; skipping GPU exports." >&2
  return 0 2>/dev/null || exit 0
fi

export SIMPLETUNER_GH200_ENABLE_UVM_HINTS=${SIMPLETUNER_GH200_ENABLE_UVM_HINTS:-1}

# UVM allocator knobs – tweak oversubscription ratio and access pattern to taste.
: "${GH200_UVM_OVERSUBSCRIPTION_RATIO:=5.0}"
: "${GH200_UVM_ACCESS_PATTERN:=gpu_first}"

export PYTORCH_CUDA_ALLOC_CONF="use_uvm:True,uvm_oversubscription_ratio:${GH200_UVM_OVERSUBSCRIPTION_RATIO},uvm_access_pattern:${GH200_UVM_ACCESS_PATTERN}"

# Keep accelerate consistent with GH200 defaults. Users can still override explicitly.
export ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION:-bf16}

cat <<EOF
[setup_gh200_env] GH200 exports applied:
  SIMPLETUNER_GH200_ENABLE_UVM_HINTS=${SIMPLETUNER_GH200_ENABLE_UVM_HINTS}
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}
  ACCELERATE_MIXED_PRECISION=${ACCELERATE_MIXED_PRECISION}

Override GH200_UVM_OVERSUBSCRIPTION_RATIO or GH200_UVM_ACCESS_PATTERN
before sourcing to fine-tune memory behaviour.
EOF
