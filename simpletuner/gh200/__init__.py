"""
GH200 support utilities for SimpleTuner.

This package exposes helpers that are only useful when running on
Grace-Hopper systems with the patched PyTorch build that enables
Unified Virtual Memory (UVM) features.
"""

from .uvm import (
    gh200_uvm_enabled,
    prefer_cpu_residency,
    prefer_gpu_residency,
    prefetch_to_device,
    optimize_batch_for_gh200,
    set_uvm_hint_override,
)
from .in_memory_backend import GH200InMemoryBackend

__all__ = [
    "gh200_uvm_enabled",
    "prefer_cpu_residency",
    "prefer_gpu_residency",
    "prefetch_to_device",
    "optimize_batch_for_gh200",
    "set_uvm_hint_override",
    "GH200InMemoryBackend",
]
