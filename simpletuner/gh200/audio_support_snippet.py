"""Reference implementation for extending GH200 batch optimization to audio tensors."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .uvm import gh200_uvm_enabled, prefer_cpu_residency, prefetch_to_device

AUDIO_KEYS: Sequence[str] = (
    "audio_latents",
    "waveform",
    "audio_samples",
    "audio_features",
)

def apply_audio_uvm_hints(batch: Mapping[str, object]) -> None:
    """Give large audio tensors CPU residency hints when GH200 mode is active."""
    if not gh200_uvm_enabled():
        return
    for key in AUDIO_KEYS:
        value = batch.get(key)
        if torch.is_tensor(value) and value.is_cuda:
            prefer_cpu_residency(value)
            prefetch_to_device(value)
        elif isinstance(value, Mapping):
            for nested in value.values():
                if torch.is_tensor(nested) and nested.is_cuda:
                    prefer_cpu_residency(nested)
                    prefetch_to_device(nested)
