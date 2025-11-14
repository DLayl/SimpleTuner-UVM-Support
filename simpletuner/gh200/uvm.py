"""
Unified Virtual Memory helpers for GH200 deployments.

These utilities are intentionally defensive: they only perform runtime
operations when both CUDA is available *and* the environment variable
``SIMPLETUNER_GH200_ENABLE_UVM_HINTS`` is set to a truthy value.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Mapping, Optional

import torch

logger = logging.getLogger("GH200UVM")

_UVM_HINT_OVERRIDE: Optional[bool] = None


def gh200_uvm_enabled() -> bool:
    """
    Return ``True`` when GH200 specific behaviour should be enabled.

    Users must opt in explicitly to avoid touching CUDA runtime symbols on
    machines that do not ship the patched PyTorch build.
    """

    if _UVM_HINT_OVERRIDE is not None:
        return _UVM_HINT_OVERRIDE
    flag = os.environ.get("SIMPLETUNER_GH200_ENABLE_UVM_HINTS", "")
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def set_uvm_hint_override(value: Optional[bool]) -> None:
    """
    Override the GH200 hint flag programmatically.

    Passing ``None`` clears the override and restores environment-based behaviour.
    """

    global _UVM_HINT_OVERRIDE
    _UVM_HINT_OVERRIDE = value


def _element_span_bytes(tensor: torch.Tensor) -> int:
    try:
        storage = tensor.untyped_storage()
        nbytes = storage.nbytes()
        if nbytes:
            return int(nbytes)
    except Exception:
        pass

    try:
        return int(tensor.element_size() * tensor.numel())
    except Exception:
        return 0


def _cudart():
    if not hasattr(torch.cuda, "cudart"):
        return None
    try:
        return torch.cuda.cudart()
    except Exception:
        return None


def _current_cuda_device() -> int:
    try:
        return int(torch.cuda.current_device())
    except Exception:
        return 0


def _apply_preferred_location(tensor: torch.Tensor, *, prefer_cpu: bool) -> bool:
    if not gh200_uvm_enabled():
        return False
    if not torch.cuda.is_available() or not tensor.is_cuda:
        return False

    try:
        if hasattr(torch.cuda, "memory_advise"):
            location = "cpu" if prefer_cpu else _current_cuda_device()
            torch.cuda.memory_advise(tensor, "set_preferred_location", device=location)
            torch.cuda.memory_advise(tensor, "set_accessed_by", device=_current_cuda_device())
            return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("torch.cuda.memory_advise failed: %s", exc)

    cudart = _cudart()
    if cudart is None:
        return False

    size = _element_span_bytes(tensor)
    if size <= 0:
        return False

    try:
        ptr = tensor.data_ptr()
    except Exception:
        try:
            ptr = tensor.storage().data_ptr()
        except Exception:
            return False

    try:
        advise = cudart.cudaMemAdvise
        preferred_location = cudart.cudaMemAdviseSetPreferredLocation
        accessed_by = cudart.cudaMemAdviseSetAccessedBy
        cpu_id = getattr(cudart, "cudaCpuDeviceId", -1)

        target = cpu_id if prefer_cpu else _current_cuda_device()
        advise(ptr, size, preferred_location, target)
        advise(ptr, size, accessed_by, _current_cuda_device())
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("cudaMemAdvise failed: %s", exc)
        return False


def prefer_cpu_residency(tensor: torch.Tensor) -> bool:
    """Hint the runtime to keep this CUDA tensor resident in CPU memory."""
    return _apply_preferred_location(tensor, prefer_cpu=True)


def prefer_gpu_residency(tensor: torch.Tensor) -> bool:
    """Hint the runtime to keep this CUDA tensor resident in GPU memory."""
    return _apply_preferred_location(tensor, prefer_cpu=False)


def prefetch_to_device(tensor: torch.Tensor, device: int | None = None) -> bool:
    """
    Attempt to prefetch a managed tensor to ``device`` asynchronously.

    Returns ``True`` on success, ``False`` otherwise.
    """

    if not gh200_uvm_enabled():
        return False
    if not torch.cuda.is_available() or not tensor.is_cuda:
        return False

    device = _current_cuda_device() if device is None else int(device)
    cudart = _cudart()
    if cudart is None:
        return False

    size = _element_span_bytes(tensor)
    if size <= 0:
        return False

    try:
        ptr = tensor.data_ptr()
    except Exception:
        try:
            ptr = tensor.storage().data_ptr()
        except Exception:
            return False

    try:
        cudart.cudaMemPrefetchAsync(ptr, size, device)
        return True
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("cudaMemPrefetchAsync failed: %s", exc)
        return False


def _iter_tensors(obj) -> Iterable[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, Mapping):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)


def optimize_batch_for_gh200(batch: Mapping[str, object]) -> None:
    """
    Apply heuristic UVM placement for common diffusion training tensors.

    Current policy:
      * Latent tensors -> prefer CPU residency (Grace RAM).
      * Text encoder outputs -> prefer GPU residency.
    """

    if not gh200_uvm_enabled():
        return

    latents = batch.get("latents")
    for tensor in _iter_tensors(latents):
        prefer_cpu_residency(tensor)

    for key in ("encoder_hidden_states", "text_embeddings", "pooled_prompt_embeds"):
        value = batch.get(key)
        for tensor in _iter_tensors(value):
            prefer_gpu_residency(tensor)
            prefetch_to_device(tensor)
