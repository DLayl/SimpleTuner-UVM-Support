# GH200 Optimization Guide

This document explains how the GH200-specific features in SimpleTuner work and how to use them.

## 1. What this integration provides

GH200 support introduces three major capabilities:

1. Unified Memory caches and batching. A new `simpletuner.gh200` module contains utilities that opt tensors into CUDA Unified Virtual Memory (UVM). VAE latent caches and text embedding caches now persist tensors in managed memory, hinting whether data should live in Grace DDR5 or Hopper HBM. The trainer also prefetches cached data back to the active GPU stream to avoid runtime faults.

2. In-memory data backend. `type: "in_memory"` backends stage entire datasets in Grace RAM at startup, enabling zero disk I/O during training while leveraging the 72 Grace CPU cores for ingestion. Compression and memory ceilings are configurable per dataset.

3. Dynamic GH200 optimizations through configuration. Training JSON files provide `cache_config` (with `*_cache_type: "unified_memory"`) and `gh200_optimizations` blocks to control UVM hints and gradient accumulation ramp-up directly, without relying solely on environment variables.

## 2. Code changes overview

### 2.1 GH200 module

Located in `simpletuner/gh200/`:

- `uvm.py` exposes `gh200_uvm_enabled()`, placement helpers (`prefer_cpu_residency`, `prefer_gpu_residency`, `prefetch_to_device`), unified batch optimization (`optimize_batch_for_gh200`), and `set_uvm_hint_override()` allowing the trainer to override env-based behaviour.
- `in_memory_backend.py` implements the `GH200InMemoryBackend`, an eager loader that reads and optionally compresses datasets into memory via parallel workers.
- `__init__.py` re-exports the GH200 hooks so other modules can import from `simpletuner.gh200`.

### 2.2 Cache integration

- `helpers/caching/vae.py` and `helpers/caching/text_embeds.py` call `_prepare_uvm_tensor`/`_prepare_text_embed_tensor` when storing, loading, or cloning cached tensors, so cached latents/text embeds reside in UVM. Prefetch hooks bring cached tensors back to the GPU before use, preventing on-demand page faults.
- The caches respond to the GH200 overrides: setting the flag to false returns to standard CPU-backed caches.

### 2.3 In-memory backend wiring

- `helpers/data_backend/builders/in_memory.py` registers the GH200 backend.
- `helpers/data_backend/builders/__init__.py` adds `"in_memory"` to the builder map.
- `helpers/data_backend/factory.py` instantiates `GH200InMemoryBackend` when `type: "in_memory"`, returning the dataset and metadata backend through the existing workflow.

### 2.4 Configuration plumbing

- `helpers/configuration/json_file.py` and `loader.py` now stash the raw configuration into `StateTracker` so GH200 features can inspect the original JSON.
- `StateTracker` gains storage for `raw_config` and `gh200_runtime_config`.
- `Trainer._configure_gh200_runtime()` reads `cache_config` and `gh200_optimizations` from the raw JSON, applies UVM overrides via `set_uvm_hint_override`, and prepares a gradient accumulation scheduler when requested. `StateTracker.set_gh200_runtime_config()` stores runtime metadata for inspection.

### 2.5 Trainer adjustments

- `Trainer` maintains a `GH200GradientRamp` instance that adjusts `gradient_accumulation_steps` on each micro-batch (before `Accelerator.accumulate`). This matches the block in `gh200_optimizations`.
- `optimize_batch_for_gh200(prepared_batch)` is still invoked for each batch to apply per-field UVM hints.

### 2.6 Environment helpers and docs

- `scripts/setup_gh200_env.sh` exports GH200 defaults for manual `bash` users.
- `gh200_diagnostic.py` allocates beyond HBM, checks environment state, and prints a summary.
- `documentation/GH200_WORKFLOW.md` provides quick-start instructions (now complemented by this guide).

## 3. Configuration schema

These keys can be added directly to a JSON config:

```json
"cache_config": {
  "vae_cache_type": "unified_memory",
  "text_cache_type": "unified_memory",
  "cache_dir": "cache/gh200",
  "persistent_cache": true
},
"gh200_optimizations": {
  "enable_uvm_hints": true,
  "enable_batch_size_rampup": true,
  "initial_effective_batch_size": 4,
  "target_effective_batch_size": 32,
  "batch_rampup_steps": 400
}
```

* `vae_cache_type` / `text_cache_type`: When set to `"unified_memory"`, caches opt tensors into GH200 UVM (even if `enable_uvm_hints` isn't explicitly set).
* `enable_uvm_hints`: Force UVM hints on/off, overriding env defaults.
* `enable_batch_size_rampup`: Enables gradient accumulation ramping.
  * `initial_effective_batch_size`: Starting effective batch size.
  * `target_effective_batch_size`: Desired size by the end; the ramp computes accumulation steps internally.
  * `batch_rampup_steps`: Micro-batch count over which the ramp progresses.

### In-memory backend

Example dataset entry:

```json
{
  "id": "train_images",
  "type": "in_memory",
  "dataset_type": "image",
  "instance_data_dir": "/workspace/DLAY-1024/subject",
  "in_memory": {
    "compression": "lz4",
    "num_workers": 64,
    "memory_limit_gb": 200
  }
}
```

* `compression`: `lz4`, `zlib`, `snappy`, or omit for raw storage.
* `num_workers`: Parallel loader threads; the backend defaults to `min(72, os.cpu_count())` if unset.
* `memory_limit_gb`: Optional guard to raise if the estimated dataset would exceed the limit.

## 4. Execution workflow

1. **Environment prep** (optional): `source scripts/setup_gh200_env.sh` to pick up defaults.
2. **Diagnostics**: Run `python gh200_diagnostic.py --oversubscription-scale 1.5` once per session to verify UVM behaviour.
3. **Configuration**: Ensure the training JSON uses `type: "in_memory"` and/or the GH200 blocks above. FSDP via Accelerate is optional; LoRA/LyCORIS runs typically use single-process bf16.
4. **Launch**: `accelerate launch --config_file config/accelerate_config_gh200.yaml simpletuner/train.py --config config/your_config.json`
5. **Monitoring**: Check the log for entries like `GH200 gradient accumulation ramp enabled` and `DEBUG: GH200 ramp adjusted gradient_accumulation_steps -> …`, plus the standard progress bar.

## 5. Best practices

- Keep `train_batch_size = 1` for Qwen edit pipelines; drive effective batch size through `gradient_accumulation_steps` and the GH200 ramp.
- While the caches prefer CPU (Grace) vs GPU (HBM) memory, `prefetch_to_device` runs before each batch to minimise page faults. You can examine managed memory behaviour with `torch.cuda.memory_stats()`.
- Remember to clean up large caches (e.g., `cache/gh200_unified`) if you run multiple experiments; use `StateTracker.delete_cache_files()` or delete the cache directory manually.
- Adjust `memory_limit_gb` on in-memory backends to maintain headroom if you run other workloads on the node.

## 6. Future improvements

Potential enhancements include exposing telemetry (periodic memory stats), surfacing GH200 options in the WebUI, and adding adaptive prefetch tuning. The current implementation provides all core functionality needed for single-node GH200 training.

