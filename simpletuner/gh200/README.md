# GH200 Module

UVM-optimized components for NVIDIA GH200 Grace Hopper systems.

## Components

- `uvm.py` – Unified Virtual Memory placement helpers (prefer_cpu_residency, prefer_gpu_residency, prefetch_to_device, optimize_batch_for_gh200).
- `in_memory_backend.py` – Grace RAM dataset backend for zero-disk-I/O training.
- `audio_support_snippet.py` – Reference implementation for audio tensor optimization.

## Usage

Configure GH200 features via the `gh200_optimizations` block in your training config:

```json
{
  "gh200_optimizations": {
    "enable_uvm_hints": true,
    "enable_batch_size_rampup": true,
    "batch_rampup_steps": 1000,
    "initial_effective_batch_size": 4,
    "target_effective_batch_size": 64
  }
}
```

The trainer automatically calls `_configure_gh200_runtime()` to set UVM hints and gradient accumulation ramping based on this configuration.

See the parent directory guide `GH200_UPSTREAM_MERGE_GUIDE_V2.md` for the full integration workflow.

## Documentation

### Current Guides
- `../GH200_UPSTREAM_MERGE_GUIDE_V2.md` – Rebase workflow and integration guidance.
- `../GH200_REBASE_CHECKLIST.md` – Validation checklist and testing procedures.
- `../docs/GH200_docs/GH200_PATCH_MANIFEST.md` – Patch inventory with delta metrics.

### Legacy Documentation
Executive summaries live in `../docs/`:
- `GH200_OPTIMIZATION_SUMMARY.md` – Key UVM optimization strategies (60-line summary).
- `GH200_MERGE_SUMMARY.md` – Original merge approach, superseded by V2 (60-line summary).

Full versions (3,684 and 1,980 lines) remain available via git history. To recover them:
```bash
git show HEAD:GH200_OPTIMIZATION_GUIDE_FINAL_V2.md > GH200_OPTIMIZATION_GUIDE_FULL.md
git show HEAD:GH200_UPSTREAM_MERGE_GUIDE.md > GH200_UPSTREAM_MERGE_GUIDE_FULL.md
```
These comprehensive guides document the optimization strategies and decision-making process that led to the current implementation.
