# GH200 SimpleTuner Rebase + Reintegration Guide (v2)

**Document Version:** 2.0  
**Last Updated:** 2025-11-14T11:56:15-05:00  
**Scope:** Full rebase of GH200 fork on top of upstream SimpleTuner `origin/main` (post-2025-10-30) with preservation of all GH200 optimizations.

---

## 1. Executive Summary

### 1.1 Why Rebase Beats Merge Now

| Metric | Upstream (since 2025-10-30) | GH200 Delta | Observation |
| --- | --- | --- | --- |
| Files touched | 209 | 12 | Upstream dwarfs fork footprint |
| Insertions | 21,880 | ~6,900 | 27× more upstream change |
| Architectural shifts | Attention backend dispatcher, audio pipeline, manual validation APIs | None | Fork lags large frameworks |

**Conclusion:** Treat GH200 as a feature extension atop a rapidly evolving upstream. Start from the complete upstream codebase, then layer GH200 patches—never the other way around.

### 1.2 Feature Preservation Rationale
- **Zero feature loss:** rebasing ensures every upstream feature (manual validation, audio loader, attention backends, ROCm knobs) is present before GH200 hooks are applied.
- **Deterministic verification:** each GH200 component is reapplied via curated patches (Section 3) so you can track exactly what changed.
- **Repeatable process:** the rebase workflow can be rerun whenever upstream ships a new batch of commits.

---

## 2. Retained Foundations from v1

### 2.1 Pre-Flight Backup Procedures
1. `git status` → ensure only expected GH200 files modified.
2. Snapshot branch: `git branch gh200-backup-$(date +%Y%m%d_%H%M%S)`.
3. Export safety patch: `git diff HEAD > gh200-current.patch`.
4. Generate rollback script (see Section 8.3).

### 2.2 Performance Baseline Capture
- Launch short GH200 job (1k steps) capturing runtime, throughput, memory stats, diagnostics log.
- Store under `baseline/` with `baseline_report.txt`, `nvidia_smi.log`, optional `nsys` traces.

### 2.3 Testing Methodology (4 Levels)
1. **Unit:** targeted scripts (caching, backend builders, GH200 helpers).
2. **Integration:** Accelerate dry runs, dataset staging, manual validation API calls.
3. **System:** full training session with monitoring (nvidia-smi, logs, webhooks).
4. **Performance:** compare before/after metrics to catch regressions.

### 2.4 Rollback Procedures
- `./ROLLBACK.sh` resets repo to backup branch.
- Keep `gh200-current.patch` for manual reapplication if needed.

---

## 3. Phase 1 – Patch Extraction (Reference)

Artifacts live in `patches/` (see `docs/GH200_PATCH_MANIFEST.md`). Use them as human-readable checklists, not `git apply` blindly.

| Patch | Scope | Purpose |
| --- | --- | --- |
| 01 | `trainer.py` | GH200GradientRamp, per-batch optimize call, runtime config |
| 02 | `vae.py`, `text_embeds.py` | UVM placement hooks for caches |
| 03 | `state_tracker.py` | Raw config & GH200 runtime storage |
| 04 | Config loaders | Propagate raw config into StateTracker |
| 05 | Data backend + builders | GH200 in-memory backend wiring |
| 06 | `gh200/` package | UVM helpers and backend implementation |
| 07 | Scripts/docs | Operator guides, diagnostics, env setup |

---

## 4. Rebase-First Workflow

### 4.1 Step 1 – Patch Extraction (completed)
- Validate `patches/*.patch` exist, verify manifest.
- Spot-check patches with `less patches/<name>.patch` before use.

### 4.2 Step 2 – Fresh Upstream Branch
```bash
cd SimpleTuner
BACKUP=$(git rev-parse --abbrev-ref HEAD)
git fetch origin main
git checkout -b gh200-reintegration origin/main
```
- Run upstream smoke tests (Section 6.1) before modifying anything.

### 4.3 Step 3 – Surgical Re-application
For each patch:
1. Open upstream file(s).
2. Open patch side-by-side (`delta`, `meld`, `code --diff`, etc.).
3. Re-implement logic respecting new upstream structures (attention backend, audio, manual validation, etc.).
4. Stage file(s), run targeted tests, note results in checklist (Section 7.2).

### 4.4 Step 4 – Extended Validation
- Execute validation scripts (Section 7.3).
- Track results in GH200_REBASE_CHECKLIST.md (Phase 3 deliverable).

---

## 5. Per-Component Integration Guidance

### 5.1 Trainer (`patches/01`) – 8 Integration Points
1. **Imports:** add `simpletuner.gh200` helpers alongside other training imports.
2. **GH200GradientRamp class:** insert near other helper classes; ensure logging hooks align with upstream logging init.
3. **Trainer.__init__ additions:** `self.gh200_ramp_scheduler`, `_gh200_last_reported_ga` defined before `parse_arguments` call.
4. **Runtime config call:** invoke `_configure_gh200_runtime()` immediately after config/accelerator initialization but before model creation.
5. **Method `_configure_gh200_runtime`:**
   - Read `StateTracker.get_raw_config()`.
   - Check `gh200_optimizations` + `cache_config` for UVM toggles.
   - Call `set_uvm_hint_override` accordingly.
   - Instantiate `GH200GradientRamp` if requested; update `StateTracker.set_gh200_runtime_config`.
6. **Dynamic GA application:** call `_apply_dynamic_gradient_accumulation()` inside training loop before `accelerate.accumulate` block.
7. **Batch optimization:** call `optimize_batch_for_gh200(prepared_batch)` immediately after `prepare_batch` to attach hints to latents/encoders/audio (Section 5.2).
8. **StateTracker interplay:** ensure manual validation, attention backend, and GH200 ramp logging co-exist (no override of upstream callbacks).

> **AttentionBackendController note:** upstream now uses controller hooks around evaluation/training transitions. When re-adding GH200 logic, avoid touching `_manual_validation_consumer` plumbing or attention controller calls. GH200 functions should be orthogonal (only configure batch size/UVM).

> **Manual validation compatibility:** gradient ramp must not change GA mid-manual-validation step; keep `_apply_dynamic_gradient_accumulation()` before `with accelerator.accumulate` so manual checkpoints observe final GA.

### 5.2 Caching / Audio Support (`patches/02`)
- `_prepare_uvm_tensor` and `_prepare_text_embed_tensor` remain as helpers but extend to audio caches by optionally calling from new audio cache code if introduced upstream.
- Extend `optimize_batch_for_gh200` (Section 6.3) to understand `audio_latents`, `waveform`, or other keys emitted by new DatasetType.AUDIO flows.

### 5.3 StateTracker & Config (`patches/03` + `04`)
- Ensure upstream `StateTracker` additions (audio metadata, manual validation fields) remain untouched; append GH200 attributes near existing config fields.
- Loaders should call `StateTracker.set_raw_config` regardless of backend (json/toml/env/cmd). In upstream, config detection changed—place setter after `json.load` or when CLI args dict is available.

### 5.4 Data Backend + Builders (`patches/05`)
- Upstream factory now supports audio buckets, validation previews, etc. Insert GH200 detection in `_is_primary_training_backend` and `from_instance_representation` without removing audio logic.
- Register `InMemoryBackendBuilder` in builders map; ensure dataset configuration validation handles `type: "in_memory"` consistent with new schema (IDs, dataset_type, caching flags).

### 5.5 GH200 Module (`patches/06`)
- Drop-in addition; confirm packaging via `simpletuner/gh200/__init__.py` with explicit exports.
- Document env variable `SIMPLETUNER_GH200_ENABLE_UVM_HINTS` + `set_uvm_hint_override` for runtime toggles.
- Update `optimize_batch_for_gh200` to include audio keys (Section 6.3 snippet).

### 5.6 Scripts & Docs (`patches/07`)
- Place GH200 docs under repo root + `documentation/` for discoverability.
- Reference Phase 3 scripts in docs (e.g., GH200 workflow now uses `test_gh200_features.sh`).

---

## 6. Testing Strategy

### 6.1 Upstream-Only Smoke Tests (before GH200 patches)
```bash
# Attention backend matrix
python train.py --config config/examples/minimal.json --attention_mechanism flash_attn_3 --max_train_steps 10

# Manual validation API (simulated via CLI flag)
python train.py --config config/examples/minimal.json --enable_manual_validation --max_train_steps 10

# Audio loader sanity
python - <<'PY'
from simpletuner.helpers.audio import load_audio
print(load_audio('tests/assets/audio/sample.wav')[0].shape)
PY
```

### 6.2 GH200-Only Validation (after patches but before combination tests)
```bash
scripts/test_gh200_features.sh
# Runs gh200_diagnostic.py, validate_uvm.py, in-memory backend staging
```

### 6.3 Combination Matrix
| Scenario | Command | Purpose |
| --- | --- | --- |
| GH200 + FlashAttention3 | `scripts/test_combined_features.sh --backend flash_attn_3` | Ensures attention dispatcher unaffected by GH200 ramp |
| GH200 + Manual Validation | `scripts/test_combined_features.sh --manual-validation` | Confirms manual checkpoints still fire |
| GH200 + Audio Dataset | configure dataset type `audio` with `in_memory` staging + run short training | validates UVM hints on new tensors |

### 6.4 Performance Regression Check
- Re-run baseline scenario; compare throughput ±5%, HBM usage, CPU RAM occupancy.
- Investigate deviations before shipping.

---

## 7. Execution Assets (Produced in Phase 3)
- `execute_gh200_rebase.sh` – orchestrates entire process.
- `GH200_REBASE_CHECKLIST.md` – manual confirmation log.
- `scripts/test_upstream_features.sh`, `scripts/test_gh200_features.sh`, `scripts/test_combined_features.sh` – validation harness.
- `gh200/audio_support_snippet.py` – reference for audio-aware batch optimization.

---

## 8. Operational Details

### 8.1 Environment Variables
| Variable | Default | Purpose |
| --- | --- | --- |
| `SIMPLETUNER_GH200_ENABLE_UVM_HINTS` | 1 | Master toggle for GH200 UVM helpers |
| `GH200_UVM_OVERSUBSCRIPTION_RATIO` | 5.0 | Controls `PYTORCH_CUDA_ALLOC_CONF` |
| `GH200_UVM_ACCESS_PATTERN` | `gpu_first` | Guides UVM placement hints |

### 8.2 Diagnostics Checklist
1. `python gh200_diagnostic.py --oversubscription-scale 1.5`
2. Inspect `gh200_diagnostic_report.json` for ✅ statuses.
3. Monitor `nvidia-smi dmon -s u` during training.

### 8.3 Rollback Script Template
```bash
#!/usr/bin/env bash
set -euo pipefail
BACKUP_BRANCH=$(cat MERGE_LOG.txt | awk -F: '/Backup branch/ {print $2}' | xargs)
if [[ -z "$BACKUP_BRANCH" ]]; then
  echo "Backup branch not found" >&2
  exit 1
fi
read -rp "Reset to $BACKUP_BRANCH? (y/N) " ans
[[ $ans == y || $ans == Y ]] || exit 1
git reset --hard "$BACKUP_BRANCH"
```

---

## 9. Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Trainer conflicts with Manual Validation/Attention Controller | Loss of upstream UX | Follow integration points; add unit tests for manual validation triggers |
| Data backend divergence (audio/in_memory) | Dataset staging failures | Reconcile schema changes first, then add GH200 builder |
| Audio tensors skipped by GH200 hints | Performance regression | Implement audio snippet (Section 7) + targeted tests |
| Docs/scripts drift | Operator confusion | Keep GH200 docs updated with script references |

---

## 10. Timeline

| Phase | Duration | Owner | Notes |
| --- | --- | --- | --- |
| Patch extraction (done) | 1h | Dustin | See `patches/` + manifest |
| Documentation v2 | 2h | Dustin | This file |
| Automation assets | 1.5h | Dustin | Scripts + checklist |
| Execution | ~20h | Team | Includes testing + perf validation |

---

## 11. Appendices

### Appendix A – Patch Reference
See `docs/GH200_PATCH_MANIFEST.md` for stats and risk levels.

### Appendix B – Audio Support Snippet (Preview)
```python
# gh200/audio_support_snippet.py
from simpletuner.gh200 import prefer_cpu_residency

def hint_audio(batch):
    for key in ("audio_latents", "audio_samples", "waveform"):
        value = batch.get(key)
        if torch.is_tensor(value):
            prefer_cpu_residency(value)
        elif isinstance(value, dict):
            for tensor in value.values():
                if torch.is_tensor(tensor):
                    prefer_cpu_residency(tensor)
```
(Full script delivered in Phase 3.)

### Appendix C – Testing Matrix Template
| Test ID | Scenario | Command | Expected Outcome |
| --- | --- | --- | --- |
| U1 | Upstream attention | `scripts/test_upstream_features.sh --attention` | Logs show backend active; exit 0 |
| G1 | GH200 diagnostic | `scripts/test_gh200_features.sh --diagnostic` | Report success |
| C1 | Combined manual validation | `scripts/test_combined_features.sh --manual` | Manual validation event triggered |

---

_End of GH200_UPSTREAM_MERGE_GUIDE_V2.md_
