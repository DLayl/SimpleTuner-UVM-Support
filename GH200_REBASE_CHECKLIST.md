# GH200 Rebase Execution Checklist

Use this document to record progress. Mark each step with ✅/⚠️/❌ and add notes.

## Section A – Pre-Flight
- [ ] `git status` clean except GH200 files
- [ ] Backup branch created (`gh200-backup-YYYYMMDD_HHMMSS`)
- [ ] `gh200-current.patch` exported
- [ ] Baseline performance run archived (`baseline/` directory)

## Section B – Upstream Preparation
1. [ ] `git checkout -b gh200-reintegration origin/main`
2. [ ] `scripts/test_upstream_features.sh` (attention/manual/audio) → Expected: exit 0, logs show backend activation + manual validation event
3. [ ] Notes:

## Section C – Patch Reapplication (per component)
| Patch | Status | Notes |
| --- | --- | --- |
| 01-trainer-gh200 | [ ] | 8 integration points verified (imports, ramp, runtime config, optimize call) |
| 02-caching-uvm | [ ] | `_prepare_uvm_tensor` + `_prepare_text_embed_tensor` wired |
| 03-state-tracker | [ ] | `set_raw_config` + `set_gh200_runtime_config` present |
| 04-configuration | [ ] | Loader + JSON store raw config |
| 05-data-backend | [ ] | `in_memory` builder registered + factory aware |
| 06-gh200-module | [ ] | `gh200` package imported by trainer/caches |
| 07-scripts-docs | [ ] | Docs reference new scripts |

## Section D – Validation Scripts
1. [ ] `scripts/test_upstream_features.sh`
2. [ ] `scripts/test_gh200_features.sh`
3. [ ] `scripts/test_combined_features.sh`
4. Notes:

## Section E – Manual Verification Points
- [ ] Manual validation button/API triggers evaluation stage
- [ ] Attention backend toggles (flash_attn_3, flex) operate with GH200 ramp enabled
- [ ] Audio dataset (DatasetType.AUDIO + in_memory) stages successfully
- [ ] GH200 diagnostic report ✅ across checks

## Section F – Performance Comparison
| Metric | Baseline | Post-rebase | Delta |
| --- | --- | --- | --- |
| Throughput (samples/s) | | | |
| Peak HBM (GB) | | | |
| Peak System RAM (GB) | | | |
| Wall-clock per 1k steps | | | |

## Section G – Troubleshooting Decision Tree
1. **Patch fails to apply** → Inspect `logs/<patch>.log`, open patch + upstream file, re-implement manually, rerun step.
2. **Upstream test fails pre-GH200** → Abort; fix upstream environment before applying patches.
3. **GH200 test fails** → Confirm env vars from `scripts/setup_gh200_env.sh`, rerun diagnostic; check `gh200_diagnostic_report.json`.
4. **Combined test fails** → Determine subsystem:
   - Manual validation error → inspect trainer logs around `_apply_dynamic_gradient_accumulation`.
   - Attention backend error → re-run with `SIMPLETUNER_LOG_LEVEL=DEBUG`, verify `AttentionBackendController` logs.
   - Audio tensor error → ensure `optimize_batch_for_gh200` handles new keys (see `gh200/audio_support_snippet.py`).
5. **Performance regression** → Compare baseline; investigate dataloader/backoff; consider adjusting GH200 ramp parameters.

## Section H – Sign-off
- [ ] Code review completed by: __________
- [ ] Tests documented in release notes
- [ ] Branch ready for PR / deployment


---

## Validation Commands Executed (2025-11-14)

### Syntax Validation
All GH200 modules and integration points verified with `python -m py_compile`:
- simpletuner/gh200/__init__.py
- simpletuner/gh200/uvm.py
- simpletuner/gh200/in_memory_backend.py
- simpletuner/gh200/audio_support_snippet.py
- simpletuner/helpers/training/trainer.py
- simpletuner/helpers/data_backend/builders/in_memory.py
- simpletuner/helpers/caching/vae.py
- simpletuner/helpers/caching/text_embeds.py

### Import Verification
```
python -c "from simpletuner.gh200 import uvm"
python -c "from simpletuner.gh200.in_memory_backend import GH200InMemoryBackend"
python -c "from simpletuner.gh200 import gh200_uvm_enabled, optimize_batch_for_gh200"
```

### Test Suite Results (GH200 Hardware)
- `scripts/test_upstream_features.sh`
- `scripts/test_gh200_features.sh`
- `scripts/test_combined_features.sh`
- `scripts/run_all_tests.sh`

### Phase 0 Verification
- Import path: `simpletuner.gh200`
- Upstream APIs present (manual validation, attention backends)
- Diagnostic flags: `--skip-allocation`, `--output`
- Audio integration: deferred (reference implementation provided)
