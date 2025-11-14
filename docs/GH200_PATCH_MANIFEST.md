# GH200 Patch Manifest

Merge base: c1ae783a21e4c42b5d90fdb362557e1099700dd6
Generated: 2025-11-14T16:38:49-05:00 (via verify_assumptions.sh)

| Patch | Scope | Delta (ins/del) | Risk | Notes |
| --- | --- | --- | --- | --- |
| patches/01-trainer-gh200.patch | simpletuner/helpers/training/trainer.py | +125 / -48 | High | GH200 gradient ramp + runtime hooks overlapping upstream attention/manual validation logic. |
| patches/02-caching-uvm.patch | helpers/caching/vae.py; helpers/caching/text_embeds.py | +72 / -6 | Medium | Adds UVM placement helpers; ensure compatibility with new caching/audio flows. |
| patches/03-state-tracker.patch | helpers/training/state_tracker.py | +18 / -0 | Low | Raw config & GH200 runtime bookkeeping. |
| patches/04-configuration.patch | helpers/configuration/json_file.py; loader.py | +4 / -0 | Low | Propagates raw config into StateTracker. |
| patches/05-data-backend.patch | helpers/data_backend/factory.py + builders | +131 / -65 | High | Introduces in-memory backend; overlaps upstream audio/dataset logic. |
| patches/06-gh200-module.patch | gh200 package | +521 / -0 | Medium | UVM utilities + in-memory backend implementation. |
| patches/07-scripts-docs.patch | GH200 docs + scripts | +6114 / -0 | Low-Medium | Documentation + diagnostics. |

## Complexity Notes
- Total delta ≈6.9k insertions / 119 deletions across 7 components.
- Highest-risk merges: trainer + data_backend.
- Testing priority: manual validation + attention backend with GH200 ramp, audio datasets, UVM diagnostics.
