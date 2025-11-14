# GH200 Training Workflow

This note captures the minimum set of steps required to run SimpleTuner on
the Grace Hopper 200 system using the custom PyTorch build with unified
virtual memory support.

## 1. Prepare the environment

1. Copy the repo to the GH200 machine (or mount it via your Jupyter workspace).
2. Source the helper script so the correct environment variables are applied:

   ```bash
   source scripts/setup_gh200_env.sh
   ```

   The script exports:

   - `SIMPLETUNER_GH200_ENABLE_UVM_HINTS=1` – turns on the new UVM placement helpers.
   - `PYTORCH_CUDA_ALLOC_CONF` – defaulting to a `5.0x` oversubscription ratio and `gpu_first` access pattern.
   - `ACCELERATE_MIXED_PRECISION=bf16` – keeps Accelerate aligned with GH200’s bf16 workflow.

   You can override `GH200_UVM_OVERSUBSCRIPTION_RATIO` or `GH200_UVM_ACCESS_PATTERN`
   before sourcing if a job needs different memory behaviour.

3. (Optional) verify the patched PyTorch build is on your `PYTHONPATH`.

## 2. Configure datasets for the in-memory backend

The GH200 has hundreds of GB of Grace DDR5 available. You can pre-stage an entire dataset using
the new `in_memory` backend. Add a block similar to the following to your training config:

```json
{
  "datasets": [
    {
      "id": "train_images",
      "type": "in_memory",
      "dataset_type": "image",
      "instance_data_dir": "/workspace/datasets/my_diffusion_set",
      "in_memory": {
        "compression": "lz4",
        "memory_limit_gb": 360,
        "num_workers": 64
      }
    }
  ]
}
```

Key tunables:

- `compression`: Choose `lz4`, `zlib`, `snappy`, or omit for raw storage.
- `memory_limit_gb`: Safety valve that keeps the loader from exceeding your headroom.
- `num_workers`: Loader parallelism for the initial staging pass.

The backend is still compatible with SimpleTuner’s metadata pipeline, so all
bucketing and caption handling continues to work as before.

## 3. Run diagnostics (recommended once per session)

Before launching a long run, sanity-check the environment with:

```bash
python gh200_diagnostic.py --oversubscription-scale 1.5
```

The script verifies:

- UVM-related environment variables.
- GPU + system metadata (architecture, RAM, driver version).
- Ability to allocate beyond HBM capacity when UVM is active (unless `--skip-allocation` is passed).

Results are printed to the console and written to `gh200_diagnostic_report.json` for auditing.

## 4. Launch training

After the environment is primed and datasets are staged (the in-memory backend loads them on first
access), use your normal launch command. Example:

```bash
accelerate launch \
  --config_file accelerate_config_gh200.yaml \
  train.py \
  --config config/gh200_in_memory_example.json
```

Because the trainer now calls `optimize_batch_for_gh200`, batches will automatically receive UVM
hints as long as `SIMPLETUNER_GH200_ENABLE_UVM_HINTS` is set.

## 5. Checklist for each new job

- [ ] Source `scripts/setup_gh200_env.sh`.
- [ ] Run `python gh200_diagnostic.py`.
- [ ] Confirm dataset config points to the `in_memory` backend.
- [ ] Ensure `PYTORCH_CUDA_ALLOC_CONF` matches the intended oversubscription policy.
- [ ] Launch Accelerate or your preferred entrypoint.
