# Ryzen CPU Results

This file records benchmark results collected on `ryzen` for discussion with
the Tenax developers. It is an informational note for the benchmark PR.

## Machine Summary

- host: `ryzen`
- backend: `cpu`
- visible JAX CPU devices: `8`
- JAX: `0.9.1`

## Command

```bash
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=8 \
  .venv/bin/python codex-benchmark/qr_dispatch_benchmark.py \
  --num-matrices 64 \
  --matrix-size 512 \
  --trials 3 \
  --split-devices 8
```

## Results

Times are in milliseconds and are means over the 3 recorded trials.

| mode | issue | tail_wait | total |
|---|---:|---:|---:|
| `loop_sync_each` | 40.639 | 658.954 | 703.772 |
| `loop_dispatch_then_sync` | 838.357 | 50.499 | 888.919 |
| `batched_vmap_jit` | 0.025 | 386.108 | 386.309 |
| `split_8_devices_dispatch_then_sync` | 89.943 | 33.446 | 123.544 |

## Observations

- The explicit 8-device split is dramatically faster than the one-device looped
  paths for this case.
- On this CPU setup, `loop_dispatch_then_sync` is actually worse than
  `loop_sync_each`, which is a useful reminder that JAX's async overlap story is
  backend-dependent and should not be assumed to help on CPU the way it helped
  on the V100 GPU runs.
- `batched_vmap_jit` remains much better than the one-device looped modes, but
  it is still far behind the explicit 8-device split.
- This benchmark is comparing multiple visible JAX CPU devices, not a pure
  single-thread versus many-thread study inside one CPU device.
