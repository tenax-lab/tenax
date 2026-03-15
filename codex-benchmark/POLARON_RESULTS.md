# Polaron Results

This file records benchmark results collected on `polaron` for discussion with
the Tenax developers. It is an informational note for a side PR, not intended
as upstream project documentation.

## Machine Summary

- host: `polaron`
- GPUs: `2 x NVIDIA V100`
- JAX: `0.9.1`
- backend: `gpu`
- devices seen by JAX: `[CudaDevice(id=0), CudaDevice(id=1)]`

## Command

```bash
python codex-benchmark/qr_size_sweep.py \
  --num-matrices 64 \
  --sizes 128 256 512 1024 \
  --trials 3 \
  --warmup 1 \
  --include-vmap-compile \
  --two-gpu-split
```

## Results

Times are in milliseconds and are means over the 3 recorded trials.

### Size 128

| mode | issue | tail_wait | total |
|---|---:|---:|---:|
| `loop_sync_each` | 59.268 | 49.389 | 117.978 |
| `loop_dispatch_then_sync` | 87.583 | 5.336 | 93.181 |
| `batched_vmap_jit` | 3.299 | 12.360 | 16.175 |
| `two_gpu_split_dispatch_then_sync` | 97.654 | 5.111 | 103.321 |
| `batched_vmap_jit_first_call` | 86.206 | 9.486 | 96.317 |
| `batched_vmap_jit_second_call` | 6.889 | 8.261 | 15.765 |

### Size 256

| mode | issue | tail_wait | total |
|---|---:|---:|---:|
| `loop_sync_each` | 37.132 | 71.238 | 121.029 |
| `loop_dispatch_then_sync` | 72.545 | 6.053 | 78.890 |
| `batched_vmap_jit` | 76.014 | 5.538 | 81.999 |
| `two_gpu_split_dispatch_then_sync` | 76.839 | 5.248 | 82.655 |
| `batched_vmap_jit_first_call` | 158.830 | 5.407 | 164.674 |
| `batched_vmap_jit_second_call` | 78.623 | 3.856 | 83.096 |

### Size 512

| mode | issue | tail_wait | total |
|---|---:|---:|---:|
| `loop_sync_each` | 99.812 | 126.972 | 241.857 |
| `loop_dispatch_then_sync` | 166.067 | 11.768 | 178.140 |
| `batched_vmap_jit` | 478.403 | 6.055 | 485.124 |
| `two_gpu_split_dispatch_then_sync` | 113.149 | 10.898 | 124.958 |
| `batched_vmap_jit_first_call` | 582.595 | 4.070 | 587.146 |
| `batched_vmap_jit_second_call` | 478.305 | 6.200 | 485.092 |

### Size 1024

| mode | issue | tail_wait | total |
|---|---:|---:|---:|
| `loop_sync_each` | 210.176 | 261.885 | 492.110 |
| `loop_dispatch_then_sync` | 372.804 | 29.343 | 402.647 |
| `batched_vmap_jit` | 3571.980 | 7.739 | 3580.610 |
| `two_gpu_split_dispatch_then_sync` | 227.691 | 26.398 | 255.268 |
| `batched_vmap_jit_first_call` | 3671.804 | 7.628 | 3680.151 |
| `batched_vmap_jit_second_call` | 3571.732 | 7.841 | 3580.549 |

## Observations

- `loop_dispatch_then_sync` consistently beats `loop_sync_each`, which is
  evidence that repeated independent `jnp.linalg.qr` calls are dispatched
  asynchronously enough for device work to overlap with later host submission.
- The single-GPU overlap is real but modest; it is not the same thing as an
  explicit sector scheduler.
- `batched_vmap_jit` is excellent for small matrices (`128`), roughly
  break-even by `256`, and dramatically worse for `512` and `1024`.
- The catastrophic `1024`-size `batched_vmap_jit` result is not mainly compile
  cost: `batched_vmap_jit_first_call` and `batched_vmap_jit_second_call` are
  both around `3.6 s`.
- After changing the two-GPU mode to pre-place inputs and interleave host
  submissions across `cuda:0` and `cuda:1`, the split path becomes clearly
  beneficial for `512` and `1024`, but it still does not reach ideal `2x`
  speedup.
- For large QR workloads, explicit independent dispatch is far better than the
  batched `vmap` formulation in this JAX/CUDA configuration.

## Tentative Relevance to Tenax

These results do not directly benchmark Tenax, but they are relevant to how a
JAX-backed dense engine might behave when used underneath symmetry-sector
linear algebra:

- Independent per-sector factorizations can benefit from asynchronous dispatch.
- Regular batching is not universally beneficial; it can become much worse for
  large matrices.
- Multi-GPU benefit exists, but it depends on explicit device placement and
  host submission structure.
