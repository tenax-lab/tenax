# JAX QR Dispatch Benchmark

This directory contains standalone JAX QR benchmarks for testing how repeated
independent `jnp.linalg.qr` calls behave when issued from a Python loop.

The main questions are whether:

- each QR call blocks immediately,
- work is enqueued asynchronously and synchronized later, or
- a batched form such as `vmap` changes the behavior materially,
- and whether explicit two-GPU splitting helps when the workload is large enough.

## Files

- `qr_dispatch_benchmark.py`: benchmark script
- `qr_size_sweep.py`: size sweep wrapper around the QR benchmark
- `POLARON_RESULTS.md`: informational notes and observed results from a dual-V100 workstation

## Run

Use the repo virtualenv so the benchmark stays independent of the Tenax package:

```bash
.venv/bin/python codex-benchmark/qr_dispatch_benchmark.py
```

Example with larger inputs:

```bash
.venv/bin/python codex-benchmark/qr_dispatch_benchmark.py \
  --num-matrices 64 \
  --matrix-size 128 \
  --trials 5
```

To separate first-call JIT compile+run from steady-state execution for the
batched `vmap` path:

```bash
.venv/bin/python codex-benchmark/qr_dispatch_benchmark.py \
  --num-matrices 32 \
  --matrix-size 512 \
  --trials 3 \
  --warmup 1 \
  --include-vmap-compile
```

## Modes

- `loop_sync_each`: do one QR and force readiness on every iteration
- `loop_dispatch_then_sync`: dispatch all QRs in a Python loop, then synchronize once
- `batched_vmap_jit`: compile one batched QR over a stack of matrices
- `two_gpu_split_dispatch_then_sync`: split the batch across two pre-placed GPU
  batches and interleave submissions across the two devices

The benchmark reports three timing columns:

- issue-phase time
- tail-wait time
- total time

The phase split should be interpreted carefully:

- issue-phase time: time spent in the Python section issuing JAX operations
- tail-wait time: time spent in the final explicit `block_until_ready(...)`
- total time: overall wall-clock time

The issue phase is not pure enqueue cost. Device work may already be running
while Python is still issuing later QR calls. The tail-wait time is only the
unfinished remainder at the point where the explicit synchronization begins.

## Optional Two-GPU Mode

If the machine has at least two visible JAX GPU devices, you can ask the
benchmark to split the batch across the first two GPUs:

```bash
.venv/bin/python codex-benchmark/qr_dispatch_benchmark.py \
  --num-matrices 64 \
  --matrix-size 128 \
  --trials 5 \
  --warmup 2 \
  --two-gpu-split
```

The two-GPU mode pre-places the split inputs onto the two devices once before
warmup/trials. The timed region measures QR dispatch and completion, not
repeated host-to-device transfer cost on every trial.

Behavior by machine type:

- no GPU: normal CPU benchmarks still work; the two-GPU mode is skipped
- one GPU: normal GPU benchmarks still work; the two-GPU mode is skipped
- two or more GPUs: the split mode is enabled and uses the first two GPU devices

## Size Sweep

For a broader size study, use:

```bash
.venv/bin/python codex-benchmark/qr_size_sweep.py
```

This runs the QR benchmark over a range of matrix sizes and prints a compact
summary for each size.

You can include the explicit first-call vs second-call `vmap` measurements in
the sweep as well:

```bash
.venv/bin/python codex-benchmark/qr_size_sweep.py \
  --sizes 128 256 512 1024 \
  --include-vmap-compile
```
