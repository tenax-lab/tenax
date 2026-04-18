---
name: tenax-benchmark
description: >
  Help users design and run performance benchmarks for Tenax algorithms across
  hardware backends (CPU, CUDA, TPU, Metal). Generates the correct CLI commands
  for python -m benchmarks.run, interprets JSON/CSV output, and helps compare
  results across backends and algorithm configurations. Use this skill when the
  user mentions benchmarks, performance, timing, speed comparison, GPU vs CPU,
  profiling, or wants to measure how fast DMRG/iDMRG/TRG/iPEPS runs. Also
  trigger for "how fast is", "performance test", "scaling study", or "which
  backend is faster".
---

# Tenax Benchmark Runner

Help users design performance experiments and run Tenax's CLI-driven benchmark
suite.

## CLI Reference

```bash
python -m benchmarks.run [OPTIONS]
```

### Core options

| Flag | Description | Values |
|------|-------------|--------|
| `-b`, `--backend` | Hardware backend | `cpu`, `cuda`, `tpu`, `metal` |
| `-a`, `--algorithm` | Algorithm(s) to benchmark | `dmrg`, `idmrg`, `trg`, `hotrg`, `ipeps`, `all` |
| `-s`, `--size` | Problem size(s) | `small`, `medium`, `large`, `all` |
| `-n`, `--trials` | Number of trials per config | Integer (default varies) |
| `-o`, `--output` | Save results to JSON | File path |
| `--csv` | Save results to CSV | File path |
| `--list-backends` | Show available backends | — |

### Examples

```bash
# Quick smoke test
python -m benchmarks.run --backend cpu --algorithm trg --size small --trials 1

# Full CPU baseline
python -m benchmarks.run --backend cpu -o benchmarks/results/cpu_baseline.json

# GPU comparison
python -m benchmarks.run --backend cuda -o benchmarks/results/cuda.json

# Specific algorithms and sizes
python -m benchmarks.run -b cpu -a dmrg idmrg -s small medium -n 5

# CSV for analysis
python -m benchmarks.run -b cpu -a all -s all --csv results.csv

# Check what's available
python -m benchmarks.run --list-backends
```

## Designing Benchmark Experiments

When the user wants to study performance, help them design a systematic
experiment.

### 1. Backend comparison

Compare the same algorithm across backends:

```bash
# CPU baseline
python -m benchmarks.run -b cpu -a dmrg -s small medium large -n 3 -o cpu_dmrg.json

# GPU
python -m benchmarks.run -b cuda -a dmrg -s small medium large -n 3 -o cuda_dmrg.json
```

Then compare wall-clock times from the JSON output. GPU advantage typically
grows with problem size (larger χ, more sites).

### 2. Algorithm scaling

Fix the backend and vary problem size to study computational scaling:

```bash
python -m benchmarks.run -b cpu -a dmrg -s small medium large -n 5 --csv dmrg_scaling.csv
```

DMRG scales as O(χ³·d·w) per site update. Students should verify this by
plotting time vs χ on a log-log plot.

### 3. Algorithm comparison

Compare different algorithms on the same problem:

```bash
python -m benchmarks.run -b cpu -a dmrg idmrg trg -s medium -n 5 --csv comparison.csv
```

### 4. JIT warmup study

JAX recompiles on first call. Run with `--trials 5` and note that the first
trial is slower (compilation). Report median or last-trial time for fair
comparison.

## Interpreting Results

The JSON output contains:
- **Timings** — wall-clock per trial
- **Parameters** — χ, L, number of sweeps, etc.
- **Device info** — backend, device name, JAX version

Guide students to:
- **Report median time**, not mean (avoids JIT warmup skew).
- **Plot time vs χ** on log-log to verify expected scaling.
- **Note memory** — GPU benchmarks may OOM at large sizes.

## Common Experiment Templates

### "Is my GPU helping?"

```bash
python -m benchmarks.run --list-backends
python -m benchmarks.run -b cpu -a dmrg -s medium -n 3 -o cpu.json
python -m benchmarks.run -b cuda -a dmrg -s medium -n 3 -o gpu.json
```

GPU typically helps when χ ≥ 64. For small χ, CPU may be faster due to
transfer overhead.

### "How does TRG scale with bond dimension?"

```bash
python -m benchmarks.run -b cpu -a trg -s small medium large -n 5 --csv trg_scaling.csv
```

TRG scales as O(χ^6) per coarse-graining step. This becomes the bottleneck
quickly.

### "Which algorithm is best for my 2D problem?"

Compare cylinder DMRG vs iPEPS:
```bash
python -m benchmarks.run -b cpu -a dmrg ipeps -s medium -n 3 --csv 2d_comparison.csv
```

Cylinder DMRG is more accurate per compute dollar for small Ly; iPEPS wins
for truly 2D problems in the thermodynamic limit.

## iPEPS Performance Tuning

### Chi ramping

Use `CTMConfig.chi_ramp` to warm up CTM convergence at small chi before
the target chi. Typical speedup: 1.2–2.1× on GPU. Example:

```python
ctm=CTMConfig(chi=32, chi_ramp=[(8, 10), (16, 10), (32, None)])
```

Each tuple is `(chi_value, max_steps)`: run up to `max_steps` CTM steps at
that chi, then move to the next. The final entry with `None` runs until
convergence at the target chi. This avoids wasting iterations at full chi
when the environment is far from converged.

## Pedagogical Notes

- Benchmarking teaches students that algorithmic complexity (O-notation) is
  only part of the story — constants, memory bandwidth, and JIT compilation
  all matter in practice.
- The CPU-vs-GPU comparison illustrates when parallelism helps: tensor
  contractions are BLAS-heavy and GPU-friendly, but only above a minimum size.
- Encourage students to always report hardware specs alongside timings.
