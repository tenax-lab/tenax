# Benchmarks

Tenax includes a CLI-driven benchmark suite that measures wall-clock
performance of every algorithm across CPU, CUDA GPU, TPU, and Apple Metal
backends.

## Quick start

```bash
# Smoke test — fastest benchmark, single trial
python -m benchmarks.run --backend cpu --algorithm trg --size small --trials 1

# Full CPU baseline (all algorithms, all sizes, 3 trials)
python -m benchmarks.run --backend cpu -o benchmarks/results/cpu_baseline.json
```

## CLI reference

```
python -m benchmarks.run [OPTIONS]
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--backend` | `-b` | `auto` | `cpu`, `cuda`, `gpu`, `tpu`, `metal`, or `auto` |
| `--algorithm` | `-a` | `all` | One or more of: `dmrg`, `idmrg`, `trg`, `hotrg`, `ipeps`, `ipeps_ad`, `all` |
| `--size` | `-s` | `all` | One or more of: `small`, `medium`, `large`, `all` |
| `--trials` | `-n` | `3` | Number of timed trials per benchmark |
| `--output` | `-o` | auto | JSON output path (default: `benchmarks/results/<backend>_<timestamp>.json`) |
| `--csv` | | none | Optional CSV output path |
| `--list-backends` | | | Print available JAX backends and exit |

Multiple algorithms and sizes can be selected in a single run:

```bash
python -m benchmarks.run -b cpu -a dmrg idmrg -s small medium -n 5
```

## Algorithms and sizes

### DMRG

Heisenberg chain via `build_mpo_heisenberg` + `build_random_mps` + `dmrg`.

| Size | L | chi | sweeps | initial bond dim |
|------|---|-----|--------|------------------|
| small | 20 | 32 | 5 | 8 |
| medium | 40 | 64 | 5 | 16 |
| large | 80 | 128 | 3 | 16 |

### iDMRG

Infinite Heisenberg chain via `build_bulk_mpo_heisenberg` + `idmrg`.

| Size | chi | iterations |
|------|-----|------------|
| small | 32 | 50 |
| medium | 64 | 100 |
| large | 128 | 50 |

### TRG

2D Ising model at the critical point via `compute_ising_tensor` + `trg`.

| Size | chi | steps |
|------|-----|-------|
| small | 8 | 15 |
| medium | 24 | 20 |
| large | 48 | 20 |

### HOTRG

2D Ising model at the critical point via `compute_ising_tensor` + `hotrg`.

| Size | chi | steps |
|------|-----|-------|
| small | 8 | 10 |
| medium | 20 | 15 |
| large | 40 | 15 |

### iPEPS (simple update)

Heisenberg model via `ipeps` with simple-update + CTM.

| Size | D | chi_ctm | steps |
|------|---|---------|-------|
| small | 2 | 8 | 50 |
| medium | 3 | 16 | 50 |
| large | 4 | 24 | 30 |

### iPEPS AD optimization

Heisenberg model via `optimize_gs_ad` with automatic differentiation through CTM.

| Size | D | chi_ctm | grad steps | learning rate |
|------|---|---------|------------|---------------|
| small | 2 | 8 | 20 | 1e-3 |
| medium | 2 | 16 | 30 | 1e-3 |
| large | 3 | 16 | 20 | 5e-4 |

## Output formats

### Summary table

Every run prints an aligned table to stdout:

```
Algorithm    Size     Backend  dtype       Warmup(s)    Mean(s)     Std(s)     Min(s) Status
trg          small    cpu      float64         0.740      0.013      0.000      0.013 OK
hotrg        small    cpu      float64        13.330     12.786      0.000     12.786 OK
```

### JSON

Full benchmark data is saved to JSON, including per-trial timings, parameter
dictionaries, device info, and algorithm-specific result values (energy,
convergence status, etc.).

```bash
python -m benchmarks.run -b cpu -a trg -s small -o results.json
cat results.json | python -m json.tool | head -20
```

### CSV

Optional flat CSV for spreadsheet analysis:

```bash
python -m benchmarks.run -b cpu -a all -s all --csv results.csv
```

Columns: `algorithm`, `size_label`, `backend`, `dtype`, `warmup_time_s`,
`mean_time_s`, `std_time_s`, `min_time_s`, `num_trials`, `error`.

### Plot existing JSON results

You can generate PNG charts directly from previously saved JSON files:

```bash
uv run --with matplotlib python -m benchmarks.plot_results
```

By default this reads `benchmarks/results/*.json` and writes plots to
`benchmarks/results/plots/` (one PNG per algorithm, grouped by size and backend).

Common options:

```bash
# Plot only DMRG/HOTRG from CPU+CUDA files
uv run --with matplotlib python -m benchmarks.plot_results \
  --algorithm dmrg hotrg \
  --backend cpu cuda

# Use minimum trial time and log-scale y axis
uv run --with matplotlib python -m benchmarks.plot_results \
  --metric min_time_s \
  --logy
```

## How timing works

1. **Setup**: `setup_fn()` builds all input tensors and config objects.
2. **Warmup**: One untimed run triggers JIT compilation; warmup wall time is
   recorded separately.
3. **Timed trials**: `setup_fn()` is called fresh each trial (important because
   DMRG mutates the MPS in-place). A device sync
   (`jnp.zeros(1).block_until_ready()`) runs before and after each trial to
   ensure GPU/TPU async dispatch is fully reflected in the wall-clock time.
4. **Error handling**: Exceptions are caught and stored in the result; the suite
   continues with the remaining benchmarks.

## Backend notes

| Backend | dtype | Notes |
|---------|-------|-------|
| `cpu` | float64 | Full x64 precision, single-threaded by default |
| `cuda` / `gpu` | float64 | Requires `tenax-tn[cuda12]` or `tenax-tn[cuda13]` |
| `tpu` | float64 | Requires `tenax-tn[tpu]` in a TPU VM |
| `metal` | float32 | Apple Silicon; limited float64 support, uses float32 automatically |
| `auto` | float64 | Lets JAX pick the best available backend |

The `--list-backends` flag prints the active JAX device info without running
any benchmarks:

```bash
python -m benchmarks.run --list-backends
```

## File layout

```
benchmarks/
  __init__.py          # package marker
  __main__.py          # python -m benchmarks alias
  run.py               # CLI entry point (argparse)
  backend.py           # backend selection, device info, dtype
  runner.py            # warmup, timing loop, BenchmarkResult dataclass
  results.py           # JSON/CSV serialization, summary table
  bench_dmrg.py        # DMRG benchmark cases
  bench_idmrg.py       # iDMRG benchmark cases
  bench_trg.py         # TRG benchmark cases
  bench_hotrg.py       # HOTRG benchmark cases
  bench_ipeps.py       # iPEPS simple-update benchmark cases
  bench_ipeps_ad.py    # iPEPS AD optimization benchmark cases
  plot_results.py      # plot JSON benchmark outputs to PNG
  results/             # default output directory for JSON results
```
