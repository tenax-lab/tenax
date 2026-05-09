# Tenax ↔ variPEPS Benchmark on Square-Lattice Heisenberg AFM — Design

**Status:** approved 2026-05-08
**Author:** YJ Kao + Claude (brainstorming session)
**Scope:** apples-to-apples benchmark of Tenax against variPEPS 1.4.2 on the
spin-½ square-lattice Heisenberg AFM ground state via implicit-AD iPEPS, covering
**accuracy parity, performance parity, and convergence-trajectory parity**.

## Goal

Produce a single report that, for each `(path, D, χ)` point on a small grid,
shows: final E/site (and Δ between libs), number of AD steps to reach
tol = 1e-6, total wall-clock, JIT-compile cost, peak memory, and an overlaid
energy-vs-step trajectory.

## Locked decisions

| Axis | Choice | Why |
|---|---|---|
| Goal | accuracy + perf + trajectory parity | full apples-to-apples report |
| Paths | both **`single_site` (1×1 + sublattice-rotated gate, unconstrained tensor)** *and* **`bipartite_2site` (2-tensor checkerboard + bare gate)** | (a) is the cheap diagnostic; (b) is the headline number |
| Tenax 1×1 ansatz | `gs_c4v=False, gs_implicit_ad=True, unit_cell="1x1"` | variPEPS has no native C4v constraint, so we drop Tenax's C4v constraint to give both libs the **same** unconstrained ansatz (matched parameter count → trajectory parity meaningful) |
| Grid | D ∈ {2, 3} × χ ∈ {16, 24} | CPU-tractable, covers regime where AD machinery matters |
| Stopping | tol 1e-6 with hard cap of 100 AD steps | physical convergence target with bounded wall-clock |
| Init — `single_site` | shared **random** init (one numpy seed, both libs load same array) | SU on the rotated gate converges to the \|↑↑⟩ saddle (E=−0.5/site), L-BFGS cannot escape it (see `ipeps_optimize.py:1389` reference-mode comment) |
| Init — `bipartite_2site` | shared **SU-converged** tensor on disk | Tenax SU on the bare gate finds a Néel-like state both libs descend from |
| dtype | complex128 both libs | variPEPS only runs at complex128 (per 2026-05-08 audit) |
| Architecture | shared protocol, two subprocess runners, one orchestrator | fair JIT cost; no `jax.config` cross-pollution; runners independently runnable |
| Location | `benchmarks/varipeps_compare/` | extends benchmarking infra without polluting `bench_ipeps_ad.py` |

Total grid: 2 paths × 2 D × 2 χ = **8 points**, each run twice (Tenax, variPEPS) = 16 subprocesses per full sweep.

## Architecture

```
[orchestrator]                      [disk]                    [tenax subproc]   [varipeps subproc]
     │                                 │                            │                  │
     │  build H gate (path-specific)   │                            │                  │
     │  build init                     │                            │                  │
     │    single_site → random          │                            │                  │
     │    bipartite_2site → Tenax SU    │                            │                  │
     │  save payload.npz ────────────► │                            │                  │
     │  spawn run_tenax    ─────────►  │ ─► load payload.npz ────►  │                  │
     │  spawn run_varipeps ──────────────────────────────────────────────────────────► │
     │                                          optimize_gs_ad     optimize_peps_network
     │                                          → JSON              → JSON
     │  ◄─ exit codes + stdout/stderr (logged) ───────────────────────────────────────│
     │  read both JSONs, merge into report.json[<key>], plot trajectories             │
```

**Process isolation** is load-bearing: subprocesses give fair JIT/cache accounting and prevent
variPEPS's import-time `jax.config` writes from polluting Tenax. **Shared init on disk** (random for
`single_site`, SU for `bipartite_2site`) makes trajectory parity meaningful from step 0. **Single
device target** (set by env var before subprocess spawn) keeps wall-clocks comparable. **Both libs
use unconstrained ansatze on both paths** — Tenax does not enforce C4v on the 1×1 path, so the
parameter counts match variPEPS exactly.

## Components

```
benchmarks/varipeps_compare/
├── protocol.py          # shared constants: GRID, TOL, MAX_STEPS, SEED, DTYPE, LBFGS_HISTORY, CTM_TOL
├── su_init.py           # build_heisenberg_gate() + run_su() → numpy arrays
├── payload.py           # save_payload(); load_payload()
├── run_tenax.py         # CLI runner, calls optimize_gs_ad with implicit AD
├── run_varipeps.py      # CLI runner, calls varipeps.optimize_peps_network
├── compare.py           # orchestrator: enumerate grid, run SU, spawn runners, merge, plot
└── results/             # gitignored; JSON per (path,D,χ,lib) + report.json + trajectory.png + run.log
```

`protocol.py` is single source of truth for the grid and shared knobs. Both runners
and the orchestrator import from it. Knobs:

- `GRID = [(path, D, χ) for path in {"single_site", "bipartite_2site"} for D in {2,3} for χ in {16,24}]`
- `TOL = 1e-6`, `MAX_STEPS = 100`, `SEED = 0`, `DTYPE = "complex128"`
- `LBFGS_HISTORY = 10`, line-search aligned to variPEPS defaults
- `CTM_TOL = 1e-8`, `CTM_MAX_ITER = 100`

## Runner JSON schema (contract)

Both `run_tenax.py` and `run_varipeps.py` emit JSON with this exact schema:

```json
{
  "lib": "tenax" | "varipeps",
  "path": "single_site" | "bipartite_2site",
  "D": 2|3, "chi": 16|24, "dtype": "complex128", "seed": 0,
  "energy_history": [E_0, E_1, ..., E_n],
  "step_times": [t_0, t_1, ..., t_n],
  "jit_compile_time": float,
  "final_energy": float,
  "num_steps": int,
  "converged": bool,
  "peak_memory_mb": float,
  "device": "cpu" | "cuda:0",
  "lib_version": "1.4.2" | "<tenax git sha>"
}
```

`energy_history[k]` is E/site after step k (post-CTM-converge, pre-optimizer-step).
`step_times[k]` is wall-clock of step k in seconds, **excluding** the JIT compile of step 0
which is recorded separately in `jit_compile_time`.

Merged `report.json` keys each grid point and includes computed deltas:
`delta_final_energy`, `delta_num_steps`, `tenax_speedup`.

## Error handling

| Class | Examples | Behavior |
|---|---|---|
| Per-point soft | variPEPS API mismatch, CTM non-convergence, hit `MAX_STEPS` | Record status in `report.json`, log to `results/<key>_<lib>.log`, continue |
| Per-point hard | OOM at high χ | Skip remaining higher-χ points for same `(path, D)` |
| Run-level fatal | variPEPS unimportable, requested device unavailable, disk full | Abort orchestrator with clear message |

Subprocess timeout per point: 30 minutes (D=3 χ=24 budget). Exceeding = hard failure.

## Testing

**Smoke test** at `tests/test_varipeps_compare.py`, marked `slow`, skipped if variPEPS unavailable:

```python
@pytest.mark.slow
@pytest.mark.skipif(not _have_varipeps(), reason="varipeps not installed")
def test_smoke_single_site_d2_chi8():
    # C4v 1x1, D=2, chi=8, MAX_STEPS=20, tol=1e-4
    # assert |E_tenax - E_varipeps| < 1e-3
    # assert both within 5e-2 of E_ref ≈ -0.6614
```

~1–2 min on CPU. Catches import-order pollution, payload schema drift, JSON schema
drift, gross numerical divergence. Does **not** validate the full grid — that's
the manual benchmark.

**Manual full benchmark:**
- `python -m benchmarks.varipeps_compare.compare --device cpu` (or `--device cuda:0`)
- Idempotent: skip already-completed points unless `--force`.
- Output: `report.json`, `trajectory.png` per point, `summary.md` parity table.

**Out of scope:** unit-testing `protocol.py` / `payload.py` (thin wrappers; YAGNI),
variPEPS internals, CPU↔GPU bit-identity, D=4 / χ=32 grid extension.

## Future extensions (deferred)

- D=4 / χ=32 on GPU (Q3 option (c)).
- Native-defaults run (Q4 option (c)) — each lib uses its own SU + optimizer defaults, comparing endpoints only.
- Honeycomb and kagome cross-checks (variPEPS supports both per memory).

## References

- Francuz, Schmoll, Csordás, Bauer & Cirac, *PRR* **7**, 013237 (Tenax `c4v_reference` mode target).
- Naumann et al., variPEPS SciPost Lect. Notes **86** (arXiv 2308.12358).
- Memory: `project_varipeps_2site_honeycomb_works.md`, `project_varipeps_ad_audit_2026_05_08.md`,
  `reference_varipeps_multisite_ad.md`, `reference_varipeps_scipost_paper.md`.
