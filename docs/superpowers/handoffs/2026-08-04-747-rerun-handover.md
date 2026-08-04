# #747 re-runs — handover to another machine

**Status:** ready to run. Nothing is blocked on code; both drivers carry the
recipe fix and the collapse detectors. This document is what a fresh operator
on different hardware needs.

**What this is.** #723/#726 established that the `1x1` CTM recipe collapses the
environment to rank-1 corners — a `chi_eff = 1` mean-field boundary whose energy
does not respond to χ, with nothing raising. #747 audited which recorded
results were produced that way. Two need re-running on GPU.

---

## What already got settled without compute

The audit (comment 1 on #747) resolved most of the issue from code paths, not
re-runs. The key fact: `python_loop_ctm_converge` has defaulted to
`recipe="2x2"` since 2026-06-10 (PR #597) and `ctm_converge_kwargs` emits no
`recipe`, so **anything measured through it was already correct**.

| result | recipe actually run | verdict |
|---|---|---|
| D=8 (#650) **dense** arm | `2x2` | **valid — do not re-run** |
| D=4 (#646) χ-scan | `2x2` | **valid — do not re-run** |
| D=8 (#650) **split** arm | `1x1` | invalid → **Run 1** |
| D=4 (#646) `A_opt` state | `1x1` | invalid → **Run 2** |
| #638 showcase | `1x1` | retired, not re-run (PR #771) |

The D=8 conclusion was **inverted**: the split arm's "χ-invariant E to 13
digits" was the collapse signature, and the dense arm's "oscillates,
`converged:false`" was the correct side doing real work.

---

## Environment

```bash
git clone https://github.com/tenax-lab/tenax.git && cd tenax
uv sync --extra cuda13          # or --extra cuda12, match the driver
uv run python -c "import jax; print(jax.devices())"
```

Requires x64 (the drivers set `jax_enable_x64` themselves) and one GPU with
**≥40 GB** for Run 1. Run 2 is small (peak ~0.3 GB).

**Device guard.** Both drivers refuse to start if any visible device is under
40 GB. That exists to stop a run silently landing on a display GPU — the
original machine has a 4 GB DGX Display card at index 3, which is why. Pass
`--allow-non-a100` on either driver to disable it; it prints the visible
devices and continues.

**The D=8 driver has a second, stricter gate**, and it is not the 40 GB one.
Its orchestrator picks GPUs from `nvidia-smi` by matching the literal string
`A100` in the device *name*, so **H100s are rejected too** — "≥40 GB" is not
the criterion there. `--allow-non-a100` drops that vendor-string requirement
as well (busy devices and anything named `Display` stay excluded on every
path, since landing on a display card is a failure on any machine).

So, on different hardware:

- A100 box → passes unchanged, do nothing.
- H100 or any other ≥40 GB card → pass `--allow-non-a100`. **Without it Run 1
  does not fail fast**: `_wait_for_free_a100s` polls every 30 s for
  `--gpu-wait-s` (default 1800 s), gives up, and the run aborts with
  `[abort] SU produced no A_opt.pkl` — half an hour to discover the hardware
  was never eligible.
- Anything smaller, or a deliberate small-scale smoke test → same flag.

Pin devices explicitly rather than relying on index order:

```bash
export CUDA_VISIBLE_DEVICES=0        # check nvidia-smi first
```

---

## Run 1 — D=8 split arm at 2x2 (the invalid physics result)

```bash
uv run python examples/heisenberg_d8_chi_scaling.py --path split \
    --outdir runs/d8_rerun_2x2          # add --allow-non-a100 off the A100 box
```

**No code change needed.** PR #768 made `ctm_split_tensor` default to
`recipe="2x2"`, so the driver now runs the correct recipe by construction.
There is no `--gs-recipe` here and none is needed: this driver's `A_opt` comes
from the SU seed, not from a `gs_*` optimization, so the collapsed-environment
contamination that Run 2 has to undo never applied to it.

### What to check, in order

1. **`corner_rank > 1` at every χ** — recorded per cell in `results.csv` since
   PR #770. If it is 1, stop: the reroute did not take and nothing downstream
   is meaningful.
2. **The driver's own χ-frozen warning stays silent.** It prints a
   `WARNING: chi-frozen energies detected` block if any two χ return a
   bit-identical energy, judged per `(D, n_devices, path)`.
3. **Compare against the dense arm's ≈ −0.6055**, which the audit shows was
   already correct. Convergence toward it confirms the inversion. A persistent
   gap is new information and worth its own issue.

> **Judge on `corner_rank`, not on whether the energy moves.** A converged
> environment is flat in χ too — that is what convergence means. The χ-frozen
> helper errs in *both* directions (see #747 comment 3): it fires on converged
> scans, and a collapsed environment is not reliably bit-identical either.
> Rank is the only sound signal.

### Also re-measure, don't just re-confirm

The recorded "≈50–100× faster" and "single-GPU wall moves χ≈112 → ≈448" were
measured on the **`1x1`** split forward. `2x2` builds plaquette projectors
instead of corner-pair ones, so the cost profile is different. The *memory*
claim (χ²·D³·d vs χ²·D⁶, ~12×) is shape-derived and should hold; the χ ceiling
depends on the new peak and needs the ladder walked again.

Budget: the recorded `1x1` split converge was ~14–21 s per χ. `2x2` measured
3–6× slower on small cells, so ~1–2 min per χ plus compile, times the ladder.

---

## Run 2 — D=4 re-optimization at `gs_recipe="2x2"`

```bash
uv run python examples/heisenberg_d4_chi_scaling.py \
    --gs-recipe 2x2 --outdir runs/d4_rerun_2x2
```

`--gs-recipe` is new (this PR); it used to be hard-coded to `1x1`. **2x2 is now
the default**, so the flag above is explicit rather than required. `1x1`
reproduces the old, invalid behaviour for bisection.

The D=4 χ *scan* is already correct and unchanged — it goes through
`python_loop_ctm_converge`. What is contaminated is `A_opt`: it was optimized
against a collapsed environment. So the recorded **+6.04e-3 vs QMC is an upper
bound mixing iPEPS truncation with a badly-optimized state**, and must not be
quoted as a D=4 truncation error until this run replaces it.

This is the cheaper run and the one that produces a genuinely new physics
number. If only one can be done, do this one.

Reference to beat: QMC E/site = **−0.669437**. The recorded (contaminated)
scan saturated at **−0.6633981** at χ≥64.

Delete or move `runs/d4_chi_scaling/A_opt.pkl` first if reusing that outdir —
`optimize_once` returns immediately when it finds one, so a stale `A_opt` from
the `1x1` run would be silently reused.

---

## Reporting back

Post to #747 with:

- `results.csv` from each run (they carry `corner_rank` per cell).
- Whether the D=8 split energy converges toward the dense −0.6055.
- The re-measured D=8 split speed and χ ceiling, flagged as **2x2** numbers so
  they are not compared against the `1x1` ones.
- The new D=4 err-vs-QMC, which supersedes +6.04e-3.

If a run shows `corner_rank == 1` anywhere, that is a regression in #723/#746 —
file it rather than working around it.

---

## Not in scope

- **#638 showcase** — retired, not re-run (PR #771). All 29 recorded cells were
  `gs_num_steps=6`, `converged=false`, with zero anchor rows, so no recipe fix
  would make them a physics result.
- **Frontier (#672)** — both arms ran `1x1`, so its reach claims describe a
  rank-1 corner; the memory/shape half survives. Lower value than Runs 1–2 and
  it depends on Run 1's new split cost profile, so it should follow them.
- **Items 4–7** (chunk×shard gate, QR-CTMRG shard NO-GO, #570 reduced-corner,
  rSVD NO-GO) — memory-locality and shape results, undisturbed by the audit.

## References

- #747 — the issue; comment 1 is the audit, comment 2 the run queue, comment 3
  the correction on what the χ-frozen detector can and cannot tell you.
- #723 / #726 / #746 — the collapse and its two fixes (fused, split).
- #766 / #767 / #769 — defects the collapse had been masking, found while
  fixing it. #767 matters here: the split AD forward can return a
  **non-converged** environment with no flag.
- PR #770 — `_ctm_diagnostics.py` (`ctm_corner_rank`, `check_ctm_env`,
  `frozen_chi_pairs`) and the per-cell `corner_rank` recording both runs rely on.
