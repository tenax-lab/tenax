# Liao 2017 PESS replication audit — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run an SU-only D-sweep (D ∈ {4, 6, 8, 10}) on the kagome S=½ AFM Heisenberg 3-PESS, measuring E/site with two independent probes — a Husimi-tree local mean-field (P1) and our existing Convention-C + CTM (P2) — and compare both against Liao 2017 PRL 118, 137202 Fig 1(a). Outcome decides whether M2b is on the critical path.

**Architecture:** Add a single new pure-JAX helper `pess_local_energy(state, h_tri)` that contracts each kagome triangle's 3-site RDM with bond-λ mean-field gauges (no CTM). Reuse existing `pess_simple_update` (PR #387 kernel, unchanged) and existing `build_pess_loss` for P2. Drive the sweep from a new benchmark script that writes a JSON report.

**Tech Stack:** JAX (complex128), pytest, existing `tenax.algorithms.pess` machinery.

**Design doc:** `docs/plans/2026-05-03-liao2017-pess-replication-design.md` (commit 60b8d08).

---

### Task 1: Pin Liao 2017 target values into a constant

**Files:**
- Create: `src/tenax/algorithms/_liao2017_targets.py`

**Why a separate module:** The benchmark script and the test both reference these numbers; centralizing avoids duplicate magic constants.

**Step 1: Create the constants file.**

Create `src/tenax/algorithms/_liao2017_targets.py` with:

```python
"""Hand-digitized 3-PESS simple-update targets from Liao 2017.

Source: H.J. Liao et al., "Gapless spin-liquid ground state in the S=1/2
kagome antiferromagnet," PRL 118, 137202 (2017), arXiv:1610.04727,
Figure 1(a) (3-PESS simple update curve, blue circles).

Values are read off the figure to ±0.001 precision; treat as a
qualitative target band, not a numerical reference.
"""

from __future__ import annotations

# {D: E/site (3-PESS simple update, S=1/2 kagome AFM Heisenberg, Δ=1)}
LIAO2017_3PESS_SU_FIG1A: dict[int, float] = {
    4: -0.4290,
    6: -0.4340,
    8: -0.4360,
    10: -0.4365,
}

# Asymptotic extrapolation reported in Fig 1(b) inset:
LIAO2017_3PESS_SU_INF: float = -0.43752  # ±0.00006

# Tolerance band for "matches Liao within figure-readout error":
LIAO2017_FIGURE_READOUT_TOL: float = 0.002
```

**Step 2: Commit.**

```bash
git add src/tenax/algorithms/_liao2017_targets.py
git commit -m "feat(pess): pin Liao 2017 PRL 118 137202 Fig 1(a) target table"
```

---

### Task 2: Write failing test for `pess_local_energy` — basic invariants

**Files:**
- Create: `tests/test_pess_local_energy.py`

**Step 1: Write the failing test.**

```python
"""Tests for the Husimi-tree local energy probe on kagome 3-PESS."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    pess_local_energy,
    pess_simple_update,
)


@pytest.mark.algorithm
def test_pess_local_energy_returns_real_finite():
    """Energy of a random 3-PESS state must be a finite real scalar."""
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    h_tri = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    e = pess_local_energy(state, h_tri)
    assert jnp.isfinite(e)
    assert jnp.abs(jnp.imag(e)) < 1e-10


@pytest.mark.algorithm
def test_pess_local_energy_d2_classical_band():
    """At D=2, SU-converged 3-PESS lies near the classical 120° energy.

    The classical Heisenberg energy per site on kagome with S=1/2 is
    -|S|^2 * J/2 = -0.125 (per nearest-neighbor pair), times the 4 NN per
    site / 2 = 2 pairs per site / kagome counting → roughly -0.25 / site
    in the J=1 convention. We assert a generous band that catches sign
    flips and gross factor errors but tolerates D=2 quantum corrections.
    """
    state = IPESSState.random(D=2, d=2, key=jax.random.PRNGKey(0))
    h_tri = kagome_triangle_xxz_hamiltonian(delta=1.0, d=2)
    state = pess_simple_update(
        state,
        h_tri,
        dt_schedule=[(0.1, 100), (0.01, 100)],
        D_max=2,
    )
    e = float(jnp.real(pess_local_energy(state, h_tri)))
    assert -0.40 < e < -0.20, f"D=2 SU energy out of band: {e}"
```

**Step 2: Run the test, verify it fails on import.**

```bash
cd /home/yjkao/tenax/.worktrees/spin1-pess-ad
uv run pytest tests/test_pess_local_energy.py -v
```

Expected: `ImportError: cannot import name 'pess_local_energy' from 'tenax.algorithms.pess'`.

---

### Task 3: Implement `pess_local_energy` in `pess.py`

**Files:**
- Modify: `src/tenax/algorithms/pess.py` — append at end of file, after `pess_simple_update`.

**Step 1: Add the helper.**

Append to `src/tenax/algorithms/pess.py`:

```python
def pess_local_energy(state: IPESSState, h_tri: np.ndarray) -> jnp.ndarray:
    """Husimi-tree mean-field energy per kagome site for a 3-PESS state.

    Computes ``<H_tri>`` on the up-triangle and the down-triangle
    independently, treating the bond-λ singular values on the *external*
    legs of each triangle as a mean-field environment (sqrt(λ) on each
    side of the RDM, full λ in the squared amplitude). The two triangle
    energies are averaged; for an SU-converged state they should agree
    by triangle symmetry. ``E/site = (E_up + E_down) / 3``.

    This is the cheapest energy estimate consistent with the bond-gauge
    information SU produces. It does not match Liao 2017's MPS-projection
    measurement quantitatively (Liao uses ``D_mps ≈ 4·D²`` on a 1D MPS
    basis, which captures more correlation), but it tracks the
    SU-correctness of the kernel without going through the
    Convention-C + CTM measurement that ``build_pess_loss`` uses.

    Args:
        state: Input :class:`IPESSState`. Typically the output of
            :func:`pess_simple_update`.
        h_tri: ``(d**3, d**3)`` Hermitian triangle Hamiltonian in the
            ``(p_a, p_b, p_c)`` row-major basis (matches
            :func:`kagome_triangle_xxz_hamiltonian`).

    Returns:
        Real scalar JAX array — the energy per kagome site.
    """
    R_a, R_b, R_c = state.R_a, state.R_b, state.R_c
    T_u, T_d = state.T_u, state.T_d
    lam_au, lam_bu, lam_cu = state.lambdas[0:3]
    lam_ad, lam_bd, lam_cd = state.lambdas[3:6]

    dtype = R_a.dtype
    d = R_a.shape[2]
    h_tri_jax = jnp.asarray(h_tri).astype(dtype).reshape(d, d, d, d, d, d)

    def _triangle_energy(
        Ra: jax.Array,
        Rb: jax.Array,
        Rc: jax.Array,
        T: jax.Array,
        lam_int_a: jax.Array,
        lam_int_b: jax.Array,
        lam_int_c: jax.Array,
        lam_ext_a: jax.Array,
        lam_ext_b: jax.Array,
        lam_ext_c: jax.Array,
    ) -> jax.Array:
        """One triangle's energy with bond gauges (axis 0 = ext, axis 1 = int)."""
        # sqrt(λ_ext) on each external leg → λ_ext in the squared amplitude.
        sqrt_la = jnp.sqrt(jnp.maximum(jnp.real(lam_ext_a), 1e-30)).astype(dtype)
        sqrt_lb = jnp.sqrt(jnp.maximum(jnp.real(lam_ext_b), 1e-30)).astype(dtype)
        sqrt_lc = jnp.sqrt(jnp.maximum(jnp.real(lam_ext_c), 1e-30)).astype(dtype)

        # Full λ_int absorbed onto axis 1 of each R (the simplex-side bond).
        Q_a = jnp.einsum("i,ijp,j->ijp", sqrt_la, Ra, lam_int_a.astype(dtype))
        Q_b = jnp.einsum("i,ijp,j->ijp", sqrt_lb, Rb, lam_int_b.astype(dtype))
        Q_c = jnp.einsum("i,ijp,j->ijp", sqrt_lc, Rc, lam_int_c.astype(dtype))

        # Triangle ket ψ[A, B, C, p_a, p_b, p_c]
        psi = jnp.einsum("Aap,Bbq,Ccr,abc->ABCpqr", Q_a, Q_b, Q_c, T)

        norm = jnp.einsum("ABCpqr,ABCpqr->", psi.conj(), psi)
        e_un = jnp.einsum(
            "ABCpqr,pqrPQR,ABCPQR->",
            psi.conj(),
            h_tri_jax,
            psi,
        )
        return jnp.real(e_un / (norm + 1e-30))

    # Up-triangle: int = up-bonds (R axis 1), ext = down-bonds (R axis 0).
    e_up = _triangle_energy(
        R_a, R_b, R_c, T_u,
        lam_au, lam_bu, lam_cu,
        lam_ad, lam_bd, lam_cd,
    )

    # Down-triangle: int = down-bonds (axis 0), ext = up-bonds (axis 1).
    # Swap R axes 0↔1 so the helper's "axis 1 = int" convention holds.
    e_dn = _triangle_energy(
        R_a.transpose(1, 0, 2),
        R_b.transpose(1, 0, 2),
        R_c.transpose(1, 0, 2),
        T_d,
        lam_ad, lam_bd, lam_cd,
        lam_au, lam_bu, lam_cu,
    )

    return (e_up + e_dn) / 3.0
```

**Step 2: Run the tests, verify both pass.**

```bash
uv run pytest tests/test_pess_local_energy.py -v
```

Expected: 2 passed.

**Step 3: Commit.**

```bash
git add src/tenax/algorithms/pess.py tests/test_pess_local_energy.py
git commit -m "feat(pess): pess_local_energy — Husimi-tree mean-field probe"
```

---

### Task 4: Export `pess_local_energy` as public API

**Files:**
- Modify: `src/tenax/__init__.py`

**Step 1: Read current entries.**

```bash
grep -n "pess_simple_update\|build_pess_loss" src/tenax/__init__.py
```

Expected to see existing lazy-import + `__all__` entries near lines 143 and 426.

**Step 2: Add `pess_local_energy` next to `pess_simple_update`.**

In the `_LAZY_IMPORTS` dict (around line 143), add after `"pess_simple_update": ...`:

```python
    "pess_local_energy": ("tenax.algorithms.pess", "pess_local_energy"),
```

In `__all__` (around line 426), add after `"pess_simple_update",`:

```python
    "pess_local_energy",
```

**Step 3: Verify the import works.**

```bash
uv run python -c "import tenax; print(tenax.pess_local_energy)"
```

Expected: `<function pess_local_energy at 0x...>`.

**Step 4: Commit.**

```bash
git add src/tenax/__init__.py
git commit -m "feat(pess): export pess_local_energy as public API"
```

---

### Task 5: Write the benchmark script

**Files:**
- Create: `examples/kagome_spin12_pess_liao2017_replication.py`

**Step 1: Create the script.**

```python
#!/usr/bin/env python3
"""Liao 2017 (PRL 118 137202) 3-PESS SU replication audit.

Runs a D-sweep at D in {4, 6, 8, 10} on the spin-1/2 kagome AFM Heisenberg
model. For each D, measures E/site two ways:

  P1 — Husimi-tree mean-field probe (`pess_local_energy`): contracts one
       triangle's 3-site RDM with bond-λ gauges as the environment. No CTM.
  P2 — Convention-C + CTM (`build_pess_loss`): maps the PESS to a
       square-iPEPS supersite, runs CTM at χ = 2*D**2, evaluates energy
       through the standard 2-site RDM helpers.

Both numbers are written to JSON alongside Liao 2017's hand-digitized
Fig 1(a) target. The diff (P1 - target, P2 - target) tells us whether
the SU kernel matches Liao (P1 ≈ target) and whether Convention-C is
the bias source (P2 above target).

Usage::

    python examples/kagome_spin12_pess_liao2017_replication.py \\
        --output examples/kagome_spin12_pess_liao2017_replication.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

from tenax.algorithms._liao2017_targets import (
    LIAO2017_3PESS_SU_FIG1A,
    LIAO2017_3PESS_SU_INF,
)
from tenax.algorithms.ipeps_config import CTMConfig
from tenax.algorithms.pess import (
    IPESSState,
    kagome_triangle_xxz_hamiltonian,
    kagome_xxz_pess_cg_gates,
    pess_local_energy,
    pess_simple_update,
)
from tenax.algorithms.pess_optimize import build_pess_loss

DELTA = 1.0
D_PHYS = 2
SU_SCHEDULE = [(0.1, 200), (0.01, 200), (0.001, 100), (0.0001, 100)]
D_LIST_DEFAULT = (4, 6, 8, 10)


def _make_ctm_config(chi: int) -> CTMConfig:
    return CTMConfig(
        chi=chi,
        max_iter=30,
        min_iter=4,
        conv_tol=1e-7,
        projector_method="svd",
        forward_gauge="phase",
        ctm_conv_method="elementwise",
        gmres_tol=1e-5,
        gmres_maxiter=80,
        gmres_restart=30,
        chi_ramp=None,
    )


def run_one(D: int, seed: int = 0, verbose: bool = False) -> dict:
    H = kagome_triangle_xxz_hamiltonian(delta=DELTA, d=D_PHYS)
    cg_gates = kagome_xxz_pess_cg_gates(delta=DELTA, d=D_PHYS)

    state = IPESSState.random(D=D, d=D_PHYS, key=jax.random.PRNGKey(seed))

    t_su = time.perf_counter()
    state = pess_simple_update(state, H, dt_schedule=SU_SCHEDULE, D_max=D)
    t_su = time.perf_counter() - t_su

    t_p1 = time.perf_counter()
    e_p1 = float(pess_local_energy(state, H))
    t_p1 = time.perf_counter() - t_p1

    chi = 2 * D * D
    config = _make_ctm_config(chi=chi)
    loss_fn = build_pess_loss(cg_gates, config)
    t_p2 = time.perf_counter()
    e_p2 = float(loss_fn(state).real)
    t_p2 = time.perf_counter() - t_p2

    target = LIAO2017_3PESS_SU_FIG1A.get(D)

    record = {
        "D": D,
        "chi": chi,
        "seed": seed,
        "su_schedule": [list(s) for s in SU_SCHEDULE],
        "e_p1_husimi": e_p1,
        "e_p2_ctm": e_p2,
        "liao2017_target": target,
        "delta_p1_target": (e_p1 - target) if target is not None else None,
        "delta_p2_target": (e_p2 - target) if target is not None else None,
        "t_su_seconds": t_su,
        "t_p1_seconds": t_p1,
        "t_p2_seconds": t_p2,
    }
    if verbose:
        print(
            f"  D={D:2d} χ={chi:3d}  P1={e_p1:.6f}  P2={e_p2:.6f}  "
            f"target={target:.6f}  Δ_P1={record['delta_p1_target']:+.4f}  "
            f"Δ_P2={record['delta_p2_target']:+.4f}",
            flush=True,
        )
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--D",
        type=int,
        nargs="*",
        default=list(D_LIST_DEFAULT),
        help="Bond dimensions (default: 4 6 8 10)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_suffix(".json"),
        help="Path to JSON results file",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"Liao 2017 PRL 118 137202 3-PESS SU replication", flush=True)
    print(f"  asymptotic target E0 = {LIAO2017_3PESS_SU_INF:.6f}", flush=True)
    print(f"  D list: {args.D}", flush=True)
    print(f"  SU schedule: {SU_SCHEDULE}", flush=True)

    results = []
    for D in args.D:
        print(f"\n=== D = {D} ===", flush=True)
        record = run_one(D=D, seed=args.seed, verbose=True)
        results.append(record)

    payload = {
        "reference": "Liao et al., PRL 118, 137202 (2017), arXiv:1610.04727",
        "asymptotic_target_E0": LIAO2017_3PESS_SU_INF,
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(results)} record(s) to {args.output}", flush=True)


if __name__ == "__main__":
    main()
```

**Step 2: Smoke-test at D=4 only (cheap, fast).**

```bash
uv run python examples/kagome_spin12_pess_liao2017_replication.py \
    --D 4 --output /tmp/liao_smoke.json
```

Expected: prints one row "D= 4 χ= 32 P1=… P2=… target=-0.4290 Δ_P1=… Δ_P2=…", writes JSON. Wall time ≈ 1–3 minutes. P1 should be in (-0.45, -0.40). P2 should be ≥ P1 (likely around -0.42).

If P1 is wildly off (e.g. positive, or below -0.5): stop and debug `pess_local_energy`.

**Step 3: Commit script.**

```bash
git add examples/kagome_spin12_pess_liao2017_replication.py
git commit -m "feat(pess): Liao 2017 PRL 3-PESS SU replication benchmark"
```

---

### Task 6: Run the full sweep and commit results JSON

**Files:**
- Create (run output): `examples/kagome_spin12_pess_liao2017_replication.json`

**Step 1: Run the full D-sweep.**

```bash
uv run python examples/kagome_spin12_pess_liao2017_replication.py
```

Expected wall time: D=4 ~2 min, D=6 ~5 min, D=8 ~10 min, D=10 ~25 min on CPU. Total ~40–50 min. The script prints per-D progress; if any D's P1 is positive or wildly off-band, stop and debug before proceeding.

**Step 2: Inspect the JSON.**

```bash
cat examples/kagome_spin12_pess_liao2017_replication.json | head -80
```

Confirm there are 4 records, each has both `e_p1_husimi` and `e_p2_ctm`, and the targets line up with `LIAO2017_3PESS_SU_FIG1A`.

**Step 3: Decide which outcome row of the design doc we landed on.**

Compute the average `delta_p1_target` and `delta_p2_target` across all four D values:

- **|Δ_P1| < 0.005** AND **|Δ_P2| > 0.005** → "P1 ≈ Liao, P2 above Liao" (Convention-C is the bias source). M2b decisively justified.
- **|Δ_P1| > 0.005** → "P1 above Liao" — there's an SU-kernel residual issue. **Stop here and re-open the SU audit.**
- **|Δ_P1| < 0.005** AND **|Δ_P2| < 0.005** → "Both ≈ Liao" — Convention-C acceptable, M2b deprioritized.

Record the chosen outcome row in the PR body when opening the PR (Task 9).

**Step 4: Commit results.**

```bash
git add examples/kagome_spin12_pess_liao2017_replication.json
git commit -m "feat(pess): Liao 2017 replication results — D in {4,6,8,10}"
```

---

### Task 7: Fix the misciting in benchmark docstring

**Files:**
- Modify: `examples/kagome_spin12_pess_ad_benchmark.py:1-24` (docstring)

**Step 1: Read the current docstring.**

```bash
sed -n '1,25p' examples/kagome_spin12_pess_ad_benchmark.py
```

**Step 2: Replace the "Liao 2019 PRX 9 031041" attribution with the correct one.**

Use the Edit tool to replace the docstring header so it cites Liao 2017 PRL 118, 137202 (kagome PESS, no AD) for the SU baseline, and notes that the AD step on top is a Tenax extension, not a Liao replication. The replacement docstring:

```python
"""Spin-½ kagome AFM Heisenberg AD-iPESS benchmark.

Pipeline:
  1. Triangle simple update (3-PESS, HOSVD truncation per simplex) —
     this matches the SU kernel of Liao et al., PRL 118, 137202 (2017),
     arXiv:1610.04727 ("Gapless spin-liquid ground state in the S=1/2
     kagome antiferromagnet"). Liao 2017 reports E/site → −0.43752(6)
     in the large-D limit (Fig 1(b) inset).
  2. AD optimization through the Tenax square-CG-iPEPS CTM
     (Convention C: PESS → 1-site square supersite, χ = 2·D²). This
     step is a Tenax extension; Liao 2017 has no AD optimization for
     kagome PESS. The AD machinery is reused from Liao 2019
     (PRX 9, 031041, "Differentiable Programming Tensor Networks"),
     which applies AD to *square-lattice* iPEPS, not kagome PESS.

For an SU-only Liao 2017 replication audit (no AD, two energy probes),
see ``kagome_spin12_pess_liao2017_replication.py``.

Usage:
    python examples/kagome_spin12_pess_ad_benchmark.py --D 2 --chi 8
    python examples/kagome_spin12_pess_ad_benchmark.py --D 4 --chi 32 --max-iter 80
    python examples/kagome_spin12_pess_ad_benchmark.py --sweep   # D ∈ {2,4,6}, χ=2D²

Output:
    JSON file at ``--output`` (default
    ``examples/kagome_spin12_pess_ad_benchmark.json``) with one entry per
    ``(D, chi)`` pair containing the per-kagome-site SU and AD energies.
"""
```

**Step 3: Commit.**

```bash
git add examples/kagome_spin12_pess_ad_benchmark.py
git commit -m "docs(pess): correct Liao reference in AD benchmark docstring"
```

---

### Task 8: Update memory entries

**Files:**
- Modify: `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_pess_su_collapse_bug.md`
- Modify: `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_pess_ad_stalls_d4.md`
- Create: `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_liao2017_replication.md`
- Modify: `/home/yjkao/.claude/projects/-home-yjkao-tenax/memory/MEMORY.md`

These edits live in the auto-memory directory, not the repo, so no git commit is involved.

**Step 1: Read the two memory files.**

```bash
cat /home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_pess_su_collapse_bug.md
cat /home/yjkao/.claude/projects/-home-yjkao-tenax/memory/project_pess_ad_stalls_d4.md
```

**Step 2: Edit `project_pess_su_collapse_bug.md`.**

Replace every occurrence of "Liao 2019" with "Liao 2017 (PRL 118, 137202)" and replace the target value `-0.420` (which is *our* SU-via-CTM number, not Liao's) with the actual Liao 2017 D=4 target `-0.4290 ± 0.001` (Fig 1(a) readout). Add a note that our `-0.420` measurement came from the Convention-C + CTM probe, not from a Liao-faithful measurement.

**Step 3: Edit `project_pess_ad_stalls_d4.md`.**

Replace "Liao -0.4324" with "Liao 2017 (PRL 118, 137202) D=4 SU target -0.4290; asymptote -0.43752". Add a sentence: "Liao 2017 has no AD baseline for kagome PESS — the AD step is a Tenax extension. The stall is failing to recover the SU energy via Tenax's CTM probe, and the SU energy itself is biased ~0.010 above Liao's value via that same probe. See `project_liao2017_replication.md` for the SU-only audit results."

**Step 4: Create `project_liao2017_replication.md`.**

```markdown
---
name: Liao 2017 PESS replication audit results
description: D-sweep at D in {4,6,8,10}, two energy probes (Husimi-tree P1 and Convention-C CTM P2) vs Liao 2017 PRL 118 137202 Fig 1(a) targets — outcome decides whether M2b is on the critical path
type: project
---

D-sweep at D ∈ {4, 6, 8, 10} of spin-1/2 kagome AFM Heisenberg 3-PESS SU
(no AD), reported in `examples/kagome_spin12_pess_liao2017_replication.json`.
Compares two energy probes against Liao 2017 PRL 118, 137202 Fig 1(a):

- **P1 (Husimi-tree)** — `pess_local_energy`, no CTM
- **P2 (Convention-C + CTM)** — `build_pess_loss` at χ = 2·D²

**Outcome row (fill in after run):** [P1 ≈ Liao, P2 above / P1 above / Both ≈]

**Why:** Pre-M2b sanity check that our SU kernel matches Liao 2017 and
that the historic "AD stalls at -0.343 vs Liao -0.4324" narrative was
comparing an apples-to-oranges target (wrong paper, wrong probe).

**How to apply:** Before opening any new line of work that depends on
"matching Liao on kagome PESS," consult this file's outcome row to see
which probe (P1 / P2) is the relevant baseline. For the AD path, P2 is
the relevant baseline (AD operates through the same CTM probe). For the
SU path, P1 is the cleaner baseline.
```

**Step 5: Update `MEMORY.md` index.**

Append (or replace if entries exist):

```markdown
- [project_liao2017_replication.md](project_liao2017_replication.md) — Liao 2017 D-sweep audit: P1 vs P2 vs Fig 1(a)
```

And ensure the entries for `project_pess_su_collapse_bug.md` and `project_pess_ad_stalls_d4.md` mention Liao 2017 (not Liao 2019) in their one-line hooks.

---

### Task 9: Open PR

**Step 1: Push the branch and open the PR.**

```bash
git push -u origin feat/spin1-xxz-pess-ad
gh pr create --title "audit: Liao 2017 PESS SU replication (D ∈ {4,6,8,10})" --body "$(cat <<'EOF'
## Summary
- Pre-M2b sanity check that our 3-PESS SU kernel matches Liao 2017 PRL 118, 137202 (the actual kagome PESS reference — *not* Liao 2019, which we'd been miscitiang).
- Adds `pess_local_energy` (Husimi-tree mean-field probe; pure JAX, no CTM).
- Adds `examples/kagome_spin12_pess_liao2017_replication.py` running a D-sweep with two energy probes (P1 Husimi-tree, P2 Convention-C CTM) and writing JSON.
- Corrects the misciting in the AD benchmark docstring.

## Outcome
Filled in from the JSON: |Δ_P1| = …, |Δ_P2| = …. Decision row: …

## Test plan
- [x] `pytest tests/test_pess_local_energy.py -v` → 2 passed
- [x] D=4 smoke run finishes in <3 min and lands in (-0.45, -0.40)
- [x] Full D ∈ {4,6,8,10} sweep completes; JSON committed
- [ ] Verify PR's Tests (Python 3.11/3.12/macOS) pass on CI
EOF
)"
```

Fill the `Outcome` block from `examples/kagome_spin12_pess_liao2017_replication.json`. The numerical values should be the average over all 4 D values.

---

## Out-of-scope reminders

- No SU kernel changes — the kernel is unchanged from PR #387.
- No AD changes — that path is orthogonal to this audit.
- No Full Update implementation — explicit non-goal.
- No MPS-projection energy — explicit non-goal.

## Reverse roll-out

If we land on the "P1 above Liao" outcome (residual SU bug), revert Task 8's claim that the SU kernel is Liao-faithful; the audit itself stays in the codebase as a regression check.
