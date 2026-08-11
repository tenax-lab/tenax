# `ipeps()` A≠B is a gauge artifact — and the root cause of #667

**Date:** 2026-08-11
**Question carried in:** does `ipeps()` return a non-uniform `(A, B)` despite
`sublattice_rotate_gate`, blocking the uniform 1x1 state that `ctm_tensor_c4v` /
#833 needs?
**Answer:** the recorded `||A-B||/||A|| = 1.72` was **mostly gauge** — but there
was a real non-uniformity underneath it, and chasing it found the root cause of
**#667** (simple update collapses the state).
**Hardware:** CPU (`JAX_PLATFORMS=cpu`), f64. Every number here is seconds-to-minutes
to reproduce; nothing needed a GPU.
**Probes:** `probe_gauge_a_vs_b.py`, `probe_h1_dimerization.py`,
`probe_h2_double_lambda.py`, `probe_h3_fullsweep_and_1x1.py`, `probe_d3_rank.py`,
`probe_h4_combined.py`, `probe_factorial.py`, `probe_unrotated_control.py`,
`probe_chi_scan.py` (session scratchpad — **not** committed; see "If someone extends this").

Everything below was first measured by reimplementing the two SU routines in a
probe with one flag flipped at a time, so each defect could be attributed
separately. **Both corrections were subsequently landed** — see "What was
landed" at the end.

## The two defects

Both live in the simple-update path and are independent of each other.

### Defect 1 — the sweep covers only half the bonds

`ipeps()` (`ipeps.py:441-449`) alternates `su_h(A,B)` and `su_v(A,B)` and nothing
else. On a checkerboard the unit cell has **four** bonds; `A.r-B.l` and `A.d-B.u`
are evolved, **`B.r-A.l` and `B.d-A.u` never are**.

Because A is therefore always the left/top site of every gate and B always the
right/bottom, the `sqrt(sigma)` of steps 7-8 lands on A's `r`,`d` legs and B's
`l`,`u` legs and nowhere else. So in the assembled lattice, half the bonds carry
the full Schmidt weight and **half carry none**.

### Defect 2 — Γ absorbs `sqrt(sigma)` that is also stored as the new λ

`ipeps_simple_update.py:94-109` sets `lam_h_new = sigma` **and** scales `A_new.r`
/ `B_new.l` by `sqrt(sigma)`. Step 1 of the next sweep re-absorbs the full
`lam_h` onto the shared leg, so that bond carries **λ^1.5**, not λ. Textbook
Vidal simple update keeps Γ bare: `Gamma_A = lam_outer^-1 U`, with λ living only
on the bond.

## What each defect does, isolated

D=2, dt=0.05, 400 steps, χ=24, rotated (ferromagnetic) gate. Metric is the 1x1
CTM energy of the reconstructed physical tensor; `|E_A - E_B|` is the uniformity
check (build the uniform ansatz from A alone, then from B alone).

| sweep | Γ | λ (normalized) | E_1x1 | \|E_A−E_B\| | rank(C1) | verdict |
|---|---|---|---|---|---|---|
| half | √σ **(shipped)** | [1, 0.0256] | −0.4999989999 | 1.8e-06 | 3 | not uniform |
| half | bare | [1, 0.3547] | −0.4626659145 | 2.4e-02 | 4 | not uniform |
| full | √σ | [1, 0.0255] | −0.5439000684 | 1.1e-16 | 6 | uniform but near-product |
| **full** | **bare** | [1, 0.1620] | **−0.6593342876** | 3.7e-15 | 24 | **uniform AND correct** |

Reference: D=2 iPEPS Heisenberg ≈ **−0.6599**; product state = −0.5.

**Neither correction works alone.** That is why this took four hypotheses: bare-Γ
tested against the half-sweep looks *worse* (−0.4627, and on the 2-site path it
produced −0.86, i.e. below the variational bound on a divergent CTM), which
refutes it if you stop there. It only pays off once the sweep is complete.

## Answering the original question

**A and B are the same physical tensor.** With the full sweep, the uniform 1x1
ansatz built from A and the one built from B agree to machine precision:

| D | \|E_1x1(A) − E_1x1(B)\| |
|---|---|
| 2 | 0.0 – 3.7e-15 |
| 3 | 1.1e-16 – 5.3e-10 |

The elementwise `||A-B||/||A||` stays ≈1.7–2.0 throughout — it is measuring the
gauge, exactly as suspected. The suspicion was right to be raised, and the
memory's warning against trusting it was correct.

**But it was not *only* gauge.** On the shipped half-sweep the state is
genuinely non-uniform, and the giveaway is not `||A-B||`:

- a spurious **dimerization** — `E_h(A left, B right) = −0.2705` vs
  `E_h(B left, A right) = −0.2457`, a 2.5e-2 gap between two bonds that a
  uniform state makes identical. Full sweep takes it to 2.4e-3.
- site A has **two different 1-site RDMs** depending on which 2-site RDM you
  trace it out of: `||rho_A(h) − rho_A(v)|| = 0.366`. A 1-site RDM is a property
  of the site; this cannot happen for a valid state.
- `rho_A` has a **negative eigenvalue, −0.0935**. Not a density matrix.
- the CTM barely converges: `diff = 6.1e-2` after 600 sweeps.

All four track the same defect and all four clear together.

### The positive control that makes the uniformity claim meaningful

Run the identical probe on the **unrotated** gate and the same measurement
reports textbook Néel — `<Sx,Sz>_A = (+0.392, +0.309)` against
`<Sx,Sz>_B = (−0.434, −0.245)`. So the probe detects non-uniformity when it is
there; "uniform" in the rotated frame is a result, not a blind spot.

## Root cause of #667, and what the broken fixed point actually is

#667 records "simple update collapses iPEPS state" and that *smaller dt is
worse*. Both are Defect 2, and the mechanism is now explicit — the shipped SU
converges to the **product state**, and all the entanglement it appears to carry
is a Trotter artifact linear in dt:

| dt | steps | λ (normalized) | E_1x1 |
|---|---|---|---|
| 0.05 | 200 / 800 | [1, 2.56e-2, 2.35e-6] | −0.5439049983 |
| 0.01 | 800 / 3000 | [1, 5.02e-3, 3.76e-9] | −0.5097508885 |
| 0.002 | 3000 | [1, 1.00e-3, 6.00e-12] | −0.5019900063 |

λ₂ ∝ dt, λ₃ ∝ dt², E → **−0.5 exactly** (the product-state energy: 2 bonds ×
−0.25). The 200-step and 800-step rows are identical to every digit, so this is
the converged fixed point, not an unfinished run. "Smaller dt is worse" is not a
Trotter-accuracy puzzle — dt *is* the only thing entangling the state.

This also disposes of a trap: at D=3 the shipped path gives λ₃ ≈ 2e-6, so a
state nominally at D=3 is really D=2. Any "D=3" result from `ipeps()` today is
not a D=3 result.

### χ-converged energies with the two corrections applied

Corner rank tracks χ (no #747-style collapse), and E is flat to 1e-12:

| D | χ=8 | χ=16 | χ=24 | χ=32 | χ=48 | rank(C1) at χ=48 |
|---|---|---|---|---|---|---|
| 2 | −0.659334287623 | −0.659334287637 | −0.659334287637 | −0.659334287637 | −0.659334287637 | 25 |
| 3 | −0.663195544316 | −0.663196089109 | −0.663196204005 | −0.663196204304 | −0.663196204307 | 48 |

D=2 lands 6e-4 above the −0.6599 reference — ordinary simple-update error, since
SU is not variationally optimal. D=3 goes lower, consistent with the dense-AD
study's −0.6602 → −0.6642 band.

**Frame control:** with both corrections the unrotated (Néel) run gives
−0.6295661257 against the rotated run's −0.6293135568 on the same 2-site CTM
path — agreeing to 2.5e-4, so the fix is not an artifact of the rotation trick.
(Both are measured on the legacy `ctm_2site`, which is itself unconverged here —
see below — which is why they sit above the χ-converged −0.6593.)

## #833 is unblocked, with one caveat

#833 needs a physical, genuinely-D=3 uniform state for `ctm_tensor_c4v`. With
both corrections, SU delivers one: A ≡ B to 1e-16, λ = [1, 0.166, **0.0156**]
(a real third Schmidt value, not 2e-6), corner rank = χ. That is the route
[[su-rotated-gate-a-neq-b]] was blocking on, and it does not require the
zero-padding route that [[ipeps-padded-warm-start-is-a-saddle]] ruled out.

The remaining gate on #833 is state *quality*, not uniformity: SU is not
variationally optimal, so this is a −0.6632 state rather than the −0.6642 the
dense AD study reaches at D=3.

## Loose ends found along the way

- **`ctm_2site` does not converge well.** Every run of the legacy dense 2-site
  CTM used by `ipeps()` stalled: `diff` 1.6e-3 to 6.5e-2 after 600 sweeps at
  conv_tol=1e-10, even on the corrected, well-behaved state. The 1-site
  `ctm_tensor(recipe="2x2")` path converged to 1e-15 on the same tensors. Possibly
  related to #780. This is why the two paths disagree by 0.03 in the control
  table above, and it is a separate issue from either defect here.
- **bare-Γ has an inverse-λ blow-up.** D=3 at dt=0.01 returned `lam = [nan, nan,
  nan]`. Dividing by λ without a pseudo-inverse cutoff is the classic simple-update
  instability; a fix must add one (`1/λ` only where `λ > tol`).
- **The fermionic path looks like it shares Defect 2 — untested.**
  `fermionic_ipeps.py:299,381` absorb `sqrt_sig` the same way. It is a *1-site* SU
  (`B = A by periodicity`, `fermionic_ipeps.py:423-424`) so Defect 1 does not
  apply as stated, but A is taken from `U_final` with `sqrt(sigma)` on `r` and
  nothing on `l`, so each bond would carry √λ rather than λ — the same one-sided
  asymmetry. Worth checking against #392 ("simple update broken for fermions,
  E→0"). **This is a code-reading lead, not a measurement.**
- `pess.py` also implements a simple update; not examined.

## What was landed

Both corrections, on `fix/667-simple-update-product-state-fixed-point`:

- `ipeps_simple_update.py` — `Gamma` stays bare; `_to_physical_tensor()` rebuilds
  the symmetric-gauge tensor; `1/lambda` becomes a pseudo-inverse with a 1e-12
  cutoff (the D=3/dt=0.01 NaN below).
- `ipeps.py` — the sweep cycles all four bonds and converts out of Vidal form
  before the CTM.
- `tests/test_su_667_product_state.py` — in the **`core`** bucket, since #667
  survived precisely because no test asserted a simple-update energy.

Deliberately **not** landed: the `ctm_2site` non-convergence, `ipeps()` ignoring
`config.unit_cell`, and the fermionic lead — all recorded below as separate work.

## If someone extends this

- The probes are in a **session-scoped scratchpad** and will be gone. They are
  small; the essential one is the 2x2 factorial — reimplement `su_h`/`su_v` with
  a `bare_gamma` flag, drive them with a 4-phase sweep
  (`su_h(A,B)`, `su_v(A,B)`, `su_h(B,A)`, `su_v(B,A)`), reconstruct the physical
  tensor as `Gamma * sqrt(lam)` on all four legs, and measure `E_1x1` from A and
  from B separately.
- **`ipeps()` never reads `config.unit_cell`** (`ipeps.py:414-421`) — it always
  runs the 2-site path from two *different* PRNG keys. Passing `unit_cell="1x1"`
  does nothing. Worth its own issue.
- No test asserts an SU energy, which is why #667 survived. A test pinning D=2 SU
  to ≈−0.659 would have caught both defects.
- Only the Heisenberg gate at D∈{2,3} was measured. Nothing here has been checked
  against another model, D>3, or the SymmetricTensor path.
