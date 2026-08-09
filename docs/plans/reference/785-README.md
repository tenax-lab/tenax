# #785 — no diagnostic predicts root-implicit gradient error in the clamped set

Reference material for [#785](https://github.com/tenax-lab/tenax/issues/785).
`785-remeasure.py` is the harness; this file is what you need to know before
trusting a number that comes out of it.

```bash
uv run python docs/plans/reference/785-remeasure.py
uv run python docs/plans/reference/785-remeasure.py --check-clamp   # detector guard
uv run python docs/plans/reference/785-remeasure.py --chi 4 6 8
```

## Why this exists

#785 has withdrawn two candidate laws (`r_next`, and an `(eps*kappa^2)^2`
scaling) after each looked monotone over a handful of states. Before adding a
third, the measurement those refutations rest on needs to hold up. Three
problems with it, all reproducible:

### 1. The issue is not reproducible as written

The probe scripts behind both of its tables were never committed. The only
squash knob in the test suite, `_site_tensor(D, d, seed, eps)`, is **not** the
state it used:

| | #785's reproduction block | `_site_tensor(eps=1e-3)` |
|---|---|---|
| `analytic g·v` | `1.196767600118e+00` | `\|grad\| = 3.339e-09` |
| rank at χ=4 | `squash1e-2` 4/4, `squash1e-3` 3/4 | 3/4, 1/4 |

`_site_tensor` scales the *entire* random part, so as `eps → 0` the state
limits to a product state and the energy and gradient collapse together
(`E0 = 0.4999999999995`, `|grad| = 3.4e-18` at `eps=1e-6`). Whatever "squashed"
meant, it was not that.

`squashed()` in the harness is the reconstruction: it squashes the **virtual
bond spectrum** (`w = (1, ε)` on each virtual leg) and leaves the dominant
block alone, so `E` and `dE/dA` stay O(1) while the half-infinite environment
picks up singular directions at O(ε²), O(ε⁴). That is a knob on the *cut*,
which is what #785 is about.

### 2. The FD reference was validated on the wrong axis

The control in the issue varies **sweep depth** (10/20/40/80/160) and finds a
plateau. That correctly rules out *truncation* of the explicit map. It does not
touch the *floating-point* floor, which is governed by `h`: a central
difference of an O(1) energy has an absolute noise floor near `ε·|E|/h ≈ 1e-11`
at `h = 1e-5`. Different confounds, and `h` was never scanned.

Scanning it on the frozen SU state — the one row that is a real physical state,
which the issue reports as a single number, 3.1e-08:

```
h        rel err
1e-3     3.473e-05
1e-4     3.687e-08     <- 8x better
1e-5     3.080e-07
1e-6     3.080e-07
```

Worse, on a near-product state (`|g·v| = 1.1e-18`) the difference **underflows
to exactly 0.0** at `h=1e-5` and `h=1e-6`, so `grad_err` reads a definitional
`1.000`; at `h=1e-2` the same row reads `2.989e+08`. There is no measurement
there, only a choice of `h`.

Note that `squash1e-6` — a row of exactly that kind — is what withdrew the
`r_next` hypothesis, on a single point.

**What the harness does instead.** It scans `h`, and picks the one where
*consecutive refinements agree best*. That disagreement is `fd_unc`, the
reference's own uncertainty, and a row with `grad_err ≤ 10·fd_unc` is reported
`MEASURABLE=False` rather than silently included. Crucially, `h` is chosen
**without looking at the analytic value** — picking the `h` that best agrees
with the quantity under test biases `grad_err` downward and measures the
harness.

It also FDs the **sweep map from a fixed start**, never a re-converged CTM
energy or a root: both are gauge/branch-discontinuous in the parameter, so the
difference *diverges* as `h` shrinks (measured −0.105 → −1.63 across
`h=1e-3..1e-6`).

### 3. The clamp knobs confound "clamp fired" with "gradient vanished"

Every synthetic knob that makes the clamp fire also drives the gradient to
zero, because both families approach a product state:

```
                 |grad|      rank     grad_err    fd_unc      measurable?
random          2.126e+00    4/4      1.5e-10    4.0e-10     at FD's own limit
bsq1e-1         3.830e-03    2/4      8.7e-05    4.6e-09     yes
bsq3e-2         1.073e-04    1/4      2.8e-05    1.0e-06     yes
bsq1e-2         3.987e-06    1/4      2.0e-06    2.1e-05     no - below fd_unc
bsq3e-3         1.077e-07    1/4      2.0e-04    2.6e-04     no
bsq1e-3         3.989e-09    1/4      1.5e-03    5.4e-03     no
bsq1e-4         3.989e-12    1/4      1.1e+00    1.0e+00     no
su_D2           2.785e-02    3/4      3.1e-07    0.0e+00     yes
```

So "clamped" and "at the noise floor" are collinear *by construction*, and a
relative `grad_err` cannot separate "the clamp damaged the gradient" from
"there is no gradient left to measure".

`su_D2` is the sole exception found: a healthy gradient (2.8e-02) **and** a
fired clamp (rank 3 of χ). The two are therefore separable in principle —
synthetic knobs just do not separate them.

**Consequence for the headline.** The 3-order spread at equal `usable_rank=3`
is `su_D2` (3.1e-08) against `squashed-1e-3` (4.4e-05): one physical state
against synthetic states of exactly the kind that turn out to be unmeasurable.

## Candidate status

| candidate | definition | verdict |
|---|---|---|
| `r_min`, `r_next`, `kappa_A` | (in the issue) | refuted there |
| `A_cancel` | `‖vjp_p(F_bar)‖ / ‖grad‖` | **refuted** — pinned at 0.500 across a 10-order `grad_err` spread |
| `B_adj` | `‖F_bar‖ / ‖y_bar‖` | **refuted** — pinned at 1.0000 |
| `L_diag` | `L_abs / ‖S_bar‖` | **refuted** — ρ = +0.429, inverts `bsq1e-1` against `bsq3e-2` |
| `L_grad` | `L_abs / ‖grad‖` | **refuted** — same |
| `L_abs` | `‖S_bar[i,i]‖` over clamped `i` | **open** — ρ = +0.886, but n = 3 distinct states |
| `usable_rank` | | binary flag only; orders nothing |

`A_cancel`'s refutation is worth keeping for its own sake. On every clamped row
`‖direct‖ + ‖adj‖ = ‖grad‖` to four digits, and `|x − y| = |x| + |y|` only when
`x` and `y` are exactly anti-parallel — so the two gradient terms **reinforce**.
There is no catastrophic cancellation in that subtraction; it is the opposite
of one. Together with `B_adj = 1.0000` and `adjoint_residual ~ 1e-15`, that
rules out the whole "conditioning of the linear algebra" family: the error
enters through neither the solve nor the assembly. Look at `F` itself, or at
the energy boundary.

`L_abs` asks the question `r_next` failed to ask — not *how much weight the
clamp destroys* but *whether the objective cares about the destroyed
directions*, via the energy's own cotangent `S_bar`:

```
state       chi  rank   grad_err     L_abs        ratio
su_D2        4    3/4   3.08e-07   2.741e-07       1.1
su_D2        6    3/6   2.66e-07   2.741e-07       1.0
bsq3e-2      4    1/4   2.81e-05   9.406e-06       3.0
bsq3e-2      6    1/6   2.81e-05   9.406e-06       3.0
bsq1e-1      4    2/4   8.67e-05   2.005e-04       0.4
bsq1e-1      6    2/6   8.67e-05   2.005e-04       0.4
```

It agrees with `grad_err` **in magnitude to within a factor of 3 across three
orders**, which is what a first-order error estimate should do. But n = 3
distinct states (six rows = three states at two χ), and `r_next` looked monotone
over *four* before `squash1e-6` killed it. **Do not act on this yet.**

What would falsify it, and should be run first:

- a state where the clamp fires but the energy has **no** weight on the clamped
  directions — `L_abs` predicts a good gradient; if `grad_err` is large it is
  dead;
- the deeply-clamped states (`usable_rank=1`), excluded here only because their
  gradients fall below the FD floor — which is where `r_next` died, and which
  is unreachable until problem 3 is fixed.

## A methodology note worth keeping

The first leverage measurement here was **wrong**, and the cross-check that
caught it is cheap enough to adopt.

`S_keep = diag(s_capped/‖s_capped‖)` is exactly diagonal and `jnp.maximum` sets
every clamped entry to the same value, which makes "the tied minimum of the
diagonal" look like the right detector. It is not: on a *full-rank* cut the
minimum is just the smallest genuine singular value, so the rule produces one
false positive per direction. `random` at χ=4 expects `[0,0,0,0]` clamped
directions and the tied-min rule reports `[1,1,1,1]`. It inflated `L_abs` by
five orders on one row and produced a confident, wrong refutation of the whole
L family.

The fix is to test against the clamp level itself (`d <= rel_floor * d[0]`).
The guard is `--check-clamp`, which cross-checks against `chi - usable_rank`
computed independently from the raw half-infinite SVD. **Run it whenever
`clamped_indices` is touched.**

The guard itself then repeated the mistake one level up, which is worth
recording because the shape keeps recurring. It first asserted that *every*
direction's clamped count equals `chi - usable_rank`, and reported
`ALL MATCH = True` — on four configurations. Extending it to all eight states
produced four mismatches:

```
bsq1e-1  chi=4  detected=[1, 1, 2, 2]   chi-usable_rank=2
su_D2    chi=4  detected=[1, 0, 0, 1]   chi-usable_rank=1
```

The detector was right and the *assertion* was wrong. `retained_rank_report`
reduces with `rank = min(rank, int(usable))` over the four directions, so
`chi - usable_rank` is `max_k(clamped_k)` — one worst-case scalar, not a
per-direction count. The correct invariant is `max(detected) == chi -
usable_rank`, which holds on all 16 rows.

Why the wrong invariant looked fine: squashing every virtual leg by the same
`w` makes the four directions equivalent, so the whole bond-squashed family
satisfies both invariants and cannot distinguish them. Only the asymmetric
states (`su_D2`, and `bsq1e-1` where the squash has not yet saturated) separate
them.

Three times in this issue now — the `r_next` law, the tied-minimum detector,
and this guard — a plausible wrong number has passed on a subset that could not
have falsified it. **Check what your control set is capable of distinguishing
before believing it.**

## Suggested order of work

1. **Break the confound.** An SU imaginary-time family across `D` would give
   healthy gradients that still clamp. #772's notes record such a family
   spanning κ from 3.3e7 to 1e12, which is the right axis. Until then the
   deeply-clamped regime cannot be measured at all.
2. **Re-measure both of #785's tables through this harness**, so every row
   carries `|g·v|`, an `h`-scan and `fd_unc`.
3. **Then** try to kill `L_abs` on the two cases above.
