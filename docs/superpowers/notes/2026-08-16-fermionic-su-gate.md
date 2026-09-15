# Fermionic simple update: the §9.1 gate, taken

**Decision: fermionic drops to v2.** `fermionic_ipeps.py` stays on the code
#881 repaired, `fpeps()` is not migrated, and Phase 3 covers the bosonic
symmetric path only (Tasks 14–15). Authorised in advance by §9.1 of the design,
so this is the gate being *taken*, not reopened under schedule pressure.

## The two gates

**Cost — fails, by ~40×, and it is the gate least likely to yield.**
Fermionic BP costs **392 ms/iteration** against dense's 18.5 ms: 11.4 s per
solve, **45.5 s per four-bond cycle** against a 1167 ms SU cycle — the gauge is
39× the step it gauges. The default 200-cycle run goes from **233 s to
~9,300 s (~2.6 hours)**. This is the block-sparse eager-dispatch wall that
#566, #618 and #630 each closed NO-GO independently; nothing in this design
changes it, and the rewrite neither causes nor fixes it.

**Layout — unexplained, and the experiment that would explain it is not cheap.**
Removing the `base_charges` pin collapses the *2-site* fermionic sweep at
**every** D tested, including the D=2 and D=4 that otherwise work — while the
pin's stated justification is 1-site-only, and the 1-site constraint is
superseded by #881. Whatever the pin is doing on the 2-site path is not what
the code says it is doing.

## Why this is a deferral and not a refutation

The gauge-exactness problem has a **diagnosis**, and it should not be lost by
being filed under "fermionic is hard".

§5.2 measures the BP gauge at **7.28e-02** displacement against a planar
witness whose floor is **6.25e-04** — 116×, and *flat in χ*, which is this
project's defect signature rather than truncation. §5.2a then localises it to
**a single Koszul sign**: `_reorder` (`ipeps_bp_gauge.py:255`) routes through
`SymmetricTensor.transpose`, and suppressing that one sign drops the
displacement to **8.93e-04** — 1.43× the floor, i.e. into the noise — makes the
D=2 gauge exactly per-leg (sign mismatches 6/16 → 0/16, rectangle identity
2.000e+00 → 5e-15), and leaves the BP fixed point untouched.

That is a diagnosis, not a fix: the correction may be a *compensating* sign
elsewhere rather than deleting this one, and choosing needs the graded-network
derivation rather than a bisection. So the fermionic gauge is probably
**correctable**, and it is still deferred — because correcting it does not touch
the cost gate, which is the one that fails by 40×.

## What must not be repeated when this is picked up

**D=2 cannot express the question.** `fermionic_ipeps.py:167` sets
`virt_charges = [i % 2 for i in range(D)]`, so at D=2 the parity sectors are
{0: 1, 1: 1} and *every* parity-preserving matrix on a virtual leg is 1×1 —
every gauge is diagonal and the sign-carrying decompositions have no
non-trivial block to act on. The first run of the §5.2a experiment was
specified at D=2 and would have reported "BP is exact on a graded network",
an artefact of sector structure. The same trap sits in the mutation suite:
"inverse not transposed on the far end" is a **no-op at D=2** (1.06e-16) and
2.86–3.37 at D=3. **Any fermionic gauge work runs at D ≥ 3.**

## What this costs today

Nothing that works. `fpeps()` keeps the path #881 repaired; no user-visible
behaviour changes. What is given up is the *unification* — bosonic SU has no
stored bond spectrum, fermionic SU still does, and the four defects #882
deletes (#667, #851, #865, #869) remain deleted only on the bosonic path.
Anything relying on that being uniform must check which engine it is on.

Bosonic symmetric (Tasks 14–15) is unaffected and proceeds: the Koszul sign
above returns the identity on a non-graded symmetry, so none of this reaches
U(1)/Z_n.
