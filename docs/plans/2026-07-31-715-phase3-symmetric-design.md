# #715 Phase 3 slice 1 — root implicit AD on `SymmetricTensor`

Status: **BUILT AND VALIDATED**, 2026-07-31. Headline result, D=2 chi=4 Z2:

| quantity | value |
|---|---|
| symmetric gradient vs dense root-implicit | **3.29e-15** relative |
| energy vs dense | 1.5e-16 |
| `norm(F(y*))` | 2.70e-13 |
| `gmres_residual` | 6.38e-15 |
| directional FD (sweep map, 12/20/30 sweeps) | 4.28e-4 / 7.67e-7 / 5.14e-9 |
| svd/eigh primitives in the graph | **none**, over six jaxprs |

Scope is the **bosonic abelian 1x1**
slice. Fermionic `FermionParity`, multisite symmetric, and production wiring
are separate slices; §8 says why each is deferred and what unblocks it.

Phase 3 is the phase #715 justifies the whole method by, so §1 states exactly
which measured cost it removes and which it does not — the issue's own framing
("where #566 and #687 get paid off") is looser than the evidence now supports.

## 1. What this actually buys, stated precisely

The #566 compile wall has been localised three times, and the final answer is
**not** the one the issue text assumes. From
`docs/superpowers/handoffs/2026-06-08-570-relocalized-not-decomposition.md`:
at D=4, chi=12 fermionic, the `svd_vjp` bucket is **61.3% of the entire
backward** (92,368 of 150,663 ops) and is *the only chi-scaling category*. But
drilling in, that bucket is not decomposition math — the singular-value
primitive and its F-matrix are ~0% of it. It is per-charge-sector **structural**
emission (block pack/unpack ~60%, gauge fixing ~25%) inside
`_truncated_svd_symmetric_traced` / `_compute_2x2_projector_symmetric` /
`_gauge_fix_symmetric_svd`, attributed to "svd_vjp" only because those ops are
emitted lexically inside those functions.

That is exactly why every cheaper-decomposition lever died: SVD-via-eigh (same
F-matrix, no win), batched per-sector decomposition (kernels 48->24, ops -0.7%),
truncated backprop (unrolled, strictly worse than implicit for compile). The
cost was never the decomposition.

**Root implicit AD is a different kind of lever, and it is the one that fits.**
It does not make the SVD cheaper to differentiate — it stops differentiating it
at all. `U*`, `S*`, `V*` become *constants* of the backward, so the whole
projector-and-SVD wrapper, structural emission included, leaves the
differentiated graph and stays forward-only. The backward becomes the VJP of
`F`, which is contractions plus Denman-Beavers matrix iterations.

Two honest caveats, recorded here so nobody later reads a claim this design did
not make:

1. **`F`'s own VJP has its own per-sector structural emission.** Block-sparse
   contractions inside `F` still pack and unpack per block. The net op-count win
   is therefore *plausible and directionally right, but unmeasured*. This slice
   is parity-gated by decision; the measurement is §8's first follow-up and it
   is the thing that converts "it runs" into "it pays".

   **What the build did establish (2026-07-31), and what it did not.** The
   *qualitative* claim is now demonstrated: the differentiated graph contains no
   `svd` and no `eigh` primitive at all, asserted over six jaxprs covering both
   the forward functions and their pullbacks. So the entire class of failure this
   method exists to remove — the degenerate-SV NaN cluster, the `lorentzian`
   regularisation, #687's accuracy floor — is structurally gone rather than
   mitigated. Sharper than expected: explicit backprop through the symmetric
   sweep does not merely NaN, it **raises** (`sector_svd` ranks singular values
   across sectors in Python, so `jax.grad` dies with `ConcretizationTypeError`
   at D=2), which makes the implicit path the only one that produces a symmetric
   gradient here at all.

   The *quantitative* cost claim remains unmeasured, and one number now argues
   against a naive reading of it: **peak RSS is 8.4 GB, essentially all in the
   GMRES solve**, where XLA compiles the ~15k-equation block-sparse VJP inside a
   `lax.while_loop`. Staged: 0.69 GB after root extraction, 2.33 after the energy
   VJP, 2.50 after the `F` VJP trace, 8.36 after the solve. `restart=10` reaches
   only 7.19 GB while being slower (93 s vs 49 s) and five orders less accurate
   (2.5e-10 vs 6.4e-15), so there is no cheap lever inside the solver. Removing
   the SVD VJP therefore did **not** remove the compile-scale problem; it moved
   it. Whether the move is a net win is exactly what §8's measurement must decide,
   and it should now measure memory alongside op count.
2. **Do not sell any of this on wall-clock.** Paper §VI.3 shows implicit losing
   to fixed-point at our D and chi. The arguments are stability (#687's floor,
   the degenerate-SV NaN class) and compile-graph size (#566).

## 2. Representation: hybrid, split at the cut

Three representations were considered; the chosen one is C.

| | environment / contractions | projector core |
|---|---|---|
| A | `SymmetricTensor` + labels | rebuilt in labels |
| B | `{sector: dense}` by hand | Phase 1 dense, per sector |
| **C** | **`SymmetricTensor` + `contract`** | **Phase 1 dense, per sector** |

C wins because it makes #715's two flagged hazards stop being work:

- *"block-diagonal quartic roots `s^L`, `s^R`"* is `_quartic_root` applied per
  block. `_denman_beavers` is `jnp.linalg.inv` plus matmuls
  (`_ctm_root_implicit_asym.py:176`) — nothing in it needs a block-sparse
  rewrite, and nothing in it puts a decomposition back into `F`.
- *"per-charge-sector null-space projectors"* is
  `jnp.linalg.svd(full_matrices=True)` per sector, then the dense slice
  `U_perp_q = U_q[:, chi_q:]`.

It also inherits two properties of the library that A and B do not:

- **Fermionic signs are free on this network.** `contractor.py:691-699`: the
  contractor deliberately applies no Koszul signs, because `FermionParity`'s
  R-symbol contributes only at physical line crossings and planar diagrams have
  none. CTM / RDM / energy networks are planar. This is what makes the
  fermionic slice (§8) small *provided* the network is expressed through
  `contract`, which B would forfeit.
- **Charge bookkeeping is the library's job**, so a *structural* mismatch —
  incompatible dimensions, a charge set that cannot pair — raises instead of
  producing a silently wrong number.

  **Partial correction (measured during Task 6).** This was originally written
  as the stronger claim that a `FlowDirection` error "becomes a raised
  exception rather than a silently mis-glued network — the #718 failure mode".
  That over-states it. `contract` enforces charge *conservation*, not physical
  correctness: on a network whose flows pair badly it keeps the
  charge-conserving components and **silently zeroes the rest** — measured at
  3.09 out of an 8.81-scale result on a `T1 x a` contraction, 120 of 1024
  entries retained. So the library catches structural errors, but a network
  that is charge-legal and still physically mis-glued returns a plausible
  wrong number, exactly as the dense path would. Approach C's advantage over B
  here is narrower than claimed; its advantages on fermionic planarity and on
  not hand-rolling charge fusion are unaffected.

  Practical consequence, and the reason this matters beyond bookkeeping: a
  random `SymmetricTensor` built on environment indices is **not** a legal CTM
  environment tensor (the double layer conserves ket and bra charges
  separately, while the fused index tracks only the difference). So
  symmetric-vs-dense parity can only be tested on environments that came from
  `initialize_ctm_tensor_env` / the sweep — never on random ones. Cf.
  [[feedback_ctm_oracle_needs_wellconditioned_state]], which is the same
  lesson from the other direction.

C also follows existing precedent: `_truncated_svd_symmetric` already groups
blocks by bond charge, runs dense per sector, and reassembles.

### The `kron` is deleted, not ported

This is the subtlest point of the port and the one most likely to be got
wrong by re-derivation.

Phase 1 and 2 attach the quartic roots with `jnp.kron(root, eye_d2)` — the
roots act on the `chi` factor of the cut leg `n = chi * d2`, with `chi` the slow
index. Under charge fusion that identity **breaks**: sector `q` of the fused leg
is a direct sum over all `(q_chi, q_d2)` with `q_chi + q_d2 = q`, so a matrix
acting only on the `chi` factor is *not* a per-sector `kron` in the fused basis.

The fix is not to reproduce it. Keep the cut leg **split** as `(chi, d2)` and
apply `s` by contraction on the `chi` leg; fuse only to feed the SVD, and split
back afterwards via the `FuseInfo` the fused index carries. The `kron`
disappears from the symmetric module entirely.

### `U_perp` is materialised, deviating from #715

#715 plans "do not materialize `U_perp`; use the complementary projector
`(1 - U* U*^dag) w`". That deviation is **declined here**. Per sector the
matrices are small, so a full SVD is cheap; and the complementary form is an
over-parametrisation whose redundant component is an extra null direction of
`d_yF` bought for nothing. Phase 1 and 2 both materialise it and both reach FD
parity, so this also keeps the three modules structurally comparable.

## 3. Components

`src/tenax/algorithms/_ctm_root_implicit_symmetric.py`, standalone, mirroring
`_ctm_root_implicit_asym.py` function-for-function. No production code is
touched.

**(a) `BondLayout`.** The symmetric analogue of "chi is an int": a per-bond map
`charge -> retained dimension`, produced by global top-chi truncation *across*
sectors at the converged point, then frozen. Every downstream shape — `u`, `v`,
`S`, `U*`, `U_perp` — keys off it. It must be static under AD, and it is the
object a wrong port corrupts silently, which is why §6 gives it a trap test.

**(b) Symmetric forward CTM.** `initialize_ctm_tensor_env` and
`_build_double_layer_tensor` are reused *without* `.todense()`.
`_ctm_root_implicit_asym.py:516-535` currently builds symmetric tensors and
immediately densifies all nine; not densifying is most of the symmetric
forward. Sweep structure, the simultaneous (not Gauss-Seidel) update, and the
element-wise convergence criterion are unchanged — corner singular values are
invariant under per-bond rotations, so a spectral criterion calls convergence
while the tensors are still moving, and the characteristic equations compare
tensors. The bond-gauge pin applies per sector.

**(c) Sector projector core.** Per direction: fuse the enlarged corner to a
matrix, group blocks by bond charge, per-sector
`jnp.linalg.svd(full_matrices=True)`, truncate globally to get `BondLayout`,
per-sector `_inv_sqrt`, per-sector gauge pin. Returns `P_left` / `P_right` as
`SymmetricTensor` plus per-sector `(U*, U_perp, S, V*, V_perp)`.

**(d) Root variables and residual.** `y = (C_tilde, E_tilde, u, S, v)` with
`C_tilde` / `E_tilde` as `SymmetricTensor` and `u` / `S` / `v` as per-sector
dense dicts. `S` stays a full matrix per sector with a genuine matrix inverse
square root: it is a *variable*, free to leave both the diagonal and the reals
(#721), and a diagonal `S` makes the in-space isometry rotation
unrepresentable. Charge conservation forbids cross-sector entries, so "general
matrix" means general *within* each sector — that block-diagonality is a
structural constraint, not a truncation, and §6 tests for leakage across it.

**(e) Adjoint.** Same GMRES-on-`d_yF` structure as Phase 1. **Verified, not
assumed:** `SymmetricTensor` flattens to a *single* leaf (the flat `_data`
buffer, per PR #87), and the L2 norm of that buffer *is* the Frobenius norm —
measured 3.700592080134 both ways on a 3-block U(1) tensor. So `gmres_pytree`
carries the correct inner product over mixed
`(SymmetricTensor, dict-of-arrays)` pytrees with no adapter.

The Phase 1 gauge argument carries over unchanged and should not be
re-litigated: an independent phase per environment tensor is an exact null
direction of `d_yF`; it is harmless because `y_bar` is orthogonal to every null
direction (E is invariant along each) and because differentiating
`F(y*(p),p) = 0` puts `d_pF` in range(`d_yF`), so the cokernel cannot reach the
gradient. Do **not** add a phase-fixing condition or a gauge quotient.

## 4. Data flow

```
A (SymmetricTensor)
  -> converge_sym                -> (env, projectors)
  -> root_parametrize_sym        -> y*, frozen consts, BondLayout
  -> E(y*, A)
  -> GMRES adjoint on d_yF
  -> dE/dA (SymmetricTensor)
```

`converge_sym` must return its projector chain, and `root_parametrize_sym` must
consume it. A cold re-pin fixes a *different* bond gauge, leaving `y*`
describing an environment it was not extracted from. A real state survives that
(the gauge is a sign; one sweep absorbs it) but a complex one does not — the
corner residual plateaus forever while the edges fall geometrically, because
the corner is the only equation built from two directions' projectors and so
the only one that sees a *relative* bond phase (#721).

## 5. Error handling

- **Empty sectors** (`chi_q = 0`) drop from the layout. **Saturated sectors**
  (`n_q == chi_q`) get genuinely zero-size `u` and `v`. Neither is exotic at
  small D; contractions must not assume non-empty operands.
- **The rank floor is relative to the global largest singular value, not the
  sector's own.** Phase 1 floors at `1e-12 * s[0]` before `S` becomes a matrix,
  because a singular `S` makes the matrix inverse square root NaN. Applying that
  per sector against each sector's own maximum would promote a small sector's
  numerical noise to a retained direction.
- Root-residual warning threshold as Phase 1, plus its `gauge_consistency`
  diagnostic — that one measures a property of the *energy* boundary, which is
  where #718 actually lived.
- Explicitly unsupported, and asserted against rather than left to fail
  obscurely: non-uniform unit cells (#667 — `initialize_ctm_tensor_env` derives
  both corner chi-legs from one `ref_axis`, so `A.l != A.r` is wrong, not merely
  untested) and 2-site.
- **`ProductSymmetry` is refused** (added 2026-08-01, from a Codex review of
  PR #729). The sector layer pins a partner charge as the raw integer `-q`,
  which is the group inverse for U(1) and, mod `n`, for `Z_n` — but charges in
  a product group are bit-packed, so `-encode(1,2)` decodes as `(-1,-3)` and
  `fuse(q,-q)` is not the identity. Those blocks violate conservation, and
  `_from_blocks_unchecked` would carry them into a contraction that silently
  zeroes the non-conserving part. Raising is the only safe option until the
  reconstruction goes through `symmetry.dual`/`fuse`.
- **`Z_n` bonds carry a non-canonical representative**, and this module
  reproduces it deliberately. A Z2 partner is labelled `-1` rather than `1`
  because `_group_blocks_by_bond_charge` fuses a *single* flow-weighted charge
  and `fuse_many` of one array skips the `% n`. This is not this module's
  invention — `tenax.linalg.svd` emits the same labels for a Z2 tensor whose
  left leg flows OUT (#733) — and the CTM cut hits it because both projectors
  are built with row and col flowing the same way. Diverging unilaterally would
  make these bonds fail to contract against the rest of the library, so the
  convention is pinned by a test that should fail and be updated in step when
  #733 is fixed.

## 6. Testing

| Tier | Gate |
|---|---|
| Structure | `BondLayout` pinned on a hand-built case; `U*^dag U_perp = 0` per sector; no cross-sector leakage in `S` |
| Dense agreement | symmetric forward energy == densified dense energy, ~1e-12; `norm(F(y*))` ~ 1e-14 |
| **Gradient** | symmetric gradient vs (a) dense root-implicit gradient after densifying, (b) directional FD. Target ~1e-9 |
| Jaxpr | no `svd` or `eigh` primitive anywhere in the backward (Phase 0 precedent) |
| **Trap** | a deliberately wrong cut-leg charge assignment must change the **state**, measured by energy — see below for why not by `norm(F)` |

Coverage is **both** Z2 (equal sector sizes) and U(1) with non-trivial charges
(unequal sectors). Testing only one hides layout bugs: per #566's D-parity
finding, equal-sector cases collapse to a single block shape and unequal ones
fragment, and only the fragmenting case exercises the layout arithmetic.

Scale D=2, chi=4-8. Tests are named so `conftest.py` auto-marking keeps the
cheap tiers in `core` and the gradient parity in the slow bucket.

### A forced layout is a different problem, not a broken one

The trap tier originally asserted that moving one retained dimension between
charge sectors leaves `norm(F)` far from zero — measured at 4.0e-1 against
2.7e-13. **That was an artifact and the claim is withdrawn** (2026-08-01, from
a Codex review of PR #729). `root_parametrize_sym` forwarded `layout_override`
to the projectors but dropped it on the sweep that advances the environment, so
the run alternated forced and natural truncations; 4.0e-1 was the floor of the
alternation, not of the forced layout.

Forwarded consistently, the forced layout **converges to a root of its own** —
`norm(F)` falls 8.8e-2 -> 2.4e-6 -> 9.3e-11 over 3/10/40 polish sweeps, and
reaches 2.2e-14 when converged from a fresh environment. That is the right
answer physically: fixing the retained charge distribution poses a different
truncation problem, and that problem has a fixed point. What the moved
dimension changes is *which* fixed point — `E = -0.10873` against `-0.10502`,
a 3.7e-3 gap next to the 1e-10 the natural layout reproduces against the dense
module.

So the layout is load-bearing, and the trap still bites; the discriminator is
the energy, not the residual. The general lesson is the one worth keeping: **a
counterfactual that fails for the reason you assumed is worth re-deriving when
the harness that forces it is itself new.** Both halves have to be forced, or
the number measures the harness.

### Two measurement traps that cost hours in Phases 1-2

- **Never FD the root or the re-converged energy.** The root parametrisation is
  gauge-*discontinuous* in `p`, so `|y_fd|` diverges as `h` shrinks
  (1.98e4 -> 1.19e5 was observed) even though every point is a valid root at
  1e-16. FD the **sweep map from a fixed start** instead.
- **Explicit backprop is a truncated reference, and complex states need more
  sweeps than real ones.** At seed 7 it reads 2.7e-3 / 3.4e-4 / 5.4e-6 / 3.1e-8
  at 8 / 12 / 20 / 30 sweeps, converging *onto* the implicit gradient. A
  too-short reference disagrees with the implicit gradient *and* agrees with FD
  to four digits at two step sizes — which reads exactly like a wrong gradient.
  Rule out sweep count before touching the adjoint.

Also: the cotangent pairing is `Re sum(g * dz)`, **unconjugated**. The conjugated
form manufactures gradient and orthogonality violations that are not there
(a fake 2.59e-2 against a true -1.5e-16).

### The technique to reach for when parity fails

Gauge-probe `F` and `E` *separately* with a finite per-bond transform `W_k`,
which cannot change anything a CTM environment computes. Ask (1) is `F` still a
root after gauging one bond, and (2) is `E` invariant under the same. That
splits "the equations are wrong" from "the objective is wrong" in one shot with
no reference value, and it is what found #718 after three wrong diagnoses. An
*exact antisymmetry* between two directions in a per-bond residual means a
transpose at a boundary, not a broken theory.

## 7. Non-goals for this slice

Fermionic; multisite symmetric; production wiring; the #566/#687 measurement.

## 8. Follow-up slices, in order

1. **Measurement.** Backward jaxpr op-count and `svd_vjp` share versus
   production `_jit_fused_fixed_point_bwd` at matched D and chi, plus accuracy
   at a point where symmetric explicit backprop floors (#687). This is what
   discharges §1's caveat 1, and it is cheap once the module exists.
2. **Fermionic `FermionParity`.** Expected small given §2's planarity argument,
   but it needs its own parity gate against a dense fermionic reference; the
   contractor's sign-free design is a claim about planar networks that should be
   tested, not trusted, at the fused cut leg.
3. **Multisite symmetric.** Blocked on #667: `_CORNER_SPECS` gives one
   `ref_axis` per corner, so each corner derives both chi-legs' charges from one
   direction. Needs two ref axes per corner plus neighbour threading before a
   non-uniform symmetric cell can be initialised at all.
4. **Production wiring.** Needs the production recipe to expose a single
   half-infinite SVD (#715 obstacle 1); `recipe="2x2"` does three SVDs per
   direction and the paper's §V equations do not apply to it verbatim.

## 9. Reference material

- Burgelman et al., arXiv:2607.15030, §V.2 (asymmetric equations), §V.3
  (covariant form), Eq. 65-88.
- Authors' implementation `leburgel/ImplicitDifferentiationPEPS.jl`,
  `src/asymmetric/fixedpoints.jl`. **Read it before deriving anything** — the
  Julia oracle pins conventions independently of the port, and Phase 1 lost
  hours adjusting placements while watching a total residual.
- `docs/plans/2026-07-29-715-phase1-modified-variables.md` and
  `docs/plans/2026-07-31-715-phase2-multisite-design.md`.
- `docs/superpowers/handoffs/2026-06-08-570-relocalized-not-decomposition.md`
  for the §1 attribution numbers.
