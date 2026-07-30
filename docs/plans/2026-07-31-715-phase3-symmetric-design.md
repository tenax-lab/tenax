# #715 Phase 3 slice 1 — root implicit AD on `SymmetricTensor`

Status: design approved, nothing built. Scope is the **bosonic abelian 1x1**
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
- **Charge bookkeeping is the library's job**, so a `FlowDirection` error
  becomes a raised exception rather than a silently mis-glued network — the
  #718 failure mode.

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

## 6. Testing

| Tier | Gate |
|---|---|
| Structure | `BondLayout` pinned on a hand-built case; `U*^dag U_perp = 0` per sector; no cross-sector leakage in `S` |
| Dense agreement | symmetric forward energy == densified dense energy, ~1e-12; `norm(F(y*))` ~ 1e-14 |
| **Gradient** | symmetric gradient vs (a) dense root-implicit gradient after densifying, (b) directional FD. Target ~1e-9 |
| Jaxpr | no `svd` or `eigh` primitive anywhere in the backward (Phase 0 precedent) |
| **Trap** | a deliberately wrong cut-leg charge assignment must **fail** |

Coverage is **both** Z2 (equal sector sizes) and U(1) with non-trivial charges
(unequal sectors). Testing only one hides layout bugs: per #566's D-parity
finding, equal-sector cases collapse to a single block shape and unequal ones
fragment, and only the fragmenting case exercises the layout arithmetic.

Scale D=2, chi=4-8. Tests are named so `conftest.py` auto-marking keeps the
cheap tiers in `core` and the gradient parity in the slow bucket.

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
