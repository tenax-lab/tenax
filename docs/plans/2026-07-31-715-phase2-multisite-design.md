# #715 Phase 2 — multisite root implicit AD: convention analysis

Status: index layer landed and pinned (2395a1e). Forward sweep, characteristic
equations and adjoint still to build. This note records the convention algebra,
which is the part that is expensive to re-derive and cheap to get subtly wrong.

## The reference is on disk

The authors' implementation is at
`/tmp/claude-1000/-home-yjkao-tenax/fad53d16-.../scratchpad/idpeps/` — cloned
during the #718 session and still present. `src/asymmetric/fixedpoints.jl` is
the whole of §V/Appendix F. PEPSKit's own index and renormalisation helpers are
at `~/.julia/packages/PEPSKit/LxZhd/src/utility/indexing.jl` and
`src/algorithms/contractions/ctmrg/`. **Read these before deriving anything.**
If the scratchpad is ever cleaned, re-clone `leburgel/ImplicitDifferentiationPEPS.jl`.

## Two half-plane conventions, and which one this module uses

Phase 1 and the paper cut the plane differently. Both are correct; mixing them
silently is not.

| | cut | glues | projector pair at `co` sits on |
|---|---|---|---|
| Phase 1 forward | **left** half, cut horizontally | `EC[k]` to `EC[k-1]` (NW to SW) | that horizontal bond |
| Paper §V.3 / PEPSKit | **upper** half, cut vertically | `EC[co]` to `EC[next(co)]` (NW to NE) | that vertical bond |

Phase 1 bridges them in `asym_root_to_covariant_convention` with a shift-by-one
and a transpose, and `test_the_two_half_plane_conventions_are_the_same_truncation`
pins the identity `M_left(k) == M_up(k-1).T`.

**Phase 2 uses the upper-half convention throughout**, because every index table
in Appendix F and every renormalisation coordinate in PEPSKit is written in it.
Carrying Phase 1's left-half convention into a unit cell would mean re-deriving
all of them, which is exactly the work the reference already did.

### The identity, checked symbolically

With `EC` axes `(chi_r, a_r, chi_d, a_d)` and

```
T(X) = X.transpose(2, 3, 0, 1).reshape(n, n)      # rows = (chi_d, a_d)
I(X) = X.reshape(n, n)                            # rows = (chi_r, a_r)
```

the two half-infinite matrices are

```
M_up(co)  = T(EC[co]) @ T(EC[next(co)])
M_left(k) = I(EC[k])  @ I(EC[k-1])
```

and since `T(X).T == I(X)`,

```
M_up(k-1).T = T(EC[k]).T @ T(EC[k-1]).T = I(EC[k]) @ I(EC[k-1]) = M_left(k).  ✓
```

That reproduces the relation Phase 1 already tests numerically, from the axis
definitions alone — which is the check that the upper-half port is the *same*
truncation and not a differently-wired one.

## Renormalisation coordinates (read off PEPSKit, not derived)

Uniform in `co`, once the upper-half convention is adopted:

```
corner[co] = P_right[prev_coordinate(co)] · EC[co]                · P_left[co]
edge[co]   = P_right[left_projector(co)]  · (edge[above(co)] ⊗ a) · P_left[co]
```

Verified against two directions each, which is enough to exclude an
accidental match:

- `renormalize_northwest_corner`: `EC[NW,r,c]`, `P_left[N,r,c]`,
  `P_right[W,r+1,c]` — and `prev_coordinate((0,r,c)) == (3,r+1,c)`. ✓
- `renormalize_southwest_corner`: `EC[SW,r,c]`, `P_left[W,r,c]`,
  `P_right[S,r,c+1]` — and `prev_coordinate((3,r,c)) == (2,r,c+1)`. ✓
- `renormalize_north_edge`: `edge[N,r-1,c]`, `P_left[N,r,c]`,
  `P_right[N,r,c-1]` — and `above((0,r,c)) == (0,r-1,c)`,
  `left_projector((0,r,c)) == (0,r,c-1)`. ✓
- `renormalize_west_edge`: `edge[W,r,c-1]`, `P_left[W,r,c]`,
  `P_right[W,r+1,c]` — and `above((3,r,c)) == (3,r,c-1)`,
  `left_projector((3,r,c)) == (3,r+1,c)`. ✓

These are the same coordinates the reference's `F1` and `F2` use
(`PR[_prev_coordinate(co)]` and `PR[_left_projector(co)]`), so the forward sweep
and the characteristic equations share one table — which is the point, since a
sweep and an equation that disagree produce a root that is not a root.

## Projectors

Unchanged from Phase 1 apart from naming. With `M[co] = A @ B`,
`A = T(EC[co])`, `B = T(EC[next(co)])`:

```
P_left[co]  = B @ Vh[:chi]† @ inv_sqrt(S)      (n x chi)   # Phase 1's P_top
P_right[co] = inv_sqrt(S) @ U[:, :chi]† @ A    (chi x n)   # Phase 1's P_bot
```

`P_right @ P_left = inv_sqrt · U†MVh† · inv_sqrt = inv_sqrt · S · inv_sqrt = 1`
exactly, for any `S`, diagonal or not — the Phase 1 argument for a genuine
matrix inverse square root carries over untouched, as does the reason it must
*not* be the Eq. 73 quartic roots (those live on the cut legs of the modified
environment, not inside the projectors).

## What is already pinned

- Every Appendix F table, entry by entry, on a 3x5 cell — large enough that
  one-step, two-step and wrap-around shifts are mutually distinguishable.
  `rightvec_invfroot_indices` is the only two-step shift and the one most
  likely to be wrong if re-derived.
- `enlarged_corner` against Phase 1's `_upper_left_quadrant` at 1x1, all four
  directions, on a twice-swept (genuinely asymmetric) environment: 1e-13.
- The 1x1 collapse to the bare `k-1` / `k+1` offsets `_covariant_pieces`
  hard-codes, so the two modules cannot drift apart.

## Remaining work

1. **Forward sweep.** `all_enlarged_corners`, `M[co]`, projectors per
   coordinate, the two renormalisation formulae above, `converge`.
   Gate: at 1x1 the energy must match the Phase 1 forward to ~1e-12. Tensor
   equality is *not* the right gate here — the two conventions differ by the
   shift and transpose above, so only gauge-invariant quantities compare.
2. **Characteristic equations.** Port `contract_asymmetric_characteristic_equation`
   (F1..F5). The shifted-cell reads are `proj_sinv_indices` for `S^-1` in the
   projectors, `leftvec_invfroot_indices` / `rightvec_invfroot_indices` for the
   quartic roots on the isometries, and `iCi[co] = s[prev(co)] · C[co] · s[co]`.
   Gate: `‖F(y*)‖ ~ 1e-16` on a converged 2x2 cell.
3. **Adjoint and gradient.** Root parametrisation over all coordinates, the
   singular `∂_yF` gauge argument (unchanged — an independent phase per
   environment tensor is still an exact null direction, now one per cell), and
   `dE/dA` per cell.
   Gates from #715: 2x2 FD parity, plus a test that a *deliberately wrong* cell
   shift fails — the tables being right must be load-bearing, not incidental.

## Trap

`‖F(y*)‖` small does **not** validate the cell shifts. A wrong shift on a
uniform unit cell (all sites equal) is invisible, because every cell holds the
same tensor. Phase 2's tests must fill the unit cell with *different* tensors —
`_site_tensor(seed=...)` varies — or they verify nothing that Phase 1 did not
already cover. This is why the 2x2 FD-parity gate is the real gate and the
1x1 agreement is only a smoke test.
