# #715 Phase 1 completion: the modified-variable formulation

Status: **derived, not implemented.** `test_gradient_parity_needs_the_modified_variables`
in `tests/test_ctm_root_implicit_asym.py` is the gate; it is a strict xfail carrying the
target number.

## Why the shortcut cannot be repaired in place

`_ctm_root_implicit_asym.py` keeps `S` diagonal and uses standard Fishman projectors.
Measured on a converged `D=2, chi=4` state:

| | directional derivative |
|---|---|
| explicit backprop (matches symmetric FD to 10 digits) | −0.81738127494 |
| root-implicit, diagonal `S` | −1.7898 |

with `‖F(y*)‖ = 2.5e-16` and `cond(∂_yF) ≈ 8e3`. Not a solver problem.

The parametrisation `U = U* + U_perp u` forces the kept-space component of the isometry
to stay exactly `U*`. The true perturbed isometry is `U*(1 + δω) + U_perp δu`. Dropping
`δω` is legitimate **only if the equations are covariant** under the environment gauge,
so that `δω` is absorbed by `δC`, `δE`.

They are not: the `s^-1/2` inside the projectors does not transform covariantly
(paper Eq. 86). So `δω` is physical here and dropping it loses a real contribution.

Nor can `δω` simply be promoted to a variable. The equation that would determine it is
the diagonality of `S` within the kept block — and linearising *that* is exactly the
`1/(s_i² - s_j²)` divergence the whole method exists to avoid. There is no third option:
the modified variables are required.

Phase 0 (C4v) gets away with the same restriction because the `eigh` formulation has no
inverse roots and Eqs. 26–28 are genuinely covariant. That contrast is the cleanest
statement of what §V.3 is for.

## The construction, in this module's conventions

Note the naming clash: the paper writes `S` for the singular values and `s = S^-1`.
Below, `s` denotes the singular values themselves, matching `all_projectors`.

### 1. Strip the roots out of the projectors

```
P_top = P̃_top s^-1/2      P̃_top = bot @ Vh†
P_bot = s^-1/2 P̃_bot      P̃_bot = U† @ top
```

so `P̃_bot @ P̃_top = U† M Vh† = s`, and a bond insertion becomes `P̃_top s^-1 P̃_bot`.
`s^-1` transforms covariantly (`s → Q^R s Q^L†` gives `s^-1 → Q^L s^-1 Q^R†`) where
`s^-1/2` does not — that single change is what buys covariance.

### 2. Modified corners and edges (paper Eq. 82)

Substituting into `_renormalised_corner` / `_renormalised_edge`:

```
C = s_a^-1/2 C̃ s_b^-1/2      C̃ = P̃_bot^(k+1) Q P̃_top^(k)
E = s_a^-1/2 Ẽ s_b^-1/2      Ẽ = P̃_bot^(k)   (T4·a) P̃_top^(k)
```

`a` and `b` are the two bonds the tensor touches. For a corner these are moves `k` and
`k+1`; for an edge both are move `k`. This matches Appendix F Table 1, which puts
`√S_{α-1}` and `√S_α` on the two legs — the shifted index is the neighbouring move.

### 3. Variables and the `y ↦ x` map

```
y = ({C̃_α}, {Ẽ_α}, {u_α}, {S_α}, {v_α})    S_α a general complex χ×χ matrix
x = ({C_α}, {E_α})                          via the relations above
```

The energy still consumes `x`, so `ў` comes from back-propagating `x̆` through the
`y ↦ x` map — the paper's Algorithm 2 line 7, and the reason it insists the map be
explicit. `ŭ = v̆ = 0` as before.

`S` **must** be a general matrix, not a vector: with `U`, `V` restricted to null-space
variations, the in-space rotation has nowhere else to go. This is the exact analogue of
Phase 0's "treat `C` as a generic complex Hermitian matrix in the reverse pass".

### 4. The cut legs (paper Eqs. 73, 87)

`M` is formed by cutting a bond, which leaves a dangling `s^-1/2` that is again
non-covariant. Replace the dangling roots by

```
s^L = (s† s)^(1/4)      s^R = (s s†)^(1/4)
```

which transform as `s^L → Q^L s^L Q^L†` and `s^R → Q^R s^R Q^R†`. For diagonal
positive `s` both reduce to `sqrt(s)`, so they are invisible at the root and only their
*variation* differs — which is the whole point.

### 5. Keeping `F` decomposition-free

`s^-1/2`, `s^L` and `s^R` are matrix functions of a varying matrix. Do **not** reach for
`eigh`: although the composite is smooth at degeneracies (a Löwner divided difference
tends to `f'(λ)`), JAX's `eigh` VJP still divides by `λ_i - λ_j` and will NaN.

Use **Newton–Schulz / Denman–Beavers** iteration instead — a fixed number of steps,
matrix multiplications only. It is polynomial in the input, so it differentiates cleanly
and `test_no_svd_in_the_differentiated_equations` keeps passing. Matrix *inverse* is
fine as-is; JAX's VJP for it is a plain contraction.

## Measured: what actually moves the error

| change | gradient error |
|---|---|
| baseline (diagonal `S`, standard projectors) | 1.2e0 |
| **`S` promoted to a general matrix + matrix `S^-1/2`** | **1.1e-2 … 2.5e-2** |
| `(S S†)^-1/4`, `(S† S)^-1/4` inside the projectors | 2.0e-1 (worse) |
| `C̃`, `Ẽ` promoted to primitive variables with `C = A C̃ A` | 2.508e-2 (**no change**) |

Two of these results narrow the remaining work sharply.

**The quartic roots do not belong in the projectors.** They are equivariant under
independent left/right rotations, but their product is not `S^-1`, so the projector
closure breaks at first order in a non-diagonal `S`.

**Step 2 above is a no-op on its own.** Switching to the modified corners and edges is a
smooth invertible change of variables, and a reparametrisation cannot change
`dE/dp` — the measured error was identical to four digits. So the modified variables are
*scaffolding*, not the fix. Useful only because the next step needs them.

**Therefore the entire remaining discrepancy sits in step 4:** the half-infinite
environment must be built from the *modified* edges with `s^L`, `s^R` on the dangling cut
legs, i.e. `M̃ ≠ M`. That changes the equations rather than the coordinates, and it is the
only ingredient of Eqs. 73–82 not yet accounted for. It is also the one part whose leg
wiring is not derivable from the closure condition alone, so it needs the figures from
Eq. 65 and Eqs. 78–80.

## Order of work

1. ~~Newton–Schulz helpers~~ **done** — `_denman_beavers` in the module, with
   `_inv_sqrt`. Note it must not call `svd` even for a scale factor;
   `test_no_svd_in_the_differentiated_equations` caught exactly that. (`_inv_sqrt`, `_quartic_root`) + tests against `eigh` on
   well-conditioned Hermitian PSD input, plus a gradient-finiteness test at degeneracy.
2. Switch `_fishman_projectors` to the tilde form; add the explicit `s^-1` bond
   insertions to the quadrants and to `half_infinite_environment`.
3. Promote `S` to a general matrix; rewrite Eqs. 78–80 accordingly (`R_S` becomes the
   full χ×χ block, not its diagonal).
4. Add the `y ↦ x` map and route `ў` through it.
5. Re-check in this order — each is a sharp oracle:
   - `‖F(y*)‖` back to ~1e-16 (it went 3.3e3 → 0.24 → 2.5e-16 as conventions were fixed
     the first time round; it localises errors to a single equation)
   - `cond(∂_yF)` still finite
   - the xfail flips green

## Then

Phase 2 (unit cells, Appendix F shifted-cell indexing for `s_α`, `s^L`, `s^R`) and
Phase 3 (`SymmetricTensor`/fermionic — block-diagonal roots, per-charge-sector null
spaces) both build directly on this and are blocked until it lands.
