# #715 Phase 1 completion

> **Read this first (supersedes most of what follows).** A discriminating
> finite-difference test shows the residual error is **not** in the characteristic
> equations:
>
> ```
> implicit AD                   = -0.810241844341
> FD of the implicit energy     = -0.817381291970    (stable across h = 1e-4..1e-6)
> reference (explicit backprop) = -0.817381274942
> ```
>
> `FD(E_implicit)` agrees with the reference to eight digits. So `E_implicit(p)` — the
> energy at the root of `F` — is the *correct function of p*. The equations, the root and
> the environment are all right; the bug is downstream, in the ~40 lines of
> `asym_root_implicit_energy_and_grad` that assemble `dE/dp = ∂_p e - F̆ ∂_p F`.
>
> This invalidates the earlier conclusion that the next step needs the paper's figures.
> It does not. Everything below about Eqs. 73-82 is still accurate as *physics*, but it is
> not where the remaining 2.5% lives. Start from the assembly.
>
> **Narrowed further by a dense-Jacobian solve** (bypassing GMRES and the real
> embedding entirely, `np.linalg.solve` on `J.T`, solve residual 2.2e-15):
>
> ```
> dense-solve implicit = -0.810241844336
> gmres      implicit  = -0.810241844341     <- identical to 11 digits
> direct term only     = -0.352819362514
> reference            = -0.817381274942
> ```
>
> So GMRES, the real embedding, the transpose convention and the ў structure are all
> exonerated. The direct term `∂_p e` is a plain VJP of the energy and is right by
> construction. **The entire error is in the indirect term `-F̆ ∂_p F`**, which is off by
> 0.0072 out of 0.4646 — 1.55% of that term alone.
>
> Since the solve is exact and `∂_yF`, `∂_pF` both come from `jacfwd` of the same `F`,
> that leaves one candidate: **`∂_pF` is incomplete**. `F` is differentiated with respect
> to `p` only through `a`, while the frozen constants `U*, U_perp, Vh*, Vh_perp,
> s_star_inv` also depend on `p` and contribute `∂F/∂consts · dconsts/dp`, which is being
> dropped. The method's justification for freezing them is that `u` absorbs the change —
> but that argument is about the *solution* `y*(p)`, not about the partial derivative, and
> `E_implicit` evaluates `parametrize`'s `y` (built with `consts(p)`) while the derivative
> tracks `y*(p; consts(p₀))`. Reconciling those two is the remaining work.
>
> **RESOLVED — the frozen constants are the bug, and the fix is specific.**
> Measured `dE/dc = -F̆ ∂_cF` for the frozen `U*`, `Vh*` (the energy has no explicit
> dependence on them, so this should vanish if freezing is valid):
>
> ```
> U*[0]:  |dE/dU*| = 1.455e-02   kept-rotation = 1.452e-02   tilt = 9.36e-04
> U*[3]:  |dE/dU*| = 1.287e-02   kept-rotation = 1.287e-02   tilt = 1.11e-04
> Vh*[3]: |dE/dVh*| = 1.736e-02  kept-rotation = 1.736e-02   tilt = 1.60e-04
> ```
>
> Not zero, and the right scale to explain the missing 7.14e-3. Decisively dominated by
> the **kept-subspace rotation** — 15x to 100x the subspace tilt.
>
> Working out what is actually invariant under that rotation, `U -> U W` with
> `S -> W† S` (note: one-sided, not `W† S W`):
>
> ```
> Π = bot Vh† S^-1 U† top      S^-1 U† -> (W† S)^-1 W† U† = S^-1 U†     invariant
> ```
>
> and symmetrically `Vh† S^-1` is invariant under `Vh -> W_R† Vh`, `S -> S W_R`.
> **Invariance requires `S^-1` adjacent to `U†` and `Vh†`, as one factor.**
>
> Every arrangement tried so far used a *symmetric* split `S^-1/2 … S^-1/2`. That
> reproduces the correct energy *value* — the two halves meet on each bond and recombine
> into `S^-1` — but it never places `S^-1` next to `U†` inside the **equations**, so `F`
> stays non-covariant under precisely the rotation that dominates the error. It also
> explains why the modified-variable test was an exact no-op: the map was written
> `C = A C̃ A` with `A = S^-1/2`, which is still the symmetric split.
>
> **Next step:** rewrite `F` so `S^-1` never appears split — root-free projectors
> `P̃_top = bot Vh†`, `P̃_bot = U† top`, with a single `S^-1` on each bond, and the
> `y -> x` map assigning the *whole* inverse to one side per bond rather than a half to
> each. Then re-measure `dE/dU*`; it should collapse toward zero, and the 2.5% with it.
>
> Things already checked and believed correct: the sign of Eq. 18; `ў = (env̆, 0, 0, 0)`
> (the energy depends on `y` only through the environment, so `ŭ = v̆ = S̆ = 0`); the
> transpose convention in the solve (`matvec(v) = vjp_y(v)[0]` applies `(∂_yF)^T`, and
> `F̆ ∂_yF = ў` is the transposed system); GMRES convergence (residual 1e-11). The most
> likely remaining suspect is that `parametrize` recomputes the frozen constants
> `U*, U_perp, Vh*, Vh_perp` at each `p`, so the `y*(p)` that `E_implicit` actually
> evaluates is the root of `F(·, p; consts(p))` while the implicit derivative is taken of
> `F(·, p; consts(p0))`. The method assumes those agree to first order because `u`
> absorbs the difference — worth verifying directly rather than assuming.



Status: **partly implemented.** General-matrix `S` has landed (84f05f1) and takes the
gradient error from 1.2e0 to 1.1e-2 … 2.5e-2. One ingredient remains — see
"Measured: what actually moves the error" below.
`test_gradient_parity_needs_the_modified_variables` in
`tests/test_ctm_root_implicit_asym.py` is the gate; it is a strict xfail carrying the
current number.

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
`S` has to become a general matrix, which is the part now implemented.

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
| quartic roots on the cut legs of `M` | 6.2e-2 / 9.4e-1 (both conventions worse) |
| **combined package**: root-free projectors + `ρ_L`,`ρ_R` in the `y→x` map + `C̃`,`Ẽ` primitive + `S̆` routed through the map | 2.0e-1 … 2.9e-1 (worse) |

The combined package was the one arrangement the frozen-constant analysis predicted
should work — the two roots are individually equivariant under the *independent* left and
right rotations that `dU*/dp` and `dVh*/dp` actually undergo. It is worse by an order of
magnitude, with the root still at ~1e-15. So that prediction is falsified too.

**Every covariance repair that could be formulated has now been tried and measured.**
`S` as a general matrix remains the only change that helped. The 2.5% residual is real,
reproducible, stable across states and χ, and localised to `∂_pF` — but not explained by
any of the arrangements above.

Two of these results narrow the remaining work sharply.

**The quartic roots do not belong in the projectors.** They are equivariant under
independent left/right rotations, but their product is not `S^-1`, so the projector
closure breaks at first order in a non-diagonal `S`.

**Step 2 above is a no-op on its own.** Switching to the modified corners and edges is a
smooth invertible change of variables, and a reparametrisation cannot change
`dE/dp` — the measured error was identical to four digits. So the modified variables are
*scaffolding*, not the fix. Useful only because the next step needs them.

### The cut-leg roots: derived, measured, **falsified**

Implemented and measured both conventions. Both are worse than doing nothing:

| cut-leg placement | gradient error |
|---|---|
| none (baseline) | 2.508e-2 |
| `s^L`/`s^R` as derived below | 6.245e-2 |
| swapped L/R | 9.447e-1 |

with the root residual still ~1e-16, so this is not a convergence artefact. **The
remaining 2.5% is not explained by the cut-leg roots as placed here.** Either the leg
assignment is wrong in some way not captured by the two-way L/R choice, or the missing
ingredient is something else entirely.

Do not retry this from the derivation below — it has been tested. The next attempt needs
the actual figures for Eq. 65 and Eqs. 78–80 to pin the wiring, or a different
hypothesis for the residual.

The derivation, for the record:


`M`'s two open legs are the right-facing legs of the two quadrants, and each carries a
dangling `A = S^-1/2` inherited from the edge it came from: the left leg from move
`k+1` (via `T1`), the right leg from move `k+3` (via `T3`). The internal bonds are fine —
two halves meet and combine into a covariant `S^-1`. So the correction is

```
M̃ = kron(X_L, 1_d2) @ M @ kron(X_R, 1_d2)
X_L = (S_{k+1} S_{k+1}†)^-1/4 @ S_{k+1}^1/2
X_R = S_{k+3}^1/2 @ (S_{k+3}† S_{k+3})^-1/4
```

i.e. divide out the dangling half-root and multiply the quartic root back in. At the root
every factor is `S^-1/2`, so `X = 1` and `M̃ = M` — consistent with the root residual
being unchanged, and confirming only the *variation* differs. Which quartic root belongs
on which side is a two-way choice; the gradient error is the oracle.

**Cost warning.** A naive `A^-1/4` as two nested Denman–Beavers loops is ~48 matrix
inversions, and `F` is evaluated hundreds of times per GMRES solve across four
directions. That made the measurement infeasible in practice. Compute the quartic roots
**once per direction outside the per-equation loop**, or use a single fused iteration for
`A^-1/4` directly, before attempting this.

**The remaining 2.5% is now unexplained.** Every ingredient of Eqs. 73–82 that could be
derived from the closure condition has been implemented and measured: general-matrix `S`
(the one that worked), the modified variables (a no-op, as a reparametrisation must be),
the quartic roots in the projectors (worse), and the quartic roots on the cut legs
(worse, both conventions). What is left is genuinely not derivable from the closure
condition alone — it needs the figures.

## Order of work

1. ~~Newton–Schulz helpers~~ **done** — `_denman_beavers` in the module, with
   `_inv_sqrt`. Note it must not call `svd` even for a scale factor;
   `test_no_svd_in_the_differentiated_equations` caught exactly that.
2. ~~Promote `S` to a general matrix; `R_S` becomes the full χ×χ block~~ **done**
   (84f05f1). This is what took the error from 1.2e0 to ~2.5e-2.
3. ~~Rebuild `half_infinite_environment` with the cut-leg roots~~ **tried, falsified**
   (6.2e-2 and 9.4e-1 for the two conventions, vs 2.5e-2 baseline). Needs the figures,
   not another derivation.
4. The tilde variables and the `y ↦ x` map come along with step 3; on their own they are
   a no-op (measured), so do not land them separately.
5. Re-check in this order — each is a sharp oracle:
   - `‖F(y*)‖` back to ~1e-16 (it went 3.3e3 → 0.24 → 2.5e-16 as conventions were fixed
     the first time round; it localises errors to a single equation)
   - `cond(∂_yF)` still finite
   - the xfail flips green

## Then

Phase 2 (unit cells, Appendix F shifted-cell indexing for `s_α`, `s^L`, `s^R`) and
Phase 3 (`SymmetricTensor`/fermionic — block-diagonal roots, per-charge-sector null
spaces) both build directly on this and are blocked until it lands.
