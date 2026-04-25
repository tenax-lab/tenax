# Native Honeycomb iPEPS CTM with AD — Design

**Status:** Design approved. Ready for implementation plan.

**Branch:** `feat/honeycomb-ctm` (worktree at `.worktrees/honeycomb-ctm`).

**References:**
- Lukin & Sotnikov, *Variational optimization of tensor-network states with the honeycomb-lattice corner transfer matrix*, PRB **107**, 054424 (2023); arXiv:2209.03428. Defines the 6-corner CTMRG on the honeycomb lattice for C₃ᵥ-symmetric uniform iPEPS.
- *Contraction algorithms for two-dimensional tensor networks: a review and benchmark*, PRE **109**, 045305 (2024); arXiv:2401.07274. §II.C extends the honeycomb CTMRG to two distinct sublattice tensors A ≠ B.
- Related Tenax design docs: `2026-04-18-python-level-ctm-ad-{design,plan}.md`, `2026-04-25-spin1-xxz-kagome-ipess-{design,plan}.md`.

## Goal

Land a native rank-4 honeycomb iPEPS CTM with implicit-AD differentiation, exposed as a public Tenax algorithm. Replaces the Kronecker-delta dummy-bond brick-wall workaround currently used by `pess_optimize.py:build_pess_loss` (commit `dfe5cbe`). General-purpose: any honeycomb iPEPS application — direct spin-1/2 Heisenberg honeycomb, kagome iPESS via supersites, future fermionic honeycomb — can consume it.

## Decisions locked during brainstorming

| Decision | Choice | Rationale |
|---|---|---|
| **Scope** | A: general-purpose honeycomb infrastructure | Enables direct honeycomb iPEPS, not just kagome iPESS. |
| **Tensor representation** | X2: rank-4 native (3 virtual + 1 physical) | Matches both reference papers; clean fit for SymmetricTensor; no degenerate projector spectrum. |
| **Sublattice scope** | S3: 2-sublattice from start, isometric SVD/eigh projectors | 2-sublattice subsumes 1-sublattice via A=B. Isometric projectors are battle-tested in Tenax; biorthogonal (Paper 2 §II.C) is a layered follow-up. |
| **Symmetry support** | Y3: Tensor-protocol-generic, dense-tested in v1 | Matches `_ctm_tensor_*.py` style. SymmetricTensor extension does not require a rewrite. |
| **Module organization** | P1: parallel `_ctm_honeycomb_*.py` family | Avoids destabilizing the existing checkerboard CTM. Consolidation with the square path is a separate post-v1 PR. |

## Architecture

**Site tensor.** Rank-4: 3 virtual bonds (labels `e0, e1, e2`, dim D) and 1 physical leg (label `phys`, dim d). Two sublattice tensors A and B. The bipartite honeycomb has each A-site connected via its 3 virtual bonds to 3 distinct B-sites (one per honeycomb edge direction), and vice versa.

**Environment.** Six-corner CTMRG (Lukin-Sotnikov). Per sublattice, 3 corner tensors `C^{(s)}_α` (rank-2, χ × χ) and 3 column tensors `L^{(s)}_α, R^{(s)}_α` (rank-3, χ × χ × D²) — one per direction α ∈ {e0, e1, e2}. For 2 sublattices: 6 corners + 12 column tensors total. Reduces to 3+6 in the uniform 1-sublattice case (A=B).

**Move topology.** One CTM iteration sweeps 3 directions. Each direction performs a *paired move* updating both sublattices' (C, L, R) for that direction simultaneously, mirroring the Paper 2 §II.C structure. Per-absorption phase fix (first-above-threshold convention) and `S_safe` NaN protection on the projector — both lifted from the existing `_ctm_projector.py` patterns. Sigma gauge applied per-absorption when `forward_gauge="sigma"`; phase gauge by default (memory: phase gauge is correct default, sigma explodes on 2-site).

**AD.** Implicit differentiation through the converged CTM fixed point — one fixed-point GMRES backward solve, JIT-fused, custom VJP via `jax.custom_vjp`. Mirrors `_ctm_energy_ad.py:ctm_energy_implicit`. Forward pass is a Python loop with chi ramp, mirroring PR #341. No direct/unrolled AD path in v1.

**Public entry point.**

```python
def honeycomb_ctm_energy_implicit(
    site_tensors: dict[Coord, Tensor],   # {(0,0): A, (1,0): B}; rank-4, labels (e0,e1,e2,phys)
    hamiltonian: jax.Array,              # bond gate; shape depends on energy_fn
    *,
    chi: int,
    max_iter: int,
    conv_tol: float,
    projector_method: str = "eigh",      # "eigh" | "svd" | "biorthogonal" (NotImplementedError stub)
    forward_gauge: str = "phase",        # "phase" | "sigma"
    chi_ramp: tuple[int, ...] | None = None,
    energy_fn: Callable | None = None,   # default: 3-edge NN bond sum; override for triangle energy
    # GMRES backward params, mirrored from existing implicit path:
    gmres_tol: float = ...,
    gmres_maxiter: int = ...,
    gmres_restart: int = ...,
    arnoldi_precheck: bool = False,
) -> jax.Array  # real scalar energy
```

## Components

### Module file layout

| File | Role |
|---|---|
| `src/tenax/algorithms/_ctm_honeycomb_env.py` | `HoneycombCTMEnv` NamedTuple (9 fields: 3 corners + 3 left + 3 right column tensors per sublattice), pytree registration. |
| `src/tenax/algorithms/_ctm_honeycomb_init.py` | `_double_layer_honeycomb(A)`, `initialize_honeycomb_env(sites, chi_init)`. |
| `src/tenax/algorithms/_ctm_honeycomb_moves.py` | `move_direction_alpha`, `honeycomb_ctm_step`, `_renormalize_honeycomb_env`, `HONEYCOMB_NEIGHBORS`. |
| `src/tenax/algorithms/_ctm_honeycomb_projector.py` | `compute_honeycomb_projector(boundary, *, method, chi, S_safe_eps)` — isometric `(P, P†)`. |
| `src/tenax/algorithms/_ctm_honeycomb_convergence.py` | `check_honeycomb_convergence(env_old, env_new, *, method, tol)`; element-wise default, SV optional. |
| `src/tenax/algorithms/_ctm_honeycomb_energy.py` | `_rdm2_bond` (2-vertex bond RDM, per direction), `_rdm1` (1-site RDM), `compute_honeycomb_energy`, `compute_honeycomb_triangle_energy` (kagome helper). |
| `src/tenax/algorithms/_ctm_honeycomb_ad.py` | `honeycomb_ctm_energy_implicit` with custom VJP and JIT-fused GMRES backward. |
| `src/tenax/algorithms/honeycomb_ctm.py` | Explicit named re-export shim (mirrors `ipeps_ctm.py` style). |

### Env keys and neighbor map

Match Tenax conventions exactly. From `_ctm_tensor_convergence.py:130-139`:

```python
Coord = tuple[int, int]

HONEYCOMB_NEIGHBORS: dict[Coord, dict[str, Coord]] = {
    (0, 0): {"e0": (1, 0), "e1": (1, 0), "e2": (1, 0)},
    (1, 0): {"e0": (0, 0), "e1": (0, 0), "e2": (0, 0)},
}
```

Multisite envs use `dict[Coord, HoneycombCTMEnv]`. `make_honeycomb_neighbors(nx, ny)` for larger unit cells is deferred follow-up.

### Public API registration

- `src/tenax/algorithms/__init__.py:_LAZY_IMPORTS` — register `honeycomb_ctm_energy_implicit`, `HoneycombCTMEnv`, `initialize_honeycomb_env`, `HONEYCOMB_NEIGHBORS`.
- `src/tenax/__init__.py:__all__` — same names.
- `README.md` — "Honeycomb iPEPS with AD" subsection with citations.
- `docs/source/algorithms.rst` — add honeycomb subsection.

## Data flow

**Forward.**

```
(A, B) rank-4 site tensors  [labels: (e0, e1, e2, phys)]
    │
    ▼
_double_layer_honeycomb:  T_A, T_B  [labels: (e0_d2, e1_d2, e2_d2), dim D²]
    │
    ▼
initialize_honeycomb_env(sites, chi_init)
    │
    ▼
forward CTM:
    for chi in chi_ramp:
        for iter in 1..max_iter:
            env = honeycomb_ctm_step(env, sites, projector_fn, forward_gauge)
            if check_honeycomb_convergence(env_old, env, conv_tol): break
        env = _renormalize_honeycomb_env(env)
    │
    ▼
energy = energy_fn(sites, env, hamiltonian)   # default: 3-edge NN bond sum
```

**Backward (implicit AD).** Custom VJP linearizes the env fixed-point `F: env -> honeycomb_ctm_step(env, sites)` and solves `(I - ∂F/∂env|env*) · v = ∂energy/∂env|env*` via JIT-fused GMRES, then applies the chain rule for the site-tensor gradient. Identical structure to `_ctm_energy_ad.py:ctm_energy_implicit`.

**Kagome iPESS integration.** `pess_optimize.py:build_pess_loss` becomes:

```python
def loss_fn(state: IPESSState):
    A_u_super, A_d_super = pess_to_honeycomb_supersites(state)   # (d, d, d, D, D, D)
    A_u = _build_honeycomb_site_tensor(A_u_super)                 # rank-4: (D, D, D, d**3)
    A_d = _build_honeycomb_site_tensor(A_d_super)
    sites = {(0, 0): A_u, (1, 0): A_d}

    return honeycomb_ctm_energy_implicit(
        sites,
        hamiltonian=H_triangle,
        chi=config.chi,
        ...,
        energy_fn=_triangle_energy_fn_intrasite,   # 1-site RDMs, sum
    )
```

The dummy-bond padding helpers (`_supersite_to_iPEPS_tensor`, `_make_supersite_indices`) are deleted in the kagome integration PR.

## Numerical safeguards

Required for AD stability — not optional:

| Safeguard | Where | Why |
|---|---|---|
| `S_safe` clamp on singular values before `1/S` | `_ctm_honeycomb_projector.py` | NaN gradients on near-zero singular values. Existing pattern. |
| Per-absorption phase fix (first-above-threshold) | `_ctm_honeycomb_moves.py` | SVD/eigh phase ambiguity makes AD non-deterministic; variPEPS convention. |
| `complex128` enforced (assertion at entry) | `honeycomb_ctm_energy_implicit` | Real float64 → non-variational drift. |
| `_renormalize_honeycomb_env` after each direction | `_ctm_honeycomb_moves.py` | Env tensors blow up exponentially. |
| `EPS` floor on energy denominator | `_ctm_honeycomb_energy.py` | RDM trace can be ~0 in transient states. |
| `forward_gauge="phase"` default | public API | Sigma gauge explodes on 2-site, phase is correct default. |
| `stop_gradient` on env in self-referential paths | `_ctm_honeycomb_ad.py` | Mirror the fix from `_ctm_energy_ad.py` to prevent NaN/vanishing grads. |

**Convergence failure policy.** Forward CTM hitting `max_iter` without `conv_tol`: emit `warnings.warn`, return energy anyway (matches existing checkerboard behavior — non-convergence is a warning, not an error, because optimizers tolerate noisy energies during outer iterations). GMRES backward failure: same warning policy. A `strict=True` flag promotes both to errors.

**Deliberately not added.** No silent dtype promotion (PR #343 removed it). No auto-shrinking χ on convergence failure. No `try/except` around projector to "recover" from rank-deficient cases (the `S_safe` clamp handles it mathematically; try/except masks bugs). No backward-compatibility shim for the dummy-bond path. No autodifferentiation through the projector phase (gauge degree of freedom; phase fix returns `stop_gradient` factor).

## Testing strategy

**Pyramid.**

- **Unit (`-m core`, fast):** per-module shape/leg-label/idempotence tests, single-step move correctness, numerical safeguards, public API smoke at D=2, χ=4.
- **Integration (`-m core`, fast):** `test_gradient_finite` (jax.grad, all 5 IPESS-equivalent grads finite, complex128); `test_fd_vs_ad_small` (complex-step FD vs AD at D=2, χ=4, rel-tol 1e-3); `test_uniform_case_recovers_lukin_sotnikov_d2` (A=B Heisenberg honeycomb at D=2, χ=8, energy within Lukin-Sotnikov Table I tolerance).
- **Regression (`-m slow`):** `test_kagome_iPESS_smoke_native_path` — same kagome iPESS smoke as `tests/test_pess_ad.py` against the new path, energy within 1e-3 of the dummy-bond hack at fixed seed.

**Reference data.**

1. Lukin-Sotnikov Table I (Heisenberg honeycomb, D ∈ [2, 7]) — primary literature reference for uniform 1-sublattice.
2. Dummy-bond hack converged energy at fixed seed/config — anchors that the rank-4 path agrees with the brick-wall path on the kagome case.
3. variPEPS reference run (your memory: `project_varipeps_2site_honeycomb_works.md` — variPEPS at χ=16 complex128). Generate one set offline, check into `tests/data/honeycomb_reference.json`.

**AD policy.** All AD tests use the new `honeycomb_ctm_energy_implicit` (mirrors PR #344 baseline migration). No xfail expected.

**What we deliberately do NOT test in v1.** SymmetricTensor (deferred per Y3); multi-unit-cell beyond 2-sublattice (YAGNI); biorthogonal projector path (one test asserts the stub raises `NotImplementedError`); fermionic.

## Validation and acceptance criteria

**Milestone M1 — Unit + AD-stability gate (must pass before code review).** All `tests/test_ctm_honeycomb_*.py` core tests pass on Python 3.11, 3.12, macOS-3.12. `test_fd_vs_ad_small` rel-tol 1e-3. `test_complex128_grad_finite` no NaN. Safeguard tests pass. Non-negotiable.

**Milestone M2 — Reference reproduction (must pass for green light to merge).**
- **M2a Uniform 1-sublattice.** Set A=B, run after short L-BFGS at D=2, χ=20 on Heisenberg honeycomb. Energy within 1% of Lukin-Sotnikov Table I.
- **M2b 2-sublattice kagome swap-out.** Run kagome iPESS end-to-end against the new path; converged energy within 1e-3 absolute of the dummy-bond path at fixed seed, D=2 d=3 χ=8; no conditioning warnings emitted.

**Milestone M3 — Performance reasonable (informational).** One JIT-compiled iteration at D=4, χ=16 on GPU completes in <2s. Compared against the dummy-bond hack baseline (~1 min/step backward) — not aiming to beat, just confirming no catastrophic regression.

**Out of scope for v1 acceptance.** Beating variPEPS energies (same ballpark suffices). Lukin-Sotnikov Kitaev (anisotropic, paper notes uniform path doesn't extend). SymmetricTensor U(1) tests. Fermionic. Larger unit cells. Biorthogonal projectors.

**PR boundaries.** This PR introduces the honeycomb CTM v1 only. A **separate, follow-up PR** rewires `pess_optimize.py` to use it and deletes the dummy-bond helpers. Biorthogonal projectors, SymmetricTensor tests, fermionic, larger unit cells — each their own future PR.

## Future consolidation with shared CTM core

Once honeycomb v1 is stable, the duplication between `_ctm_tensor_*.py` (square checkerboard) and `_ctm_honeycomb_*.py` becomes the natural target for refactor.

**Intended refactor.** Introduce a `LatticeGeometry` interface (Python protocol or dataclass) capturing geometry-specific behavior: number of CTM directions, site tensor rank and leg labels, double-layer construction, move topology, projector boundary shapes. `SquareCheckerboardGeometry` and `HoneycombGeometry` implementations; CTM core dispatches on geometry. Existing `_ctm_tensor_*.py` and `_ctm_honeycomb_*.py` become thin wrappers around a shared `_ctm_core_*.py` family. Public APIs unchanged.

**Triggering criteria.** Don't refactor preemptively. Trigger when **at least one** is true:

1. A third lattice geometry is on the active roadmap (triangular, native kagome 6-fold) — extract before adding it to avoid 3-way duplication.
2. A material AD/projector improvement needs to land in both paths — extracting first prevents two parallel changes.
3. Tenax has been stable on honeycomb v1 for ≥ 6 weeks of real use, the abstractions have proven natural, and duplication is causing review friction.

If none trigger by ~6 months out, revisit whether consolidation is paying for itself. **Premature abstraction is worse than duplication.**

**PR shape.** Pure-refactor PR (no behavior changes). Same M1+M2 acceptance — existing tests prove invariance. PR description: "consolidates CTM core; no public-API changes."

**Closing the loop.** The honeycomb v1 PR description references this section explicitly:

> This PR introduces a parallel `_ctm_honeycomb_*.py` family alongside the existing checkerboard CTM. The duplication is intentional: see "Future consolidation" in `docs/plans/2026-04-25-honeycomb-ctm-design.md` for the post-v1 refactor plan. Reviewers: please flag duplication you'd want addressed in a follow-up consolidation PR rather than as a v1 blocker.

## Tradeoffs and risks

- **Why X2 native rank-4 over X1a non-uniform-D rank-5.** The reference papers use rank-4 native. X1a is a different algorithm (square 4-corner CTM with one trivial direction), not a port. X1a was preferred when I mistakenly thought it matched the papers; once verified that Lukin-Sotnikov uses rank-4 with `A^s_{ijk}` and 6-corner topology, X2 became the only faithful choice.
- **Why isometric projectors over biorthogonal.** Tenax's existing isometric SVD/eigh path with phase gauge + complex128 + `S_safe` is battle-tested at χ=16 in 2-sublattice checkerboard (analogous regime). Biorthogonal projectors are substantial new infrastructure; defer until empirical evidence shows they're needed. Pluggable projector layer keeps the option open.
- **Why dense-only tests in v1 with protocol-generic code.** Y3: pay a small upfront cost (`tensor.contract(...)` over `jnp.einsum(...)`) to get SymmetricTensor "for free" later, without requiring a parallel symmetric-only port.
- **Risk: 2-sublattice biorthogonality required at high χ.** If isometric projectors prove unstable in the honeycomb 2-sublattice regime in ways they aren't in checkerboard, biorthogonal becomes mandatory. Mitigation: pluggable projector interface; biorthogonal can land in a follow-up PR without rewriting topology.
- **Risk: env tensor convention drift.** With per-sublattice (C, L, R) tuples and per-direction labels (`e0/e1/e2`), the leg-flow conventions differ from the existing checkerboard u/d/l/r. Mitigation: explicit `_HONEYCOMB_EDGE_SPECS` dict with flow conventions, tested against round-trip identities.
- **Risk: kagome triangle energy semantics.** The triangle Hamiltonian basis (`np.kron(np.kron(s_a, s_b), s_c)`) must match the supersite physical-leg fusion order in `_build_honeycomb_site_tensor`. Mitigation: shared fusion-order constant referenced from both paths; regression test in M2b catches mismatches.
