# Multisite-CTM RDM Brute-Force Diagnostic — Design

**Status:** design approved 2026-05-06; implementation pending (will be planned via `superpowers:writing-plans`).
**Branch:** `feat/multisite-kagome-pess` (worktree `.worktrees/multisite-kagome-pess`).
**Parent plan:** `docs/plans/2026-05-05-multisite-kagome-pess.md` — Phase C.3 BLOCKED. This design specifies the diagnostic that resumes Phase C.3.

## Goal

Localise which of three suspects in `compute_energy_pess_3site_multisite`'s consumer chain produces unphysical infinite-lattice RDMs at D=4, χ=16:

1. The multisite-CTM environment construction (`_ctm_tensor_multisite` and friends).
2. `_rdm{2x1,1x2}_tensor_2site` consuming those envs across **dim-1 v-w iPEPS placeholder bonds**.
3. `_rdm_3site_marginal_vw_{row,col}` consuming those envs.

The wavefunction-fidelity test passes at D∈{1..6} on the 1-cell PBC torus, so the encoding itself is correct. The blocker is somewhere in the chain `multisite-CTM forward → environments → 2-site / marginalised-3-site RDM helpers` on the **infinite lattice**, manifesting at D≥2 χ=16 in the C.3 probe (`E/site = -0.916` at L-BFGS step 7, ~0.48 below the spin-1/2 AFH variational floor).

## Decision tree from the diagnostic

| Observation | Localisation |
|---|---|
| At D=1, all 6 RDMs match brute-force at the 3×3 PBC torus to 1e-10 | encoding + consumer helpers are correct at product state; bug is D≥2-only |
| At D=1, **any** RDM disagrees | direct consumer-helper bug — fix here first |
| At D=2,3,4: 4 NN RDMs converge to brute-force as χ→∞, but 2 marginalised v-w RDMs do not | bug isolated to `_rdm_3site_marginal_vw_{row,col}` |
| At D=2,3,4: marginalised v-w RDMs converge but 4 NN RDMs do not | bug isolated to `_rdm{2x1,1x2}_tensor_2site` under dim-1 placeholder bonds |
| Both RDM families fail to converge | bug upstream in multisite-CTM environments themselves |
| Marginalisation-consistency invariant fails (3-site ρ_uvw marginalised over u ≠ ρ_vw via direct 2-site call on the dim-1 v-w bond) | bug is environment-side (the two paths share envs) |
| Structural invariants (Herm / PSD / trace) all pass; per-bond ⟨H⟩ stays in [-3/4, 1/4] | qualitative sanity OK; bug is quantitative — magnitude of correlations |
| Any structural invariant fires | hard test gate flags it; investigate that invariant first |

## Scope

**A + C combined:**

- **A.** Audit-only probe — diagnostic table over a (D, χ, state) ladder. JSON output. No asserts. Matches the existing `kagome_pess_multisite_phase_c3_*.py` naming + JSON pattern.
- **C.** Strict pytest gates — structural-invariants test file with hard asserts.

Dropped from earlier proposal: 1-cell-vs-2×2 torus-size scaling. The multisite-CTM tiles a 2D lattice with diagonal-stripe sublattices `(x+y) mod 3 ∈ {u,v,w}`; the minimum balanced PBC torus is 3×3. The "1-cell torus" used by the WF-fidelity test is a different topology (iPESS-cell direct closure) and does not generalise as a torus-size sweep.

## Lattice tiling — derivation

Walking the kagome `neighbor_map` from each direction:

| direction | u→ | v→ | w→ |
|---|---|---|---|
| right | v | w | u |
| left | w | u | v |
| top | w | u | v |
| bottom | v | w | u |

`right` and `bottom` both advance sublattice index by +1 (mod 3); `left` and `top` advance by −1. Verified all 12 entries match the 2D lattice with sublattice = `(x + y) mod 3` ↦ `{0:u, 1:v, 2:w}`.

**Convention:** `top` is `−y`, `bottom` is `+y` (y increases downward). Verified against `neighbor_map[u][top] = w = (0+(−1)) mod 3 = 2` and `neighbor_map[u][bottom] = v = (0+1) mod 3 = 1`.

**3×3 PBC torus sublattice layout:**

```
(0,0)=u (1,0)=v (2,0)=w
(0,1)=v (1,1)=w (2,1)=u
(0,2)=w (1,2)=u (2,2)=v
```

9 sites total, 3 of each sublattice, balanced. Wavefunction dim `d^9 = 512` at d=2.

Next size up is 6×6 = 36 sites = `d^36 ≈ 7×10^10` — infeasible for brute-force.

## Brute-force contractions

### 3×3 multisite-tile PBC torus

New helper `_contract_multisite_3x3_torus(sites: dict[str, jnp.ndarray]) -> jnp.ndarray`. Inputs are the 3 multisite tensors from `pess_to_kagome_3site_multisite`. Builds 9 rank-5 tensors at the 9 lattice positions (with `S_u`, `S_v`, `S_w` placed per the `(x+y) mod 3` rule), wires bonds via the `neighbor_map`, contracts to a rank-9 wavefunction `(d, d, d, d, d, d, d, d, d)` indexed by the 9 physical legs in row-major position order.

**Contraction cost:** peak intermediate ~`D^perimeter ≤ D^6 = 4096` at D=4. Tractable at D∈{1..4} via `jnp.einsum` with `opt_einsum` ordering.

### 3×3 iPESS PBC torus

New helper `_contract_ipess_3x3_torus(state: IPESSState) -> jnp.ndarray`. 9 R + 9 T_u + 9 T_d tensors arranged as the kagome cell tiles a 3×3 stripe pattern. Same Convention-C gauge as `_contract_ipess_one_cell_pbc` (`sqrt(λ)` on each R's T_d-side, full `λ` on T_u-side).

**Encoding-fidelity sanity gate (3×3 generalisation of the existing 1-cell gate):**

```python
@pytest.mark.core
@pytest.mark.parametrize("D", [1, 2, 3])
def test_3site_multisite_3x3_torus_fidelity_matches_ipess(D):
    state = IPESSState.random(D=D, d=2, key=jax.random.PRNGKey(0))
    psi_ipess = _contract_ipess_3x3_torus(state)
    sites = pess_to_kagome_3site_multisite(...)
    psi_ms = _contract_multisite_3x3_torus(sites)
    fidelity = |<psi_ipess|psi_ms>|² / (||psi_ipess||² · ||psi_ms||²)
    assert_allclose(fidelity, 1.0, atol=1e-12)
```

If this fails the brute-force can't be a reference. Test is independent of CTM — purely wavefunction-direct.

### Brute-force RDM extraction

`_brute_force_rdm_from_torus_psi(psi, sites_to_keep: tuple[int, ...]) -> jnp.ndarray`. Given the rank-9 wavefunction and a tuple of physical-leg indices to keep, returns `ρ = Tr_{rest}(|ψ⟩⟨ψ|) / ⟨ψ|ψ⟩`. Two-line `einsum` + reshape.

For the 6 bonds we audit:

- 4 NN: 4 distinct (u, v) and (u, w) site-pairs at adjacent lattice positions on the 3×3 tile. Pick representatives via the `(x+y) mod 3` map and the `neighbor_map` direction.
- 2 marginalised-3-site v-w: take any (u, v, w) triple that the `_rdm_3site_marginal_vw_{row,col}` helpers reference — the `_row` helper traces a 1×3 horizontal block, the `_col` helper traces a 3×1 vertical block.

## Structural invariants (strict pytest gates)

For each of the 6 RDMs the energy formula consumes — at D=2, χ=16, on an SU-warmstarted state — assert:

1. **Hermiticity:** `‖ρ - ρ†‖_F ≤ 1e-10`.
2. **PSD:** `min(eigvalsh(ρ)) ≥ -1e-10`.
3. **Trace 1:** `|tr(ρ) − 1| ≤ 1e-8`.
4. **Marginalisation-consistency:** marginalise the 3-site `ρ_uvw` (sourced from the same multisite-CTM environments that back `_rdm_3site_marginal_vw_*`) over u; compare to the v-w 2-site RDM extracted via the *direct* `_rdm{2x1,1x2}_tensor_2site` call across the dim-1 v-w iPEPS bond. At converged CTM both probe the same physical v-w correlator and must agree to 1e-8.
5. **Per-bond ⟨H⟩ in spectrum bound:** `eigvalsh(H_pair)` for spin-1/2 isotropic XXZ at δ=1 is `{−3/4, 1/4, 1/4, 1/4}`. Each bond's `tr(ρ · H_pair)` must satisfy `−3/4 − 1e-8 ≤ ⟨H⟩ ≤ 1/4 + 1e-8`. The C.3 blocker (`⟨H⟩/bond ≈ −0.46`) is inside this bound, so this gate alone won't catch the current pathology — it's an early-warning for stronger violations.
6. **D=1 brute-force exact equality:** at D=1 the encoded state is product, so each of the 6 multisite-CTM RDMs must equal the corresponding 3×3-torus brute-force RDM to 1e-10.

## File layout

```
examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py    [new, audit probe]
tests/test_pess_3site_multisite_rdm_invariants.py                  [new, strict gates]
```

The probe imports brute-force helpers from the test file (test files are importable as modules). No `src/` changes.

### Audit probe CLI

```
python examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.py \
    --D-ladder 1,2,3,4 \
    --chi-ladder 8,16,32 \
    --state-kind both \   # random | su | both
    --seed 0 \
    --output examples/kagome_pess_multisite_phase_c3_rdm_brute_force_diag.json
```

Outputs JSON with one entry per (D, χ, state_kind, bond_id):

```json
{
  "D": 2, "chi": 16, "state_kind": "su", "bond_id": "uv-h",
  "frobenius_delta_brute_vs_ctm": 0.0123,
  "energy_brute": -0.21, "energy_ctm": -0.45, "energy_delta": -0.24,
  "rho_brute_eigvals": [...], "rho_ctm_eigvals": [...]
}
```

Plus a printed table for human inspection.

### Test categories

| Test | D | χ | Marker | Wall budget |
|---|---|---|---|---|
| 3×3 multisite-vs-iPESS WF-fidelity (3 D values) | 1, 2, 3 | — | `core` | < 5 s |
| D=1 brute-force vs CTM RDM equality (gate #6) | 1 | 8 | `core` | < 30 s |
| Structural invariants 1-3 (Herm/PSD/trace) | 2 | 16 | `algorithm` | < 60 s |
| Marginalisation-consistency (gate #4) | 2 | 16 | `algorithm` | < 60 s |
| Per-bond ⟨H⟩ spectrum bound (gate #5) | 2 | 16 | `algorithm` | < 30 s |

Tests live in `tests/test_pess_3site_multisite_rdm_invariants.py`. The auto-marker rules in `conftest.py` are filename-based (`core`, `algorithm`, `slow`); this filename does not trip any auto-marker, so explicit `@pytest.mark.core` / `@pytest.mark.algorithm` decorators are used. CI required (`-m core`) gets the cheap two; full suite gets all five. Total CI time delta: ~30 s.

### Audit probe wall budget

12 (D, χ) points × (~5 s SU + ~3-25 s CTM forward + ~2 s RDM extractions) ≈ 5 min single-CPU. Acceptable for a diagnostic.

## Out of scope

- No `src/` changes. Diagnostic + invariants test only.
- No fix in this PR — once the diagnostic localises the bug, that's a follow-up PR with this probe as the localising witness.
- No symmetric-tensor or fermionic path coverage. Pure dense, U(1)-trivial.
- No long L-BFGS run. Probe runs on SU-warmstarted states (deterministic, what C.3 builds on).

## Resume-on-failure

- **Gate #4 (marginalisation-consistency) fires while #1-3 pass:** bug is environment-construction-side; investigate `_ctm_tensor_multisite` and how envs feed `_rdm_3site_marginal_vw_*`.
- **Gates #1-3 fire on a specific RDM:** bug is in the helper producing that RDM. The 4-NN-vs-marginalised split tells us which helper.
- **All gates pass at D=2 χ=16 but the audit probe shows large brute-force-vs-CTM deltas:** the bug is quantitative-correlation, not qualitative-pathology. Re-run audit at D=4 χ=32 to see if the deltas grow with D and χ.

## Cross-references

- Parent plan: `docs/plans/2026-05-05-multisite-kagome-pess.md`
- Memory: `~/.claude/projects/-home-yjkao-tenax/memory/project_kagome_3site_multisite_pivot.md`
- C.3 audit-trail probes: `examples/kagome_pess_multisite_phase_c3_{probe,tight_ctm_probe,hz_diag}.py`
- WF-fidelity test (encoding gate, generalised here to 3×3): `tests/test_pess_3site_multisite_wavefunction.py`
- Multisite-CTM dispatch: `src/tenax/algorithms/_ctm_tensor_convergence.py::ctm_multisite`
- Energy chain: `src/tenax/algorithms/_pess_multisite_energy.py::compute_energy_pess_3site_multisite`
- PR #398 (parent): https://github.com/tenax-lab/tenax/pull/398
