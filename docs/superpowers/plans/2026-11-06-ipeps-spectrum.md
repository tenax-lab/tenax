# iPEPS Excitation Spectrum and Spectral Function Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `compute_spectrum` and `spectral_function` to tenax, validated at D=2 against a window oracle and the published dispersion, and run the D=3, D=4 and J2=0.3 benchmarks.

**Architecture:** A private module `_ipeps_excitation_ctm.py` holds the raw-array mixed contractions and the second derivative (finite difference on the ket, reverse on the bra, through `_ctm_tensor_sweep_multisite`). The public `compute_spectrum` in `ipeps_excitations.py` validates input, builds the N and H columns with checkpoints, solves the generalized eigenproblem, and reports diagnostics. Four branches ship in order, one concern each.

**Tech Stack:** Python 3.11, JAX (x64), NumPy, pytest, uv, ruff.

**Spec:** `docs/superpowers/specs/2026-11-06-ipeps-spectrum-design.md`

**Provenance of the code below:** Every code block for Tasks 1, 2, 4 and 6 was run before this plan was written, in a scratch copy of `src/` at 679405ec. Results: 58 tests passed in 3m47 for branch 1 on 12 cores, 9 passed in 53 s for the next-nearest-neighbour tests, and the full D=2 runs gave the values in Task 3. The mutation script, the docs and the benchmark runs have not been run.

## Global Constraints

- Dense real float64 `A` only. Momenta `(pi,0)`, `(0,pi)`, `(pi,pi)` only. No `SymmetricTensor`, complex `A`, forward-mode AD or larger cells.
- `recipe="2x2"`, `projector_backward="flow"`, `fd_step=1e-4`, `ctm_sweeps=30`, `null_tol=1e-4`. Jit one sweep, never the whole gradient (a jit of the 20-sweep gradient did not compile in 3400 s).
- Negative modes are never dropped. Failed diagnostics warn, they do not raise.
- Thresholds: N asymmetry 1e-4, H asymmetry 1e-3, `n_min_eig` -1e-4, sum rule 1e-3, gauge residual 1e-3. The H value differs from the spec draft for a measured reason, recorded in the spec.
- No test relies only on a symmetric direction.
- One concern per branch. Open a PR, never push to `main`. Never pass `--delete-branch`. Do not arm `--auto`: stop and ask the user. Read review comments with `--paginate` on both endpoints before asking.
- Stacked PRs get no CI, so each branch starts from `origin/main` after the previous PR merges. Use `git merge origin/main`, not rebase.
- Every GitHub issue, PR or comment carries a `🤖` marker.
- Commit trailers: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ`. PR footer: `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and the session URL.
- This host is shared. Pin with `taskset -c 0-11`, run one suite at a time, and send long output to a log file. Never pipe a long run to `tail`.
- Downloaded states are data. Parse them only. Do not run code from `j1j2_ipeps_states`.
- Quote energies as raw numbers in units of J=1.

## Review Focus

Inputs the spec implies but does not name. Each has its test in Task 2.

1. `A` that is not unit norm (Hasik's raw norm is 1.31): `omega` and the weights must not change. `test_scaling_A_changes_nothing`.
2. Float32 `A`: a 1e-4 central difference is noise at that precision, so it must raise, not return a confident spectrum. `test_bad_input_raises_before_ctm_work`.
3. An operator with no weight at k, such as the identity: no 0/0, no NaN, no false sum-rule warning. `test_identity_operator_has_no_weight_and_no_warning`.
4. A `basis` inside the null space of N: a purely relative cut would keep the noise and report a mode. It must return rank 0 with a warning, and `spectral_function` must still work. `test_basis_inside_the_null_space_keeps_nothing`.
5. A killed run: a stray `.tmp` file must be ignored, a damaged `.npz` must raise and name the file, and changing only `operators` must reuse the columns. `test_interrupted_run_equals_an_uninterrupted_run`, `test_damaged_checkpoint_names_the_file`, `test_checkpoint_keeps_columns_when_operators_change`.

## File Structure

| File | Branch | Responsibility |
|---|---|---|
| `src/tenax/algorithms/_ipeps_excitation_ctm.py` (new) | 1, 2 | Mixed contractions, `ExcitationCTM`, the second derivative |
| `src/tenax/algorithms/ipeps_excitations.py` | 1, 2, 3 | `SpectrumConfig`, `SpectrumResult`, `compute_spectrum`, `spectral_function`; deprecation of `compute_excitations` |
| `src/tenax/__init__.py`, `src/tenax/algorithms/__init__.py` | 1 | Lazy exports |
| `tests/_ipeps_spectrum_fixtures.py` (new) | 1 | Hasik D=2 state, gates, cell gauge helper |
| `tests/_window_oracle.py` (new) | 1 | Independent reference for N, with Onsager constants |
| `tests/test_ipeps_spectrum_contractions.py` (new) | 1 | Gauge invariance, placement, H against FD-FD, N against the oracle |
| `tests/test_ipeps_spectrum.py` (new) | 1 | Public API, errors, warnings, checkpoints |
| `tests/test_ipeps_spectrum_full.py` (new) | 1 | Slow: full D=2 spectra |
| `tests/test_ipeps_spectrum_nnn.py` (new) | 2 | Mixed diagonal RDM and `gate_nnn` |
| `tests/test_ipeps_spectrum_benchmark.py` (new) | 4 | State parser and U(1) scan |
| `benchmarks/ipeps_spectrum_hasik.py` (new) | 4 | Benchmark entry point |
| `docs/guide/algorithms/ad_excitations.md`, `docs/api/algorithms.rst`, `README.md`, `.claude/skills/tenax-ipeps-workflow/SKILL.md` | 1, 3, 4 | Docs |

---

## Branch 1: `feat/ipeps-spectrum`

### Task 1: Mixed contractions and the second derivative

**Files:**
- Create: `src/tenax/algorithms/_ipeps_excitation_ctm.py`
- Create: `tests/_ipeps_spectrum_fixtures.py`, `tests/_window_oracle.py`, `tests/test_ipeps_spectrum_contractions.py`
- Modify: `tests/conftest.py` (`_FILE_MARKERS`)

**Interfaces:**
- Consumes: `_ctm_tensor_multisite`, `_ctm_tensor_sweep_multisite`, `ctm_tensor` (`_ctm_tensor_convergence.py`), `_build_double_layer_tensor` (`_ctm_tensor_init.py`), `_wrap_as_dense_tensor` (`_ipeps_optimize_shared.py`), `compute_energy_ctm_tensor_multisite` (`_ctm_tensor_energy.py`), `_as_dense_env` (`ipeps_excitations.py`).
- Produces:
  - `CELL: list[tuple[int, int]]`, `NEIGHBORS: dict`, `cell_phases(a: int, b: int) -> dict`, `unpack_env(env, D) -> dict`
  - `window_1x1(E, K, Kb)`, `mixed_rdm_h(EL, ER, K1, Kb1, K2, Kb2)`, `mixed_rdm_v(ET, EB, K1, Kb1, K2, Kb2)`, `z00(c1, c2, c3, c4)`, `z10(top, bot)`, `z01(left, right)`
  - `ExcitationCTM(A, gate, chi, sweeps)` with attributes `A, D, d, gate, ctm_converged, envs0`, and methods `densities(phases, eps, X, Y) -> (e, l)` and `ket_derivative(phases, X, h) -> (g_h, g_n)`
  - `_converged_env(sites, chi) -> (envs, converged: bool)`, a module-level seam the tests patch
  - Fixtures: `hasik_d2()`, `rotated_gate()`, `heisenberg_gate()`, `apply_op(A, op)`, `regauge_cell(envs, rng)`, `SY_REAL`, `SY_COMPLEX`, `SX`, `SZ`, `E_D2`

- [ ] **Step 1: Create the branch and install hooks**

The worktree `../tenax-spectrum` is on `docs/ipeps-spectrum-spec`, which holds the spec and this plan. Branch from it, so the docs ship with the code.

```bash
cd /home/yjkao/tenax-spectrum
git fetch origin
git merge origin/main
git switch -c feat/ipeps-spectrum
uv run pre-commit install
```

- [ ] **Step 2: Write the fixtures**

Create `tests/_ipeps_spectrum_fixtures.py`:

```python
"""Hasik's D=2 J2=0 Heisenberg state, for the compute_spectrum tests.

Source: github.com/jurajHasik/j1j2_ipeps_states (CC BY 4.0), j2=0.0, D=2,
chi_opt48, in the sublattice-rotated frame. peps-torch lists E=-0.66023109348.
"""

import jax.numpy as jnp
import numpy as np

from tenax.algorithms._ipeps_excitation_ctm import CELL, NEIGHBORS

_D2_ENTRIES = [
    1.0, -1.6194835639214339e-02, 1.7282585382645214e-02, 4.1985986298862715e-01,
    1.7282585382645214e-02, 4.1985986298862715e-01, -4.4700030141459453e-05, 3.0093367495500688e-02,
    1.7282585382645214e-02, 4.1985986298862715e-01, -4.4720562436946177e-05, 3.0093429238755503e-02,
    -4.4720562436946177e-05, 3.0093429238755503e-02, -2.5360940365643496e-05, 1.6170125837876856e-03,
    1.7282585382645214e-02, 4.1985986298862715e-01, -4.4720562436946177e-05, 3.0093429238755503e-02,
    -4.4720562436946177e-05, 3.0093429238755503e-02, -2.5360940365643496e-05, 1.6170125837876856e-03,
    -4.4700030141459453e-05, 3.0093367495500688e-02, -2.5360940365643496e-05, 1.6170125837876856e-03,
    -2.5360940365643496e-05, 1.6170125837876856e-03, -1.9191912448893797e-06, 7.7277947587816155e-05,
]  # fmt: skip

SY_REAL = np.array([[0.0, 0.5], [-0.5, 0.0]])  # i S^y
SY_COMPLEX = np.array([[0.0, -0.5j], [0.5j, 0.0]])
SX = np.array([[0.0, 0.5], [0.5, 0.0]])
SZ = np.diag([0.5, -0.5])
E_D2 = -0.66023109348


def hasik_d2() -> np.ndarray:
    """The state, normalised to unit Frobenius norm, in ``(u, d, l, r, s)``."""
    A = np.array(_D2_ENTRIES).reshape(2, 2, 2, 2, 2)
    return A / np.linalg.norm(A)


def heisenberg_gate() -> np.ndarray:
    """``S.S`` as ``(s1, s2, s1', s2')``: next-nearest neighbours, same sublattice."""
    return np.array(
        [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, 0.0],
            [0.0, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ]
    ).reshape(2, 2, 2, 2)


def rotated_gate() -> np.ndarray:
    """``S.S`` in the sublattice-rotated frame: nearest neighbours."""
    H = heisenberg_gate().reshape(4, 4)
    W = np.kron(np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    return (W @ H @ W.T).reshape(2, 2, 2, 2)


def apply_op(A: np.ndarray, op: np.ndarray) -> np.ndarray:
    return np.tensordot(A, op, axes=([4], [1]))


def regauge_cell(envs, rng):
    """One invertible matrix per link of the cell, so shared bonds stay closed.

    A link joins ``c`` to its right or bottom neighbour in one of the four edge
    rows. A contraction that pairs environments of the wrong cell sites sees two
    unrelated matrices and changes value.
    """
    chi = envs[(0, 0)]["C1"].shape[0]

    def gauge():
        g = rng.normal(size=(chi, chi)) + 3 * np.eye(chi)
        return g, np.linalg.inv(g)

    top = {c: gauge() for c in CELL}
    right = {c: gauge() for c in CELL}
    bottom = {c: gauge() for c in CELL}
    left = {c: gauge() for c in CELL}
    out = {}
    for c in CELL:
        E = {k: np.asarray(v) for k, v in envs[c].items()}
        west, north = NEIGHBORS[c]["left"], NEIGHBORS[c]["top"]
        out[c] = {
            "C1": left[north][0].T @ E["C1"] @ top[west][0],
            "T1": np.einsum("ab,bxyd,dc->axyc", top[west][1], E["T1"], top[c][0]),
            "C2": top[c][1] @ E["C2"] @ right[north][0],
            "T2": np.einsum("ab,bxyd,dc->axyc", right[north][1], E["T2"], right[c][0]),
            "C3": right[c][1] @ E["C3"] @ bottom[c][1].T,
            "T3": np.einsum("ab,bxyd,dc->axyc", bottom[west][1], E["T3"], bottom[c][0]),
            "C4": left[c][1] @ E["C4"] @ bottom[west][0],
            "T4": np.einsum("ab,bxyd,dc->axyc", left[north][1], E["T4"], left[c][0]),
        }
    return {c: {k: jnp.asarray(v) for k, v in out[c].items()} for c in CELL}
```

- [ ] **Step 3: Write the window oracle**

Create `tests/_window_oracle.py`:

```python
"""Window oracle: C(r) = <O^dag_r O_0> on rectangular CTM windows, by direct sum.

A reference for N_k that shares no code with compute_spectrum: it does no
differentiation and no perturbed CTM. It is converged only at D=2. Certified on
the Ising PEPS against Onsager in test_ipeps_spectrum_contractions.py.
"""

import numpy as np
from scipy.special import ellipk


class Window:
    """All tensors are plain numpy; site tensors are closed (u, d, l, r) of dim D^2."""

    def __init__(self, env):
        self.C1, self.C2, self.C3, self.C4, self.T1, self.T2, self.T3, self.T4 = [
            np.asarray(x) for x in env
        ]

    def left(self, h):
        L = self.C1.T  # (t, a)
        for _ in range(h):
            # (t, x.., a) (a, x, a') -> (t, x.., x, a')
            L = np.tensordot(L, self.T4, axes=(L.ndim - 1, 0))
        return np.tensordot(L, self.C4, axes=(L.ndim - 1, 0))  # (t, x1..xh, b)

    def right(self, h):
        R = self.C2  # (t, a)
        for _ in range(h):
            R = np.tensordot(R, self.T2, axes=(R.ndim - 1, 0))
        return np.tensordot(R, self.C3, axes=(R.ndim - 1, 0))  # (t, r1..rh, k=b)

    def apply(self, L, col):
        """Absorb one column; col[i] is the closed site tensor of row i."""
        h = len(col)
        # (t, x1..xh, b) -> (t', u, x1..xh, b)
        L = np.tensordot(self.T1, L, axes=(0, 0))  # (u, t', x1..xh, b)
        L = np.moveaxis(L, 1, 0)  # (t', u, x1..xh, b)
        for i, a in enumerate(col):
            # L = (t', u, r1..r_i, x_{i+1}..x_h, b); contract u and x_{i+1}
            xpos = 2 + i
            L = np.tensordot(L, a, axes=((1, xpos), (0, 2)))  # a: (u,d,l,r)
            # now (t', r1..r_i, x_{i+2}..x_h, b, d, r_new)
            L = np.moveaxis(L, -2, 1)  # d -> slot 1
            L = np.moveaxis(L, -1, 2 + i)  # r_new -> after r1..r_i
        # (t', d, r1..rh, b) with T3 (b, d, b')
        L = np.tensordot(L, self.T3, axes=((h + 2, 1), (0, 1)))
        return L  # (t', r1..rh, b')

    def close(self, L, h):
        return np.tensordot(L, self.right(h), axes=L.ndim)


def correlators(env, aa, ket, bra, both, R):
    """C[(m, n)] for |m|,|n| <= R, r = pos(bra) - pos(ket), y pointing down.

    aa/ket/bra/both are closed double-layer tensors: no operator, O on the ket,
    O^dag on the bra, and both on one site.  C(-r) = conj C(r) fills the rest.
    """
    w = Window(env)
    C = {}
    for n in range(R + 1):
        h = n + 1
        Rc = w.right(h)
        L0 = w.left(h)
        for ket_row, bra_row in [(0, n)] if n == 0 else [(0, n), (n, 0)]:
            sgn = 1 if ket_row == 0 else -1  # bra below ket: +n; bra above: -n
            col = [aa] * h
            nv = w.apply(L0, col)
            if ket_row == 0:  # the m = 0 column holds both insertions
                c0 = list(col)
                if n == 0:
                    c0[0] = both
                else:
                    c0[0], c0[n] = ket, bra
                val = np.tensordot(w.apply(L0, c0), Rc, axes=nv.ndim)
                C[(0, n)] = val / np.tensordot(nv, Rc, axes=nv.ndim)
            c1 = list(col)
            c1[ket_row] = ket
            v = w.apply(L0, c1)
            cb = list(col)
            cb[bra_row] = bra
            for m in range(1, R + 1):
                val = np.tensordot(w.apply(v, cb), Rc, axes=nv.ndim)
                norm = np.tensordot(w.apply(nv, col), Rc, axes=nv.ndim)
                C[(m, sgn * n)] = val / norm
                if m < R:
                    v = w.apply(v, col)
                    nv = w.apply(nv, col)
                    # keep the scale bounded on long sweeps
                    s = np.linalg.norm(nv)
                    v, nv = v / s, nv / s
    for (m, n), val in list(C.items()):
        C[(-m, -n)] = np.conj(val)
    return C


def closed_double_layer(K, Kb):
    """``(u, d, l, r)`` of dimension ``D^2``, with the physical leg traced."""
    D2 = K.shape[0] ** 2
    return np.einsum("udlrs,UDLRs->uUdDlLrR", K, Kb).reshape(D2, D2, D2, D2)


def norm_form(env, A, B, one_site_expectation, a, b, R):
    """``N_k(B, B)`` at ``k = (a pi, b pi) != 0``: the sum, less the disconnected part.

    ``one_site_expectation`` is ``<B>_ket <B>_bra``, the constant that ``C(r)``
    approaches at large ``r``.
    """
    C = correlators(
        env,
        closed_double_layer(A, A),
        closed_double_layer(B, A),
        closed_double_layer(A, B),
        closed_double_layer(B, B),
        R,
    )
    return sum(
        np.cos(np.pi * (a * m + b * n)) * (np.real(value) - one_site_expectation)
        for (m, n), value in C.items()
    )


def ising_peps(beta):
    """``|psi|^2`` of this PEPS is the classical 2D Ising partition function."""
    W = np.array(
        [
            [np.exp(beta / 2), np.exp(-beta / 2)],
            [np.exp(-beta / 2), np.exp(beta / 2)],
        ]
    )
    e, V = np.linalg.eigh(W)
    M = V @ np.diag(np.sqrt(e)) @ V.T
    A = np.einsum("su,sd,sl,sr->udlrs", M, M, M, M)
    return A / np.linalg.norm(A)


def onsager_c10(beta):
    """Exact nearest-neighbour correlation of the 2D Ising model."""
    t = np.tanh(2 * beta)
    kappa = 2 * np.sinh(2 * beta) / np.cosh(2 * beta) ** 2
    return 0.5 / t * (1 + (2 / np.pi) * (2 * t**2 - 1) * ellipk(kappa**2))
```

- [ ] **Step 4: Write the failing tests**

Create `tests/test_ipeps_spectrum_contractions.py`:

```python
"""Contractions and second derivatives behind compute_spectrum.

No test here relies only on a symmetric direction. A reversed T4/C4 ring in the
first version of these contractions matched the energy to 10 digits and S^y A to
7 digits on a C4v state, and failed only on random vectors.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_spectrum_fixtures import (
    E_D2,
    apply_op,
    hasik_d2,
    regauge_cell,
    rotated_gate,
)
from _window_oracle import (
    closed_double_layer,
    correlators,
    ising_peps,
    norm_form,
    onsager_c10,
)

from tenax.algorithms._ctm_tensor_convergence import _ctm_tensor_multisite, ctm_tensor
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor_multisite
from tenax.algorithms._ipeps_excitation_ctm import (
    CELL,
    NEIGHBORS,
    ExcitationCTM,
    cell_phases,
    mixed_rdm_h,
    mixed_rdm_v,
    unpack_env,
    window_1x1,
    z00,
    z01,
    z10,
)
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.algorithms.ipeps_excitations import _as_dense_env


@pytest.fixture(scope="module")
def ctm():
    return ExcitationCTM(hasik_d2(), rotated_gate(), chi=10, sweeps=30)


def _random_sites(rng, shape):
    return {c: jnp.asarray(rng.normal(size=shape)) for c in CELL}


def _environments(ctm):
    return {c: unpack_env(ctm.envs0[c], ctm.D) for c in CELL}


def test_every_window_is_gauge_invariant_on_the_cell(ctm):
    rng = np.random.default_rng(0)
    E = _environments(ctm)
    G = regauge_cell(E, rng)
    kets, bras = _random_sites(rng, ctm.A.shape), _random_sites(rng, ctm.A.shape)
    for c in CELL:
        cx, cy = NEIGHBORS[c]["right"], NEIGHBORS[c]["bottom"]
        cxy = NEIGHBORS[cx]["bottom"]
        pairs = [
            (window_1x1(E[c], kets[c], bras[c]), window_1x1(G[c], kets[c], bras[c])),
            (
                z00(E[cxy]["C1"], E[cy]["C2"], E[c]["C3"], E[cx]["C4"]),
                z00(G[cxy]["C1"], G[cy]["C2"], G[c]["C3"], G[cx]["C4"]),
            ),
            (z10(E[cy], E[c]), z10(G[cy], G[c])),
            (z01(E[cx], E[c]), z01(G[cx], G[c])),
            (
                mixed_rdm_h(E[c], E[cx], kets[c], bras[c], kets[cx], bras[cx]),
                mixed_rdm_h(G[c], G[cx], kets[c], bras[c], kets[cx], bras[cx]),
            ),
            (
                mixed_rdm_v(E[c], E[cy], kets[c], bras[c], kets[cy], bras[cy]),
                mixed_rdm_v(G[c], G[cy], kets[c], bras[c], kets[cy], bras[cy]),
            ),
        ]
        for plain, gauged in pairs:
            np.testing.assert_allclose(gauged, plain, rtol=1e-8, atol=0)


def test_energy_density_matches_the_published_energy(ctm):
    zero = jnp.zeros_like(ctm.A)
    energy, _ = ctm.densities(cell_phases(1, 0), 0.0, zero, zero)
    assert float(energy) == pytest.approx(E_D2, abs=1e-9)


def test_energy_density_matches_the_multisite_energy_on_a_staggered_state(ctm):
    """Bra = ket, not uniform: the bond placement against the existing code path."""
    rng = np.random.default_rng(2)
    phases = cell_phases(1, 1)
    Y = rng.normal(size=ctm.A.shape)
    sites = {c: ctm.A + 0.02 * phases[c] * Y / np.linalg.norm(Y) for c in CELL}
    wrapped = {c: _wrap_as_dense_tensor(sites[c]) for c in CELL}
    envs = _ctm_tensor_multisite(
        wrapped, NEIGHBORS, 10, max_iter=200, conv_tol=1e-10, _warn_unconverged=False
    )
    E = {c: unpack_env(envs[c], ctm.D) for c in CELL}
    reference = compute_energy_ctm_tensor_multisite(
        wrapped, envs, NEIGHBORS, jnp.asarray(ctm.gate)
    )
    assert float(ctm.energy_density(E, sites, sites)) == pytest.approx(
        float(reference), abs=1e-9
    )


def _unit(rng, shape):
    B = rng.normal(size=shape)
    return B / np.linalg.norm(B)


@pytest.mark.parametrize("ab", [(1, 0), (1, 1)], ids=["X", "M"])
def test_h_matches_a_double_finite_difference_on_random_pairs(ctm, ab):
    rng = np.random.default_rng(7)
    phases = cell_phases(*ab)
    step = 1e-3
    X, Y = _unit(rng, ctm.A.shape), _unit(rng, ctm.A.shape)
    g_h, _ = ctm.ket_derivative(phases, jnp.asarray(X), 1e-4)

    def energy(eps, delta):
        return float(
            ctm.densities(phases, eps, jnp.asarray(X), delta * jnp.asarray(Y))[0]
        )

    reference = (
        energy(step, step)
        - energy(step, -step)
        - energy(-step, step)
        + energy(-step, -step)
    ) / (4 * step**2)
    assert float(np.vdot(Y, np.asarray(g_h))) == pytest.approx(reference, abs=1e-4)


@pytest.mark.parametrize("ab", [(1, 0), (1, 1)], ids=["X", "M"])
def test_n_matches_the_window_oracle_on_random_directions(ctm, ab):
    rng = np.random.default_rng(1)
    A = np.asarray(ctm.A)
    site = _wrap_as_dense_tensor(ctm.A)
    env = _as_dense_env(
        ctm_tensor(site, 10, max_iter=800, conv_tol=1e-11, return_meta=True)[0]
    )
    E = unpack_env(ctm.envs0[(0, 0)], ctm.D)
    z = float(window_1x1(E, ctm.A, ctm.A))
    phases = cell_phases(*ab)
    B = _unit(rng, A.shape)
    B_j = jnp.asarray(B)
    disconnected = (
        float(window_1x1(E, B_j, ctm.A)) / z * float(window_1x1(E, ctm.A, B_j)) / z
    )
    reference = norm_form(env, A, B, disconnected, *ab, R=6)
    _, g_n = ctm.ket_derivative(phases, B_j, 1e-4)
    assert float(np.vdot(B, np.asarray(g_n))) == pytest.approx(reference, abs=1e-4)


def test_window_oracle_reproduces_onsager():
    beta = 0.3
    A = ising_peps(beta)
    B = apply_op(A, np.diag([1.0, -1.0]))
    env = _as_dense_env(
        ctm_tensor(
            _wrap_as_dense_tensor(jnp.asarray(A)),
            16,
            max_iter=500,
            conv_tol=1e-12,
            return_meta=True,
        )[0]
    )
    C = correlators(
        env,
        closed_double_layer(A, A),
        closed_double_layer(B, A),
        closed_double_layer(A, B),
        closed_double_layer(B, B),
        1,
    )
    assert C[(1, 0)].real == pytest.approx(onsager_c10(beta), abs=1e-10)
```

Add to `_FILE_MARKERS` in `tests/conftest.py`, directly before `"test_tensor.py": "core",`:

```python
    # compute_spectrum: raw-array contractions and second derivatives at D=2,
    # chi=10, on a few columns (about 4 min for both files on 12 cores).
    "test_ipeps_spectrum_contractions.py": "core",
    "test_ipeps_spectrum.py": "core",
    # Full D=2 spectra; each test carries an explicit ``slow``.
    "test_ipeps_spectrum_full.py": "algorithm",
```

- [ ] **Step 5: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_spectrum_contractions.py -q --no-cov`
Expected: collection error, `ModuleNotFoundError: No module named 'tenax.algorithms._ipeps_excitation_ctm'`.

- [ ] **Step 6: Write the module**

Create `src/tenax/algorithms/_ipeps_excitation_ctm.py`:

```python
"""Independent ket and bra iPEPS on a 2x2 cell, for H_k and N_k at k = (a pi, b pi).

Private contractions behind :func:`tenax.algorithms.ipeps_excitations.compute_spectrum`.
Every closed network here must stay invariant under random gauges on the
environment bonds; see ``tests/test_ipeps_spectrum.py``.  A reversed T4/C4 ring
matches the energy to 10 digits on a C4v state, so a symmetric check is not enough.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_tensor_multisite,
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
from tenax.core.tensor import DenseTensor

CELL = [(x, y) for x in range(2) for y in range(2)]
NEIGHBORS = {
    (x, y): {
        "left": ((x - 1) % 2, y),
        "right": ((x + 1) % 2, y),
        "top": (x, (y - 1) % 2),
        "bottom": (x, (y + 1) % 2),
    }
    for x, y in CELL
}


def cell_phases(a: int, b: int) -> dict:
    """Phases ``(-1)^(a x + b y)`` of the momentum ``(a pi, b pi)`` on the cell."""
    return {c: (-1.0) ** (a * c[0] + b * c[1]) for c in CELL}


def _ordered(t, labels):
    have = [str(i.label) for i in t.indices]
    return jnp.transpose(t.todense(), [have.index(label) for label in labels])


def unpack_env(env, D: int) -> dict:
    """Raw arrays with each double-layer edge leg split into (ket, bra)."""

    def edge(T):
        return T.reshape(T.shape[0], D, D, T.shape[2])

    return {
        "C1": _ordered(env.C1, ["c1_d", "c1_r"]),
        "C2": _ordered(env.C2, ["c2_l", "c2_d"]),
        "C3": _ordered(env.C3, ["c3_u", "c3_l"]),
        "C4": _ordered(env.C4, ["c4_r", "c4_u"]),
        "T1": edge(_ordered(env.T1, ["t1_l", "u2", "t1_r"])),
        "T2": edge(_ordered(env.T2, ["t2_u", "r2", "t2_d"])),
        "T3": edge(_ordered(env.T3, ["t3_r", "d2", "t3_l"])),
        "T4": edge(_ordered(env.T4, ["t4_d", "l2", "t4_u"])),
    }


def window_1x1(E, K, Kb):
    """Closed one-site window: Z11."""
    return jnp.einsum(
        E["C1"], [0, 1],
        E["T1"], [1, 2, 3, 4],
        E["C2"], [4, 5],
        E["T4"], [0, 7, 8, 6],
        K, [2, 9, 7, 10, 11],
        Kb, [3, 12, 8, 13, 11],
        E["T2"], [5, 10, 13, 14],
        E["C4"], [6, 15],
        E["T3"], [15, 9, 12, 16],
        E["C3"], [14, 16],
        [],
        optimize="greedy",
    )  # fmt: skip


def mixed_rdm_h(EL, ER, K1, Kb1, K2, Kb2):
    """Horizontal bond, K1 left and K2 right: ``(s1, s2, s1_bra, s2_bra)``."""
    L = jnp.einsum(
        EL["C1"], [0, 1],
        EL["T1"], [1, 2, 3, 4],
        EL["T4"], [0, 6, 7, 5],
        EL["C4"], [5, 8],
        EL["T3"], [8, 9, 10, 11],
        K1, [2, 9, 6, 12, 13],
        Kb1, [3, 10, 7, 14, 15],
        [4, 12, 14, 11, 13, 15],
        optimize="greedy",
    )  # fmt: skip
    R = jnp.einsum(
        ER["C2"], [0, 1],
        ER["T1"], [2, 3, 4, 0],
        ER["T2"], [1, 5, 6, 7],
        ER["C3"], [7, 8],
        ER["T3"], [9, 10, 11, 8],
        K2, [3, 10, 12, 5, 13],
        Kb2, [4, 11, 14, 6, 15],
        [2, 12, 14, 9, 13, 15],
        optimize="greedy",
    )  # fmt: skip
    return jnp.einsum(L, [0, 1, 2, 3, 4, 5], R, [0, 1, 2, 3, 6, 7], [4, 6, 5, 7])


def mixed_rdm_v(ET, EB, K1, Kb1, K2, Kb2):
    """Vertical bond, K1 top and K2 bottom: ``(s1, s2, s1_bra, s2_bra)``."""
    U = jnp.einsum(
        ET["C1"], [0, 1],
        ET["T1"], [1, 2, 3, 4],
        ET["C2"], [4, 5],
        ET["T4"], [0, 7, 8, 6],
        ET["T2"], [5, 9, 10, 11],
        K1, [2, 12, 7, 9, 13],
        Kb1, [3, 14, 8, 10, 15],
        [6, 12, 14, 11, 13, 15],
        optimize="greedy",
    )  # fmt: skip
    D_ = jnp.einsum(
        EB["T4"], [0, 1, 2, 3],
        EB["C4"], [3, 4],
        EB["T3"], [4, 5, 6, 7],
        EB["C3"], [8, 7],
        EB["T2"], [9, 10, 11, 8],
        K2, [12, 5, 1, 10, 13],
        Kb2, [14, 6, 2, 11, 15],
        [0, 12, 14, 9, 13, 15],
        optimize="greedy",
    )  # fmt: skip
    return jnp.einsum(U, [0, 1, 2, 3, 4, 5], D_, [0, 1, 2, 3, 6, 7], [4, 6, 5, 7])


def z00(c1, c2, c3, c4):
    return jnp.einsum(
        c1, [0, 1], c2, [1, 2], c3, [2, 3], c4, [0, 3], [], optimize="greedy"
    )


def z10(top, bot):
    """``C1 T1 C2`` of the row below the cut, closed with ``C3 T3 C4`` of the row above."""
    return jnp.einsum(
        top["C1"], [0, 1],
        top["T1"], [1, 2, 3, 4],
        top["C2"], [4, 5],
        bot["C3"], [5, 6],
        bot["T3"], [7, 2, 3, 6],
        bot["C4"], [0, 7],
        [],
        optimize="greedy",
    )  # fmt: skip


def z01(left, right):
    """``C1 T4 C4`` of the column right of the cut, closed with ``C2 T2 C3`` of the column left."""
    return jnp.einsum(
        left["C1"], [0, 1],
        right["C2"], [1, 2],
        right["T2"], [2, 3, 4, 5],
        right["C3"], [5, 6],
        left["C4"], [7, 6],
        left["T4"], [0, 3, 4, 7],
        [],
        optimize="greedy",
    )  # fmt: skip


class ExcitationCTM:
    """Energy and log-norm densities of a ket and bra that differ by cell phases."""

    def __init__(self, A, gate, chi, sweeps):
        self.A = jnp.asarray(A)
        self.D, self.d = self.A.shape[0], self.A.shape[-1]
        self.gate = jnp.asarray(gate).reshape((self.d,) * 4)
        self.chi, self.sweeps = chi, sweeps
        site = _wrap_as_dense_tensor(self.A)
        self._template = _build_double_layer_tensor(site).indices
        envs, self.ctm_converged = _converged_env({c: site for c in CELL}, chi)
        self.envs0 = jax.tree_util.tree_map(jax.lax.stop_gradient, envs)
        self._sweep = jax.jit(
            lambda e, dl: _ctm_tensor_sweep_multisite(
                e, dl, NEIGHBORS, chi, True, "svd", "flow", "2x2"
            )[0]
        )

    def _double_layer(self, K, Kb):
        D2 = self.D**2
        data = jnp.einsum("udlrs,UDLRs->uUdDlLrR", K, Kb).reshape(D2, D2, D2, D2)
        return DenseTensor(data, self._template)

    def environments(self, kets, bras):
        dl = {c: self._double_layer(kets[c], bras[c]) for c in CELL}
        e = self.envs0
        for _ in range(self.sweeps):
            e = self._sweep(e, dl)
        return {c: unpack_env(e[c], self.D) for c in CELL}

    def energy_density(self, E, kets, bras):
        total = 0.0
        for c in CELL:
            for direction, fn in (("right", mixed_rdm_h), ("bottom", mixed_rdm_v)):
                n = NEIGHBORS[c][direction]
                rho = fn(E[c], E[n], kets[c], bras[c], kets[n], bras[n])
                total = total + _bond_value(rho, self.gate)
        return total / len(CELL)

    def log_norm_density(self, E, kets, bras):
        total = 0.0
        for c in CELL:
            cx, cy = NEIGHBORS[c]["right"], NEIGHBORS[c]["bottom"]
            cxy = NEIGHBORS[cx]["bottom"]
            total = (
                total
                + jnp.log(jnp.abs(window_1x1(E[c], kets[c], bras[c])))
                + jnp.log(
                    jnp.abs(z00(E[cxy]["C1"], E[cy]["C2"], E[c]["C3"], E[cx]["C4"]))
                )
                - jnp.log(jnp.abs(z10(E[cy], E[c])))
                - jnp.log(jnp.abs(z01(E[cx], E[c])))
            )
        return total / len(CELL)

    def densities(self, phases, eps, X, Y):
        """``(energy density, log-norm density)`` at ket ``A + eps phi X``, bra ``A + phi Y``."""
        kets = {c: self.A + eps * phases[c] * X for c in CELL}
        bras = {c: self.A + phases[c] * Y for c in CELL}
        E = self.environments(kets, bras)
        return self.energy_density(E, kets, bras), self.log_norm_density(E, kets, bras)

    def ket_derivative(self, phases, X, h):
        """``d/deps d/dY`` of both densities at the base point: central difference on the ket."""
        zero = jnp.zeros_like(self.A)
        grads = []
        for sign in (1.0, -1.0):
            _, pull = jax.vjp(lambda Y: self.densities(phases, sign * h, X, Y), zero)
            grads.append(
                (
                    pull((jnp.ones(()), jnp.zeros(())))[0],
                    pull((jnp.zeros(()), jnp.ones(())))[0],
                )
            )
        scale = 2 * h
        return (grads[0][0] - grads[1][0]) / scale, (grads[0][1] - grads[1][1]) / scale


def _bond_value(rho, gate):
    return jnp.einsum("ijkl,klij->", rho, gate) / jnp.einsum("ijij->", rho)


def _converged_env(sites, chi):
    envs, info, _ = _ctm_tensor_multisite(
        sites, NEIGHBORS, chi, max_iter=600, conv_tol=1e-11,
        _return_status=True, _warn_unconverged=False,
    )  # fmt: skip
    return envs, bool(info.converged)
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_spectrum_contractions.py -q --no-cov > /tmp/spectrum_contractions.log 2>&1; grep -E "passed|failed" /tmp/spectrum_contractions.log`
Expected: `8 passed`, about 2 min. A failure here is a finding. Investigate it. Do not loosen a tolerance to pass.

- [ ] **Step 8: Commit**

```bash
git add src/tenax/algorithms/_ipeps_excitation_ctm.py tests/_ipeps_spectrum_fixtures.py tests/_window_oracle.py tests/test_ipeps_spectrum_contractions.py tests/conftest.py
git commit -m "$(cat <<'EOF'
feat(excitations): mixed contractions and second derivative on a 2x2 cell

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
```

### Task 2: `compute_spectrum`, `spectral_function`, checkpoints and exports

**Files:**
- Modify: `src/tenax/algorithms/ipeps_excitations.py` (imports at lines 18-26, append at the end)
- Modify: `src/tenax/__init__.py` (lazy map at line 358, `__all__` at line 615)
- Modify: `src/tenax/algorithms/__init__.py` (lazy map at line 99, `__all__` at line 250)
- Create: `tests/test_ipeps_spectrum.py`

**Interfaces:**
- Consumes from Task 1: `ExcitationCTM`, `cell_phases`, `_converged_env`, and the fixtures. Existing: `_as_dense_array`.
- Produces:
  - `SpectrumConfig(chi, ctm_sweeps=30, fd_step=1e-4, null_tol=1e-4, checkpoint_dir=None)`
  - `SpectrumResult` with `omega, weights, structure_factor, rank, energy_density, n_asymmetry, h_asymmetry, n_min_eig, n_negative_modes, sum_rule_deficit, gauge_residual, null_dim`
  - `compute_spectrum(A, gate, momentum, operators, config, *, basis=None) -> SpectrumResult`
  - `spectral_function(result, name, omega, eta=0.1) -> np.ndarray`
  - Private, used by tests: `_gauge_vectors(A, a, b)`, `_spectrum_momentum(momentum) -> (a, b)`, `_warn_spectrum(result)`, `_spectrum_fingerprint(A, gate, ab, config, basis)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ipeps_spectrum.py`:

```python
"""compute_spectrum and spectral_function: results, errors, warnings, checkpoints."""

import warnings

import numpy as np
import pytest
from _ipeps_spectrum_fixtures import (
    SY_COMPLEX,
    SY_REAL,
    SZ,
    apply_op,
    hasik_d2,
    rotated_gate,
)

import tenax.algorithms._ipeps_excitation_ctm as ctm_module
from tenax import SpectrumConfig, SpectrumResult, compute_spectrum, spectral_function
from tenax.algorithms._ipeps_excitation_ctm import ExcitationCTM
from tenax.algorithms.ipeps_excitations import (
    _gauge_vectors,
    _spectrum_momentum,
    _warn_spectrum,
)
from tenax.core.tensor import SymmetricTensor

X_POINT = (np.pi, 0.0)
M_POINT = (np.pi, np.pi)


def _basis(*vectors):
    """Orthonormal columns: the given vectors and one random direction."""
    A = hasik_d2()
    rng = np.random.default_rng(11)
    stack = [v.ravel() for v in vectors] + [rng.normal(size=A.size)]
    Q, _ = np.linalg.qr(np.stack(stack, axis=1))
    return Q


@pytest.fixture(scope="module")
def basis():
    A = hasik_d2()
    return _basis(apply_op(A, SY_REAL), A)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return tmp_path_factory.mktemp("spectrum")


def _config(checkpoint, name, **kwargs):
    return SpectrumConfig(chi=10, checkpoint_dir=checkpoint / name, **kwargs)


def _run(checkpoint, basis, momentum=X_POINT, operators=None, name="x", **kwargs):
    operators = {"Sy": SY_REAL} if operators is None else operators
    return compute_spectrum(
        hasik_d2(),
        rotated_gate(),
        momentum,
        operators,
        _config(checkpoint, name, **kwargs),
        basis=basis,
    )


class _Stop(Exception):
    pass


@pytest.fixture
def no_ctm(monkeypatch):
    """Fail if a call gets as far as the expensive part."""

    def refuse(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(ctm_module, "_converged_env", refuse)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("momentum", "name", "expected"),
    [(X_POINT, "x", 0.203989), (M_POINT, "m", 0.808568)],
    ids=["X", "M"],
)
def test_structure_factor_and_sum_rule(checkpoint, basis, momentum, name, expected):
    result = _run(checkpoint, basis, momentum, name=name)
    assert result.structure_factor["Sy"] == pytest.approx(expected, abs=2e-5)
    assert result.weights["Sy"].sum() == pytest.approx(expected, abs=2e-5)
    assert abs(result.sum_rule_deficit["Sy"]) < 1e-6
    assert result.gauge_residual is None


def test_complex_operator_equals_its_real_form(checkpoint, basis):
    result = _run(checkpoint, basis, operators={"re": SY_REAL, "im": SY_COMPLEX})
    np.testing.assert_allclose(result.weights["im"], result.weights["re"], atol=1e-12)
    assert result.structure_factor["im"] == pytest.approx(
        result.structure_factor["re"], abs=1e-12
    )


def test_identity_operator_has_no_weight_and_no_warning(checkpoint, basis):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = _run(checkpoint, basis, operators={"id": np.eye(2)})
    assert abs(result.structure_factor["id"]) < 1e-6
    assert result.sum_rule_deficit["id"] == 0.0
    assert np.isfinite(result.weights["id"]).all()


def test_operator_outside_the_basis_warns(checkpoint, basis):
    with pytest.warns(UserWarning, match="outside basis"):
        _run(checkpoint, basis, operators={"Sz": SZ})


def test_scaling_A_changes_nothing(checkpoint, basis):
    plain = _run(checkpoint, basis)
    scaled = compute_spectrum(
        3.0 * hasik_d2(),
        rotated_gate(),
        X_POINT,
        {"Sy": SY_REAL},
        _config(checkpoint, "x"),
        basis=basis,
    )
    np.testing.assert_allclose(scaled.omega, plain.omega, atol=1e-12)
    np.testing.assert_allclose(scaled.weights["Sy"], plain.weights["Sy"], atol=1e-12)


def test_basis_inside_the_null_space_keeps_nothing():
    A = hasik_d2()
    with pytest.warns(UserWarning, match="no direction survives"):
        result = compute_spectrum(
            A,
            rotated_gate(),
            X_POINT,
            {"id": np.eye(2)},
            SpectrumConfig(chi=10),
            basis=A.reshape(-1, 1),
        )
    assert result.rank == 0
    assert result.omega.shape == (0,)
    assert spectral_function(result, "id", np.linspace(0, 1, 3)).tolist() == [
        0.0,
        0.0,
        0.0,
    ]


@pytest.mark.parametrize(("a", "b"), [(1, 0), (0, 1), (1, 1)])
def test_gauge_span_has_the_measured_rank(a, b):
    A = hasik_d2()
    assert np.linalg.matrix_rank(_gauge_vectors(A, a, b), tol=1e-10) == 2 * 2**2 - 1


def test_a_and_gauge_vectors_are_null_in_n():
    # Random combination: a symmetric direction alone hides a wrong contraction.
    A = hasik_d2()
    rng = np.random.default_rng(5)
    ctm = ExcitationCTM(A, rotated_gate(), 10, 30)
    phases = ctm_module.cell_phases(1, 0)
    vectors = _gauge_vectors(A, 1, 0)
    combined = vectors[:, 1:] @ rng.normal(size=vectors.shape[1] - 1)
    for v in (A.ravel(), combined / np.linalg.norm(combined)):
        _, g_n = ctm.ket_derivative(phases, np.reshape(v, A.shape), 1e-4)
        assert float(np.linalg.norm(np.asarray(g_n))) < 1e-3


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("momentum", "expected"),
    [
        ((np.pi, 0.0), (1, 0)),
        ((0.0, np.pi), (0, 1)),
        ((np.pi, np.pi), (1, 1)),
        ((-np.pi, 0.0), (1, 0)),
        ((3 * np.pi, np.pi), (1, 1)),
        ([np.pi, 2 * np.pi], (1, 0)),
    ],
)
def test_momentum_is_taken_modulo_two_pi(momentum, expected):
    assert _spectrum_momentum(momentum) == expected


@pytest.mark.parametrize(
    "momentum",
    [(0.0, 0.0), (2 * np.pi, 0.0), (np.pi / 2, np.pi / 2), (np.pi,), (np.pi, 0.0, 0.0)],
)
def test_momentum_outside_the_three_names_them(momentum):
    with pytest.raises(ValueError, match=r"\(pi, 0\), \(0, pi\), \(pi, pi\)"):
        _spectrum_momentum(momentum)


# ---------------------------------------------------------------------------
# Errors: each raises before any CTM work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"A": 1j * hasik_d2()}, ValueError, "real"),
        ({"A": hasik_d2().astype(np.float32)}, ValueError, "float64"),
        ({"A": hasik_d2()[0]}, ValueError, "shape"),
        ({"gate": np.eye(3)}, ValueError, "gate"),
        ({"operators": {"bad": np.eye(3)}}, ValueError, "bad"),
        ({"basis": np.ones((5, 2))}, ValueError, "shape"),
        ({"basis": np.ones((32, 2))}, ValueError, "orthonormal"),
        ({"momentum": (0.0, 0.0)}, ValueError, "Gamma"),
    ],
)
def test_bad_input_raises_before_ctm_work(no_ctm, change, error, message):
    arguments = {
        "A": hasik_d2(),
        "gate": rotated_gate(),
        "momentum": X_POINT,
        "operators": {"Sy": SY_REAL},
        "basis": None,
    } | change
    basis = arguments.pop("basis")
    with pytest.raises(error, match=message):
        compute_spectrum(
            arguments["A"],
            arguments["gate"],
            arguments["momentum"],
            arguments["operators"],
            SpectrumConfig(chi=10),
            basis=basis,
        )


def test_symmetric_tensor_is_refused(no_ctm):
    A = object.__new__(SymmetricTensor)  # the refusal must not touch its contents
    with pytest.raises(NotImplementedError, match="SymmetricTensor"):
        compute_spectrum(A, rotated_gate(), X_POINT, {}, SpectrumConfig(chi=10))


def test_many_columns_without_checkpoints_warns(no_ctm):
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3, 3, 3, 2))
    with pytest.warns(UserWarning, match="162 columns"), pytest.raises(_Stop):
        compute_spectrum(A, rotated_gate(), X_POINT, {}, SpectrumConfig(chi=10))


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------


def _clean_result(**changes):
    fields = {
        "omega": np.array([1.0]),
        "weights": {"Sy": np.array([0.2])},
        "structure_factor": {"Sy": 0.2},
        "rank": 1,
        "energy_density": -0.66,
        "n_asymmetry": 0.0,
        "h_asymmetry": 0.0,
        "n_min_eig": 0.0,
        "n_negative_modes": 0,
        "sum_rule_deficit": {"Sy": 0.0},
        "gauge_residual": 0.0,
        "null_dim": 7,
    }
    return SpectrumResult(**(fields | changes))


def test_clean_result_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_spectrum(_clean_result())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"rank": 0}, "no direction survives"),
        ({"n_asymmetry": 2e-4}, "N asymmetry"),
        ({"h_asymmetry": 2e-3}, "H asymmetry"),
        ({"n_min_eig": -2e-4}, "lowest N eigenvalue"),
        ({"n_negative_modes": 2}, "raise chi"),
        ({"sum_rule_deficit": {"Sy": 2e-3}}, "sum rule deficit"),
        ({"sum_rule_deficit": {"Sy": -2e-3}}, "sum rule deficit"),
        ({"gauge_residual": 2e-3}, "gauge residual"),
    ],
)
def test_each_diagnostic_warns_above_its_threshold(change, message):
    with pytest.warns(UserWarning, match=message):
        _warn_spectrum(_clean_result(**change))


def test_unconverged_ctm_warns(checkpoint, basis, monkeypatch):
    real = ctm_module._converged_env
    monkeypatch.setattr(ctm_module, "_converged_env", lambda *a: (real(*a)[0], False))
    with pytest.warns(UserWarning, match="did not converge"):
        _run(checkpoint, basis)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_finished_run_resumes_without_derivatives(checkpoint, basis, monkeypatch):
    first = _run(checkpoint, basis)

    def refuse(*args, **kwargs):
        raise AssertionError("resumed run recomputed a column")

    monkeypatch.setattr(ExcitationCTM, "ket_derivative", refuse)
    second = _run(checkpoint, basis)
    np.testing.assert_array_equal(second.omega, first.omega)


def test_interrupted_run_equals_an_uninterrupted_run(
    tmp_path, checkpoint, basis, monkeypatch
):
    uninterrupted = _run(checkpoint, basis)
    real = ExcitationCTM.ket_derivative
    calls = []

    def interrupt(self, *args):
        if len(calls) == 1:
            raise KeyboardInterrupt
        calls.append(1)
        return real(self, *args)

    monkeypatch.setattr(ExcitationCTM, "ket_derivative", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, basis)
    assert len(list((tmp_path / "x").glob("column_*.npz"))) == 1
    (tmp_path / "x" / "leftover.tmp").write_bytes(b"partial")

    monkeypatch.setattr(ExcitationCTM, "ket_derivative", real)
    resumed = _run(tmp_path, basis)
    np.testing.assert_allclose(resumed.omega, uninterrupted.omega, atol=1e-10)


def test_checkpoint_keeps_columns_when_operators_change(checkpoint, basis, monkeypatch):
    _run(checkpoint, basis)

    def refuse(*args, **kwargs):
        raise AssertionError("changing operators must not recompute a column")

    monkeypatch.setattr(ExcitationCTM, "ket_derivative", refuse)
    result = _run(checkpoint, basis, operators={"id": np.eye(2)})
    assert "id" in result.weights


def test_checkpoint_from_another_run_raises_before_ctm_work(
    checkpoint, basis, monkeypatch
):
    _run(checkpoint, basis)

    def refuse(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(ctm_module, "_converged_env", refuse)
    with pytest.raises(ValueError, match="different run"):
        _run(checkpoint, basis, fd_step=2e-4)


def test_damaged_checkpoint_names_the_file(tmp_path, basis):
    directory = tmp_path / "x"
    directory.mkdir()
    (directory / "column_00000.npz").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="column_00000.npz"):
        _run(tmp_path, basis)


# ---------------------------------------------------------------------------
# spectral_function
# ---------------------------------------------------------------------------


def test_spectral_function_is_a_lorentzian_sum():
    result = _clean_result(
        omega=np.array([1.0, 2.0]), weights={"Sy": np.array([0.2, 0.1])}
    )
    assert spectral_function(result, "Sy", 1.0, eta=0.5) == pytest.approx(
        0.2 / (0.5 * np.pi) + 0.1 * (0.5 / np.pi) / 1.25
    )
    grid = np.linspace(-2000.0, 2000.0, 800001)
    S = spectral_function(result, "Sy", grid, eta=0.1)
    assert S.shape == grid.shape
    assert np.trapezoid(S, grid) == pytest.approx(0.3, abs=1e-3)


def test_spectral_function_refuses_bad_arguments():
    result = _clean_result()
    with pytest.raises(KeyError, match="Sy"):
        spectral_function(result, "Sq", np.zeros(1))
    with pytest.raises(ValueError, match="eta"):
        spectral_function(result, "Sy", np.zeros(1), eta=0.0)


def test_negative_modes_stay_in_the_spectral_function():
    result = _clean_result(omega=np.array([-1.0]), n_negative_modes=1)
    assert spectral_function(result, "Sy", -1.0, eta=0.1) > 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_spectrum.py -q --no-cov`
Expected: collection error, `ImportError: cannot import name 'SpectrumConfig' from 'tenax'`.

- [ ] **Step 3: Add the imports**

In `src/tenax/algorithms/ipeps_excitations.py`, replace

```python
from dataclasses import dataclass

import jax
```

with

```python
import hashlib
import os
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import jax
```

and add this line directly after `from tenax.algorithms._einsum_compat import einsum_promoted`:

```python
from tenax.algorithms._ipeps_excitation_ctm import ExcitationCTM, cell_phases
```

- [ ] **Step 4: Append the implementation**

Append to the end of `src/tenax/algorithms/ipeps_excitations.py`:

```python
# ---------------------------------------------------------------------------
# Excitation spectrum and spectral function (2x2 cell, summed momentum)
# ---------------------------------------------------------------------------


@dataclass
class SpectrumConfig:
    """Configuration for :func:`compute_spectrum`.

    Attributes:
        chi: CTM bond dimension. One value cannot show convergence: compare at
            least two.
        ctm_sweeps: CTM sweeps differentiated, from the converged environment.
        fd_step: Central-difference step on the ket direction.
        null_tol: Eigenvectors of N below ``null_tol`` times the largest
            eigenvalue are dropped.
        checkpoint_dir: Write each finished column here, and resume from it.
    """

    chi: int
    ctm_sweeps: int = 30
    fd_step: float = 1e-4
    null_tol: float = 1e-4
    checkpoint_dir: Path | None = None


@dataclass
class SpectrumResult:
    """Result of :func:`compute_spectrum`.

    Attributes:
        omega: Excitation energies above the ground state, ascending.
        weights: Per operator, ``|<lambda|O|0>|^2`` for each entry of ``omega``.
        structure_factor: Per operator, the static ``s(k)``.
        rank: Size of the subspace kept from N.
        energy_density: Energy per site at the base point.
        n_asymmetry: Largest ``|N - N^T|``.
        h_asymmetry: Largest ``|H - H^T|``.
        n_min_eig: Lowest eigenvalue of N over the largest.
        n_negative_modes: Number of ``omega < 0``. These stay in ``omega``.
        sum_rule_deficit: Per operator, ``1 - sum(weights) / s(k)``.
        gauge_residual: Largest singular value of N on ``A`` and the gauge
            vectors. ``None`` when ``basis`` is given.
        null_dim: Number of eigenvalues of N below the cut.
    """

    omega: np.ndarray
    weights: dict[str, np.ndarray]
    structure_factor: dict[str, float]
    rank: int
    energy_density: float
    n_asymmetry: float
    h_asymmetry: float
    n_min_eig: float
    n_negative_modes: int
    sum_rule_deficit: dict[str, float]
    gauge_residual: float | None
    null_dim: int


def _spectrum_momentum(momentum) -> tuple[int, int]:
    """Map a momentum to ``(a, b)`` with ``k = (a pi, b pi)``, modulo 2 pi."""
    allowed = "(pi, 0), (0, pi), (pi, pi)"
    k = np.asarray(momentum, dtype=float)
    units = k / np.pi
    if k.shape != (2,) or np.abs(units - np.round(units)).max() > 1e-8:
        raise ValueError(f"momentum must be one of {allowed}; got {momentum!r}")
    a, b = (int(x) % 2 for x in np.round(units))
    if (a, b) == (0, 0):
        raise ValueError(
            f"momentum must be one of {allowed}; got {momentum!r}. Gamma is "
            "refused: the method needs k != 0. In the sublattice-rotated frame "
            "physical Gamma is (pi, pi)."
        )
    return a, b


def _spectrum_array(A) -> np.ndarray:
    if isinstance(A, SymmetricTensor):
        raise NotImplementedError(
            "compute_spectrum does not support SymmetricTensor: use a dense A, "
            "and `basis` to restrict to a charge sector."
        )
    A = np.asarray(_as_dense_array(A))
    if np.iscomplexobj(A):
        raise ValueError("compute_spectrum needs a real A; got complex.")
    if A.dtype != np.float64:
        raise ValueError(f"compute_spectrum needs float64 A; got {A.dtype}.")
    if A.ndim != 5 or len(set(A.shape[:4])) != 1:
        raise ValueError(f"A must have shape (D, D, D, D, d); got {A.shape}.")
    return A / np.linalg.norm(A)


def _spectrum_basis(basis, size: int) -> np.ndarray:
    if basis is None:
        return np.eye(size)
    basis = np.asarray(basis)
    if np.iscomplexobj(basis):
        raise ValueError("basis must be real.")
    if basis.ndim != 2 or basis.shape[0] != size or basis.shape[1] > size:
        raise ValueError(
            f"basis must have shape ({size}, n) with n <= {size}; got {basis.shape}."
        )
    if np.abs(basis.T @ basis - np.eye(basis.shape[1])).max() > 1e-10:
        raise ValueError("basis columns must be orthonormal.")
    return basis


def _spectrum_gate(gate, d: int, name: str) -> np.ndarray:
    gate = np.asarray(_as_dense_array(gate))
    if gate.size != d**4 or gate.ndim not in (2, 4):
        raise ValueError(
            f"{name} must have shape ({d}, {d}, {d}, {d}) or ({d**2}, {d**2}); "
            f"got {gate.shape}."
        )
    return gate.reshape((d,) * 4)


def _spectrum_operators(operators, d: int) -> dict[str, np.ndarray]:
    checked = {}
    for name, op in operators.items():
        op = np.asarray(op)
        if op.shape != (d, d):
            raise ValueError(f"operator {name!r} must be {d}x{d}; got {op.shape}.")
        checked[name] = op
    return checked


def _gauge_vectors(A: np.ndarray, a: int, b: int) -> np.ndarray:
    """``A`` and the bond-gauge directions at ``k = (a pi, b pi)``, as columns."""
    D = A.shape[0]
    px, py = (-1.0) ** a, (-1.0) ** b
    vectors = [A.ravel()]
    for p in range(D):
        for q in range(D):
            X = np.zeros((D, D))
            X[p, q] = 1.0
            vectors.append(
                (
                    np.einsum("udlrs,rq->udlqs", A, X)
                    - px * np.einsum("pl,udlrs->udprs", X, A)
                ).ravel()
            )
            vectors.append(
                (
                    np.einsum("udlrs,dq->uqlrs", A, X)
                    - py * np.einsum("pu,udlrs->pdlrs", X, A)
                ).ravel()
            )
    return np.stack(vectors, axis=1)


def _gauge_residual(N: np.ndarray, A: np.ndarray, a: int, b: int) -> float:
    U, s, _ = np.linalg.svd(_gauge_vectors(A, a, b), full_matrices=False)
    Q = U[:, s > 1e-10 * s[0]]
    return float(np.linalg.norm(Q.T @ N @ Q, 2))


def _spectrum_fingerprint(A, gate, ab, config, basis) -> str:
    digest = hashlib.sha256()
    for array in (A, gate, basis):
        digest.update(str(array.shape).encode())
        # Rounded: rescaling A before the internal normalisation must still resume.
        digest.update(np.round(np.asarray(array, dtype=np.float64), 12).tobytes())
    digest.update(repr((ab, config.chi, config.ctm_sweeps, config.fd_step)).encode())
    return digest.hexdigest()


def _load_columns(directory: Path | None, fingerprint: str) -> dict[int, tuple]:
    """Finished columns in ``directory``. A mismatch or a damaged file raises."""
    columns: dict[int, tuple] = {}
    if directory is None or not directory.exists():
        return columns
    for path in sorted(directory.glob("column_*.npz")):
        try:
            with np.load(path) as data:
                found = str(data["fingerprint"])
                column = (data["h"], data["n"])
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise ValueError(f"Checkpoint {path} is damaged: {exc}") from exc
        if found != fingerprint:
            raise ValueError(
                f"Checkpoint {path} is from a different run: A, the gates, the "
                "momentum, chi, ctm_sweeps, fd_step or basis changed. Use a new "
                "checkpoint_dir."
            )
        columns[int(path.stem.removeprefix("column_"))] = column
    return columns


def _save_column(directory: Path, index: int, fingerprint: str, h, n) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    with os.fdopen(handle, "wb") as stream:
        np.savez(stream, fingerprint=fingerprint, h=h, n=n)
    os.replace(temporary, directory / f"column_{index:05d}.npz")


def _reduce_spectrum(N, H, null_tol):
    """Solve in the kept subspace: ``(omega, modes, cut, null_dim, n_min_eig)``."""
    w, V = np.linalg.eigh(N)
    # A floor, so a basis inside the null space keeps nothing instead of noise.
    cut = null_tol * max(w[-1], 1e-2)
    keep = w > cut
    U = V[:, keep] / np.sqrt(w[keep])
    reduced = U.T @ H @ U
    omega, c = np.linalg.eigh(0.5 * (reduced + reduced.T))
    null_dim = int((np.abs(w) <= cut).sum())
    return omega, U @ c, cut, null_dim, float(w[0] / max(w[-1], 1e-2))


def compute_spectrum(
    A,
    gate,
    momentum,
    operators,
    config: SpectrumConfig,
    *,
    basis=None,
) -> SpectrumResult:
    """Excitation spectrum and spectral weights of a one-site iPEPS on a 2x2 cell.

    Solves ``(H_k - E0 N_k) v = omega N_k v``, with both matrices as mixed second
    derivatives of the energy and log-norm densities through the CTM. The
    momentum sum is complete, unlike :func:`compute_excitations`.

    ``A``, the gates, ``operators`` and ``momentum`` must share a frame. Failed
    diagnostics warn: see :class:`SpectrumResult`.

    Args:
        A: Real site tensor, ``(u, d, l, r, s)``. It is normalised internally.
        gate: Nearest-neighbour gate, used on the horizontal and vertical bonds.
        momentum: One of ``(pi, 0)``, ``(0, pi)``, ``(pi, pi)``, modulo ``2 pi``.
        operators: Maps a name to a ``d x d`` matrix, for the spectral weights.
        config: See :class:`SpectrumConfig`.
        basis: Optional orthonormal columns that restrict the tangent vectors.

    Returns:
        :class:`SpectrumResult`.
    """
    a, b = _spectrum_momentum(momentum)
    A = _spectrum_array(A)
    d = A.shape[-1]
    gate = _spectrum_gate(gate, d, "gate")
    operators = _spectrum_operators(operators, d)
    full_space = basis is None
    basis = _spectrum_basis(basis, A.size)
    n_columns = basis.shape[1]
    directory = None if config.checkpoint_dir is None else Path(config.checkpoint_dir)
    fingerprint = _spectrum_fingerprint(A, gate, (a, b), config, basis)
    columns = _load_columns(directory, fingerprint)
    if directory is None and n_columns > 100:
        warnings.warn(
            f"compute_spectrum: {n_columns} columns and no checkpoint_dir. A "
            "killed run loses them all.",
            stacklevel=2,
        )

    ctm = ExcitationCTM(A, gate, config.chi, config.ctm_sweeps)
    if not ctm.ctm_converged:
        warnings.warn(
            f"compute_spectrum: the CTM did not converge at chi={config.chi}.",
            stacklevel=2,
        )
    phases = cell_phases(a, b)
    zero = jnp.zeros_like(ctm.A)
    energy, _ = ctm.densities(phases, 0.0, zero, zero)

    H, N = np.zeros((n_columns,) * 2), np.zeros((n_columns,) * 2)
    for i in range(n_columns):
        if i not in columns:
            X = jnp.asarray(basis[:, i].reshape(A.shape))
            g_h, g_n = ctm.ket_derivative(phases, X, config.fd_step)
            columns[i] = (
                basis.T @ np.asarray(g_h).ravel(),
                basis.T @ np.asarray(g_n).ravel(),
            )
            if directory is not None:
                _save_column(directory, i, fingerprint, *columns[i])
        H[:, i], N[:, i] = columns[i]

    n_asymmetry = float(np.abs(N - N.T).max())
    h_asymmetry = float(np.abs(H - H.T).max())
    N = 0.5 * (N + N.T)
    H = 0.5 * (H + H.T)
    omega, modes, cut, null_dim, n_min_eig = _reduce_spectrum(N, H, config.null_tol)
    rank = modes.shape[1]

    weights: dict[str, np.ndarray] = {}
    structure_factor: dict[str, float] = {}
    deficit: dict[str, float] = {}
    for name, op in operators.items():
        B = np.tensordot(A, op, axes=([4], [1])).ravel()
        parts = (B.real, B.imag) if np.iscomplexobj(B) else (B,)
        w_total, s_total = np.zeros(rank), 0.0
        for part in parts:
            c = basis.T @ part
            if np.linalg.norm(part - basis @ c) > 1e-8 * max(np.linalg.norm(B), 1e-300):
                warnings.warn(
                    f"compute_spectrum: operator {name!r} applied to A has weight "
                    "outside basis; its s(k) and weights cover the part inside.",
                    stacklevel=2,
                )
            w_total = w_total + (modes.T @ (N @ c)) ** 2
            s_total = s_total + float(c @ N @ c)
        weights[name], structure_factor[name] = w_total, s_total
        deficit[name] = 0.0 if s_total <= cut else 1.0 - float(w_total.sum()) / s_total

    gauge_residual = _gauge_residual(N, A, a, b) if full_space else None
    result = SpectrumResult(
        omega=omega,
        weights=weights,
        structure_factor=structure_factor,
        rank=rank,
        energy_density=float(energy),
        n_asymmetry=n_asymmetry,
        h_asymmetry=h_asymmetry,
        n_min_eig=n_min_eig,
        n_negative_modes=int((omega < 0).sum()),
        sum_rule_deficit=deficit,
        gauge_residual=gauge_residual,
        null_dim=null_dim,
    )
    _warn_spectrum(result)
    return result


def _warn_spectrum(result: SpectrumResult) -> None:
    problems = []
    if result.rank == 0:
        problems.append("no direction survives null_tol: basis lies in the null space")
    if result.n_asymmetry > 1e-4:
        problems.append(f"N asymmetry {result.n_asymmetry:.2e}")
    # Looser than N: one noisy reverse-mode column gives 6.6e-4 at D=2 chi=10 M,
    # with omega unchanged to 5 digits across fd_step and ctm_sweeps.
    if result.h_asymmetry > 1e-3:
        problems.append(f"H asymmetry {result.h_asymmetry:.2e}")
    if result.n_min_eig < -1e-4:
        problems.append(f"lowest N eigenvalue ratio {result.n_min_eig:.2e}")
    if result.n_negative_modes:
        problems.append(
            f"{result.n_negative_modes} negative modes, kept in omega: raise chi"
        )
    for name, value in result.sum_rule_deficit.items():
        if abs(value) > 1e-3:
            problems.append(f"sum rule deficit {value:.2e} for {name!r}")
    if result.gauge_residual is not None and result.gauge_residual > 1e-3:
        problems.append(f"gauge residual {result.gauge_residual:.2e}")
    for problem in problems:
        warnings.warn(f"compute_spectrum: {problem}.", stacklevel=3)


def spectral_function(result: SpectrumResult, name: str, omega, eta: float = 0.1):
    """Lorentzian-broadened ``S(k, omega)`` of one operator.

    Args:
        result: From :func:`compute_spectrum`.
        name: A key of ``result.weights``.
        omega: Frequency grid.
        eta: Half-width of the Lorentzian. Must be positive.

    Returns:
        ``S`` on the grid, with the shape of ``omega``. It integrates to the sum
        of the weights over all frequencies.
    """
    if name not in result.weights:
        raise KeyError(
            f"No operator {name!r}; this result has {sorted(result.weights)}."
        )
    if eta <= 0:
        raise ValueError(f"eta must be positive; got {eta}.")
    grid = np.asarray(omega, dtype=float)[..., None]
    lorentzian = (eta / np.pi) / ((grid - result.omega) ** 2 + eta**2)
    return lorentzian @ result.weights[name]
```

- [ ] **Step 5: Add the exports**

In both `src/tenax/__init__.py` and `src/tenax/algorithms/__init__.py`, add to the lazy map directly after the `"ExcitationConfig": (...)` line:

```python
    "SpectrumConfig": ("tenax.algorithms.ipeps_excitations", "SpectrumConfig"),
    "SpectrumResult": ("tenax.algorithms.ipeps_excitations", "SpectrumResult"),
    "compute_spectrum": ("tenax.algorithms.ipeps_excitations", "compute_spectrum"),
    "spectral_function": ("tenax.algorithms.ipeps_excitations", "spectral_function"),
```

and to `__all__` directly after `"compute_excitations",` (four-space indent):

```python
    "SpectrumConfig",
    "SpectrumResult",
    "compute_spectrum",
    "spectral_function",
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_spectrum.py tests/test_ipeps_spectrum_contractions.py -q --no-cov > /tmp/spectrum_core.log 2>&1; grep -E "passed|failed" /tmp/spectrum_core.log`
Expected: `58 passed`, about 4 min.

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/ipeps_excitations.py src/tenax/__init__.py src/tenax/algorithms/__init__.py tests/test_ipeps_spectrum.py
git commit -m "$(cat <<'EOF'
feat(excitations): compute_spectrum and spectral_function

Solves the generalized eigenproblem at (pi,0), (0,pi), (pi,pi) on a 2x2 cell
with the complete momentum sum, with spectral weights, diagnostics and
per-column checkpoints.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
```

### Task 3: Slow test, docs, mutation count and the PR

**Files:**
- Create: `tests/test_ipeps_spectrum_full.py`
- Modify: `src/tenax/algorithms/ipeps_excitations.py` (module docstring)
- Modify: `docs/api/algorithms.rst` (line 292), `docs/guide/algorithms/ad_excitations.md` (before `## References`, line 303), `.claude/skills/tenax-ipeps-workflow/SKILL.md` (Stage 4), `README.md` (line 39)

**Interfaces:**
- Consumes from Task 2: `compute_spectrum`, `SpectrumConfig`, `SpectrumResult`, `spectral_function`.
- Produces: open PR for branch 1, and a new issue for S(k,w).

- [ ] **Step 1: Write the slow test**

Create `tests/test_ipeps_spectrum_full.py`:

```python
"""Full D=2 spectrum: X and M, chi=10 against chi=16."""

import numpy as np
import pytest
from _ipeps_spectrum_fixtures import SX, SY_REAL, SZ, hasik_d2, rotated_gate

from tenax import SpectrumConfig, compute_spectrum

OPERATORS = {"Sy": SY_REAL, "Sx": SX, "Sz": SZ}


@pytest.mark.slow
@pytest.mark.parametrize(
    ("momentum", "omega_low", "weight_top"),
    [((np.pi, 0.0), 2.293, 0.1904), ((np.pi, np.pi), 0.6487, 0.7744)],
    ids=["X", "M"],
)
def test_full_spectrum(momentum, omega_low, weight_top):
    results = [
        compute_spectrum(
            hasik_d2(), rotated_gate(), momentum, OPERATORS, SpectrumConfig(chi=chi)
        )
        for chi in (10, 16)
    ]
    for result in results:
        assert result.omega[0] == pytest.approx(omega_low, abs=2e-3)
        assert result.weights["Sy"].max() == pytest.approx(weight_top, abs=2e-3)
        assert result.n_negative_modes == 0
        assert result.n_min_eig > -1e-4
        assert result.gauge_residual < 1e-3
        for deficit in result.sum_rule_deficit.values():
            assert abs(deficit) < 1e-3
    np.testing.assert_allclose(results[0].omega[:3], results[1].omega[:3], atol=1e-3)
```

- [ ] **Step 2: Run the slow test**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_spectrum_full.py -q --no-cov > /tmp/spectrum_full.log 2>&1; grep -E "passed|failed" /tmp/spectrum_full.log`
Expected: `2 passed`. Each `compute_spectrum` call took about 200 s at chi=10, and chi=16 costs more, so allow about 30 min.

Measured at chi=10 before this plan: X: `omega` 2.293, 2.9559, 3.4587, top S^y weight 0.1904, `gauge_residual` 4e-12. M: `omega` 0.6487, 2.7959, 2.9615, top S^y weight 0.7744, `gauge_residual` 3e-12. Rank 23 and null dimension 9 at both.

- [ ] **Step 3: Update the module docstring**

In `src/tenax/algorithms/ipeps_excitations.py`, replace

```
5. High-level ``compute_excitations()`` for the dispersion relation
"""
```

with

```
5. High-level ``compute_excitations()`` for the dispersion relation
6. ``compute_spectrum()`` and ``spectral_function()``: the complete momentum sum
   and spectral weights at (pi,0), (0,pi) and (pi,pi), following Ponsioen, Hasik
   & Corboz, PRB 108, 195111 (2023)
"""
```

- [ ] **Step 4: Update the API page**

In `docs/api/algorithms.rst`, add directly after `.. autofunction:: tenax.algorithms.ipeps_excitations.make_momentum_path`:

```rst

.. autoclass:: tenax.algorithms.ipeps_excitations.SpectrumConfig
   :members:
   :no-index:

.. autoclass:: tenax.algorithms.ipeps_excitations.SpectrumResult
   :members:
   :no-index:

.. autofunction:: tenax.algorithms.ipeps_excitations.compute_spectrum

.. autofunction:: tenax.algorithms.ipeps_excitations.spectral_function
```

- [ ] **Step 5: Add the guide section**

In `docs/guide/algorithms/ad_excitations.md`, insert directly before `## References`:

````markdown
## Spectrum and Spectral Function

`compute_spectrum` includes the complete momentum sum through the CTM, and gives
spectral weights. `compute_excitations` truncates that sum to nearest neighbours
(#1052), and its lowest energies on published Heisenberg states are negative.

### Example

```python
import numpy as np
from tenax import SpectrumConfig, compute_spectrum, spectral_function

Sy = np.array([[0.0, 0.5], [-0.5, 0.0]])  # i S^y: the real form
config = SpectrumConfig(chi=16, checkpoint_dir="runs/x_chi16")
result = compute_spectrum(A, gate, (np.pi, 0.0), {"Sy": Sy}, config)

result.omega                  # excitation energies above the ground state
result.weights["Sy"]          # |<lambda|S^y|0>|^2 for each omega
result.structure_factor["Sy"] # s(k)

grid = np.linspace(0.0, 5.0, 501)
S = spectral_function(result, "Sy", grid, eta=0.1)  # Lorentzian half-width eta
```

`A`, the gates, the operators and the momentum must share one frame. In the
sublattice-rotated frame, the physical Gamma point is `(pi, pi)`.

### Limits

- Dense real `A` only. A `SymmetricTensor` or a complex `A` raises.
- Only `(pi, 0)`, `(0, pi)` and `(pi, pi)`, because the 2x2 cell reaches only
  those. Gamma raises: the method needs k != 0.
- The cost is one column for each entry of `A`. At D=3, chi=18 one H column takes
  about 30 s on 12 CPU cores. At D=4, chi=16 it takes about 135 s, which is about
  19 h per momentum for H alone. The norm adds up to the same again.
- Use `checkpoint_dir` on any long run. A rerun resumes, and changing only
  `operators` reuses all columns.
- `basis` restricts the tangent space, for example to a charge sector. An operator
  whose `O A` has weight outside `basis` warns, and its results cover only the
  part inside.

### chi convergence

One run cannot show convergence, so compare at least two chi values. The diagnostics
do not replace this. At D=3, chi=18 is too small: it gives negative modes (-4.42 and
-0.30 at X), and their Rayleigh quotients are positive at chi=27 and about +6 at
chi=54. Fixed chi=18 eigenvectors evaluated at larger chi give:

| chi | 18 | 27 | 36 | 54 |
|---|---|---|---|---|
| omega(X) | 2.2641 | 2.2665 | 2.2676 | 2.2679 |
| omega(M) | 0.2688 | | 0.3056 | 0.3083 |

Negative modes stay in `omega`. The warning tells you to raise chi.

### Diagnostics

| Field | Meaning | Warns if |
|---|---|---|
| `energy_density` | energy per site | the CTM did not converge |
| `n_asymmetry` | largest entry of N - N^T | above 1e-4 |
| `h_asymmetry` | largest entry of H - H^T | above 1e-3 |
| `n_min_eig` | lowest N eigenvalue over the largest | below -1e-4 |
| `n_negative_modes` | modes with omega < 0 | any |
| `sum_rule_deficit` | per operator, 1 - sum(weights)/s(k) | above 1e-3 |
| `gauge_residual` | largest singular value of N on A and the gauge vectors, `basis=None` only | above 1e-3 |
| `null_dim` | eigenvalues of N below the cut | never |

The H limit is looser because of one noisy reverse-mode column: at D=2, chi=10, M it
gives 6.6e-4, while omega stays at 0.64873 across `fd_step` 1e-3, 1e-4 and 30, 60
sweeps.

### Validation

Hasik's D=2 J2=0 state (`j1j2_ipeps_states`, CC BY 4.0), chi=10:

| | X = (pi, 0) | M = (pi, pi) | Reference |
|---|---|---|---|
| Lowest omega | 2.293 | 0.6487 | X: series ~2.18, QMC ~2.16; M: 0 as D grows |
| s(k) for S^y | 0.20399 | 0.80857 | window oracle in random directions, within 1e-4 |
| Top S^y weight | 0.1904 | 0.7744 | |
| Sum rule deficit | 1e-13 | 2e-15 | |

````

- [ ] **Step 6: Update the skill**

In `.claude/skills/tenax-ipeps-workflow/SKILL.md`, insert directly before `### Interpreting results` in Stage 4:

````markdown
### Spectrum with spectral weights

`compute_excitations` truncates the momentum sum, so its low energies are not
reliable (#1052). For (pi,0), (0,pi) and (pi,pi) use `compute_spectrum`, and compare
at least two chi values:

```python
from tenax import SpectrumConfig, compute_spectrum, spectral_function

config = SpectrumConfig(chi=16, checkpoint_dir="runs/x_chi16")
result = compute_spectrum(A_opt, H_rot, (np.pi, 0.0), {"Sy": Sy_real}, config)
S = spectral_function(result, "Sy", np.linspace(0, 5, 501), eta=0.1)
```

Negative modes mean chi is too small: raise it. See the
[guide](../../../docs/guide/algorithms/ad_excitations.md) for cost and diagnostics.

````

- [ ] **Step 7: Update the README**

In `README.md`, add directly after the `**Quasiparticle excitations**` line:

```markdown
- **Spectral functions** — iPEPS spectra with weights and S(k,ω) at (π,0), (0,π), (π,π) ([guide](docs/guide/algorithms/ad_excitations.md#spectrum-and-spectral-function))
```

- [ ] **Step 8: Commit before mutating**

```bash
git add tests/test_ipeps_spectrum_full.py src/tenax/algorithms/ipeps_excitations.py docs/api/algorithms.rst docs/guide/algorithms/ad_excitations.md .claude/skills/tenax-ipeps-workflow/SKILL.md README.md
git commit -m "$(cat <<'EOF'
docs(excitations): compute_spectrum guide, API, skill and slow test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
```

- [ ] **Step 9: Run the mutation count**

Write `$CLAUDE_JOB_DIR/tmp/mutate_spectrum.py` (or any path outside the repo), and run it from the worktree. It restores each file from memory, not with `git checkout`, and turns off bytecode so that a same-size mutant is not served stale.

```python
"""Mutation count for compute_spectrum. Run from the worktree root, after committing."""

import pathlib
import subprocess
import sys

M = "src/tenax/algorithms/_ipeps_excitation_ctm.py"
P = "src/tenax/algorithms/ipeps_excitations.py"
TESTS = ["tests/test_ipeps_spectrum_contractions.py", "tests/test_ipeps_spectrum.py"]
MUTANTS = [
    ("ring reversal: T4 in window_1x1", M, 'E["T4"], [0, 7, 8, 6],', 'E["T4"], [6, 7, 8, 0],'),
    ("z10: T3 reversed", M, 'bot["T3"], [7, 2, 3, 6],', 'bot["T3"], [6, 2, 3, 7],'),
    ("z01: T4 reversed", M, 'left["T4"], [0, 3, 4, 7],', 'left["T4"], [7, 3, 4, 0],'),
    ("rdm_h: ket and bra of K2 swapped", M, "K2, [3, 10, 12, 5, 13],", "K2, [4, 10, 12, 5, 13],"),
    ("placement: the bond uses one site's env", M, "n = NEIGHBORS[c][direction]\n", "n = c\n"),
    ("log norm: Z01 not divided", M, "- jnp.log(jnp.abs(z01(E[cx], E[c])))", "+ jnp.log(jnp.abs(z01(E[cx], E[c])))"),
    ("phases: a and b swapped", M, "(a * c[0] + b * c[1])", "(b * c[0] + a * c[1])"),
    ("second derivative: h not doubled", M, "scale = 2 * h", "scale = h"),
    ("frozen projectors", M, '"svd", "flow", "2x2"', '"svd", "frozen", "2x2"'),
    ("H and N columns swapped", P, "H[:, i], N[:, i] = columns[i]", "N[:, i], H[:, i] = columns[i]"),
    ("weights not squared", P, "(modes.T @ (N @ c)) ** 2", "(modes.T @ (N @ c))"),
    ("no floor on the N cut", P, "cut = null_tol * max(w[-1], 1e-2)", "cut = null_tol * w[-1]"),
    ("A not normalised", P, "return A / np.linalg.norm(A)", "return A"),
    ("gauge: px ignored", P, "px, py = (-1.0) ** a, (-1.0) ** b", "px, py = 1.0, (-1.0) ** b"),
    ("float32 accepted", P, "if A.dtype != np.float64:", "if False:"),
    ("orthonormality unchecked", P, "if np.abs(basis.T @ basis - np.eye(basis.shape[1])).max() > 1e-10:", "if False:"),
    ("fingerprint unchecked", P, "if found != fingerprint:", "if False:"),
    ("Gamma accepted", P, "if (a, b) == (0, 0):", "if False:"),
    ("column not saved", P, "_save_column(directory, i, fingerprint, *columns[i])", "pass"),
    ("Lorentzian width", P, "(eta / np.pi) / ((grid - result.omega) ** 2 + eta**2)", "(1.0 / np.pi) / ((grid - result.omega) ** 2 + eta**2)"),
    ("deficit guard removed", P, "0.0 if s_total <= cut else", "1.0 - float(w_total.sum()) / s_total if True else"),
]

survived = []
for label, name, old, new in MUTANTS:
    path = pathlib.Path(name)
    original = path.read_text()
    assert original.count(old) == 1, f"anchor not unique for {label}: {original.count(old)}"
    try:
        path.write_text(original.replace(old, new))
        for pyc in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
            pyc.unlink()
        run = subprocess.run(
            [sys.executable, "-m", "pytest", *TESTS, "-x", "-q", "--no-cov", "-p", "no:cacheprovider"],
            capture_output=True,
            text=True,
        )
    finally:
        path.write_text(original)
    killed = run.returncode == 1
    if run.returncode not in (0, 1):
        print(run.stdout[-1500:], run.stderr[-1500:])
    print(f"{'KILLED' if killed else 'SURVIVED':8} {label} (rc={run.returncode})", flush=True)
    if not killed:
        survived.append(label)
print(f"{len(MUTANTS) - len(survived)} of {len(MUTANTS)} killed; survivors: {survived}")
```

Run: `PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu taskset -c 0-11 uv run python "$CLAUDE_JOB_DIR/tmp/mutate_spectrum.py" > "$CLAUDE_JOB_DIR/tmp/mutate_spectrum.log" 2>&1`

Allow up to about 1 h. Then run `git status --short` and confirm that no source file changed. For each survivor, decide whether it is an equivalent mutant (state why) or a test gap (add a test in the same style, commit it, and rerun that mutant). The ring reversal must be killed. `frozen projectors` may survive: flow and frozen agreed to every printed digit at chi=16 and differed at chi=4, so at chi=10 it may be close to equivalent; if so, say so instead of adding a test. Put the count and the named mutants in the PR body.

- [ ] **Step 10: Run the core marker and lint**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_spectrum.py tests/test_ipeps_spectrum_contractions.py tests/test_ipeps_excitations.py -m core -q --no-cov > /tmp/spectrum_core.log 2>&1; grep -E "passed|failed" /tmp/spectrum_core.log`
Expected: all selected tests pass, and the two new files are selected by `-m core`.

Run: `uv run pre-commit run --files $(git diff --name-only origin/main...HEAD)`
Expected: all hooks pass.

- [ ] **Step 11: Open the S(k,w) issue and the PR**

```bash
git push -u origin feat/ipeps-spectrum
gh issue create --title "iPEPS spectral function S(k,w) validated against PRB 108, 195111" --body "$(cat <<'EOF'
> 🤖 **AI-generated issue** — written by Claude Code, posted by @yingjerkao.

Tracks the workflow in docs/superpowers/specs/2026-11-06-ipeps-spectrum-design.md.

- [ ] compute_spectrum and spectral_function, J2=0
- [ ] gate_nnn
- [ ] deprecate compute_excitations
- [ ] benchmarks: D=3, D=4, J2 energy, Fig. 1

Other momenta and larger cells stay in #1052.
EOF
)"
gh pr create --base main --title "feat(excitations): compute_spectrum and spectral_function" --body "$(cat <<'EOF'
## Summary
- Adds compute_spectrum and spectral_function: H_k and N_k as mixed second derivatives through the CTM, at (pi,0), (0,pi), (pi,pi) on a 2x2 cell.
- N comes from the CTM log-norm, H from the energy density. Finite difference on the ket, reverse on the bra.
- Per-column checkpoints with a run fingerprint, and diagnostics that warn.

## Validation
State the measured values from the slow test and the mutation count here, with the ring reversal named.

## Not in this PR
gate_nnn, the deprecation of compute_excitations, and the benchmarks each get their own PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
```

Replace the sentence under `## Validation` with the numbers read from the logs, using the `PATCH` endpoint (`gh pr edit` is broken in this repository: `gh api repos/tenax-lab/tenax/pulls/<n> -X PATCH -f body=...`). Add `Refs #<issue>`.

- [ ] **Step 12: Read reviews and stop**

When CI is green and the Codex verdict has arrived:

```bash
gh api repos/tenax-lab/tenax/pulls/<n>/reviews  --paginate --jq '.[] | "[\(.state)] \(.user.login)\n\(.body)"'
gh api repos/tenax-lab/tenax/pulls/<n>/comments --paginate --jq '.[] | "\(.path):\(.line)\n\(.body)"'
gh api repos/tenax-lab/tenax/issues/<n>/comments --paginate --jq '.[] | "\(.user.login)\n\(.body)"'
```

Verify each finding before acting on it or dismissing it. Report to the user and wait. Do not arm `--auto`. Branch 2 starts after this PR merges.

---

## Branch 2: `feat/ipeps-spectrum-nnn`

### Task 4: `gate_nnn` with the mixed diagonal RDM

**Files:**
- Modify: `src/tenax/algorithms/_ipeps_excitation_ctm.py`, `src/tenax/algorithms/ipeps_excitations.py`, `tests/conftest.py`, `docs/guide/algorithms/ad_excitations.md`
- Create: `tests/test_ipeps_spectrum_nnn.py`

**Interfaces:**
- Consumes: `ExcitationCTM`, `compute_spectrum`, `_spectrum_fingerprint`, `regauge_cell`, `_rdm_diagonal_tensor`.
- Produces: `mixed_rdm_diag(Es, kets, bras, *, main: bool)` with `Es, kets, bras` ordered `(TL, TR, BL, BR)`, returning `(s1, s2, s1_bra, s2_bra)`. `ExcitationCTM(A, gate, chi, sweeps, gate_nnn=None)`. `compute_spectrum(A, gate, momentum, operators, config, *, gate_nnn=None, basis=None)`. `gate_nnn` is not rotated: next-nearest neighbours share a sublattice.

- [ ] **Step 1: Create the branch**

```bash
git fetch origin && git switch -c feat/ipeps-spectrum-nnn origin/main
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_ipeps_spectrum_nnn.py`:

```python
"""The next-nearest-neighbour gate in compute_spectrum: the mixed diagonal RDM."""

import jax.numpy as jnp
import numpy as np
import pytest
from _ipeps_spectrum_fixtures import (
    SY_REAL,
    apply_op,
    hasik_d2,
    heisenberg_gate,
    regauge_cell,
    rotated_gate,
)

from tenax import SpectrumConfig, compute_spectrum
from tenax.algorithms._ctm_tensor_energy import _rdm_diagonal_tensor
from tenax.algorithms._ipeps_excitation_ctm import (
    CELL,
    NEIGHBORS,
    ExcitationCTM,
    cell_phases,
    mixed_rdm_diag,
    unpack_env,
)
from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor

J2 = 0.3


@pytest.fixture(scope="module")
def ctm():
    return ExcitationCTM(
        hasik_d2(), rotated_gate(), 10, 30, gate_nnn=J2 * heisenberg_gate()
    )


def _plaquette(mapping, c):
    cx, cy = NEIGHBORS[c]["right"], NEIGHBORS[c]["bottom"]
    return tuple(mapping[x] for x in (c, cx, cy, NEIGHBORS[cx]["bottom"]))


@pytest.mark.parametrize("main", [True, False], ids=["main", "anti"])
def test_diagonal_rdm_is_gauge_invariant_on_the_cell(ctm, main):
    rng = np.random.default_rng(0)
    E = {c: unpack_env(ctm.envs0[c], ctm.D) for c in CELL}
    G = regauge_cell(E, rng)
    kets = {c: jnp.asarray(rng.normal(size=ctm.A.shape)) for c in CELL}
    bras = {c: jnp.asarray(rng.normal(size=ctm.A.shape)) for c in CELL}
    for c in CELL:
        ks, bs = _plaquette(kets, c), _plaquette(bras, c)
        np.testing.assert_allclose(
            mixed_rdm_diag(_plaquette(G, c), ks, bs, main=main),
            mixed_rdm_diag(_plaquette(E, c), ks, bs, main=main),
            rtol=1e-8,
            atol=0,
        )


@pytest.mark.parametrize("main", [True, False], ids=["main", "anti"])
def test_diagonal_rdm_at_bra_equal_ket_matches_the_existing_one(ctm, main):
    E = unpack_env(ctm.envs0[(0, 0)], ctm.D)
    A = ctm.A
    rho = mixed_rdm_diag((E,) * 4, (A,) * 4, (A,) * 4, main=main)
    rho = rho / jnp.einsum("ijij->", rho)
    reference = _rdm_diagonal_tensor(_wrap_as_dense_tensor(A), ctm.envs0[(0, 0)])
    np.testing.assert_allclose(rho, reference, atol=1e-12)


def test_both_diagonals_close_to_the_same_number(ctm):
    """Open then trace: each site sees its own ket and bra at any position."""
    rng = np.random.default_rng(3)
    E = unpack_env(ctm.envs0[(0, 0)], ctm.D)
    kets = tuple(jnp.asarray(rng.normal(size=ctm.A.shape)) for _ in range(4))
    bras = tuple(jnp.asarray(rng.normal(size=ctm.A.shape)) for _ in range(4))
    main = mixed_rdm_diag((E,) * 4, kets, bras, main=True)
    anti = mixed_rdm_diag((E,) * 4, kets, bras, main=False)
    assert float(jnp.einsum("ijij->", main)) == pytest.approx(
        float(jnp.einsum("ijij->", anti)), rel=1e-10
    )


def test_energy_density_adds_two_diagonals_per_site(ctm):
    nearest = ExcitationCTM(hasik_d2(), rotated_gate(), 10, 30)
    zero = jnp.zeros_like(ctm.A)
    phases = cell_phases(1, 0)
    with_nnn, _ = ctm.densities(phases, 0.0, zero, zero)
    without, _ = nearest.densities(phases, 0.0, zero, zero)
    rho = _rdm_diagonal_tensor(_wrap_as_dense_tensor(ctm.A), ctm.envs0[(0, 0)])
    diagonal = jnp.einsum("ijkl,klij->", rho, jnp.asarray(heisenberg_gate()))
    assert float(with_nnn) == pytest.approx(
        float(without) + 2 * J2 * float(diagonal), abs=1e-9
    )


def test_h_with_the_nnn_gate_matches_a_double_finite_difference(ctm):
    rng = np.random.default_rng(7)
    phases = cell_phases(1, 0)
    X = rng.normal(size=ctm.A.shape)
    Y = rng.normal(size=ctm.A.shape)
    X, Y = X / np.linalg.norm(X), Y / np.linalg.norm(Y)
    step = 1e-3

    def energy(eps, delta):
        return float(
            ctm.densities(phases, eps, jnp.asarray(X), delta * jnp.asarray(Y))[0]
        )

    reference = (
        energy(step, step)
        - energy(step, -step)
        - energy(-step, step)
        + energy(-step, -step)
    ) / (4 * step**2)
    g_h, _ = ctm.ket_derivative(phases, jnp.asarray(X), 1e-4)
    assert float(np.vdot(Y, np.asarray(g_h))) == pytest.approx(reference, abs=1e-4)


def test_compute_spectrum_keeps_the_sum_rule_with_nnn(tmp_path):
    A = hasik_d2()
    stack = [apply_op(A, SY_REAL).ravel(), A.ravel()]
    basis, _ = np.linalg.qr(np.stack(stack, axis=1))
    config = SpectrumConfig(chi=10, checkpoint_dir=tmp_path / "nnn")
    result = compute_spectrum(
        A,
        rotated_gate(),
        (np.pi, 0.0),
        {"Sy": SY_REAL},
        config,
        gate_nnn=J2 * heisenberg_gate(),
        basis=basis,
    )
    assert abs(result.sum_rule_deficit["Sy"]) < 1e-6
    with pytest.raises(ValueError, match="different run"):
        compute_spectrum(
            A,
            rotated_gate(),
            (np.pi, 0.0),
            {"Sy": SY_REAL},
            config,
            gate_nnn=0.5 * heisenberg_gate(),
            basis=basis,
        )


def test_bad_nnn_gate_shape_raises():
    with pytest.raises(ValueError, match="gate_nnn"):
        compute_spectrum(
            hasik_d2(),
            rotated_gate(),
            (np.pi, 0.0),
            {},
            SpectrumConfig(chi=10),
            gate_nnn=np.eye(3),
        )
```

Add to `_FILE_MARKERS` in `tests/conftest.py`, directly after `"test_ipeps_spectrum_full.py": "algorithm",`:

```python
    # gate_nnn in compute_spectrum: the mixed diagonal RDM at D=2, chi=10.
    "test_ipeps_spectrum_nnn.py": "core",
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_spectrum_nnn.py -q --no-cov`
Expected: collection error, `ImportError: cannot import name 'mixed_rdm_diag'`.

- [ ] **Step 4: Edit the module**

Apply these replacements to `src/tenax/algorithms/_ipeps_excitation_ctm.py`.

Replace

```python
def z00(c1, c2, c3, c4):
```

with

```python
def _phys(open_, a, b):
    """Einsum labels of the ket and bra physical legs: traced, or left open."""
    return (a, b) if open_ else (a, a)


def mixed_rdm_diag(Es, kets, bras, *, main: bool):
    """Diagonal pair on a plaquette: ``(s1, s2, s1_bra, s2_bra)``.

    ``Es``, ``kets`` and ``bras`` are ordered ``(TL, TR, BL, BR)``.  ``main``
    leaves TL and BR open (TL=s1, BR=s2); otherwise TR and BL are open
    (TR=s1, BL=s2).
    """
    (E_tl, E_tr, E_bl, E_br), (k_tl, k_tr, k_bl, k_br), (b_tl, b_tr, b_bl, b_br) = (
        Es, kets, bras,
    )  # fmt: skip
    s, t = _phys(main, 10, 13)
    TL = jnp.einsum(
        E_tl["C1"], [0, 1],
        E_tl["T1"], [1, 2, 3, 4],
        E_tl["T4"], [0, 5, 6, 7],
        k_tl, [2, 8, 5, 9, s],
        b_tl, [3, 11, 6, 12, t],
        [4, 7, 8, 11, 9, 12] + ([s, t] if main else []),
        optimize="greedy",
    )  # fmt: skip
    s, t = _phys(not main, 10, 13)
    TR = jnp.einsum(
        E_tr["T1"], [0, 1, 2, 3],
        E_tr["C2"], [3, 4],
        E_tr["T2"], [4, 5, 6, 7],
        k_tr, [1, 8, 9, 5, s],
        b_tr, [2, 11, 12, 6, t],
        [0, 7, 8, 11, 9, 12] + ([s, t] if not main else []),
        optimize="greedy",
    )  # fmt: skip
    s, t = _phys(not main, 10, 13)
    BL = jnp.einsum(
        E_bl["T4"], [0, 1, 2, 3],
        E_bl["C4"], [3, 4],
        E_bl["T3"], [4, 5, 6, 7],
        k_bl, [8, 5, 1, 9, s],
        b_bl, [11, 6, 2, 12, t],
        [7, 0, 8, 11, 9, 12] + ([s, t] if not main else []),
        optimize="greedy",
    )  # fmt: skip
    s, t = _phys(main, 10, 13)
    BR = jnp.einsum(
        E_br["T3"], [0, 1, 2, 3],
        E_br["C3"], [4, 3],
        E_br["T2"], [5, 6, 7, 4],
        k_br, [8, 1, 9, 6, s],
        b_br, [11, 2, 12, 7, t],
        [0, 5, 8, 11, 9, 12] + ([s, t] if main else []),
        optimize="greedy",
    )  # fmt: skip
    if main:
        left = jnp.einsum(TL, [0, 1, 2, 3, 4, 5, 6, 7], BL, [8, 1, 2, 3, 9, 10], [0, 4, 5, 8, 9, 10, 6, 7])
        right = jnp.einsum(TR, [0, 1, 2, 3, 4, 5], BR, [6, 1, 2, 3, 7, 8, 9, 10], [0, 4, 5, 6, 7, 8, 9, 10])
        return jnp.einsum(left, [0, 1, 2, 3, 4, 5, 6, 7], right, [0, 1, 2, 3, 4, 5, 8, 9], [6, 8, 7, 9])
    left = jnp.einsum(TL, [0, 1, 2, 3, 4, 5], BL, [6, 1, 2, 3, 7, 8, 9, 10], [0, 4, 5, 6, 7, 8, 9, 10])
    right = jnp.einsum(TR, [0, 1, 2, 3, 4, 5, 6, 7], BR, [8, 1, 2, 3, 9, 10], [0, 4, 5, 8, 9, 10, 6, 7])
    return jnp.einsum(left, [0, 1, 2, 3, 4, 5, 8, 9], right, [0, 1, 2, 3, 4, 5, 6, 7], [6, 8, 7, 9])


def z00(c1, c2, c3, c4):
```

Replace

```python
    def __init__(self, A, gate, chi, sweeps):
```

with

```python
    def __init__(self, A, gate, chi, sweeps, gate_nnn=None):
```

Replace

```python
        self.gate = jnp.asarray(gate).reshape((self.d,) * 4)
```

with

```python
        self.gate = jnp.asarray(gate).reshape((self.d,) * 4)
        self.gate_nnn = (
            None if gate_nnn is None else jnp.asarray(gate_nnn).reshape((self.d,) * 4)
        )
```

Replace

```python
                total = total + _bond_value(rho, self.gate)
```

with

```python
                total = total + _bond_value(rho, self.gate)
            if self.gate_nnn is not None:
                cx, cy = NEIGHBORS[c]["right"], NEIGHBORS[c]["bottom"]
                cxy = NEIGHBORS[cx]["bottom"]
                sites = (c, cx, cy, cxy)
                Es = tuple(E[x] for x in sites)
                ks = tuple(kets[x] for x in sites)
                bs = tuple(bras[x] for x in sites)
                for main in (True, False):
                    total = total + _bond_value(
                        mixed_rdm_diag(Es, ks, bs, main=main), self.gate_nnn
                    )
```

- [ ] **Step 5: Edit the public code**

Apply these replacements to `src/tenax/algorithms/ipeps_excitations.py`.

Replace

```python
    *,
    basis=None,
) -> SpectrumResult:
```

with

```python
    *,
    gate_nnn=None,
    basis=None,
) -> SpectrumResult:
```

Replace

```python
        config: See :class:`SpectrumConfig`.
        basis: Optional
```

with

```python
        config: See :class:`SpectrumConfig`.
        gate_nnn: Optional next-nearest-neighbour gate, used on both diagonals.
        basis: Optional
```

Replace

```python
    gate = _spectrum_gate(gate, d, "gate")
    operators =
```

with

```python
    gate = _spectrum_gate(gate, d, "gate")
    if gate_nnn is not None:
        gate_nnn = _spectrum_gate(gate_nnn, d, "gate_nnn")
    operators =
```

Replace

```python
    fingerprint = _spectrum_fingerprint(A, gate, (a, b), config, basis)
```

with

```python
    fingerprint = _spectrum_fingerprint(A, gate, gate_nnn, (a, b), config, basis)
```

Replace

```python
def _spectrum_fingerprint(A, gate, ab, config, basis) -> str:
    digest = hashlib.sha256()
    for array in (A, gate, basis):
```

with

```python
def _spectrum_fingerprint(A, gate, gate_nnn, ab, config, basis) -> str:
    digest = hashlib.sha256()
    for array in (A, gate, np.zeros(0) if gate_nnn is None else gate_nnn, basis):
```

Replace

```python
    ctm = ExcitationCTM(A, gate, config.chi, config.ctm_sweeps)
```

with

```python
    ctm = ExcitationCTM(A, gate, config.chi, config.ctm_sweeps, gate_nnn=gate_nnn)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_spectrum_nnn.py tests/test_ipeps_spectrum.py -q --no-cov > /tmp/spectrum_nnn.log 2>&1; grep -E "passed|failed" /tmp/spectrum_nnn.log`
Expected: `9 passed` from the new file, and no failure in the other. The first took 53 s when measured.

- [ ] **Step 7: Update the guide**

In `docs/guide/algorithms/ad_excitations.md`, add at the end of the `### Example` block under `## Spectrum and Spectral Function`, after the closing fence:

```markdown
For a J1-J2 model pass `gate_nnn`, used on both diagonals. Next-nearest neighbours
share a sublattice, so in the rotated frame this is `J2 * S.S` without the rotation.
The diagonal RDM is a larger contraction, so each column costs more than the figures
under Limits.
```

- [ ] **Step 8: Add the diagonal mutants and run them**

Use the script from Task 3 Step 9 with this `MUTANTS` list and with `TESTS = ["tests/test_ipeps_spectrum_nnn.py"]`:

```python
M = "src/tenax/algorithms/_ipeps_excitation_ctm.py"
MUTANTS = [
    ("diag: BL T4 reversed", M, 'E_bl["T4"], [0, 1, 2, 3],', 'E_bl["T4"], [3, 1, 2, 0],'),
    ("diag: TR ket and bra swapped", M, "k_tr, [1, 8, 9, 5, s],", "k_tr, [2, 8, 9, 5, s],"),
    ("diag: BR env from TL", M, "(E_tl, E_tr, E_bl, E_br), (k_tl", "(E_br, E_tr, E_bl, E_tl), (k_tl"),
    ("two diagonals -> one", M, "for main in (True, False):", "for main in (True,):"),
]
```

Before running, read the formatted file and confirm each anchor exists exactly once (the script asserts this). Report the count in the PR.

- [ ] **Step 9: Commit and open the PR**

```bash
git add src/tenax/algorithms/_ipeps_excitation_ctm.py src/tenax/algorithms/ipeps_excitations.py tests/test_ipeps_spectrum_nnn.py tests/conftest.py docs/guide/algorithms/ad_excitations.md
git commit -m "$(cat <<'EOF'
feat(excitations): gate_nnn in compute_spectrum, with a mixed diagonal RDM

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
git push -u origin feat/ipeps-spectrum-nnn
gh pr create --base main --title "feat(excitations): gate_nnn in compute_spectrum" --body "$(cat <<'EOF'
## Summary
- Adds the mixed diagonal RDM and gate_nnn, used on both diagonals of every site.
- At bra = ket the RDM matches _rdm_diagonal_tensor; with random sites both diagonals close to the same number.

Mutation count: state it here.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
```

Then follow Task 3 Step 12: read the reviews, report, and stop.

---

## Branch 3: `chore/deprecate-compute-excitations`

### Task 5: Deprecate `compute_excitations`

**Files:**
- Modify: `src/tenax/algorithms/ipeps_excitations.py` (`compute_excitations`), `pyproject.toml` (`filterwarnings`), `tests/test_ipeps_excitations.py`, `docs/guide/algorithms/ad_excitations.md` (`## Excitation Spectrum`, line 248), `.claude/skills/tenax-ipeps-workflow/SKILL.md`, `examples/heisenberg_ipeps_excitations.py`

**Interfaces:**
- Consumes: `compute_spectrum`.
- Produces: `compute_excitations` emits `DeprecationWarning` naming `compute_spectrum`. It does not delegate, because v1 cannot reach every momentum it accepts.

- [ ] **Step 1: Create the branch**

```bash
git fetch origin && git switch -c chore/deprecate-compute-excitations origin/main
```

- [ ] **Step 2: Write the failing test**

Add to class `TestExcitationEnergies` in `tests/test_ipeps_excitations.py`:

```python
    def test_compute_excitations_is_deprecated(
        self, small_peps_and_env, heisenberg_gate
    ):
        A, env, d = small_peps_and_env
        E_gs = float(compute_energy_ctm(A, env, heisenberg_gate, d))
        config = ExcitationConfig(num_excitations=1, null_space_tol=1e-2)
        with pytest.warns(DeprecationWarning, match="compute_spectrum"):
            compute_excitations(
                A, env, heisenberg_gate, E_gs, [(np.pi, 0.0)], config
            )
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_excitations.py -k deprecated -q --no-cov`
Expected: FAIL, `DID NOT WARN`.

- [ ] **Step 4: Add the warning**

In `compute_excitations`, as the first statement after the docstring:

```python
    warnings.warn(
        "compute_excitations is deprecated: it truncates the momentum sum to "
        "nearest neighbours (#1052). Use compute_spectrum at (pi,0), (0,pi) and "
        "(pi,pi). Other momenta are not available yet.",
        DeprecationWarning,
        stacklevel=2,
    )
```

Add this to the docstring, directly after its first line:

```
    .. deprecated:: 0.9.3
        Use :func:`compute_spectrum`. This function truncates the momentum sum
        to nearest neighbours (#1052).
```

Run `git describe --tags --abbrev=0` first, and use the next patch version if `0.9.3` is not it.

- [ ] **Step 5: Silence it in the test bucket**

In `pyproject.toml`, add to `filterwarnings`, after the `chi_ramp` entry:

```toml
    # compute_excitations is deprecated in favour of compute_spectrum (#1052).
    # test_compute_excitations_is_deprecated un-suppresses it via ``pytest.warns``.
    "ignore:compute_excitations is deprecated:DeprecationWarning",
```

- [ ] **Step 6: Update the docs**

In `docs/guide/algorithms/ad_excitations.md`, add directly below `## Excitation Spectrum`:

```markdown
```{warning}
`compute_excitations` is deprecated: it truncates the momentum sum (#1052), and its
lowest energies on published Heisenberg states are negative (-0.10 at D=2 and -0.27
at D=3 at (pi,0)). Use [`compute_spectrum`](#spectrum-and-spectral-function) where
its momenta apply.
```
```

In `.claude/skills/tenax-ipeps-workflow/SKILL.md`, in Stage 4, add directly above the first code block:

```markdown
`compute_excitations` is deprecated (#1052). Use it only for a rough scan at
momenta `compute_spectrum` cannot reach.
```

In `examples/heisenberg_ipeps_excitations.py`, add to the top docstring:

```
compute_excitations is deprecated (#1052). This example stays because it is the only
route to a full Brillouin-zone path, and its low energies are not reliable.
```

- [ ] **Step 7: Run the tests**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 uv run pytest tests/test_ipeps_excitations.py tests/test_ipeps_excitations_gauge.py -q --no-cov > /tmp/excitations.log 2>&1; grep -E "passed|failed" /tmp/excitations.log`
Expected: all pass.

- [ ] **Step 8: Commit and open the PR**

```bash
git add src/tenax/algorithms/ipeps_excitations.py pyproject.toml tests/test_ipeps_excitations.py docs/guide/algorithms/ad_excitations.md .claude/skills/tenax-ipeps-workflow/SKILL.md examples/heisenberg_ipeps_excitations.py
git commit -m "$(cat <<'EOF'
chore(excitations): deprecate compute_excitations in favour of compute_spectrum

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
git push -u origin chore/deprecate-compute-excitations
gh pr create --base main --title "chore(excitations): deprecate compute_excitations" --body "$(cat <<'EOF'
## Summary
- compute_excitations emits a DeprecationWarning naming compute_spectrum.
- It does not delegate: compute_spectrum cannot reach every momentum it accepts.
- Its truncated-sum helpers are removed with it in a later release.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
```

Then follow Task 3 Step 12: read the reviews, report, and stop.

---

## Branch 4: `bench/ipeps-spectrum`

### Task 6: Benchmark script

**Files:**
- Create: `benchmarks/ipeps_spectrum_hasik.py`, `tests/test_ipeps_spectrum_benchmark.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: `compute_spectrum(..., gate_nnn=)`, `ExcitationCTM`, `cell_phases`.
- Produces: `load_state(path) -> np.ndarray`, `u1_residuals(A, angles=721) -> dict`, and a CLI with tasks `u1`, `energy`, `spectrum`.

- [ ] **Step 1: Create the branch**

```bash
git fetch origin && git switch -c bench/ipeps-spectrum origin/main
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_ipeps_spectrum_benchmark.py`:

```python
"""The benchmark's state parser and U(1) scan."""

import json

import numpy as np
import pytest

from benchmarks.ipeps_spectrum_hasik import load_state, u1_residuals


def test_load_state_reads_the_leg_order_s_u_l_d_r(tmp_path):
    path = tmp_path / "state.json"
    entries = ["1 0 1 0 0 3.0", "0 1 0 0 1 4.0"]
    site = {"siteId": "A", "physDim": 2, "auxDim": 2, "entries": entries}
    path.write_text(json.dumps({"sites": [site]}))
    A = load_state(path)
    assert A.shape == (2, 2, 2, 2, 2)
    # s=1, u=0, l=1, d=0, r=0 is A[u, d, l, r, s] = A[0, 0, 1, 0, 1]
    assert A[0, 0, 1, 0, 1] == pytest.approx(3.0 / 5.0)
    assert A[1, 0, 0, 1, 0] == pytest.approx(4.0 / 5.0)


def _u1_symmetric_tensor():
    rng = np.random.default_rng(0)
    charge = np.array([0, 1])
    physical = np.array([1, -1])
    A = np.zeros((2, 2, 2, 2, 2))
    for u, d, left, r, s in np.ndindex(2, 2, 2, 2, 2):
        if charge[u] + charge[d] + charge[left] + charge[r] - 2 == physical[s]:
            A[u, d, left, r, s] = rng.normal()
    return A / np.linalg.norm(A)


def test_u1_scan_finds_a_symmetric_tensor():
    residuals = u1_residuals(_u1_symmetric_tensor())
    assert residuals["best_xz"] < 1e-12
    assert residuals["best_angle"] == pytest.approx(0.0)


def test_u1_scan_finds_it_in_another_bond_gauge():
    A = _u1_symmetric_tensor()
    theta = 0.7
    V = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = np.einsum("au,bd,cl,er,udlrs->abces", V, V, V, V, A)
    assert u1_residuals(rotated)["best_xz"] < 1e-12


def test_u1_scan_rejects_a_random_tensor():
    A = np.random.default_rng(1).normal(size=(2, 2, 2, 2, 2))
    assert u1_residuals(A / np.linalg.norm(A))["best_xz"] > 0.1
```

Add to `_FILE_MARKERS` in `tests/conftest.py`, directly after `"test_ipeps_spectrum_nnn.py": "core",`:

```python
    # The benchmark's state parser and U(1) scan: no CTM.
    "test_ipeps_spectrum_benchmark.py": "core",
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_spectrum_benchmark.py -q --no-cov`
Expected: collection error, `ModuleNotFoundError: No module named 'benchmarks.ipeps_spectrum_hasik'`.

- [ ] **Step 4: Write the script**

Create `benchmarks/ipeps_spectrum_hasik.py`:

```python
"""Benchmark compute_spectrum on Hasik's J1-J2 states.

States: github.com/jurajHasik/j1j2_ipeps_states (CC BY 4.0), the JSON files in
single-site_pg-C4v-A1/. They are data: this script only parses them, and never
runs code from that repository.

Example::

    uv run python benchmarks/ipeps_spectrum_hasik.py state.json energy --chi 36 --j2 0.3
    uv run python benchmarks/ipeps_spectrum_hasik.py state.json u1
    uv run python benchmarks/ipeps_spectrum_hasik.py state.json spectrum --chi 36 \
        --momentum X --checkpoint-dir runs/d3_chi36_x --out d3_chi36_x.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SY_REAL = np.array([[0.0, 0.5], [-0.5, 0.0]])
SX = np.array([[0.0, 0.5], [0.5, 0.0]])
SZ = np.diag([0.5, -0.5])
MOMENTA = {"X": (np.pi, 0.0), "Y": (0.0, np.pi), "M": (np.pi, np.pi)}


def load_state(path: Path) -> np.ndarray:
    """Parse entries ``"s u l d r value"`` into ``A[u, d, l, r, s]``, normalised."""
    site = json.loads(Path(path).read_text())["sites"][0]
    D = site["auxDim"]
    A = np.zeros((D, D, D, D, site["physDim"]))
    for entry in site["entries"]:
        s, u, left, d, r, value = entry.split()[:6]
        A[int(u), int(d), int(left), int(r), int(s)] = float(value)
    return A / np.linalg.norm(A)


def heisenberg_gate() -> np.ndarray:
    return np.array(
        [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, -0.25, 0.5, 0.0],
            [0.0, 0.5, -0.25, 0.0],
            [0.0, 0.0, 0.0, 0.25],
        ]
    ).reshape(2, 2, 2, 2)


def rotated_gate() -> np.ndarray:
    W = np.kron(np.eye(2), np.array([[0.0, 1.0], [-1.0, 0.0]]))
    H = heisenberg_gate().reshape(4, 4)
    return (W @ H @ W.T).reshape(2, 2, 2, 2)


def u1_residuals(A: np.ndarray, angles: int = 721) -> dict[str, float]:
    """How far ``A`` is from invariance under a U(1) with any bond generators.

    The residual is ``|P G A| / |G A|``, with ``P`` the projector off the span of
    an independent ``Q`` on each of the four legs. Zero means a charge sector
    exists in some bond gauge. The physical axis ``G`` is scanned in the real
    xz plane, and ``i S^y`` is tested separately.
    """
    D = A.shape[0]
    columns = []
    for p in range(D):
        for q in range(D):
            E = np.zeros((D, D))
            E[p, q] = 1.0
            columns += [
                np.einsum("xu,udlrs->xdlrs", E, A).ravel(),
                np.einsum("xd,udlrs->uxlrs", E, A).ravel(),
                np.einsum("xl,udlrs->udxrs", E, A).ravel(),
                np.einsum("xr,udlrs->udlxs", E, A).ravel(),
            ]
    U, singular, _ = np.linalg.svd(np.stack(columns, axis=1), full_matrices=False)
    Q = U[:, singular > 1e-10 * singular[0]]

    def residual(generator):
        b = np.tensordot(A, generator, axes=([4], [1])).ravel()
        return float(np.linalg.norm(b - Q @ (Q.T @ b)) / np.linalg.norm(b))

    scan = [
        (residual(np.cos(t) * SZ + np.sin(t) * SX), float(t))
        for t in np.linspace(0.0, np.pi, angles)
    ]
    best, angle = min(scan)
    return {"best_xz": best, "best_angle": angle, "i_sy": residual(SY_REAL)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("task", choices=["u1", "energy", "spectrum"])
    parser.add_argument("--chi", type=int, default=36)
    parser.add_argument("--j2", type=float, default=0.0)
    parser.add_argument("--momentum", choices=sorted(MOMENTA), default="X")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    A = load_state(args.state)
    print(f"state {args.state}: D={A.shape[0]}, {A.size} columns")
    if args.task == "u1":
        print(u1_residuals(A))
        return

    from tenax import SpectrumConfig, compute_spectrum
    from tenax.algorithms._ipeps_excitation_ctm import ExcitationCTM, cell_phases

    gate_nnn = args.j2 * heisenberg_gate() if args.j2 else None
    if args.task == "energy":
        ctm = ExcitationCTM(A, rotated_gate(), args.chi, 1, gate_nnn=gate_nnn)
        zero = np.zeros_like(A)
        energy, _ = ctm.densities(cell_phases(1, 0), 0.0, zero, zero)
        print(f"chi={args.chi} J2={args.j2} energy per site {float(energy):.10f}")
        return

    start = time.time()
    result = compute_spectrum(
        A,
        rotated_gate(),
        MOMENTA[args.momentum],
        {"Sy": SY_REAL, "Sx": SX, "Sz": SZ},
        SpectrumConfig(chi=args.chi, checkpoint_dir=args.checkpoint_dir),
        gate_nnn=gate_nnn,
    )
    print(f"{time.time() - start:.0f} s, rank {result.rank}")
    print("lowest omega", np.round(result.omega[:6], 4))
    print("s(k)", result.structure_factor)
    print("sum rule deficit", result.sum_rule_deficit)
    print(
        f"energy {result.energy_density:.10f} negative modes "
        f"{result.n_negative_modes} n_min_eig {result.n_min_eig:.2e} "
        f"gauge residual {result.gauge_residual}"
    )
    if args.out is not None:
        np.savez(
            args.out,
            omega=result.omega,
            **{f"w_{name}": w for name, w in result.weights.items()},
            **{f"s_{name}": s for name, s in result.structure_factor.items()},
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `JAX_PLATFORMS=cpu uv run pytest tests/test_ipeps_spectrum_benchmark.py -q --no-cov`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/ipeps_spectrum_hasik.py tests/test_ipeps_spectrum_benchmark.py tests/conftest.py
git commit -m "$(cat <<'EOF'
feat(benchmarks): compute_spectrum on Hasik's J1-J2 states

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
```

### Task 7: Run the benchmarks and record the results

Each run below is long, and its result goes in the guide whether it passes or not. A miss gets investigated, not retuned. Report only values read from the logs.

**Files:**
- Modify: `docs/guide/algorithms/ad_excitations.md` (add `### Benchmarks` under `## Spectrum and Spectral Function`)

**Interfaces:**
- Consumes: `benchmarks/ipeps_spectrum_hasik.py`.
- Produces: the benchmark table, and a go or no-go decision for Fig. 1 by the user.

- [ ] **Step 1: Fetch the states into an empty data directory**

Fetch only the JSON and `.dat` data files. Do not fetch or run the repository's code.

```bash
mkdir -p "$HOME/tenax-sma/hasik_states" && cd "$HOME/tenax-sma/hasik_states"
BASE=https://raw.githubusercontent.com/jurajHasik/j1j2_ipeps_states/main/single-site_pg-C4v-A1
for f in j20.0/state_1s_A1_j20.0_D2_chi_opt48 j20.0/state_1s_A1_j20.0_D3_chi_opt108 j20.0/state_1s_A1_j20.0_D4_chi_opt97 j20.3/state_1s_A1_j20.3_D4_chi_opt96; do
  curl -sfL -O "$BASE/$f.json"; curl -sfL -O "$BASE/$f.dat"
done
ls -la
```

Expected: 8 non-empty files. `S=$HOME/tenax-sma/hasik_states` below.

- [ ] **Step 2: Benchmark 2, the U(1) scan (minutes)**

Run: `for f in $S/*.json; do JAX_PLATFORMS=cpu uv run python benchmarks/ipeps_spectrum_hasik.py "$f" u1; done`

Measured before this plan, with `best_xz` the residual at the best axis, where a synthetic U(1) tensor gives 1e-15 and a random tensor 0.64:

| State | `best_xz` | Reading |
|---|---|---|
| J2=0, D=2 | 3.8e-4 | close to U(1) |
| J2=0, D=3 | 0.66 | none |
| J2=0, D=4 | 0.052 | none at this accuracy |
| J2=0.3, D=4 (Fig. 1) | 0.71 | none |

So the sector saving that the spec expected does not apply to the Fig. 1 state. Confirm these values and record them.

- [ ] **Step 3: Benchmark 3, the J2 energy (minutes to an hour)**

Run: `JAX_PLATFORMS=cpu taskset -c 0-11 nohup uv run python benchmarks/ipeps_spectrum_hasik.py $S/state_1s_A1_j20.3_D4_chi_opt96.json energy --chi 32 --j2 0.3 > bench3.log 2>&1 &`

Pass: within 1e-5 of the `chi_obs=32` row of `state_1s_A1_j20.3_D4_chi_opt96.dat`, -0.5549696161. A run at chi=36 did not finish its environment in 280 s, so allow time.

- [ ] **Step 4: Benchmark 1, D=3 at chi=36 (about 1 to 2 days per momentum)**

Run each momentum with a checkpoint directory, so a kill costs one column:

```bash
JAX_PLATFORMS=cpu taskset -c 0-11 nohup uv run python benchmarks/ipeps_spectrum_hasik.py $S/state_1s_A1_j20.0_D3_chi_opt108.json spectrum --chi 36 --momentum X --checkpoint-dir runs/d3_chi36_x --out d3_chi36_x.npz > d3_chi36_x.log 2>&1 &
```

Repeat with `--momentum M`. Pass: no negative modes, and omega(X) near 2.268. The first full re-solve at this chi: the numbers in the spec are fixed-vector evaluations. Measure and report the cost per column from the checkpoint timestamps after 5 columns. If the projection is above 3 days per momentum, stop and ask the user.

- [ ] **Step 5: Ask the user before Benchmarks 4 and 5**

Report the measured cost per column and project D=4, 512 columns. Before this plan the projection for H at chi=36 was about 9 days per momentum on 12 CPU cores, with up to twice that for N. Ask whether to run D=4 at J2=0 and Fig. 1, on which machine, and at which chi. The user decides. Do not start them unasked.

- [ ] **Step 6: If approved, run Benchmarks 4 and 5**

Benchmark 4, D=4 J2=0, with `$S/state_1s_A1_j20.0_D4_chi_opt97.json`: pass if omega(X) is below the D=3 value and toward 2.18, and the gap at M is smaller than at D=3.

Benchmark 5, Fig. 1, at the chi the user chose in Step 5:

```bash
JAX_PLATFORMS=cpu taskset -c 0-11 nohup uv run python benchmarks/ipeps_spectrum_hasik.py $S/state_1s_A1_j20.3_D4_chi_opt96.json spectrum --chi "$CHI" --j2 0.3 --momentum X --checkpoint-dir runs/fig1 --out fig1.npz > fig1.log 2>&1 &
```

Frame: the state is sublattice-rotated. U = i sigma^y commutes with S^y and flips S^x and S^z on one sublattice. So physical (pi,0) is rotated X for S^y, and rotated Y = (0,pi) for S^x and S^z. C4v acts on the lattice only, so S^x and S^z at Y should equal their values at X, and one run at X should give all three. Check this first on the D=2 state, with `--momentum Y` at chi=10, against X: `Sy` 0.203989, `Sx` 0.203142, `Sz` 0.095381.

Before comparing, read the caption of Fig. 1 in arXiv:2306.13327 and state which component or sum its s = 0.182 means.

Pass: s within 1% of 0.182, and the peak within 0.05 of 1.05 at eta=0.1.

- [ ] **Step 7: Record the results and open the PR**

Add a `### Benchmarks` table to the guide, with chi, machine, cost per column, and pass or miss for each step. Include the U(1) table from Step 2.

```bash
git add docs/guide/algorithms/ad_excitations.md
git commit -m "$(cat <<'EOF'
docs(excitations): compute_spectrum benchmark results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
git push -u origin bench/ipeps-spectrum
gh pr create --base main --title "feat(benchmarks): compute_spectrum benchmarks and results" --body "$(cat <<'EOF'
## Summary
- Adds benchmarks/ipeps_spectrum_hasik.py, and the results table in the guide.

State each benchmark's measured value, pass or miss here.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013vsTDi8oyvbhqwqrfLQqnQ
EOF
)"
```

Then follow Task 3 Step 12: read the reviews, report, and stop.

### Task 8: The forward-mode follow-up issue

- [ ] **Step 1: Open the issue**

```bash
gh issue create --title "Forward-mode second derivative through the CTM: findings and open causes" --body "$(cat <<'EOF'
> 🤖 **AI-generated issue** — written by Claude Code, posted by @yingjerkao.

compute_spectrum uses a finite difference on the ket because forward mode did not work. D=2, chi=10, measured 2026-11-05:

- A recursive Lorentzian custom_jvp SVD matches FD to 1e-9 on a random matrix.
- In the sweep it needs a safe sqrt of the zeroed Fishman S (_ctm_tensor_projector_2x2.py:885-886: 17 of 40 values below the floor give inf*0). First-order forward is then correct: FD over forward equals FD-FD at -0.061971.
- Reverse over that rule is exact for S^y A (0.4976434, 0.0388188) but gives about 1e3 in random directions, where FD-FD gives about 0.1. A rank-aware mask does not change this. Cause not found.
- Forward under jit diverges (1e36 at one sweep, NaN by ten). Cause not found.
- Reverse over reverse gives NaN.

The ket derivative is behind one function, ExcitationCTM.ket_derivative, so a working rule can replace it.
EOF
)"
```

---

## Self-Review Record

- Spec coverage: API (Task 2, Task 4), method and bond placement (Task 1), errors, diagnostics, checkpoints (Task 2), core and slow tests (Tasks 1-3), mutation count (Task 3), delivery in four branches, docs, both issues (Tasks 3, 8), benchmarks 1-5 (Task 7).
- Changes from the spec, all recorded in its Revisions section: H asymmetry threshold 1e-3; `A` normalised internally; float64 required; an absolute floor on the N cut; benchmark 2 answered during planning.
