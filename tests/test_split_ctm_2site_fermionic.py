"""Fermionic (graded) 2-site split-CTM forward parity — #463 Phase 4.

The split 2×2 absorb kernels build double-layer corners / edges via graded
``A.bar()`` alone; they do NOT carry the order-dependent Koszul signs that the
*fused* 2-site path applies (the ``*_2plaq_fused`` variants gated by
``_env_is_fermionic``).  As a result, a raw fermionic split sweep diverges from
the proven fused sweep — the corner GROW alone is already wrong (not just the
edge).  See the module docstring of ``_split_ctm_tensor_moves`` and #641.

Phase 4 routes fermionic envs through the fused sweep on merged envs, then
re-splits the output edges back to ket/bra halves (``merge → fused → resplit``).

These tests pin that behaviour:

* ``test_fermionic_split_sweep_matches_fused`` — one split sweep, merged to the
  fused representation, must equal one fused sweep from the *same* shared env, to
  machine precision, for every corner and edge.  Convergence- and
  init-independent: the fused sweep is the trusted oracle (validated vs JW-ED in
  #674), so per-sweep parity ⟹ (by induction) converged-env parity ⟹ correct
  energy.
* ``test_fermionic_edge_resplit_roundtrip`` — the load-bearing new primitive: a
  fused edge, re-split into ket/bra halves and merged back, reproduces itself.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FermionParity = pytest.importorskip("tenax.core.symmetry").FermionParity

from tenax.algorithms._ctm_tensor_convergence import (
    CHECKERBOARD_NEIGHBORS as NB,
)
from tenax.algorithms._ctm_tensor_convergence import (
    _ctm_tensor_sweep_multisite,
)
from tenax.algorithms._ctm_tensor_init import _build_double_layer_tensor
from tenax.algorithms._split_ctm_tensor_convergence import (
    _initialize_split_multisite_env,
    _split_ctm_sweep_multisite_2x2,
)
from tenax.algorithms._split_ctm_tensor_energy import _split_env_to_tensor_standard
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import SymmetricTensor


def _make_fp_site(D=2, seed=0):
    """Uniform-compatible FermionParity iPEPS site (``A.l`` charges == ``A.r``)."""
    sym = FermionParity()
    ch = np.array(([0, 1] * D)[:D], dtype=np.int32)
    phys = np.array([0, 1], dtype=np.int32)

    def idx(c, flow, label):
        return TensorIndex.from_charges(sym, c, flow, label=label)

    return SymmetricTensor.random_normal(
        (
            idx(ch, FlowDirection.OUT, "u"),
            idx(ch, FlowDirection.IN, "d"),
            idx(ch, FlowDirection.OUT, "l"),
            idx(ch, FlowDirection.IN, "r"),
            idx(phys, FlowDirection.IN, "phys"),
        ),
        jax.random.PRNGKey(seed),
    )


def _one_minus_fidelity(x, y):
    """1 − scale/phase-invariant overlap fidelity, aligning axes by label."""
    lx = list(x.labels())
    ly = list(y.labels())
    assert sorted(lx) == sorted(ly), f"label sets differ:\n {lx}\n {ly}"
    perm = tuple(lx.index(lbl) for lbl in ly)
    ax = np.asarray(x.transpose(perm).todense()).ravel()
    ay = np.asarray(y.todense()).ravel()
    nx = np.linalg.norm(ax)
    ny = np.linalg.norm(ay)
    assert nx > 0 and ny > 0
    return 1.0 - abs(np.vdot(ax, ay)) / (nx * ny)


def _build_shared_fermionic_envs(chi, n_warmup=4):
    """A generic (non-trivial) fermionic split env + its fused merge.

    Uniform A=B checkerboard: enough to exercise the graded corner/edge grows
    while keeping the block-sparse forward cheap.
    """
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_ctm_sweep_multisite,
    )

    A = _make_fp_site(seed=11)
    site_tensors = {(0, 0): A, (1, 0): A}
    bars = {c: T.bar() for c, T in site_tensors.items()}
    split_envs = _initialize_split_multisite_env(site_tensors, chi, chi)
    for _ in range(n_warmup):
        split_envs = _split_ctm_sweep_multisite(
            split_envs, site_tensors, bars, NB, chi, chi, True, "2x2"
        )
    fused_envs = {c: _split_env_to_tensor_standard(e) for c, e in split_envs.items()}
    return site_tensors, bars, split_envs, fused_envs


def test_fermionic_split_sweep_matches_fused():
    """One fermionic split 2×2 sweep == one fused sweep on the same shared env."""
    chi = 6
    site_tensors, bars, split_envs, fused_envs = _build_shared_fermionic_envs(chi)

    # One fused sweep (the trusted oracle).
    dls = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    fused_new, _eps, _sS = _ctm_tensor_sweep_multisite(
        fused_envs, dls, NB, chi, True, recipe="2x2"
    )

    # One split sweep (fermionic path); lossless interlayer bond.
    split_new = _split_ctm_sweep_multisite_2x2(
        split_envs, site_tensors, bars, NB, chi, 4 * chi
    )

    for c in site_tensors:
        merged = _split_env_to_tensor_standard(split_new[c])
        ref = fused_new[c]
        for field in ("C1", "C2", "C3", "C4", "T1", "T2", "T3", "T4"):
            got = getattr(merged, field)
            exp = getattr(ref, field)
            assert _one_minus_fidelity(got, exp) < 1e-10, (
                f"cell {c} {field}: 1-F={_one_minus_fidelity(got, exp):.3e}"
            )


def test_fermionic_ctm_split_2site_public_path():
    """End-to-end: ``ctm_split_tensor_2site`` runs for a fermionic pair and its
    energy is finite/real.

    Guards the public wiring for fermionic envs — the convergence loop, the
    fermionic dispatch inside ``_split_ctm_sweep_multisite_2x2``, the post-sweep
    ``_renormalize_split_env``, and the split energy path (which merges to
    fused).  Rigorous per-move correctness is covered by
    ``test_fermionic_split_sweep_matches_fused``; here we only assert the entry
    point produces a valid graded environment and a sane energy.
    """
    import jax.numpy as jnp

    from tenax.algorithms._split_ctm_tensor_convergence import ctm_split_tensor_2site
    from tenax.algorithms._split_ctm_tensor_energy import (
        compute_energy_split_ctm_tensor_2site,
    )
    from tenax.algorithms.fermionic_ipeps import FPEPSConfig, spinless_fermion_gate

    A = _make_fp_site(seed=1)
    B = _make_fp_site(seed=2)
    chi = 6
    gate = spinless_fermion_gate(FPEPSConfig(D=2, t=1.0, V=0.5))

    env_A, env_B = ctm_split_tensor_2site(A, B, chi, max_iter=6, conv_tol=0.0)

    # Graded environment preserved (fermionic SymmetricTensors, all fields set).
    for env in (env_A, env_B):
        assert _split_env_is_fermionic_ok(env)

    E = compute_energy_split_ctm_tensor_2site(A, B, env_A, env_B, gate, d=2)
    assert jnp.isfinite(E)
    assert abs(float(jnp.imag(E))) < 1e-8


def _split_env_is_fermionic_ok(env) -> bool:
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _split_env_is_fermionic,
    )

    fields = (
        env.C1,
        env.C2,
        env.C3,
        env.C4,
        env.T1_ket,
        env.T1_bra,
        env.T2_ket,
        env.T2_bra,
        env.T3_ket,
        env.T3_bra,
        env.T4_ket,
        env.T4_bra,
    )
    return _split_env_is_fermionic(env) and all(f is not None for f in fields)


def test_fermionic_edge_resplit_roundtrip():
    """Fused edge → re-split into ket/bra halves → merge back == original."""
    from tenax.algorithms._split_ctm_tensor_convergence import (
        _tensor_env_to_split_standard,
    )

    chi = 6
    site_tensors, _bars, split_envs, fused_envs = _build_shared_fermionic_envs(chi)
    A = site_tensors[(0, 0)]
    fused = fused_envs[(0, 0)]

    resplit = _tensor_env_to_split_standard(fused, A, chi_I=4 * chi)
    remerged = _split_env_to_tensor_standard(resplit)

    for field in ("T1", "T2", "T3", "T4"):
        got = getattr(remerged, field)
        exp = getattr(fused, field)
        assert _one_minus_fidelity(got, exp) < 1e-12, (
            f"{field}: 1-F={_one_minus_fidelity(got, exp):.3e}"
        )


def _rel_corner_gap(merged_env, ref_env):
    """Max relative-norm gap over the four corners (scale-SENSITIVE).

    Corners are not resplit in the split representation, so the merge-back
    preserves their absolute scale exactly — unlike edges, whose SVD resplit is
    only scale/direction-faithful (see ``test_fermionic_edge_resplit_roundtrip``,
    which checks fidelity, not magnitude).  Corners are therefore the honest
    probe of whether the fused env was normalized.
    """
    worst = 0.0
    for field in ("C1", "C2", "C3", "C4"):
        got_t, exp_t = getattr(merged_env, field), getattr(ref_env, field)
        perm = tuple(list(got_t.labels()).index(lbl) for lbl in exp_t.labels())
        got = np.asarray(got_t.transpose(perm).todense()).ravel()
        exp = np.asarray(exp_t.todense()).ravel()
        worst = max(worst, np.linalg.norm(got - exp) / np.linalg.norm(exp))
    return worst


def test_fermionic_split_sweep_honors_renormalize_false():
    """The fermionic ``merge → fused → resplit`` sweep honors ``renormalize=False``.

    Regression for the hard-coded ``renormalize=True`` in the fused routing
    (#690): a caller passing ``renormalize=False`` must get the *raw* fixed-point
    map — identical to a raw (``renormalize=False``) fused sweep on the merged
    environment — not a fused env that was silently normalized before resplit.

    Scale-sensitive on purpose: the bug leaves *direction* unchanged (the fused
    sweep runs either way) and only rescales, so the fidelity-based parity test
    above cannot see it.  We key on the corners, whose absolute scale survives
    the resplit round-trip exactly, and additionally assert the flag genuinely
    toggles behaviour (``renormalize=True`` does *not* reproduce the raw oracle).
    """
    chi = 6
    site_tensors, bars, split_envs, fused_envs = _build_shared_fermionic_envs(chi)

    # Oracle: one RAW (renormalize=False) fused sweep on the merged env.
    dls = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
    fused_raw, _eps, _sS = _ctm_tensor_sweep_multisite(
        fused_envs, dls, NB, chi, False, recipe="2x2"
    )

    # renormalize=False must reproduce the raw fused sweep (scale included).
    split_raw = _split_ctm_sweep_multisite_2x2(
        split_envs, site_tensors, bars, NB, chi, 4 * chi, renormalize=False
    )
    # renormalize=True must NOT — proving the flag is actually consulted.
    split_norm = _split_ctm_sweep_multisite_2x2(
        split_envs, site_tensors, bars, NB, chi, 4 * chi, renormalize=True
    )

    for c in site_tensors:
        gap_raw = _rel_corner_gap(
            _split_env_to_tensor_standard(split_raw[c]), fused_raw[c]
        )
        gap_norm = _rel_corner_gap(
            _split_env_to_tensor_standard(split_norm[c]), fused_raw[c]
        )
        assert gap_raw < 1e-9, (
            f"cell {c}: renormalize=False corner gap {gap_raw:.3e} vs the raw "
            "fused oracle (fused env was normalized despite the flag)"
        )
        assert gap_norm > 1e-3, (
            f"cell {c}: renormalize=True corner gap {gap_norm:.3e} — the "
            "renormalize flag has no effect on the fused routing"
        )
