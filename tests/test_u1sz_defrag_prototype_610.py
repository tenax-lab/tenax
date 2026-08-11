"""Faithfulness guard for the #610 C-lever prototype (Stage 2 prereq)."""

import importlib.util
import pathlib

import jax
import numpy as np
import pytest

from examples.u1sz_defrag_prototype_610 import sector_dropping_truncation
from tenax import compute_energy_ctm_tensor
from tenax.algorithms._ctm_diagnostics import ctm_corner_rank
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

# ``compute_energy_ctm_tensor`` returns ``E_h + E_v`` with ``H = S_i . S_j``,
# whose eigenvalues are -3/4 (singlet) and +1/4 (triplet).  Two bonds, so any
# genuine expectation value lies in [-1.5, +0.5] whatever the state and however
# poorly converged the environment: an inaccurate RDM is still a density
# matrix.  Leaving this interval means the RDM is not one, which is a different
# and worse failure than an inaccurate energy.
E_MIN, E_MAX = -1.5, 0.5


def test_prototype_env_is_not_collapsed_and_energy_is_physical():
    """The #610 sector-dropping prototype must not destroy the environment.

    This replaces an assertion of ``-2.0 < e < 0.0`` that was passing for the
    wrong reason.  ``ctm_tensor``'s ``recipe`` default changed ``1x1`` -> ``2x2``
    in #765 (fixing #723), and this test passes no ``recipe``, so it silently
    switched environments.  Measured at D=3, chi=12, the same call under each::

        recipe   criterion      corner_rank   energy      old assertion
        1x1      0.000000e+00   1             -0.018152   passes
        2x2      1.256895e-01   5             +0.140116   fails

    The ``1x1`` criterion is *exactly* zero because a rank-1 corner is an
    absorbing state: the environment stops changing, so the sweep reports
    convergence.  That is the #723/#726 chi_eff=1 mean-field collapse, the same
    defect behind the #747/#771 retractions.  So the old green was produced by a
    collapsed environment, and #765 did not break this test -- it removed the
    thing that was making it pass.

    Hence the two assertions that carry weight here:

    * **The corner is not rank-1.**  ``ctm_corner_rank`` has shipped since #747
      for exactly this, and would have caught the old environment rather than
      blessing it.
    * **The energy is physical**, not negative.  ``A`` is ``random_normal`` and
      is never optimised -- the fixture's own docstring says AFM correlations
      "emerge from optimization", which this test does not do -- so
      ``<S_i . S_j> ~ 0`` with no preferred sign, and the old upper bound of 0.0
      excluded the entire allowed positive half.

    Convergence is deliberately **not** asserted, and this test is no longer
    named as though it were: under ``2x2`` the criterion does not reach
    ``conv_tol`` even at 400 sweeps and is not monotone (3.9e-2 at 200, back up
    to 8.8e-2 at 400).  See #853.
    """
    jax.config.update("jax_enable_x64", True)
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    with sector_dropping_truncation(keep={-1, 0, 1}):
        env, _ = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7)
        for name in env._fields:
            t = getattr(env, name)
            assert np.all(np.isfinite(np.asarray(t._data))), f"{name} non-finite"
        e = float(compute_energy_ctm_tensor(A, env, heisenberg_gate_u1sz()))

    rank = ctm_corner_rank(env)
    assert rank > 1, (
        f"CTM environment collapsed to a rank-{rank} corner: this is a "
        f"chi_eff=1 mean-field boundary, not a corner transfer matrix (#723, "
        f"#726, #747). Any energy read off it is meaningless, and this test "
        f"passed on exactly such an environment before #765 fixed the "
        f"ctm_tensor recipe default."
    )
    assert np.isfinite(e), "prototype energy non-finite"
    assert E_MIN <= e <= E_MAX, (
        f"prototype energy {e} is outside the physically attainable range "
        f"[{E_MIN}, {E_MAX}] for E_h + E_v with H = S_i . S_j. An unconverged "
        f"environment gives an inaccurate energy, not an impossible one, so "
        f"this means the RDM is not a valid density matrix. See #853."
    )


#: The doubled *virtual* legs of the environment tensors.  These carry the
#: site tensor's own charge structure (D=3 with pattern [0, +1, -1] fuses to
#: +/-2) and are **not** chi bonds, so the prototype neither does nor should
#: drop them.
_D2_LEGS = {"u2", "d2", "l2", "r2"}

KEEP = {-1, 0, 1}


def _chi_bond_charges(env) -> dict[str, list[int]]:
    """Charge set of every **chi bond** in the environment, by ``tensor.leg``."""
    out = {}
    for name in env._fields:
        t = getattr(env, name)
        for idx in t.indices:
            if idx.label in _D2_LEGS:
                continue
            out[f"{name}.{idx.label}"] = sorted(
                {int(c) for c in np.asarray(idx.charges).ravel()}
            )
    return out


def test_every_chi_bond_carries_only_kept_charges():
    """The prototype's actual contract: no chi bond outside ``keep``.

    Added after review of the guard rewrite (#855): ``rank > 1`` and a physical
    energy interval are both satisfied by a **baseline** environment, and the
    block-count guard below asserts *counts* (5 -> 3, 19 -> 9), which can fall
    without the excluded charges being gone.  So nothing in this file verified
    the thing the prototype exists to do.

    Measured per leg at D=3 chi=12, which is also the negative control::

        leg          dim   dropping=False       dropping=True
        C1.c1_r       12   [-2,-1, 0, 1, 2]     [-1, 0, 1]
        C1.c1_d       12   [-2,-1, 0, 1, 2]     [-1, 0, 1]
        T1.t1_l       12   [-2,-1, 0, 1, 2]     [-1, 0, 1]
        T1.t1_r       12   [-2,-1, 0, 1, 2]     [-1, 0, 1]
        T1.u2          9   [-2,-1, 0, 1, 2]     [-2,-1, 0, 1, 2]   <- D^2 leg

    Every chi bond is exactly ``keep`` under the prototype and carries the full
    five-sector structure without it, so this discriminates.  ``T1.u2`` is the
    site's fused virtual leg rather than a chi bond and is excluded -- dropping
    it would be wrong, and a check that demanded it would fail on correct
    behaviour.
    """
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    with sector_dropping_truncation(keep=KEEP):
        env, _diff = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7)

    charges = _chi_bond_charges(env)
    assert charges, "no chi bonds found -- the leg-classification is wrong"
    offenders = {leg: cs for leg, cs in charges.items() if set(cs) - KEEP}
    assert not offenders, (
        f"chi bonds carry charges outside keep={sorted(KEEP)}: {offenders}. "
        f"The prototype's truncation did not take effect on those legs, so an "
        f"environment that looks healthy by rank and energy is really the "
        f"un-dropped baseline."
    )


def test_prototype_actually_drops_sectors():
    # Under the prototype, the env block counts must drop toward the Gate-A
    # static prediction: corners 5->3, edges 19->9 at D=3 chi=12.
    A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
    with sector_dropping_truncation(keep={-1, 0, 1}):
        env, _ = ctm_tensor(A, chi=12, max_iter=8, conv_tol=1e-6)
    nblocks = {n: len(getattr(env, n)._block_keys) for n in env._fields}
    # edges must have strictly fewer blocks than the fragmented baseline (19)
    assert nblocks["T1"] < 19, f"edge sectors not dropped: {nblocks}"
    assert nblocks["C1"] <= 5, f"corner sectors not dropped: {nblocks}"


@pytest.mark.xfail(
    reason=(
        "#610 (stale obstruction, independent of #700): with the current "
        "env-init chi-bond charge structure the documented mixed-generation "
        "chi-bond mismatch ('does not match previous') no longer trips — the "
        "surgical-drop backward VJP traces without raising. #700's vacuum-tiling "
        "fix does not restore the obstruction (it was pinning a since-changed "
        "env-init detail, not the #700 collapse). The real #610 structural fix "
        "(env-init + per-direction refresh emitting a uniform sector set) is "
        "what would flip this test."
    ),
    strict=False,
)
def test_backward_trace_does_not_survive_surgical_drop():
    """Gate-B Step-0 obstruction guard (#610): documents that the SURGICAL
    traced-path sector drop (filtered backward base_charges + greedy-backfill-
    suppressed traced SVD that honestly emits a smaller chi_new) STILL cannot be
    traced through the multisite 2x2 backward.

    Root cause (NO-GO-by-obstruction): the multisite sweep refreshes only the
    legs along the current absorption direction per move, while env-init seeds
    every chi bond at the full 5-sector {-2,-1,0,1,2} structure.  A later
    direction's enlarged-corner build then contracts a freshly-dropped (3-sector)
    leg against a perpendicular leg still carrying the full env-init bond ->
    opt_einsum size mismatch.  This pins the obstruction so a future structural
    fix (env-init + per-direction refresh emitting a uniform sector set) is
    recognised as the thing that flips this test.
    """
    jax.config.update("jax_enable_x64", True)
    spec = importlib.util.spec_from_file_location(
        "probe_backward_jaxpr_566",
        pathlib.Path(__file__).resolve().parents[1]
        / "examples"
        / "probe_backward_jaxpr_566.py",
    )
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    A = probe.make_site("u1sz", 3)
    chi = 12

    # Baseline (no prototype) traces fine.
    jx = probe.backward_vjp_jaxpr(A, chi, "auto")
    assert hasattr(jx, "jaxpr") or hasattr(jx, "eqns")

    # IMPORTANT: the monkeypatch only takes effect if the patched CTM-step is
    # traced COLD under the patch. The baseline call above seeds the jax.jit
    # compilation cache (identical input avals + same _make_jit_ctm_step
    # closure), so without clearing it the prototype-wrapped call would reuse the
    # un-patched cached trace and silently NOT raise. Clear caches so the patched
    # path re-traces — mirrors the standalone `--defrag` probe, which runs cold.
    jax.clear_caches()

    # Under the surgical drop the single-sweep VJP fails to trace with the
    # documented mixed-generation chi-bond mismatch.
    with pytest.raises(ValueError, match="does not match previous"):
        with sector_dropping_truncation(keep={-1, 0, 1}):
            probe.backward_vjp_jaxpr(A, chi, "auto")
