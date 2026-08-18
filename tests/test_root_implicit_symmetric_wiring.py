"""Wiring for ``ctm_ad_mode="root_implicit_symmetric"`` (#715 Phase 3).

The engine has been built and tested since #729; what was missing was the
optimizer contract, and every gap in it is a *type* gap rather than a numerical
one:

* the engine always returns three values -- there is no ``return_diagnostics``
  flag -- so the loop's two-tuple unpack broke outright;
* its gradient is a ``SymmetricTensor``, and the loop's NaN guard called
  ``jnp.isfinite`` on it, which raises;
* the loop flattened its parameter with ``.todense()``, which would have
  discarded the block structure that is the whole point of Phase 3;
* nothing in the library produces a symmetric initial state, so ``su_init``
  and the random fallback both hand back a dense one.

So the tests that matter here are cheap and structural, and they run against a
**stub engine**: what has to be checked is that the right *object* reaches the
engine and the right object comes back, not that the physics is right -- the
physics is covered by ``test_ctm_root_implicit_symmetric.py``. One end-to-end
descent test carries the real engine and is marked ``slow``; a single symmetric
gradient at D=2, chi=4 costs ~180 s and 4.8 GB (#731).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import ZnSymmetry
from tenax.core.tensor import DenseTensor, SymmetricTensor


def _cfg(*, steps=2, chi=4, mode="root_implicit_symmetric", **ctm_kw):
    base = dict(chi=chi, max_iter=50, conv_tol=1e-10, ctm_ad_mode=mode)
    base.update(ctm_kw)
    return iPEPSConfig(
        max_bond_dim=2,
        unit_cell="1x1",
        ctm=CTMConfig(**base),
        gs_num_steps=steps,
        gs_optimizer="adam",
        gs_learning_rate=1e-2,
        gs_line_search=False,
        gs_metric_precond=False,
    )


def _sym_site(seed: int = 2) -> SymmetricTensor:
    """A small Z2 site tensor with a genuinely non-trivial block structure.

    Built block-wise rather than by projecting a dense array: a random dense
    buffer has entries the charges cannot carry, and ``from_dense`` rejects it
    -- correctly, since silently dropping them would be a different tensor.
    Leg dimensions and flows match ``_convergent_site_tensor`` in
    ``test_ctm_root_implicit_symmetric.py``, which is what the slow test below
    uses, so the two exercise the same layout.
    """
    sym = ZnSymmetry(2)

    def leg(flow, lbl):
        return TensorIndex(
            symmetry=sym,
            sectors=np.array([0, 1]),
            multiplicities=np.array([1, 1]),
            flow=flow,
            label=lbl,
        )

    A = SymmetricTensor.random_normal_np(
        (
            leg(FlowDirection.IN, "u"),
            leg(FlowDirection.OUT, "d"),
            leg(FlowDirection.IN, "l"),
            leg(FlowDirection.OUT, "r"),
            leg(FlowDirection.OUT, "phys"),
        ),
        np.random.RandomState(seed),
    )
    return A * (1.0 / (A.norm() + 1e-12))


def _gate():
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    h = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    return h.reshape(2, 2, 2, 2)


# ------------------------------------------------------------------ #
# A stub engine: the loop's contract, without 180 s of CTM             #
# ------------------------------------------------------------------ #


class _StubEngine:
    """Stands in for ``sym_root_implicit_energy_and_grad``.

    Records what it was handed, and returns the **three**-tuple the real one
    returns, with a ``SymmetricTensor`` gradient on the input's own indices.
    """

    def __init__(self, *, grad_factory=None, energies=None):
        self.calls: list = []
        self.kwargs: list = []
        self._grad_factory = grad_factory or (lambda A: A * 0.1)
        self._energies = list(energies) if energies is not None else None

    def __call__(self, A, gate, **kw):
        self.calls.append(A)
        self.kwargs.append(kw)
        if self._energies:
            energy = jnp.asarray(self._energies.pop(0))
        else:
            energy = jnp.asarray(-0.1 - 0.01 * len(self.calls))
        return energy, self._grad_factory(A), {"converged": True}


@pytest.fixture
def wired(monkeypatch):
    """``optimize_gs_ad_root_implicit`` with the engine and the final CTM stubbed.

    ``_final_env`` converges an ordinary forward CTM and evaluates the energy
    on it; both are real work and neither is what these tests are about.
    """
    import tenax.algorithms._ctm_python_loop as loop_mod
    import tenax.algorithms._ctm_root_implicit_symmetric as sym_mod
    import tenax.algorithms._ctm_tensor_energy as energy_mod
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    engine = _StubEngine()
    monkeypatch.setattr(sym_mod, "sym_root_implicit_energy_and_grad", engine)
    monkeypatch.setattr(
        loop_mod,
        "python_loop_ctm_converge",
        lambda sites, nb, **kw: ({(0, 0): None}, {}),
    )
    monkeypatch.setattr(
        energy_mod, "compute_energy_ctm_tensor", lambda *a, **k: jnp.asarray(-0.5)
    )
    return optimize_gs_ad_root_implicit, engine


# ------------------------------------------------------------------ #
# The initial state: the one thing a caller must supply                #
# ------------------------------------------------------------------ #


def test_a_symmetric_run_without_A_init_says_exactly_what_is_missing():
    """``su_init`` and the random fallback both build *dense* states.

    Falling back to one of them would silently optimise a dense lift of a
    problem the caller posed symmetrically -- and it would "work", which is
    worse.
    """
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    with pytest.raises(ValueError, match="needs an explicit A_init"):
        optimize_gs_ad_root_implicit(_gate(), None, _cfg())


def test_a_dense_A_init_is_refused_and_names_the_mode_that_takes_it():
    from tenax.algorithms._ipeps_optimize_shared import _wrap_as_dense_tensor
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    dense = _wrap_as_dense_tensor(jnp.zeros((2, 2, 2, 2, 2)))
    assert isinstance(dense, DenseTensor)
    with pytest.raises(TypeError, match="root_implicit'"):
        optimize_gs_ad_root_implicit(_gate(), dense, _cfg())


def test_a_symmetric_A_init_is_refused_on_the_DENSE_mode():
    """The mirror, so neither error can be satisfied by a blanket cast."""
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    with pytest.raises(TypeError, match="root_implicit_symmetric"):
        optimize_gs_ad_root_implicit(_gate(), _sym_site(), _cfg(mode="root_implicit"))


# ------------------------------------------------------------------ #
# The parameter must stay symmetric                                    #
# ------------------------------------------------------------------ #


def test_the_optimizer_parameter_never_stops_being_a_symmetric_tensor(wired):
    """``.todense()`` on the parameter would undo Phase 3 silently.

    The run would still converge to *something* -- a dense state with the same
    entries -- so nothing else in the suite would notice.
    """
    optimize, engine = wired
    A = _sym_site()

    optimize(_gate(), A, _cfg(steps=3))

    assert len(engine.calls) >= 3, engine.calls
    for seen in engine.calls:
        assert isinstance(seen, SymmetricTensor), type(seen).__name__
        assert seen._block_keys == A._block_keys
        assert seen.indices == A.indices


def test_the_returned_state_is_the_symmetric_tensor_that_was_optimised(wired):
    optimize, _engine = wired
    A = _sym_site()

    A_opt, _env, _E = optimize(_gate(), A, _cfg(steps=2))

    assert isinstance(A_opt, SymmetricTensor)
    assert A_opt._block_keys == A._block_keys
    # It really moved -- otherwise "stayed symmetric" is trivially true.
    assert float(jnp.linalg.norm(A_opt._data - A._data)) > 0.0


def test_the_engine_is_asked_not_to_trace_its_backward_jaxpr(wired):
    """Six ``make_jaxpr`` traces of ~40k-equation programs, per step.

    Affordable once in a test that asserts on them; pure overhead in a loop.
    """
    optimize, engine = wired
    optimize(_gate(), _sym_site(), _cfg(steps=2))

    assert engine.kwargs, "the engine was never called"
    for kw in engine.kwargs:
        assert kw.get("collect_backward_jaxpr") is False, kw


def test_no_rank_clamp_override_is_forwarded_to_an_engine_that_lacks_it(wired):
    """``rel_floor`` is a dense-1x1-only knob and is now refused, not dropped."""
    optimize, _engine = wired
    with pytest.raises(NotImplementedError, match="rel_floor"):
        optimize(_gate(), _sym_site(), _cfg(rel_floor=1e-8))


def test_a_multisite_unit_cell_is_refused_on_the_symmetric_mode():
    """The combination that used to be harmless and stopped being so.

    ``root_implicit_variant`` dispatches the symmetric mode on ``ctm_ad_mode``
    alone, so ``root_implicit_symmetric`` + ``unit_cell="2site"`` returns
    ``"symmetric"`` and sails past the ``variant == "cell"`` rejection. While
    the symmetric arm was unwired that cost nothing -- both routes hit a
    ``NotImplementedError``. Wiring it made the combination *run*: the 1-site
    engine on one ``A_init``, the configured unit cell silently dropped, and an
    energy returned for a different physical model.
    """
    import dataclasses

    from tenax.algorithms.ipeps_optimize_root_implicit import (
        root_implicit_variant,
        validate_root_implicit_config,
    )

    cfg = dataclasses.replace(_cfg(), unit_cell="2site")
    # The dispatcher still says "symmetric" -- that is its documented job, and
    # pinning it here is what makes the guard below the thing being tested.
    assert root_implicit_variant(cfg) == "symmetric"

    with pytest.raises(NotImplementedError, match="symmetric engine is 1x1 only"):
        validate_root_implicit_config(cfg)


def test_the_symmetric_mode_still_accepts_a_1x1_cell():
    """The guard must not over-reject the only cell that works."""
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        validate_root_implicit_config,
    )

    validate_root_implicit_config(_cfg())


def test_the_dense_variant_still_accepts_the_rank_clamp():
    """The guard must not over-reject: ``rel_floor`` is legitimate on 1x1 dense."""
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        validate_root_implicit_config,
    )

    validate_root_implicit_config(_cfg(mode="root_implicit", rel_floor=1e-8))


# ------------------------------------------------------------------ #
# The NaN guard, through the pytree                                    #
# ------------------------------------------------------------------ #


def _with_data(A, data):
    """Rebuild ``A`` with a new packed buffer, keeping its block structure.

    Through the pytree rather than through a private constructor, so this
    helper cannot drift from what the optimizer loop itself does.
    """
    import jax

    leaves, treedef = jax.tree.flatten(A)
    assert len(leaves) == 1, f"SymmetricTensor is meant to be a single leaf: {leaves}"
    return jax.tree.unflatten(treedef, [data])


def test_a_non_finite_symmetric_gradient_is_counted_and_masked(monkeypatch):
    """``jnp.isfinite`` on a ``SymmetricTensor`` raises; the guard goes by leaves.

    This is not a hypothetical: a NaN gradient on a physical state is #772, and
    it is the failure this guard was written for on the dense path.  Reaching
    it with a symmetric gradient used to raise ``TypeError`` from inside the
    guard itself -- the rescue path crashing instead of rescuing.
    """
    import tenax.algorithms._ctm_python_loop as loop_mod
    import tenax.algorithms._ctm_root_implicit_symmetric as sym_mod
    import tenax.algorithms._ctm_tensor_energy as energy_mod
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    A = _sym_site()
    n_entries = int(A._data.size)

    def poisoned(A_in):
        # One NaN entry: *partly* masked, which the loop must report as a step
        # that still moved, not as a no-op.
        return _with_data(A_in, A_in._data.at[0].set(jnp.nan) * 0.1)

    engine = _StubEngine(grad_factory=poisoned)
    monkeypatch.setattr(sym_mod, "sym_root_implicit_energy_and_grad", engine)
    monkeypatch.setattr(
        loop_mod,
        "python_loop_ctm_converge",
        lambda sites, nb, **kw: ({(0, 0): None}, {}),
    )
    monkeypatch.setattr(
        energy_mod, "compute_energy_ctm_tensor", lambda *a, **k: jnp.asarray(-0.5)
    )

    with pytest.warns(RuntimeWarning, match="non-finite gradient entries") as rec:
        A_opt, _env, _E = optimize_gs_ad_root_implicit(_gate(), A, _cfg(steps=2))

    messages = " ".join(str(w.message) for w in rec)
    # The count must come from the block buffer, not from a dense shape: those
    # differ, and quoting the wrong one makes "all entries masked" wrong too.
    assert f"of {n_entries} non-finite" in messages, messages
    assert "NOT a no-op" in messages, messages
    assert isinstance(A_opt, SymmetricTensor)


def test_a_symmetric_tensor_really_is_a_single_leaf_pytree():
    """The assumption every helper above rests on, asserted rather than trusted.

    If this stopped holding, ``_grad_l2_norm``, ``clip_by_global_norm`` and the
    NaN mask would each silently mean something different.
    """
    import jax

    A = _sym_site()
    leaves = jax.tree.leaves(A)
    assert len(leaves) == 1, [getattr(x, "shape", x) for x in leaves]
    assert leaves[0].shape == A._data.shape
    # And the leaf's L2 norm IS the Frobenius norm, which is what makes
    # ``clip_by_global_norm`` and ``_normalize_params`` mean the right thing.
    assert float(jnp.linalg.norm(leaves[0])) == pytest.approx(float(A.norm()))


# ------------------------------------------------------------------ #
# End to end, with the real engine                                     #
# ------------------------------------------------------------------ #


@pytest.mark.slow
def test_the_symmetric_optimizer_descends_the_energy():
    """The claim that matters, and the only one a stub cannot make.

    Two Adam steps on the Z2 fixture the engine's own tests use.  Marked
    ``slow`` for cost, not for flakiness: one symmetric gradient at D=2, chi=4
    is ~180 s and 4.8 GB even after #731.
    """
    from test_ctm_root_implicit_symmetric import _convergent_site_tensor

    import tenax
    from tenax.algorithms.ipeps_optimize_root_implicit import (
        optimize_gs_ad_root_implicit,
    )

    A = _convergent_site_tensor()
    cfg = _cfg(steps=2, chi=4)

    energies: list[float] = []
    import tenax.algorithms._ctm_root_implicit_symmetric as sym_mod

    real = sym_mod.sym_root_implicit_energy_and_grad

    def recording(A_in, gate, **kw):
        e, g, d = real(A_in, gate, **kw)
        energies.append(float(jnp.real(e)))
        return e, g, d

    sym_mod.sym_root_implicit_energy_and_grad = recording
    try:
        A_opt, env, E = optimize_gs_ad_root_implicit(tenax.heisenberg_gate(), A, cfg)
    finally:
        sym_mod.sym_root_implicit_energy_and_grad = real

    print(f"symmetric root-implicit descent: {energies} -> reported E={E}")
    assert isinstance(A_opt, SymmetricTensor)
    assert len(energies) >= 2, energies
    assert energies[-1] < energies[0], energies
    assert np.isfinite(E)
