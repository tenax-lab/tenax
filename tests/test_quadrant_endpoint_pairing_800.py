"""The root-implicit quadrants pair the endpoints the production CTM does (#800).

#800 item 2 alleged that ``_upper_left_quadrant`` pairs ``C1.d`` with ``T4.u``
where ``_build_enlarged_corner(..., position="top_left")`` makes ``C1.d``
adjacent to ``T4.d``, with the C4/T3 pairing reversed the same way. The issue
flagged it *unverified*, and said why it is worth checking anyway: every
endpoint has dimension ``chi``, so shape and fixed-point tests cannot see a
swap, and a wrong contraction that still converges to *a* fixed point is the
#700 / #702 failure shape, both of which were real.

**The allegation is false**, and this file is the check rather than the
assertion. What the reading misses is :func:`swap_env_convention`: the
root-implicit module stores every tensor in the frame of its own direction --
which is what makes ``rotate_env`` a pure relabel -- while ``CTMTensorEnv``
closes the same ring with ``C4`` transposed and ``T3``, ``T4`` reversed, and
labels the geometrically *down* end of T4 ``t4_u``. Compared through that map
the two contractions agree bit for bit.

Two things make the check non-vacuous, and both matter more than the verdict:

* the environment is **deliberately asymmetric** -- non-symmetric corners,
  non-palindromic edges, and ``chi != D**2``. A symmetric corner or a
  palindromic edge is invariant under exactly the swap under test, which is how
  #718 hid for months;
* the test measures its own **sensitivity**, by evaluating the alternative
  pairing the issue describes and asserting it differs by an amount this
  comparison would have caught.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tenax.algorithms._ctm_root_implicit_asym as M
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms._ctm_tensor_projector_2x2 import _build_enlarged_corner
from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.symmetry import U1Symmetry
from tenax.core.tensor import DenseTensor

jax.config.update("jax_enable_x64", True)

_D, _d, _CHI = 2, 2, 5  # chi != D**2 on purpose: it separates the two leg kinds


def _site(rng):
    sym = U1Symmetry()
    ch = np.zeros(_D, dtype=np.int32)
    pch = np.zeros(_d, dtype=np.int32)

    def ix(c, flow, label):
        return TensorIndex.from_charges(sym, c.copy(), flow, label=label)

    idx = (
        ix(ch, FlowDirection.OUT, "u"),
        ix(ch, FlowDirection.IN, "d"),
        ix(ch, FlowDirection.OUT, "l"),
        ix(ch, FlowDirection.IN, "r"),
        ix(pch, FlowDirection.IN, "phys"),
    )
    return DenseTensor(jnp.asarray(rng.standard_normal((_D, _D, _D, _D, _d))), idx)


@pytest.fixture(scope="module")
def fixture():
    """An asymmetric environment, its CTM-convention twin, and a double layer."""
    rng = np.random.RandomState(4)
    A = _site(rng)
    template = initialize_ctm_tensor_env(A, _CHI)

    a_t = _build_double_layer_tensor(A)
    labels = list(a_t.labels())
    a_t = a_t.transpose(tuple(labels.index(x) for x in ("u2", "d2", "l2", "r2")))

    d2 = _D * _D
    a = jnp.asarray(rng.standard_normal((d2, d2, d2, d2)))
    a_dense = DenseTensor(a, a_t.indices)

    shapes = {
        "C1": (_CHI, _CHI),
        "C2": (_CHI, _CHI),
        "C3": (_CHI, _CHI),
        "C4": (_CHI, _CHI),
        "T1": (_CHI, d2, _CHI),
        "T2": (_CHI, d2, _CHI),
        "T3": (_CHI, d2, _CHI),
        "T4": (_CHI, d2, _CHI),
    }
    env = M.AsymEnv(
        **{k: jnp.asarray(rng.standard_normal(v)) for k, v in shapes.items()}
    )
    return env, M._to_ctm_env(env, template), a, a_dense


def _as_np(t, order):
    lbls = list(t.labels())
    return np.asarray(t.transpose(tuple(lbls.index(x) for x in order)).todense())


def _rel(x, y):
    return float(np.max(np.abs(x - y)) / max(np.max(np.abs(y)), 1e-300))


def test_the_fixture_can_actually_see_a_transposed_endpoint(fixture):
    """The guard that keeps everything below from being vacuous.

    A symmetric corner satisfies ``C == C.T`` and a palindromic edge satisfies
    ``T == reverse(T)``, so either one is *invariant* under the swap this file
    exists to detect. ``initialize_ctm_tensor_env`` returns a symmetric ``C4``
    and palindromic ``T3``/``T4``, and production single-site CTM converges to
    symmetric corners even from a random ``A`` -- which is exactly why #718
    survived for months.
    """
    env, _ctm, _a, _a_dense = fixture
    for name in ("C1", "C2", "C3", "C4"):
        X = getattr(env, name)
        assert float(jnp.linalg.norm(X - X.T)) > 1.0, name
    for name in ("T1", "T2", "T3", "T4"):
        X = getattr(env, name)
        rev = jnp.transpose(X, (2, 1, 0))
        assert float(jnp.linalg.norm(X - rev)) > 1.0, name


def test_the_upper_left_quadrant_matches_the_production_enlarged_corner(fixture):
    env, ctm, a, a_dense = fixture

    mine = np.asarray(M._upper_left_quadrant(env, a))  # (chi_r, a_r, chi_d, a_d)
    theirs = _as_np(
        _build_enlarged_corner(ctm.C1, ctm.T1, ctm.T4, a_dense, position="top_left"),
        ("chi_R", "r2", "chi_B", "d2"),
    )

    assert mine.shape == theirs.shape
    assert _rel(mine, theirs) < 1e-13, _rel(mine, theirs)


def test_the_lower_left_quadrant_matches_the_production_enlarged_corner(fixture):
    env, ctm, a, a_dense = fixture

    mine = np.asarray(M._lower_left_quadrant(env, a))  # (chi_u, a_u, chi_r, a_r)
    theirs = _as_np(
        _build_enlarged_corner(ctm.C4, ctm.T3, ctm.T4, a_dense, position="bottom_left"),
        ("chi_T", "u2", "chi_R", "r2"),
    )

    assert mine.shape == theirs.shape
    # Looser than the upper-left's bit-for-bit agreement purely because the two
    # contraction orders reassociate differently here.
    assert _rel(mine, theirs) < 1e-13, _rel(mine, theirs)


def test_the_comparison_would_have_caught_the_pairing_the_issue_describes(fixture):
    """Sensitivity, measured rather than asserted by hope.

    Without this the two tests above pass on any comparison loose enough to
    ignore an endpoint swap -- and since every endpoint has dimension ``chi``,
    a swap costs nothing in shape.
    """
    env, ctm, a, a_dense = fixture
    oracle = _as_np(
        _build_enlarged_corner(ctm.C1, ctm.T1, ctm.T4, a_dense, position="top_left"),
        ("chi_R", "r2", "chi_B", "d2"),
    )

    # #800's reading: pair C1 axis 0 with T4 axis 0 rather than axis 2.
    alt = np.asarray(jnp.einsum("ce,efg,cih,fjik->gkhj", env.C1, env.T1, env.T4, a))
    assert _rel(alt, oracle) > 1e-1, _rel(alt, oracle)


def test_dropping_the_convention_swap_is_also_visible(fixture):
    """And the swap itself is load-bearing, not decoration.

    ``swap_env_convention`` is an involution, so applying it once more before
    ``_to_ctm_env`` hands the production code the module's own frame -- the
    pre-#718 state. If that were indistinguishable, the agreement above would
    say nothing about which convention is in force.
    """
    env, _ctm, a, a_dense = fixture
    mine = np.asarray(M._upper_left_quadrant(env, a))

    unswapped = M._to_ctm_env(M.swap_env_convention(env), _template_of(fixture))
    theirs = _as_np(
        _build_enlarged_corner(
            unswapped.C1, unswapped.T1, unswapped.T4, a_dense, position="top_left"
        ),
        ("chi_R", "r2", "chi_B", "d2"),
    )
    assert _rel(theirs, mine) > 1e-1, _rel(theirs, mine)


def _template_of(fixture):
    """Rebuild the env template the fixture used, for the unswapped comparison."""
    rng = np.random.RandomState(4)
    return initialize_ctm_tensor_env(_site(rng), _CHI)
