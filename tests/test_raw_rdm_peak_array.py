"""The raw 2-site RDM builders hold no array larger than their network needs.

The raw vertical RDM (``_rdm1x2``, and its copies ``_rdm1x2_2site`` and
``ipeps_excitations._rdm1x2_with_open_tensors``) used to join the two sites
before it absorbed the right edge and the bottom row.  That array held both
sites' ket and bra legs next to two chi and two D^2 legs: chi^2 D^4 d^4
entries.  The horizontal builders never hold both sites' physical legs next to
chi legs.

For the archived spin-1 kagome PESS state (d = 27 for the three spins of the
unit cell, D = 2) at chi = 16, that array has 2,176,782,336 entries (17.4 GB).
On the XLA CPU backend (jax 0.10.2) the next ``jnp.einsum``, which takes it as
an operand, then returned zeros and raised no error.  The vertical bond
contributed 0, and ``compute_energy_ctm`` gave -0.1771777483 per site instead
of -0.3543554967; the #845 warning was the only sign.  The same einsum on the
same data with the operand cut to 2,040,733,440 entries is correct, and a
one-hot operand of each size, float64 and float32, splits the same way.  So
the threshold lies between those two sizes, at about 2**31 entries (measured
2026-11-12).  The zeros come from the transpose that XLA makes of the operand
before the dot.  Under ``jax.jit`` in float32, that transpose alone returns
zeros at the larger size and is correct at the smaller one, and the dot alone,
on an operand already in its layout, is correct at the larger size.  Turning
off XLA's YNNPACK fusion does not change the result.

The failure needs a 17 GB operand, so these tests do not run it.  Like
``test_einsum_mixed_dtype.py`` (#813), they inspect the jaxpr on small shapes:
no builder may produce an array larger than the largest one its network needs.
The horizontal builders, which never had the problem, show that the bound can
be met.  Each shape makes a different factor large (d, D or chi), so an order
that grows too fast in any one of them breaks the bound at one shape at least.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms import ipeps_rdm
from tenax.algorithms.ipeps_config import CTMEnvironment
from tenax.algorithms.ipeps_excitations import (
    _rdm1x2_with_open_tensors,
    _rdm2x1_with_open_tensors,
)
from tenax.algorithms.ipeps_rdm import (
    _build_double_layer_open,
    _rdm1x2,
    _rdm1x2_2site,
    _rdm2x1,
    _rdm2x1_2site,
)

# (chi, D, d).  Within each shape chi, D^2 and d differ, so a wrong pairing
# between two kinds of leg is a shape error, and only a swap between two legs
# of the same kind is left for the values to catch.
SHAPES = [
    pytest.param(3, 2, 5, id="d-heavy"),
    pytest.param(4, 3, 2, id="D-heavy"),
    pytest.param(13, 2, 2, id="chi-heavy"),
]

ONE_SITE = [_rdm2x1, _rdm1x2]
TWO_SITE = [_rdm2x1_2site, _rdm1x2_2site]
OPEN = [_rdm2x1_with_open_tensors, _rdm1x2_with_open_tensors]
VERTICAL = [_rdm1x2, _rdm1x2_2site, _rdm1x2_with_open_tensors]


def _network_bound(chi, D, d):
    """Entries in the largest array a 2-site RDM network has to hold.

    ``ao`` itself, ``(D^2, D^2, D^2, D^2, d, d)``; a boundary row with its
    three D^2 legs open (C1-T1-C2 with T4 and T2 attached), chi^2 D^6; a half
    network with one site absorbed, chi^2 D^2 d^2; and the RDM, d^4.
    """
    return max(D**8 * d**2, chi**2 * D**6, chi**2 * D**2 * d**2, d**4)


def _old_vertical_peak(chi, D, d):
    """The largest arrays of the old vertical order: the joined sites,
    chi^2 D^4 d^4, and T4 with the bottom site, chi^2 D^6 d^2."""
    return max(chi**2 * D**4 * d**4, chi**2 * D**6 * d**2)


def _largest_array(fn, *args):
    """Entries in the largest array that an equation of ``fn``'s jaxpr makes.

    Nested jaxprs count too: each ``jnp.einsum`` is a jitted call.
    """

    def walk(jaxpr):
        n = 0
        for eqn in jaxpr.eqns:
            n = max([n, *(math.prod(v.aval.shape) for v in eqn.outvars)])
            for param in eqn.params.values():
                for sub in param if isinstance(param, (tuple, list)) else (param,):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns"):
                        n = max(n, walk(sub))
        return n

    return walk(jax.make_jaxpr(fn)(*args).jaxpr)


def _random(rng, shape):
    return jnp.asarray(rng.standard_normal(shape) + 1j * rng.standard_normal(shape))


def _env(rng, chi, D):
    corners = [_random(rng, (chi, chi)) for _ in range(4)]
    edges = [_random(rng, (chi, D * D, chi)) for _ in range(4)]
    return CTMEnvironment(*corners, *edges)


def _site(rng, D, d):
    return _random(rng, (D, D, D, D, d))


def _call(builder, rng, chi, D, d):
    """``(fn, args)``: the builder on random inputs, ``fn`` taking arrays only."""
    if builder in ONE_SITE:
        args = (_site(rng, D, d), _env(rng, chi, D))
        return (lambda A, env: builder(A, env, d)), args
    if builder in TWO_SITE:
        args = (
            _site(rng, D, d),
            _site(rng, D, d),
            _env(rng, chi, D),
            _env(rng, chi, D),
        )
        return (lambda A, B, env_A, env_B: builder(A, B, env_A, env_B, d)), args
    args = (
        _build_double_layer_open(_site(rng, D, d)),
        _build_double_layer_open(_site(rng, D, d)),
        _env(rng, chi, D),
    )
    return (lambda ao1, ao2, env: builder(ao1, ao2, env, d)), args


@pytest.mark.parametrize(
    "builder", ONE_SITE + TWO_SITE + OPEN, ids=lambda b: b.__name__
)
@pytest.mark.parametrize("chi,D,d", SHAPES)
def test_no_raw_rdm_builder_holds_an_array_past_its_network_bound(builder, chi, D, d):
    """An array that grows like chi^2 D^4 d^4 passes 2**31 entries at chi = 16
    for d = 27, and XLA's CPU einsum then returns zeros (module docstring)."""
    bound = _network_bound(chi, D, d)
    # The regime: the old vertical order must break the bound at this shape,
    # or this shape could not tell the two orders apart.
    assert _old_vertical_peak(chi, D, d) > bound
    fn, args = _call(builder, np.random.default_rng(0), chi, D, d)
    peak = _largest_array(fn, *args)
    # Every builder holds a half network, so a smaller peak means the walk
    # missed arrays, and the upper bound below would pass for nothing.
    assert peak >= chi**2 * D**2 * d**2, f"the jaxpr walk saw only {peak:,d} entries"
    assert peak <= bound, (
        f"{builder.__name__} makes an array of {peak:,d} entries; its network "
        f"needs at most {bound:,d} (chi={chi}, D={D}, d={d})"
    )


# One np.einsum over the documented 1x2 network: C1, T1, C2, T4, T2 and ao_top
# for the top site; T4, T2, ao_bot, C4, T3 and C3 for the bottom one.  The
# stacked edges join top t4_u to bottom t4_d (f) and top t2_d to bottom t2_u
# (g); ao_top.d joins ao_bot.u (d).  Output (ket1, ket2, bra1, bra2).
_NETWORK_1X2 = "ab,buc,ce,alf,erg,udlrst,fmh,gni,dqmnwx,hj,jqk,ik->swtx"


def _np_ao(A):
    A = np.asarray(A)
    shape = (A.shape[0] ** 2,) * 4 + (A.shape[4],) * 2
    return np.einsum("udlrs,UDLRt->uUdDlLrRst", A, A.conj()).reshape(shape)


def _oracle_1x2(ao_top, ao_bot, env_top, env_bot):
    """The vertical RDM from one np.einsum; it shares no code with the builders."""
    C1, C2, _, _, T1, T2, _, T4 = (np.asarray(x) for x in env_top)
    _, _, C3, C4, _, T2_bot, T3, T4_bot = (np.asarray(x) for x in env_bot)
    # The greedy path caps each intermediate at the largest operand by default.
    # At the chi-heavy shape no pair fits under that cap, and numpy then
    # contracts eight operands in one naive loop, which does not finish.
    return np.einsum(
        _NETWORK_1X2,
        *(C1, T1, C2, T4, T2, ao_top, T4_bot, T2_bot, ao_bot, C4, T3, C3),
        optimize=("greedy", 2**40),
    )


@pytest.mark.parametrize("builder", VERTICAL, ids=lambda b: b.__name__)
@pytest.mark.parametrize("chi,D,d", SHAPES)
def test_vertical_rdm_matches_the_documented_network(builder, chi, D, d, monkeypatch):
    """The contraction order may change; the network may not.

    Every field is random and complex, with no lattice symmetry, so a swap of
    two legs of the same kind changes the values.
    """
    # The contraction is under test, not the normaliser, whose validity check
    # fires on a random environment.
    monkeypatch.setattr(
        ipeps_rdm, "_normalise_rdm_for_energy", lambda mat, context="": mat
    )
    fn, args = _call(builder, np.random.default_rng(1), chi, D, d)
    got = np.asarray(fn(*args)).reshape((d,) * 4)
    if builder is _rdm1x2:
        A, env = args
        ref = _oracle_1x2(_np_ao(A), _np_ao(A), env, env)
    elif builder is _rdm1x2_2site:
        A, B, env_A, env_B = args
        ref = _oracle_1x2(_np_ao(A), _np_ao(B), env_A, env_B)
    else:
        ao1, ao2, env = args
        ref = _oracle_1x2(np.asarray(ao1), np.asarray(ao2), env, env)
    err = float(np.max(np.abs(got - ref)) / np.max(np.abs(ref)))
    assert err < 1e-12, f"{builder.__name__} is off the network by {err:.2e}"
