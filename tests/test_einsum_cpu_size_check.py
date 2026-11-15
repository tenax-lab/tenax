"""On CPU, ``einsum_promoted`` refuses a contraction that holds an array of
2**31 entries or more, and every einsum of the raw RDM builders goes through it
(#1193).

XLA's CPU backend (jax 0.10.2) returns zeros, and raises no error, when it
transposes some arrays of more than 2**31 entries.  ``jnp.einsum`` makes such a
transpose of an operand before a dot.  The raw vertical RDM builder met it on
the spin-1 kagome PESS state (d = 27, D = 2) at chi = 16, and the vertical bond
then gave 0 (``test_raw_rdm_peak_array.py``).  Measured on that transpose
(float32, a one-hot input): 2**31 entries are correct and 2**31 + 4096 entries
are zeros, so the check refuses 2**31 entries and more.

No test makes an array of 2**31 entries.  The tests trace with
``jax.eval_shape``, or call on small operands.  The refusals pin the CPU
device, so they hold on a machine whose default device is a GPU.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms import _einsum_compat
from tenax.algorithms._einsum_compat import einsum_promoted
from tenax.algorithms.ipeps_config import CTMEnvironment
from tenax.algorithms.ipeps_excitations import (
    _rdm1x2_with_open_tensors,
    _rdm2x1_with_open_tensors,
)
from tenax.algorithms.ipeps_rdm import (
    _rdm1x2,
    _rdm1x2_2site,
    _rdm2x1,
    _rdm2x1_2site,
)

LIMIT = 2**31
SD = jax.ShapeDtypeStruct

# A path that makes the outer product of the first two operands first.
_OUTER_FIRST = [(0, 1), (0, 1), (0, 1)]
# a = b = c = 2**8 and d = 2**7: on _OUTER_FIRST the outer product (a, b, c, d)
# has 2**31 entries, while no operand has more than 2**16 and the output is a
# scalar.  The default path holds at most 2**16 entries.
_FOUR = "ab,cd,ac,bd->"
_FOUR_SHAPES = [(2**8, 2**8), (2**8, 2**7), (2**8, 2**8), (2**8, 2**7)]


@pytest.fixture
def on_cpu():
    """Run on the CPU device, the only backend that the check is for."""
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _traced(subscripts, *shapes, **kwargs):
    """The output shape of ``einsum_promoted`` traced on abstract float32
    operands; nothing is allocated."""
    out = jax.eval_shape(
        lambda *ops: einsum_promoted(subscripts, *ops, **kwargs),
        *(SD(shape, jnp.float32) for shape in shapes),
    )
    return out.shape


@pytest.mark.parametrize(
    "subscripts,shapes,kwargs",
    [
        # The output has 2**31 entries; the operands have 2**16 and 2**15.
        pytest.param("i,j->ij", [(2**16,), (2**15,)], {}, id="output"),
        # An operand has 2**31 entries; the output has 2**16.
        pytest.param("ij,j->i", [(2**16, 2**15), (2**15,)], {}, id="operand"),
        # One operand, transposed, as in #1193.
        pytest.param("ij->ji", [(2**16, 2**15)], {}, id="transpose"),
        pytest.param(
            _FOUR, _FOUR_SHAPES, {"optimize": _OUTER_FIRST}, id="intermediate"
        ),
    ],
)
def test_refuses_an_array_of_2pow31_entries(on_cpu, subscripts, shapes, kwargs):
    with pytest.raises(ValueError, match=r"2,147,483,648 entries.*#1193"):
        _traced(subscripts, *shapes, **kwargs)


@pytest.mark.parametrize(
    "subscripts,shapes,kwargs,out",
    [
        # 2**31 - 1 entries (a prime) in an operand and in the output.
        pytest.param(
            "i,j->ij", [(LIMIT - 1,), (1,)], {}, (LIMIT - 1, 1), id="just-below"
        ),
        # The operands of the refused intermediate case, on the default path.
        pytest.param(_FOUR, _FOUR_SHAPES, {}, (), id="default-path"),
    ],
)
def test_allows_fewer_than_2pow31_entries(on_cpu, subscripts, shapes, kwargs, out):
    assert _traced(subscripts, *shapes, **kwargs) == out


def test_eager_calls_are_checked(monkeypatch):
    """An eager call on operands committed to the CPU is checked too, whatever
    the default device.  The limit is lowered to 64 entries, so that the test
    makes no large array."""
    monkeypatch.setattr(_einsum_compat, "_CPU_SIZE_LIMIT", 64)
    cpu = jax.devices("cpu")[0]
    a = jax.device_put(jnp.ones(16), cpu)
    b = jax.device_put(jnp.ones(8), cpu)
    with pytest.raises(ValueError, match="#1193"):
        einsum_promoted("i,j->ij", a, b)
    assert einsum_promoted("i,j->ij", a[:7], b).shape == (7, 8)


def test_other_backends_are_not_checked(monkeypatch):
    """The check is for XLA's CPU backend only: on another platform the
    contraction goes to ``jnp.einsum`` unchecked."""
    monkeypatch.setattr(_einsum_compat, "_call_platform", lambda *args: "gpu")
    assert _traced("i,j->ij", (2**16,), (2**15,)) == (2**16, 2**15)


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


@pytest.mark.parametrize(
    "optimize", ["auto", _OUTER_FIRST], ids=["default", "outer-first"]
)
def test_the_checked_path_is_the_path_that_runs(on_cpu, optimize):
    """The check reads the sizes from a path and gives that path to
    ``jnp.einsum``, so the largest array of the jaxpr is the largest of that
    path."""
    ops = [jnp.ones(shape) for shape in [(4, 4), (4, 2), (4, 4), (4, 2)]]
    _, info = jnp.einsum_path(_FOUR, *ops, optimize=optimize)
    largest = max(info.size_list)
    # Only the outer-first path holds the 128-entry outer product, so a call
    # that ran the other path fails the last assert.
    assert (largest == 4 * 4 * 4 * 2) == (optimize == _OUTER_FIRST)

    def fn(*operands):
        return einsum_promoted(_FOUR, *operands, optimize=optimize)

    assert _largest_array(fn, *ops) == largest


# Forms of the RDM builders, with chi = 5, D**2 = 4 and d = 3, and _FOUR, whose
# default path does not start with the first two operands.
_SPECS = [
    ("ab,buc->auc", [(5, 5), (5, 4, 5)]),
    ("ucxdj,udxrst->crjst", [(4, 5, 4, 4, 5), (4, 4, 4, 4, 3, 3)]),
    ("ab,buc,ce->aue", [(5, 5), (5, 4, 5), (5, 5)]),
    ("auc,axg,gdj->ucxdj", [(5, 4, 5), (5, 4, 5), (5, 4, 5)]),
    ("ijkl,klij->", [(3, 3, 3, 3), (3, 3, 3, 3)]),
    ("ijij->", [(3, 3, 3, 3)]),
    (_FOUR, [(4, 4), (4, 2), (4, 4), (4, 2)]),
]


@pytest.mark.parametrize("subscripts,shapes", _SPECS, ids=[s for s, _ in _SPECS])
@pytest.mark.parametrize("complex_first", [False, True], ids=["real", "mixed"])
def test_results_are_unchanged_bit_for_bit(on_cpu, subscripts, shapes, complex_first):
    """The check changes no bit: the result is that of the call before the
    check, ``jnp.einsum`` on the promoted operands."""
    rng = np.random.default_rng(0)
    ops = [jnp.asarray(rng.standard_normal(shape)) for shape in shapes]
    if complex_first:
        ops[0] = ops[0] * np.exp(0.7j)
    ref_ops = ops
    if len(ops) > 1:
        dtype = jnp.result_type(*ops)
        ref_ops = [op.astype(dtype) for op in ops]
    np.testing.assert_array_equal(
        np.asarray(einsum_promoted(subscripts, *ops)),
        np.asarray(jnp.einsum(subscripts, *ref_ops)),
    )


BUILDERS = [
    _rdm2x1,
    _rdm1x2,
    _rdm2x1_2site,
    _rdm1x2_2site,
    _rdm2x1_with_open_tensors,
    _rdm1x2_with_open_tensors,
]


def _traced_builder(builder, chi, D, d):
    """The RDM shape of ``builder`` traced on abstract complex inputs."""
    c128 = jnp.complex128
    site = SD((D, D, D, D, d), c128)
    env = CTMEnvironment(
        *[SD((chi, chi), c128)] * 4, *[SD((chi, D * D, chi), c128)] * 4
    )
    if builder in (_rdm2x1, _rdm1x2):
        args = (site, env)
    elif builder in (_rdm2x1_2site, _rdm1x2_2site):
        args = (site, site, env, env)
    else:
        ao = SD((D * D,) * 4 + (d, d), c128)
        args = (ao, ao, env)
    return jax.eval_shape(lambda *a: builder(*a, d), *args).shape


def _network_bound(chi, D, d):
    """Entries in the largest array of a raw RDM builder, as
    ``_network_bound`` of ``test_raw_rdm_peak_array.py``: the double-layer
    site, D^8 d^2; a boundary row with three D^2 legs open, chi^2 D^6; a half
    network with one site absorbed, chi^2 D^2 d^2; and the RDM, d^4."""
    return max(D**8 * d**2, chi**2 * D**6, chi**2 * D**2 * d**2, d**4)


@pytest.mark.parametrize(
    "D,d,first",
    [
        # The spin-1 kagome state of #1193 (d > D^2): the half network.
        pytest.param(2, 27, 859, id="D2-d27"),
        # Spin 1/2 at D = 6 (d < D^2): the boundary row.
        pytest.param(6, 2, 215, id="D6-d2"),
    ],
)
@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_rdm_builders_refuse_from_the_first_chi_past_the_limit(
    on_cpu, builder, D, d, first
):
    """Every raw builder passes at ``first - 1`` and is refused at ``first``,
    the first chi whose network bound has 2**31 entries.  The guide quotes
    both values of ``first``."""
    assert _network_bound(first - 1, D, d) < LIMIT <= _network_bound(first, D, d)
    assert _traced_builder(builder, first - 1, D, d) == (d,) * 4
    size = _network_bound(first, D, d)
    with pytest.raises(ValueError, match=rf"{size:,d} entries.*#1193"):
        _traced_builder(builder, first, D, d)


def _einsum_calls(path: Path, base: str | None, name: str) -> list[int]:
    """Lines of the calls ``base.name(...)``, or ``name(...)`` if ``base`` is
    None."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    lines = []
    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        if base is None and isinstance(f, ast.Name) and f.id == name:
            lines.append(n.lineno)
        elif (
            base is not None
            and isinstance(f, ast.Attribute)
            and f.attr == name
            and isinstance(f.value, ast.Name)
            and f.value.id == base
        ):
            lines.append(n.lineno)
    return lines


@pytest.mark.parametrize("mod", ["ipeps_rdm.py", "ipeps_excitations.py"])
def test_every_einsum_of_the_rdm_builders_is_checked(mod):
    """A raw ``jnp.einsum`` in these modules would skip the size check."""
    path = Path(__file__).resolve().parents[1] / "src" / "tenax" / "algorithms" / mod
    assert _einsum_calls(path, None, "einsum_promoted"), f"{mod}: no einsum found"
    offenders = _einsum_calls(path, "jnp", "einsum")
    assert offenders == [], (
        f"{mod} has raw jnp.einsum at lines {offenders}; use einsum_promoted, "
        "which on CPU refuses a contraction of 2**31 entries or more (#1193)"
    )
