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

``ipeps()`` knows the largest array of its energy step from chi, D and d, so
on CPU it compares that bound with the limit before its simple update, and
raises at once rather than after the run.

No test makes an array of 2**31 entries.  The tests trace with
``jax.eval_shape``, or call on small operands, or lower the limit.  The
refusals pin the CPU device, so they hold on a machine whose default device is
a GPU.
"""

from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms import _einsum_compat, ipeps_rdm
from tenax.algorithms._einsum_compat import einsum_promoted
from tenax.algorithms.ipeps import ipeps
from tenax.algorithms.ipeps_config import CTMConfig, CTMEnvironment, iPEPSConfig
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

# The module, not the function: ``tenax.algorithms`` caches its lazy names, so
# after ``from tenax.algorithms import ipeps`` has run before the submodule
# loaded, ``import tenax.algorithms.ipeps as m`` gives the function ``ipeps``.
ipeps_module = importlib.import_module("tenax.algorithms.ipeps")

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
    the first chi whose bound, ``_raw_rdm_peak_entries``, has 2**31 entries.
    The guide quotes both values of ``first``."""
    bound = ipeps_rdm._raw_rdm_peak_entries
    assert bound(first - 1, D, d) < LIMIT <= bound(first, D, d)
    assert _traced_builder(builder, first - 1, D, d) == (d,) * 4
    size = bound(first, D, d)
    with pytest.raises(ValueError, match=rf"{size:,d} entries.*#1193"):
        _traced_builder(builder, first, D, d)


# One shape for each term of the bound, (chi, D, d), with the index of the
# term that is the largest there: the double-layer site D^8 d^2, a boundary
# row chi^2 D^6, a half network chi^2 D^2 d^2, and the RDM d^4.
_REGIMES = [
    pytest.param(2, 3, 2, 0, id="site"),  # 26244, 2916, 144, 16
    pytest.param(8, 2, 2, 1, id="boundary-row"),  # 1024, 4096, 1024, 16
    pytest.param(8, 1, 5, 2, id="half-network"),  # 25, 64, 1600, 625
    pytest.param(2, 1, 6, 3, id="rdm"),  # 36, 4, 144, 1296
]


@pytest.mark.parametrize("chi,D,d,term", _REGIMES)
@pytest.mark.parametrize("builder", BUILDERS, ids=lambda b: b.__name__)
def test_the_bound_is_the_largest_array_of_every_builder(
    on_cpu, monkeypatch, builder, chi, D, d, term
):
    """With the limit lowered to the bound, every builder is refused and the
    message gives the bound; with one entry more, it passes.  So the bound is
    the largest array of each builder, at a shape where each term in turn is
    the largest."""
    terms = [D**8 * d**2, chi**2 * D**6, chi**2 * D**2 * d**2, d**4]
    assert sorted(terms)[-2] < terms[term] == max(terms)
    bound = ipeps_rdm._raw_rdm_peak_entries(chi, D, d)
    monkeypatch.setattr(_einsum_compat, "_CPU_SIZE_LIMIT", bound)
    with pytest.raises(ValueError, match=rf"{bound:,d} entries.*#1193"):
        _traced_builder(builder, chi, D, d)
    monkeypatch.setattr(_einsum_compat, "_CPU_SIZE_LIMIT", bound + 1)
    assert _traced_builder(builder, chi, D, d) == (d,) * 4


class _SweepRan(Exception):
    """The stub simple update ran: ``ipeps()`` got past its size check."""


def _stub_sweep(monkeypatch):
    """Replace the simple update of ``ipeps()`` with a stub that raises
    ``_SweepRan``."""

    def sweep(*args, **kw):
        raise _SweepRan

    monkeypatch.setattr(ipeps_module, "_simple_update_checkerboard_sweep", sweep)


def _small_ipeps(monkeypatch, limit, gate=None, **kwargs):
    """``ipeps()`` at chi = 9, D = 2 and d = 5, with the CPU limit lowered to
    ``limit`` and the simple update replaced by a stub that raises
    ``_SweepRan``.  The gate is zero by default: the run stops before the
    gate acts.  The energy step of this run holds 8100 entries at most, in
    the half network, chi^2 D^2 d^2."""
    monkeypatch.setattr(_einsum_compat, "_CPU_SIZE_LIMIT", limit)
    _stub_sweep(monkeypatch)
    config = iPEPSConfig(max_bond_dim=2, num_imaginary_steps=1, ctm=CTMConfig(chi=9))
    gate = jnp.zeros((5,) * 4) if gate is None else gate
    return ipeps(gate, None, config, **kwargs)


def test_ipeps_refuses_before_the_simple_update(on_cpu, monkeypatch):
    """On CPU, ``ipeps()`` compares the bound of its energy step with the
    limit before the simple update runs, and the message names the ways out.
    With ``below-the-limit`` below, it pins the bound of this run at 8100."""
    bound = ipeps_rdm._raw_rdm_peak_entries
    # chi, D and d each change the bound at this shape, so the message shows
    # which values ipeps() gives to the bound.
    assert bound(9, 2, 5) == 8100
    assert 8100 not in (bound(2, 2, 5), bound(9, 5, 2), bound(9, 2, 2))
    with pytest.raises(
        ValueError,
        match=r"chi=9, D=2 and d=5.* 8,100 entries.*#1193.*compute_energy=False",
    ):
        _small_ipeps(monkeypatch, 8100)


@pytest.mark.parametrize(
    "limit,kwargs,platform",
    [
        pytest.param(8101, {}, "cpu", id="below-the-limit"),
        pytest.param(8100, {"compute_energy": False}, "cpu", id="no-energy-step"),
        pytest.param(8100, {}, "gpu", id="off-cpu"),
    ],
)
def test_ipeps_runs_when_its_energy_step_is_allowed(
    on_cpu, monkeypatch, limit, kwargs, platform
):
    """Below the limit, with ``compute_energy=False`` (no CTM and no energy
    step), or off CPU, ``ipeps()`` goes on to its simple update."""
    monkeypatch.setattr(_einsum_compat, "_call_platform", lambda *args: platform)
    with pytest.raises(_SweepRan):
        _small_ipeps(monkeypatch, limit, **kwargs)


def _platform_by_commitment(committed, default):
    """A ``_call_platform`` with its own rule, the platform of a committed
    leaf and else that of the default device, but with the two platforms
    named by the test: this host has a CPU only."""

    def platform(args, kwargs, pinned):
        leaves = jax.tree.leaves((args, kwargs))
        if any(getattr(x, "committed", False) for x in leaves):
            return committed
        return default

    return platform


@pytest.mark.parametrize(
    "committed,default,expected",
    [
        pytest.param("cpu", "gpu", ValueError, id="gate-on-cpu"),
        pytest.param("gpu", "cpu", _SweepRan, id="gate-on-other-platform"),
    ],
)
def test_ipeps_reads_the_platform_of_the_gate_too(
    on_cpu, monkeypatch, committed, default, expected
):
    """The sites of a random start are not committed, so the simple update
    runs where a committed gate is.  The check reads that platform: a gate
    on CPU is checked when the default device is not a CPU, and a gate on
    another platform is not checked when it is."""
    monkeypatch.setattr(
        _einsum_compat, "_call_platform", _platform_by_commitment(committed, default)
    )
    gate = jax.device_put(jnp.zeros((5,) * 4), jax.devices("cpu")[0])
    assert gate.committed
    with pytest.raises(expected):
        _small_ipeps(monkeypatch, 8100, gate=gate)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param((np.int32(215), np.int32(6), np.int32(2)), id="int32"),
        pytest.param((215.0, 6.0, 2.0), id="float"),
    ],
)
def test_the_bound_is_a_python_int(args):
    """An ``int32`` chi, D or d does not overflow, and the bound is an
    ``int``, which the message formats with ``:,d``.  At chi = 215, D = 6
    and d = 2 the boundary row has 2,156,673,600 entries, past the int32
    range."""
    size = ipeps_rdm._raw_rdm_peak_entries(*args)
    assert type(size) is int
    assert size == 2_156_673_600


def test_ipeps_refuses_at_the_real_limit_with_an_int32_chi(on_cpu, monkeypatch):
    """At the real limit, ``ipeps()`` refuses chi = 215 at D = 6 and d = 2,
    the first refused chi of the guide, also when chi is an ``np.int32``."""
    _stub_sweep(monkeypatch)
    config = iPEPSConfig(
        max_bond_dim=6, num_imaginary_steps=1, ctm=CTMConfig(chi=np.int32(215))
    )
    with pytest.raises(
        ValueError, match=r"chi=215, D=6 and d=2.* 2,156,673,600 entries"
    ):
        ipeps(jnp.zeros((2,) * 4), None, config)


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
