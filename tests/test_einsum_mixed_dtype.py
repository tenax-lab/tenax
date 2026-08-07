"""Multi-operand mixed-dtype contractions must not emit a real-by-real GEMM
with a complex output type (#813).

On CUDA that lowering dies with::

    INTERNAL: GEMM is not supported by cublasLt and legacy cublas fallback
    is removed.

It took out 22 gauge tests, which *looked* like a gauge-invariance regression
in the #748 family and was not -- the error is raised before any assertion
runs.

These tests are deliberately **backend-independent**: they inspect the emitted
jaxpr rather than executing the contraction, because the crash only reproduces
on CUDA and CI is CPU-only.  A GPU-only test would guard nothing in the
required gate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._einsum_compat import einsum_promoted

_SPEC = "ab,buc,ce->aue"
_CHI, _D2 = 8, 4


def _operands(*complex_flags):
    rng = np.random.RandomState(0)
    shapes = [(_CHI, _CHI), (_CHI, _D2, _CHI), (_CHI, _CHI)]
    out = []
    for shape, is_c in zip(shapes, complex_flags):
        a = rng.standard_normal(shape)
        out.append(jnp.asarray(a * np.exp(0.7j) if is_c else a))
    return out


def _bad_dots(jaxpr):
    """dot_general equations with all-real operands but a complex output.

    This is the exact shape cuBLASLt refuses: XLA is asked for a real-by-real
    GEMM that must produce complex output, because JAX propagated the final
    result type onto an intermediate whose own operands are both real.
    """
    bad = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name != "dot_general":
            continue
        ins = [v.aval.dtype for v in eqn.invars if hasattr(v, "aval")]
        out = eqn.outvars[0].aval.dtype
        if np.issubdtype(out, np.complexfloating) and not any(
            np.issubdtype(d, np.complexfloating) for d in ins
        ):
            bad.append((ins, out))
    return bad


def test_plain_einsum_emits_the_unsupported_dot():
    """Pins the defect itself, so the fix below is not guarding a phantom.

    If a future JAX stops propagating the complex result type onto a real
    intermediate, this fails and ``einsum_promoted`` can be retired.
    """
    a, b, c = _operands(False, False, True)
    jaxpr = jax.make_jaxpr(lambda x, y, z: jnp.einsum(_SPEC, x, y, z))(a, b, c)
    assert _bad_dots(jaxpr.jaxpr), (
        "expected jnp.einsum to emit a real-by-real dot_general with a complex "
        "output; if it no longer does, #813's workaround is obsolete"
    )


def test_promoted_einsum_emits_no_unsupported_dot():
    """The fix: every dot_general sees operands matching its output type."""
    a, b, c = _operands(False, False, True)
    jaxpr = jax.make_jaxpr(lambda x, y, z: einsum_promoted(_SPEC, x, y, z))(a, b, c)
    assert _bad_dots(jaxpr.jaxpr) == [], (
        f"real-by-real dot_general with complex output survived: "
        f"{_bad_dots(jaxpr.jaxpr)}"
    )


@pytest.mark.parametrize(
    "flags", [(False, False, True), (True, False, False), (True, True, True)]
)
def test_promoted_einsum_matches_jnp_einsum_numerically(flags):
    """Promotion must be a pure lowering change, never a numerical one."""
    ops = _operands(*flags)
    got = einsum_promoted(_SPEC, *ops)
    ref = np.einsum(_SPEC, *[np.asarray(o) for o in ops])
    assert np.max(np.abs(np.asarray(got) - ref)) < 1e-12


def test_all_real_operands_are_not_promoted_to_complex():
    """A blanket cast-to-complex would double flops and memory on every
    real-valued contraction in the RDM builders.  Common-dtype promotion is a
    no-op when the operands already agree."""
    ops = _operands(False, False, False)
    assert einsum_promoted(_SPEC, *ops).dtype == jnp.float64


def _multi_operand_jnp_einsums(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "einsum"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "jnp"
        and len(n.args) >= 4  # subscripts + 3 operands
    ]


@pytest.mark.parametrize("mod", ["ipeps_rdm.py", "ipeps_excitations.py"])
def test_rdm_builders_do_not_reintroduce_raw_multi_operand_einsum(mod):
    """The 15 converted sites must stay converted.

    Only these two modules are pinned: they are the ones the #813 failures
    actually exercised.  Other modules still hold multi-operand einsums and
    remain exposed -- see the issue.
    """
    path = Path(__file__).resolve().parents[1] / "src" / "tenax" / "algorithms" / mod
    offenders = _multi_operand_jnp_einsums(path)
    assert offenders == [], (
        f"{mod} has raw multi-operand jnp.einsum at lines {offenders}; use "
        "einsum_promoted (#813) or these crash on CUDA with mixed dtypes"
    )
