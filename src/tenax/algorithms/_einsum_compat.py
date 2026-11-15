"""``jnp.einsum`` with operands promoted to a common dtype first (#813), and on
CPU a size check (#1193).

Multi-operand contractions that mix real and complex operands crash on CUDA::

    INTERNAL: GEMM is not supported by cublasLt and legacy cublas fallback
    is removed.

The cause is not the mixed dtypes themselves but *where* JAX puts the
promotion.  Given ``einsum("ab,buc,ce->aue", C1, T1, C2)`` with only ``C2``
complex, the emitted jaxpr is::

    a:f64[8,8]  b:f64[8,4,8]  c:c128[8,8]
    d:c128[4,8,8] = dot_general[preferred_element_type=complex128] b a
    e:c128[4,8,8] = dot_general[preferred_element_type=complex128] d c

The *first* ``dot_general`` contracts two **real** operands while carrying the
final complex result type.  XLA is therefore asked for a real-by-real GEMM
with a complex output, cuBLASLt has no such kernel, and the legacy cuBLAS
fallback that used to absorb it has been removed (jaxlib 0.10.2).

Binary contractions never hit this -- JAX promotes the two operands directly,
so each ``dot_general`` sees operands matching its output type.  It takes three
or more operands for a real-by-real intermediate to inherit a complex result
type, which is why it surfaced in the RDM builders rather than anywhere simpler.

Promoting up front restores that invariant for every intermediate.  This is
semantically identical to calling ``jnp.einsum`` directly -- einsum promotes
anyway -- and the cast is a no-op when the dtypes already agree, so all-real
contractions stay real and pay nothing.

On CPU, XLA (jax 0.10.2) returns zeros, and raises no error, when it transposes
some arrays of more than 2**31 entries, and ``jnp.einsum`` makes such a
transpose of an operand before a dot (#1193).  Measured on the transpose that
#1193 met, 2**31 entries are correct and 2**31 + 4096 entries are zeros.  So on
CPU, a contraction in which an operand, an intermediate or the output has 2**31
entries or more raises ``ValueError``.  The sizes come from the contraction
path, and that same path goes to ``jnp.einsum``, so the check reads the path
that runs.  It costs one ``opt_einsum`` path per call, once per trace under
``jit``.  Every einsum of the raw RDM builders (``ipeps_rdm``,
``ipeps_excitations``) goes through here for this check, binary ones too.

The check does not see two things.  Under ``jax.vmap`` an operand has its
per-example shape, so the batched arrays are larger than the checked ones.
Under ``jax.jit`` an operand is a tracer with no device, so the check takes the
default device.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._block_sparse_jit import _call_platform

__all__ = ["einsum_promoted"]

# On CPU, a contraction that holds an array of this many entries or more is
# refused (module docstring).
_CPU_SIZE_LIMIT = 2**31


def einsum_promoted(subscripts, *operands, **kwargs):
    """``jnp.einsum`` with every operand cast to their common dtype first.

    Drop-in replacement for ``jnp.einsum``.  See the module docstring for why
    this is needed on CUDA for three-or-more-operand mixed-dtype contractions.

    All-real operands remain real: ``jnp.result_type`` returns the real dtype
    and ``astype`` is then a no-op, so nothing is promoted to complex
    needlessly.

    Raises:
        ValueError: On CPU, if an operand, an intermediate or the output has
            2**31 entries or more (#1193; see the module docstring).
    """
    if len(operands) > 1:
        dtype = jnp.result_type(*operands)
        operands = tuple(jnp.asarray(op).astype(dtype) for op in operands)
    if _on_cpu(operands):
        path, info = jnp.einsum_path(
            subscripts, *operands, optimize=kwargs.pop("optimize", "auto")
        )
        largest = max([*(jnp.size(op) for op in operands), *info.size_list])
        if largest >= _CPU_SIZE_LIMIT:
            shapes = [tuple(jnp.shape(op)) for op in operands]
            raise ValueError(
                f"einsum {subscripts!r} on shapes {shapes}: an operand, an "
                f"intermediate or the output has {largest:,d} entries, 2**31 "
                "or more.  XLA on CPU returns zeros, and raises no error, when "
                "it transposes some arrays this large (#1193).  Use a smaller "
                "chi or D, or a GPU."
            )
        kwargs["optimize"] = path
    return jnp.einsum(subscripts, *operands, **kwargs)


def _on_cpu(operands) -> bool:
    """Whether the contraction runs on XLA's CPU backend.

    A tracer has no device: under ``jit`` the call runs where the whole
    program runs, and ``_call_platform`` takes that from the default device.
    """
    concrete = [op for op in operands if not isinstance(op, jax.core.Tracer)]
    return _call_platform(concrete, {}, None) == "cpu"
