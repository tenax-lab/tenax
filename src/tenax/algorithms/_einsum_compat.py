"""``jnp.einsum`` with operands promoted to a common dtype first (#813).

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
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["einsum_promoted"]


def einsum_promoted(subscripts, *operands, **kwargs):
    """``jnp.einsum`` with every operand cast to their common dtype first.

    Drop-in replacement for ``jnp.einsum``.  See the module docstring for why
    this is needed on CUDA for three-or-more-operand mixed-dtype contractions.

    All-real operands remain real: ``jnp.result_type`` returns the real dtype
    and ``astype`` is then a no-op, so nothing is promoted to complex
    needlessly.
    """
    if len(operands) > 1:
        dtype = jnp.result_type(*operands)
        operands = tuple(jnp.asarray(op).astype(dtype) for op in operands)
    return jnp.einsum(subscripts, *operands, **kwargs)
