"""Optional cuTensorNet backend for GPU-accelerated tensor contractions.

Uses NVIDIA cuTensorNet (from cuQuantum SDK) for multi-tensor contractions
on GPU. Falls back gracefully when cuTensorNet is not available.

Includes a ``jax.custom_vjp`` wrapper so cuTensorNet contractions are
compatible with JAX automatic differentiation (iPEPS AD, etc.).

Requires: ``pip install cuquantum-cu13`` (or cu12 with matching CUDA toolkit).
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

# Lazy import — don't fail at module load time
_cutn_available: bool | None = None
_cutn_contract = None
_cutn_Network = None


def is_available() -> bool:
    """Check if cuTensorNet is available and a CUDA GPU is present."""
    global _cutn_available, _cutn_contract, _cutn_Network

    if _cutn_available is not None:
        return _cutn_available

    try:
        from cuquantum.tensornet import Network, contract

        # Check for CUDA device
        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)

        if not has_gpu:
            _cutn_available = False
            return False

        _cutn_contract = contract
        _cutn_Network = Network
        _cutn_available = True
        logger.info("cuTensorNet backend available")
        return True
    except (ImportError, RuntimeError):
        _cutn_available = False
        return False


def _cutn_contract_raw(subscripts: str, arrays: list) -> jax.Array:
    """Raw cuTensorNet contraction (no AD support).

    Falls back to jnp.einsum if arrays contain JAX tracers (during AD
    backward pass) since DLPack cannot convert traced arrays.
    """
    # Check for tracers — DLPack doesn't work with JAX tracers
    if any(isinstance(a, jax.core.Tracer) for a in arrays):
        return jnp.einsum(subscripts, *arrays)

    import cupy as cp
    from cuquantum.tensornet import contract as cutn_contract_fn

    cp_arrays = []
    for arr in arrays:
        if hasattr(arr, "__dlpack__"):
            cp_arrays.append(cp.from_dlpack(arr))
        else:
            cp_arrays.append(cp.asarray(arr))

    result_cp = cutn_contract_fn(subscripts, *cp_arrays)
    return jnp.from_dlpack(result_cp)


# ---------------------------------------------------------------------------
# JAX-differentiable cuTensorNet contraction via custom_vjp
# ---------------------------------------------------------------------------


def _parse_subscripts(subscripts: str) -> tuple[list[str], str]:
    """Parse 'abc,def->adf' into (['abc', 'def'], 'adf')."""
    input_part, output_part = subscripts.split("->")
    return input_part.split(","), output_part


def _vjp_subscripts(subscripts: str, wrt_idx: int) -> str:
    """Derive the einsum subscript for dL/d(input[wrt_idx]).

    Given forward: subscripts = 'abc,apd,bpxe,def->cxf'
    The VJP w.r.t. input[i] contracts the output gradient with all
    OTHER inputs, producing a tensor with input[i]'s indices.

    Returns the einsum subscript for this VJP contraction.
    """
    input_subs, output_sub = _parse_subscripts(subscripts)
    target_sub = input_subs[wrt_idx]

    # Build VJP subscripts: output_grad + all other inputs -> target shape
    vjp_inputs = [output_sub]
    for j, sub in enumerate(input_subs):
        if j != wrt_idx:
            vjp_inputs.append(sub)

    return ",".join(vjp_inputs) + "->" + target_sub


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def contract_ad(subscripts: str, *arrays: jax.Array) -> jax.Array:
    """cuTensorNet contraction with JAX AD support.

    Forward: uses cuTensorNet for GPU-accelerated contraction.
    Backward: computes VJPs via cuTensorNet contractions of the
    output gradient with the other input tensors.

    Args:
        subscripts: Einsum subscript string.
        *arrays:    Input JAX arrays.

    Returns:
        JAX array with the contraction result.
    """
    return _cutn_contract_raw(subscripts, list(arrays))


def _contract_ad_fwd(
    subscripts: str, *arrays: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    result = _cutn_contract_raw(subscripts, list(arrays))
    return result, arrays  # save inputs for backward


def _contract_ad_bwd(
    subscripts: str, residuals: tuple[jax.Array, ...], g: jax.Array
) -> tuple[jax.Array, ...]:
    arrays = residuals

    grads = []
    for i in range(len(arrays)):
        vjp_sub = _vjp_subscripts(subscripts, i)
        # Contract output gradient with all other inputs
        vjp_arrays = [g] + [arrays[j] for j in range(len(arrays)) if j != i]
        grad_i = _cutn_contract_raw(vjp_sub, vjp_arrays)
        grads.append(grad_i)

    return tuple(grads)


contract_ad.defvjp(_contract_ad_fwd, _contract_ad_bwd)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------


def contract_with_plan(
    subscripts: str,
    arrays: list[Any],
    plan_cache: dict | None = None,
) -> jax.Array:
    """Contract with optional plan caching.

    Uses cuTensorNet for the contraction. NOT AD-compatible —
    use ``contract_ad`` for differentiable contractions.
    """
    return _cutn_contract_raw(subscripts, arrays)
