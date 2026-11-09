"""Raw-array views of CTM environments for the dense-only code paths.

``optimize_gs_ad`` and ``ctm_tensor`` return Tensor-protocol objects: a
``DenseTensor`` site tensor and a ``CTMTensorEnv`` whose 8 fields are
``DenseTensor``s.  The excitation contractions and the raw-array result of
``ctm()`` use plain ``jax.Array``s and a raw-array ``CTMEnvironment``.  The
fields of a ``CTMTensorEnv`` share the leg and flow convention of the raw
``CTMEnvironment`` (verified to machine precision against
``compute_energy_ctm``), so ``.todense()`` on each field is correct and keeps
the converged environment.

For a ``DenseTensor``, ``.todense()`` returns the stored buffer, so it is free.
A ``SymmetricTensor`` would be densified here, against the project rule on
``todense()`` on the symmetric path (CLAUDE.md / AGENTS.md): CTM edges scale as
``chi * D**2 * chi``, and a block-sparse production run could run out of memory
or silently bypass the symmetry machinery.  So it is refused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms.ipeps_config import CTMEnvironment
from tenax.core.tensor import SymmetricTensor, Tensor


def _as_dense_array(x: jax.Array | Tensor) -> jax.Array:
    """Return a raw ``jax.Array`` from an array or ``DenseTensor``.

    Raises ``NotImplementedError`` for ``SymmetricTensor`` — the excitation
    path is dense-only and must not silently densify a block-sparse tensor.
    """
    if isinstance(x, SymmetricTensor):
        raise NotImplementedError(
            "compute_excitations does not support SymmetricTensor inputs: the "
            "excitation contraction path is dense-only and densifying a "
            "block-sparse tensor would defeat the symmetry machinery (and can "
            "OOM at large chi/D). Convert the ground state to a dense iPEPS "
            "before computing excitations, or track symmetric-aware "
            "excitations as a follow-up."
        )
    if isinstance(x, Tensor):  # DenseTensor: todense() returns the stored buffer
        return x.todense()
    return jnp.asarray(x)


def _as_dense_env(env) -> CTMEnvironment:
    """Coerce a CTM environment to a raw-array ``CTMEnvironment``.

    Accepts the legacy raw-array ``CTMEnvironment`` (returned unchanged) or the
    Tensor-protocol ``CTMTensorEnv`` produced by ``optimize_gs_ad`` (each of the
    8 corner/edge fields is converted via ``.todense()``).
    """
    fields = tuple(env)
    if len(fields) != 8:
        raise ValueError(
            "compute_excitations expects an 8-tensor CTM environment "
            f"(4 corners + 4 edges); got {len(fields)} fields. Split-CTM "
            "environments are not supported by the excitation path."
        )
    if any(isinstance(f, Tensor) for f in fields):
        return CTMEnvironment(*(_as_dense_array(f) for f in fields))
    return env if isinstance(env, CTMEnvironment) else CTMEnvironment(*fields)
