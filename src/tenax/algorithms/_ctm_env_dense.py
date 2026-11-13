"""Raw-array views of CTM environments for the dense-only code paths.

``optimize_gs_ad`` returns a ``DenseTensor`` site tensor and a ``CTMTensorEnv``
whose 8 fields are ``DenseTensor``s, and ``ctm_tensor`` returns the
``CTMTensorEnv``.  The excitation contractions and the raw-array results of
``ctm()``, ``ctm_2site()`` and ``ipeps()`` use plain ``jax.Array``s and a
raw-array ``CTMEnvironment``.  The fields of a ``CTMTensorEnv`` share the leg
and flow convention of the raw ``CTMEnvironment`` when each stores its legs in
the order ``CTMTensorEnv`` documents, so ``.todense()`` on each field is then
correct and keeps the converged environment.  ``.todense()`` goes by axis
position, and the 2x2 sweep of ``ctm_tensor`` stores each corner with its two
legs in the reverse order, so ``_as_dense_env`` passes a ``CTMTensorEnv``
through ``_in_documented_leg_order`` first.  A transposed corner is invisible on
a state with lattice symmetry, whose corners are symmetric matrices, and moves
the energy on any other.

For a ``DenseTensor``, ``.todense()`` returns the stored buffer, so it is free.
A ``SymmetricTensor`` would be densified here, against the project rule on
``todense()`` on the symmetric path (CLAUDE.md / AGENTS.md): CTM edges scale as
``chi * D**2 * chi``, and a block-sparse production run could run out of memory
or silently bypass the symmetry machinery.  So it is refused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_tensor_init import (
    _STD_CORNER_SPECS,
    _STD_EDGE_SPECS,
    CTMTensorEnv,
)
from tenax.algorithms.ipeps_config import CTMEnvironment
from tenax.core.tensor import SymmetricTensor, Tensor


def _as_dense_array(x: jax.Array | Tensor) -> jax.Array:
    """Return a raw ``jax.Array`` from an array or ``DenseTensor``.

    Raises ``NotImplementedError`` for ``SymmetricTensor``: the raw-array paths
    are dense-only and must not silently densify a block-sparse tensor.
    """
    if isinstance(x, SymmetricTensor):
        raise NotImplementedError(
            "A SymmetricTensor cannot be converted to a raw array here: this "
            "path is dense-only, and densifying a block-sparse tensor would "
            "defeat the symmetry machinery (and can OOM at large chi/D). "
            "Convert the state to a dense iPEPS first."
        )
    if isinstance(x, Tensor):  # DenseTensor: todense() returns the stored buffer
        return x.todense()
    return jnp.asarray(x)


def _as_dense_env(env) -> CTMEnvironment:
    """Coerce a CTM environment to a raw-array ``CTMEnvironment``.

    Accepts the legacy raw-array ``CTMEnvironment`` (returned unchanged) or the
    Tensor-protocol ``CTMTensorEnv`` produced by ``optimize_gs_ad`` and
    ``ctm_tensor``.  A ``CTMTensorEnv`` is put in the leg order it documents, by
    label (``_in_documented_leg_order``), before each of the 8 corner/edge fields
    is converted via ``.todense()``.  A ``CTMEnvironment`` is read by position.
    """
    fields = tuple(env)
    if len(fields) != 8:
        raise ValueError(
            "Expected an 8-tensor CTM environment (4 corners + 4 edges); got "
            f"{len(fields)} fields. Split-CTM environments are not supported "
            "on this raw-array path."
        )
    if isinstance(env, CTMTensorEnv):
        # ``.todense()`` goes by position, and the 2x2 sweep stores each corner
        # with its two legs reversed.
        fields = tuple(_in_documented_leg_order(env))
    if any(isinstance(f, Tensor) for f in fields):
        return CTMEnvironment(*(_as_dense_array(f) for f in fields))
    return env if isinstance(env, CTMEnvironment) else CTMEnvironment(*fields)


# The leg order ``CTMTensorEnv`` documents, per field: the labels that lead each
# row of the tables that build the initial environment.
_DOCUMENTED_LEGS = {
    **{name: spec[:2] for name, spec in _STD_CORNER_SPECS.items()},
    **{name: spec[:3] for name, spec in _STD_EDGE_SPECS.items()},
}


def _in_documented_leg_order(env: CTMTensorEnv) -> CTMTensorEnv:
    """Return ``env`` with every field's legs in the order ``CTMTensorEnv`` documents.

    ``.todense()`` converts each field by axis position, and the raw consumers
    (``ipeps_rdm``, ``ipeps_excitations``) contract by position in the documented
    order: ``C1`` is ``(c1_d, c1_r)``, ``T1`` is ``(t1_l, u2, t1_r)``, and so on
    (the leading labels of ``_STD_CORNER_SPECS`` and ``_STD_EDGE_SPECS``).  The
    2x2 sweep of ``ctm_tensor`` returns the edges in that order but each corner
    with its two legs reversed, ``C1`` as ``(c1_r, c1_d)``.  The labels and flows
    are right, and ``compute_energy_ctm_tensor``, which goes by label, is
    unaffected; but a raw environment made by position holds every corner
    transposed.  That is invisible on a state with lattice symmetry, whose
    corners are symmetric matrices, and wrong on any other.  ``_as_dense_env``
    calls this on every ``CTMTensorEnv``.

    Fields are permuted by label, never by position.  A field already in order
    comes back unchanged (the same object).

    Raises:
        ValueError: A field whose labels are not exactly the expected ones.
    """
    fields = {}
    for name, field in env._asdict().items():
        want = _DOCUMENTED_LEGS[name]
        have = field.labels()
        if have != want:
            if len(have) != len(want) or set(have) != set(want):
                raise ValueError(
                    f"CTMTensorEnv.{name} has the legs {have}; expected exactly "
                    f"the labels {want}, in any order."
                )
            field = field.permute_legs(tuple(have.index(label) for label in want))
        fields[name] = field
    return CTMTensorEnv(**fields)
