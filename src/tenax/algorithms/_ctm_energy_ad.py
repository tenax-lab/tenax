"""CTM-to-energy AD wrappers: Python-loop forward with configurable backward."""

from __future__ import annotations

__all__ = ["ctm_energy_explicit"]

import jax
import jax.numpy as jnp

from tenax.algorithms._ctm_python_loop import (
    Coord,
    _make_jit_ctm_step,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    CTMTensorEnv,
    initialize_ctm_tensor_env,
)


def ctm_energy_explicit(
    site_tensors: dict[Coord, object],
    neighbors: dict[Coord, dict[str, Coord]],
    gate,
    *,
    chi: int = 20,
    warmup_steps: int = 3,
    backprop_steps: int = 20,
    projector_method: str = "eigh",
    renormalize: bool = True,
    projector_backward: str = "auto",
    env_init: dict[Coord, CTMTensorEnv] | None = None,
    energy_fn=None,
) -> jnp.ndarray:
    """Compute iPEPS energy with explicit-differentiation backward.

    Forward: warmup (no grad) + checkpointed CTM sweeps.
    Backward: standard JAX autodiff through checkpointed sweeps.
    """
    jit_step = _make_jit_ctm_step(neighbors)

    envs = (
        env_init
        if env_init is not None
        else {c: initialize_ctm_tensor_env(A, chi) for c, A in site_tensors.items()}
    )

    # Warmup: no gradient tracking
    for _ in range(warmup_steps):
        envs = jax.lax.stop_gradient(
            jit_step(
                site_tensors,
                envs,
                chi=chi,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
        )

    # Backprop phase: checkpointed sweeps
    for _ in range(backprop_steps):
        envs = jax.checkpoint(
            lambda st, e: jit_step(
                st,
                e,
                chi=chi,
                projector_method=projector_method,
                renormalize=renormalize,
                projector_backward=projector_backward,
            )
        )(site_tensors, envs)

    coord = next(iter(envs))
    env = envs[coord]
    A = site_tensors[coord]
    if energy_fn is not None:
        return energy_fn(A, env, gate)
    return compute_energy_ctm_tensor(A, env, gate)
