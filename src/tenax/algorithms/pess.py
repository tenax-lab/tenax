"""iPESS (infinite Projected Entangled Simplex State) on the kagome lattice."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

D_PHYS_DEFAULT = 3  # spin-1


@dataclass(frozen=True)
class IPESSState:
    """Kagome iPESS parameters.

    R_a, R_b, R_c: rank-3 site tensors, shape (D, D, d). Index order is
        (leg-to-T_u, leg-to-T_d, physical).
    T_u, T_d: rank-3 simplex tensors, shape (D, D, D). Index order is
        (leg-to-R_a, leg-to-R_b, leg-to-R_c).
    lambdas: 6 bond singular-value vectors of length D, ordered
        (a-up, b-up, c-up, a-down, b-down, c-down).
    """

    R_a: jax.Array
    R_b: jax.Array
    R_c: jax.Array
    T_u: jax.Array
    T_d: jax.Array
    lambdas: tuple[jax.Array, ...]

    @classmethod
    def random(
        cls,
        D: int,
        d: int = D_PHYS_DEFAULT,
        key: jax.Array | None = None,
        scale: float = 0.1,
    ) -> IPESSState:
        if key is None:
            key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 5)

        def cmplx(k, shape):
            re = jax.random.normal(k, shape) * scale
            im = jax.random.normal(jax.random.fold_in(k, 1), shape) * scale
            return (re + 1j * im).astype(jnp.complex128)

        return cls(
            R_a=cmplx(keys[0], (D, D, d)),
            R_b=cmplx(keys[1], (D, D, d)),
            R_c=cmplx(keys[2], (D, D, d)),
            T_u=cmplx(keys[3], (D, D, D)),
            T_d=cmplx(keys[4], (D, D, D)),
            lambdas=tuple(jnp.ones(D) for _ in range(6)),
        )
