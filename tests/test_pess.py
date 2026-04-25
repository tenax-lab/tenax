import jax
import jax.numpy as jnp

from tenax.algorithms.pess import IPESSState


def test_ipess_state_shapes():
    D, d = 4, 3
    key = jax.random.PRNGKey(0)
    state = IPESSState.random(D=D, d=d, key=key)
    assert state.R_a.shape == (D, D, d)
    assert state.R_b.shape == (D, D, d)
    assert state.R_c.shape == (D, D, d)
    assert state.T_u.shape == (D, D, D)
    assert state.T_d.shape == (D, D, D)
    assert all(lam.shape == (D,) for lam in state.lambdas)
    assert state.R_a.dtype == jnp.complex128
    assert state.T_u.dtype == jnp.complex128
