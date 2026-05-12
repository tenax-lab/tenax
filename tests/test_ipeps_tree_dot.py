"""Regression tests for ``_tree_dot`` (line-search inner product).

The optimizer plumbing in ``optimize_gs_ad`` uses ``_tree_dot`` to compute
slopes, CG betas, and Wolfe direction derivatives. The contract is:

    _tree_dot(a, b) = Re( sum_leaves <conj(a_leaf), b_leaf> )

These tests pin that contract so the NumPy host-materialisation rewrite
(perf follow-up to ``project_f3_landed_line_search_next.md``) cannot
silently drift from the JAX-based reference.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.ipeps_optimize import _tree_dot


def _jax_reference(a, b) -> float:
    """The previous JAX-only implementation, kept here as a reference."""
    import jax

    leaves_a = jax.tree.leaves(a)
    leaves_b = jax.tree.leaves(b)
    return float(
        jnp.real(sum(jnp.sum(jnp.conj(la) * lb) for la, lb in zip(leaves_a, leaves_b)))
    )


def test_tree_dot_real_single_leaf():
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    # 1*4 + 2*5 + 3*6 = 32
    assert _tree_dot(a, b) == pytest.approx(32.0)
    assert _tree_dot(a, b) == pytest.approx(_jax_reference(a, b))


def test_tree_dot_complex_takes_conj_of_first_arg():
    """<conj(a), b> != <a, b> for complex inputs."""
    a = jnp.array([1.0 + 1j, 2.0])
    b = jnp.array([1.0, 1.0 + 2j])
    # conj(a) . b = (1-1j)*1 + 2*(1+2j) = 1-1j + 2+4j = 3+3j; .real = 3.0
    assert _tree_dot(a, b) == pytest.approx(3.0)
    assert _tree_dot(a, b) == pytest.approx(_jax_reference(a, b))


def test_tree_dot_tuple_pytree():
    """Two-leaf tuple (matches the 2-site iPEPS (A, B) gradient shape)."""
    a = (jnp.array([1.0, 2.0]), jnp.array([[1.0, 0.0], [0.0, 1.0]]))
    b = (jnp.array([3.0, 4.0]), jnp.array([[5.0, 0.0], [0.0, 5.0]]))
    # leaf0: 1*3 + 2*4 = 11; leaf1: trace(diag(1,1) @ diag(5,5)) = 10; total = 21
    assert _tree_dot(a, b) == pytest.approx(21.0)
    assert _tree_dot(a, b) == pytest.approx(_jax_reference(a, b))


def test_tree_dot_dict_pytree():
    a = {"x": jnp.array([1.0, 2.0]), "y": jnp.array([3.0])}
    b = {"x": jnp.array([4.0, 5.0]), "y": jnp.array([6.0])}
    # 1*4 + 2*5 + 3*6 = 32
    assert _tree_dot(a, b) == pytest.approx(32.0)
    assert _tree_dot(a, b) == pytest.approx(_jax_reference(a, b))


def test_tree_dot_empty_pytree_is_zero():
    """An empty pytree (no leaves) returns 0.0 without crashing."""
    assert _tree_dot((), ()) == 0.0
    assert _tree_dot({}, {}) == 0.0


def test_tree_dot_returns_python_float():
    """Caller code is ``float(_tree_dot(...))``; the return must already
    be a Python float so Python-level comparisons (``<``, ``max``) work
    without an implicit JAX materialise."""
    out = _tree_dot(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    assert isinstance(out, float)
    assert not isinstance(out, jnp.ndarray)


def test_tree_dot_accepts_numpy_arrays():
    """The optimizer sometimes passes already-materialised NumPy arrays
    (e.g. a host-cached previous gradient). Don't re-fetch via JAX."""
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    assert _tree_dot(a, b) == pytest.approx(11.0)
