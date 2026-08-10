"""The dense CTM entry points must accept complex site tensors (#842).

``ctm_2site`` seeded its ``lax.while_loop`` carry with ``env_A.C1.dtype``,
while the body assigned ``_dense_svd(..., compute_uv=False)`` -- which is
always real.  On a complex site tensor the carry therefore went in complex128
and came out float64, and ``lax.while_loop``'s carry-type invariance rejected
it outright::

    TypeError: while_loop body function carry input and carry output must
    have equal types

``ctm`` had already been fixed, in the same file, 150 lines up, with a comment
describing this exact failure.  ``ctm_split`` is a Python loop seeded with
``prev_sv = None``, so it never had the constraint.

These tests cover all three entry points rather than only the one that was
broken: the defect is a property of the seed pattern, not of ``ctm_2site``,
and fixing one of two identical loops is how it survived the first time.
Real states are unaffected either way -- which is why nothing caught it.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import pytest

from tenax.algorithms.ipeps import heisenberg_gate, ipeps, sublattice_rotate_gate
from tenax.algorithms.ipeps_config import CTMConfig, iPEPSConfig
from tenax.algorithms.ipeps_ctm_convergence import ctm, ctm_2site, ctm_split

D = 2
DPHYS = 2
CFG = CTMConfig(chi=4, chi_I=4, max_iter=6, conv_tol=1e-10)


def _site(seed: int, *, dtype) -> jax.Array:
    """A random ``(D, D, D, D, d)`` site tensor of the requested dtype."""
    key = jax.random.PRNGKey(seed)
    real = jax.random.normal(key, (D, D, D, D, DPHYS))
    if jnp.issubdtype(dtype, jnp.complexfloating):
        imag = jax.random.normal(jax.random.fold_in(key, 1), (D, D, D, D, DPHYS))
        return (real + 1j * imag).astype(dtype)
    return real.astype(dtype)


# ---------------------------------------------------------------------------
# The regression itself.
# ---------------------------------------------------------------------------


def test_ctm_2site_accepts_a_complex_site_tensor():
    """The exact #842 repro: this raised ``TypeError`` before the fix."""
    A = _site(0, dtype=jnp.complex128)
    B = _site(7, dtype=jnp.complex128)

    env_A, env_B = ctm_2site(A, B, CFG)

    for env in (env_A, env_B):
        assert jnp.isfinite(jnp.abs(env.C1)).all()
        assert jnp.isfinite(jnp.abs(env.T1)).all()


@pytest.mark.parametrize("entry", ["ctm", "ctm_2site", "ctm_split"])
def test_every_dense_ctm_entry_point_accepts_complex(entry):
    """All three, not just the one that was broken.

    #842 exists because a fix landed on one of two identical loops.  A test
    that only covers ``ctm_2site`` sets the same trap for whoever adds the
    next entry point.
    """
    A = _site(0, dtype=jnp.complex128)
    B = _site(7, dtype=jnp.complex128)

    if entry == "ctm":
        ctm(A, CFG)
    elif entry == "ctm_2site":
        ctm_2site(A, B, CFG)
    else:
        ctm_split(A, CFG)


def test_complex_input_is_not_silently_realified():
    """A fix that dropped the imaginary part would pass every test above.

    The carry mismatch is fixed by seeding the *singular values* real, not by
    making the environment real: the environment of a complex state is complex,
    and casting it away would turn a loud ``TypeError`` into a quiet wrong
    answer.

    Checked on the **edge** tensors, not the corners.  The corners of a valid
    environment come out numerically real here (``|imag| ~ 2e-16``) because the
    double layer is ``a = A (x) A*``, so a corner assertion would fail against a
    perfectly correct implementation.  The edges carry the imaginary part --
    measured ``max|T.imag|`` between 6e-2 and 8e-1 across seeds and chi.
    """
    A = _site(0, dtype=jnp.complex128)
    B = _site(7, dtype=jnp.complex128)

    env_A, env_B = ctm_2site(A, B, CFG)

    for name, env in (("A", env_A), ("B", env_B)):
        assert jnp.iscomplexobj(env.C1), f"env_{name}.C1 lost its complex dtype"
        assert jnp.iscomplexobj(env.T1), f"env_{name}.T1 lost its complex dtype"
        edge_imag = max(
            float(jnp.max(jnp.abs(getattr(env, f"T{i}").imag))) for i in (1, 2, 3, 4)
        )
        assert edge_imag > 1e-3, (
            f"env_{name} is complex-typed but its edges are numerically real "
            f"(max|T.imag| = {edge_imag:.3g}) -- the imaginary part of the "
            f"state was discarded somewhere in the sweep"
        )


def test_reported_convergence_info_is_real_on_complex_input():
    """``diff`` is a magnitude; it must stay real however complex the state is.

    The seed and the body have to agree on *real*, not merely on some common
    type -- seeding ``diff`` complex would satisfy ``while_loop`` and then make
    ``float(info.diff)`` in ``ipeps()`` raise or warn (#839).
    """
    A = _site(0, dtype=jnp.complex128)
    B = _site(7, dtype=jnp.complex128)

    _eA, _eB, info = ctm_2site(A, B, CFG, return_meta=True)

    assert not jnp.iscomplexobj(info.diff)
    assert float(info.diff) >= 0.0
    assert int(info.n_iter) > 0


def test_ipeps_accepts_a_complex_initial_state():
    """Reachability: #842 is hit through public API, not just the internal call.

    ``num_imaginary_steps=0`` goes straight from the supplied state to
    ``ctm_2site``, which is the shortest path from user input to the raise.
    """
    gate = sublattice_rotate_gate(heisenberg_gate())
    config = iPEPSConfig(
        max_bond_dim=D,
        num_imaginary_steps=0,
        dt=0.05,
        unit_cell="1x1",
        ctm=CFG,
    )
    initial = (_site(0, dtype=jnp.complex128), _site(7, dtype=jnp.complex128))

    with warnings.catch_warnings():
        # A 6-sweep CTM on a random state need not converge; #839's warning is
        # correct here and is not what this test is about.
        warnings.simplefilter("ignore", UserWarning)
        E, _tensors, _envs = ipeps(gate, initial, config)

    assert jnp.isfinite(jnp.real(E))
