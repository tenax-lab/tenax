"""iPEPS AD optimization benchmark cases."""

from __future__ import annotations

import jax.numpy as jnp

from tenax import CTMConfig, iPEPSConfig, optimize_gs_ad, sublattice_rotate_gate

_SIZES = {
    "small": {"D": 2, "chi_ctm": 8, "gs_steps": 20, "lr": 1e-3},
    "medium": {"D": 2, "chi_ctm": 16, "gs_steps": 30, "lr": 1e-3},
    "large": {"D": 3, "chi_ctm": 16, "gs_steps": 20, "lr": 5e-4},
}

# Reference-mode (Francuz et al. PRR 7, 013237) implicit-AD path requires
# unit_cell="1x1" + gs_c4v=True + gs_implicit_ad=True + the dense C4v CTM
# fixed-point backward.  Smaller sweep budget since each step pays the
# Krylov adjoint cost.
_REFERENCE_SIZES = {
    "small": {"D": 2, "chi_ctm": 8, "gs_steps": 20},
    "medium": {"D": 2, "chi_ctm": 16, "gs_steps": 20},
}


def _heisenberg_gate(dtype=jnp.float64) -> jnp.ndarray:
    """Build H = Sz*Sz + 0.5*(S+*S- + S-*S+) as (2,2,2,2) tensor."""
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    return H.reshape(2, 2, 2, 2)


def get_benchmarks(dtype=jnp.float64) -> list[dict]:
    benchmarks = []
    for size_label, p in _SIZES.items():

        def _make_fns(p=p, dtype=dtype):
            def setup():
                gate = _heisenberg_gate(dtype=dtype)
                config = iPEPSConfig(
                    max_bond_dim=p["D"],
                    ctm=CTMConfig(chi=p["chi_ctm"]),
                    gs_learning_rate=p["lr"],
                    gs_num_steps=p["gs_steps"],
                )
                return (gate, None, config)

            def run(gate, A_init, config):
                return optimize_gs_ad(gate, A_init, config)

            return setup, run

        setup_fn, run_fn = _make_fns()
        benchmarks.append(
            {
                "name": "ipeps_ad",
                "size_label": size_label,
                "params": p,
                "setup_fn": setup_fn,
                "run_fn": run_fn,
            }
        )

    for size_label, p in _REFERENCE_SIZES.items():

        def _make_reference_fns(p=p, dtype=dtype):
            def setup():
                gate = sublattice_rotate_gate(_heisenberg_gate(dtype=dtype))
                config = iPEPSConfig(
                    max_bond_dim=p["D"],
                    ctm=CTMConfig(
                        chi=p["chi_ctm"],
                        ctm_ad_mode="c4v_reference",
                        adjoint_solver="bicgstab",
                        adjoint_maxiter=200,
                        adjoint_tol=1e-5,
                        # Tikhonov-damp (I - J^T) to keep the Krylov backward
                        # well-conditioned near the CTM fixed point. Without
                        # this, the residual blows up as the optimizer
                        # approaches the true ground state.
                        adjoint_tikhonov=1e-3,
                    ),
                    gs_optimizer="lbfgs",
                    gs_num_steps=p["gs_steps"],
                    gs_implicit_ad=True,
                    gs_c4v=True,
                    # su_init=False: SU converges to the |↑↑⟩ product state of
                    # the sublattice-rotated gate, which is a gradient-zero
                    # extremum at E=-0.5 per site. L-BFGS cannot escape it.
                    # Random init lets the optimizer break free.
                    su_init=False,
                )
                return (gate, None, config)

            def run(gate, A_init, config):
                return optimize_gs_ad(gate, A_init, config)

            return setup, run

        setup_fn, run_fn = _make_reference_fns()
        benchmarks.append(
            {
                "name": "ipeps_ad_c4v_reference",
                "size_label": size_label,
                "params": p,
                "setup_fn": setup_fn,
                "run_fn": run_fn,
            }
        )

    return benchmarks
