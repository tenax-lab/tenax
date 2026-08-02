"""GSPMD multi-GPU sharding for dense HOTRG (large-chi coarse-graining).

HOTRG is forward-only (no AD-through-SVD backward), and it forms a chi^6
``T_merged`` intermediate that dominates memory while the HOSVD operand is only
chi^4. Sharding a free leg of ``T_merged`` keeps that chi^6 wall at ~1/N per
device -> ideal per-device memory relief and a higher chi ceiling (measured
2.00x on 2xA100; see docs/.../2026-07-03-hotrg-trg-multigpu-feasibility.md).

Sharding is a pure layout hint: ``HOTRGConfig(device_mesh=mesh)`` must give the
same free energy as single-device. Parity runs on fake CPU devices via a
subprocess (matching tests/test_ctm_sharding_parity.py); the per-device memory
relief itself is a GPU high-water measurement (examples/probe_hotrg_multigpu.py).
"""

import os
import subprocess
import sys


def _run_subproc(n_dev, *args):
    env = dict(
        os.environ,
        XLA_FLAGS=f"--xla_force_host_platform_device_count={n_dev}",
        JAX_PLATFORMS="cpu",
    )
    return subprocess.run(
        [
            sys.executable,
            "tests/_hotrg_sharding_parity_subproc.py",
            *[str(a) for a in args],
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_hotrg_sharded_matches_single_device():
    """HOTRG free energy under a 2-device GSPMD mesh equals single-device to
    <1e-9 on a well-conditioned state (beta=0.6, chi=16: measured ~2e-16).

    Sharding is exact up to floating-point reassociation. On a hard state
    (small chi near Tc) the sharded HOSVD can pick a marginally different but
    equally-valid truncation among near-degenerate singular values, so the gap
    grows to O(1e-4) — but it collapses with chi (chi=32 at beta=0.3 is
    bit-identical). This case exercises the truncation regime while staying in
    the tight-parity regime, so it still catches a real sharding bug.
    """
    r = _run_subproc(2, 0.6, 16, 6, 1e-9)
    assert r.returncode == 0, (
        f"HOTRG sharded parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"
    )


def test_hotrg_flag_off_is_unchanged():
    """device_mesh=None (default) is a literal no-op: bit-identical to omitting
    it. Single device, no fake-device env needed."""
    from tenax.algorithms.hotrg import HOTRGConfig, hotrg
    from tenax.algorithms.trg import compute_ising_tensor

    T = compute_ising_tensor(0.3)
    f_default = float(hotrg(T, HOTRGConfig(max_bond_dim=8, num_steps=6)))
    f_none = float(hotrg(T, HOTRGConfig(max_bond_dim=8, num_steps=6, device_mesh=None)))
    assert f_default == f_none
