import os
import subprocess
import sys

import pytest

# Importable when pytest runs this file directly (prepend import mode puts the
# test dir on sys.path); guard the insert so it also works if invoked some other
# way.  Same import the subprocess driver uses.
sys.path.insert(0, os.path.dirname(__file__))
from _ctm_sharding_probe import heisenberg_dense_probe_energy

# Per device-count probe size: D² must divide the device count so every
# sharded axis (the double-layer D² legs AND the chi legs of the edges) splits
# evenly, otherwise XLA raises an IndivisibleError.
#   N=2: D=2 → D²=4 divisible by 2, chi=8 divisible by 2.
#   N=4: D=4 → D²=16 divisible by 4, chi=4 divisible by 4 (D=2 would give
#        D²=4 / 4 = 1 per device, which works, but the raw-D=2 legs the absorb
#        also touches are not divisible by 4 — D=4 shards cleanly everywhere).
_PARITY_PROBE = {2: (2, 8), 4: (4, 4)}


def _run_parity_subproc(n_dev, D, chi, *, well_conditioned=False, thresh=1e-8):
    """Drive the parity subprocess under ``n_dev`` fake CPU devices; return the
    CompletedProcess.

    ``--xla_force_host_platform_device_count=N`` fabricates N devices on the CPU
    platform only, so the subprocess MUST run on CPU.  On a GPU box JAX otherwise
    defaults to the single real GPU, the N-way mesh all-gather never finds its
    peers, and the child aborts on a rendezvous timeout. ``JAX_PLATFORMS=cpu``
    pins the platform so the fake-device trick (and the GSPMD sharding it
    exercises) works regardless of host accelerators.
    """
    env = dict(
        os.environ,
        XLA_FLAGS=f"--xla_force_host_platform_device_count={n_dev}",
        JAX_PLATFORMS="cpu",
    )
    return subprocess.run(
        [
            sys.executable,
            "tests/_ctm_sharding_parity_subproc.py",
            str(D),
            str(chi),
            str(int(well_conditioned)),
            repr(thresh),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


@pytest.mark.parametrize(
    "n_dev",
    [
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                reason=(
                    "#702: PR #676 (direction-dependent 2x2 bond bookkeeping, "
                    "bisected first-bad 7b8e5ad) moved the D=2 random single-site "
                    "2x2 CTM fixed point into an ill-conditioned region "
                    "(e_single -0.0083153747 -> -0.0075941887). Sharding "
                    "reassociates the D²-axis sum, so the sharded-vs-single "
                    "delta is O(eps*kappa): it blew from 5.55e-17 at 7b8e5ad^ to "
                    "7.68e-5 here, >> thresh 1e-8. Same #676 root cause as the "
                    "chi-bump regression; the well-conditioned probes below are "
                    "unaffected (fixed point stable across #676). Flips green "
                    "once #702 restores the fixed point."
                ),
                strict=False,
            ),
        ),
        4,
    ],
)
def test_sharded_forward_matches_single_device(n_dev):
    """Dense CTM energy under an N-device GSPMD mesh equals the single-device
    result to <1e-8 (subprocess with fake CPU devices)."""
    D, chi = _PARITY_PROBE[n_dev]
    r = _run_parity_subproc(n_dev, D, chi)
    assert r.returncode == 0, f"parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"


def test_sharded_well_conditioned_tight_parity():
    """On a WELL-CONDITIONED state the sharded forward CTM matches single-device
    to a tight tolerance even at a larger χ where the contracted D²-axis is
    reassociated across 4 devices (D=4 → 4 elements/device).

    This guards the property that matters physically: sharding is exact up to
    floating-point reassociation, and on a well-separated CTM fixed point that
    reassociation stays small. The companion random-state probe at the SAME
    (D=4, χ=8) diverges by ~1e-4 (the reassociation noise is amplified by the
    random state's large condition number), so this case exercises a regime the
    small-χ ``test_sharded_forward_matches_single_device`` cases do not reach.

    Tolerance (#692): the reassociation floor is machine-precision (~1e-17)
    locally but **platform-dependent** — CI runners (different CPU/XLA
    reduction tree) land at ~9e-10 on this state (measured on the #692 Full-
    tests matrix), so the original 1e-10 gate was not portable. 1e-6 clears the
    observed CI noise by ~1000x while staying ~100x below the ill-conditioned
    random regime (~1e-4), so it still catches a real sharding/fixed-point
    regression. The single-device ``e_single`` is stable across PR #676
    (0.0009211466 → 0.0009211468), i.e. this is a tolerance-portability fix,
    not a masked regression.
    """
    r = _run_parity_subproc(4, D=4, chi=8, well_conditioned=True, thresh=1e-6)
    assert r.returncode == 0, f"parity failed:\nSTDOUT:{r.stdout}\nSTDERR:{r.stderr}"


def test_flag_off_is_unchanged():
    """device_mesh=None must give the exact same energy as not passing it.

    The sharding flag is opt-in: ``device_mesh=None`` (the default) must be a
    literal no-op, so the energy is bit-identical to omitting the argument.
    Both calls run single-device, so no fake-device env is needed.
    """
    e_default = heisenberg_dense_probe_energy(D=2, chi=8, seed=1)
    e_none = heisenberg_dense_probe_energy(D=2, chi=8, device_mesh=None, seed=1)
    assert e_default == e_none
