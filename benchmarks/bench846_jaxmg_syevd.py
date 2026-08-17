"""#846: does a multi-GPU eigensolver beat single-GPU ``eigh`` on the dense CTM?

The dense 2-site CTM spends most of a sweep in Hermitian eigendecompositions of
size ``n = chi * D**2`` -- measured at ~89% (D=4, chi=32) and ~81% (D=6, chi=48)
on one GPU.  This benchmark asks whether ``jaxmg``'s cuSolverMg ``syevd`` beats
``jnp.linalg.eigh`` at those sizes, which is the only thing that decides #846.

Two things this script refuses to do, both learned the hard way:

* **It never reports a timing it has not validated.**  A failing cuSolverMg call
  returns all-NaN eigenvalues *quickly*, which reads as a 4x speedup if you time
  it without checking the values.  That happened; the number was void.
* **It preflights the environment and says why it cannot run**, rather than
  emitting a misleading failure.  ``jaxmg`` needs peer-to-peer access between
  devices and a CUDA-12 JAX build; neither is obvious from the error it raises.

Run::

    python benchmarks/bench846_jaxmg_syevd.py

Requires (optional extra, never a hard dependency)::

    pip install "jax[cuda12]" jaxmg

Hardware: peer-to-peer capable GPUs.  GeForce cards do **not** qualify -- NVIDIA
disables P2P on consumer parts, and cuSolverMg then fails with
``CUSOLVER_STATUS_EXECUTION_FAILED`` (status 6).  Use data-center GPUs (A100 /
H100) or any pair with NVLink.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
import time

import numpy as np

# n = chi * D**2 for the (D, chi) the dense CTM actually runs at, plus the
# per-device tile width T_A.  T_A must divide the per-device shard and should be
# >= 128 or cuSolverMg is extremely slow; these assume 2 devices.
CASES = [
    # (D, chi, n, T_A)
    (4, 32, 512, 128),
    (6, 48, 1728, 288),
    (8, 64, 4096, 512),
]


def _fail(msg: str, code: int = 2) -> None:
    print(f"\nCANNOT EVALUATE: {msg}", file=sys.stderr)
    sys.exit(code)


def preflight() -> None:
    """Refuse to produce numbers the hardware cannot support."""
    try:
        import jax
    except ImportError:
        _fail("jax is not installed.")

    devs = jax.devices()
    gpus = [d for d in devs if d.platform == "gpu"]
    if len(gpus) < 2:
        _fail(f"need >= 2 GPUs, found {len(gpus)}: {devs}")

    # jaxmg selects its binary directory as cu{cuda_major} but the PyPI wheel
    # ships only cu12.  On a CUDA 13 jax this dies with a bare OSError on
    # libcyclic.so and then a misleading "No FFI handler registered".
    try:
        import jaxmg
    except ImportError:
        _fail('jaxmg is not installed.  pip install "jax[cuda12]" jaxmg')
    except OSError as exc:  # pragma: no cover - environment dependent
        _fail(
            f"jaxmg native libraries failed to load ({exc}).\n"
            "The PyPI wheel ships cu12 binaries only and picks its directory "
            "from the CUDA major version, so a CUDA 13 jax looks for a cu13/ "
            'directory that is not there.  Install "jax[cuda12]".'
        )

    have = os.path.dirname(jaxmg.__file__)
    shipped = sorted(os.path.basename(p) for p in glob.glob(os.path.join(have, "cu*")))
    print(f"jaxmg binary dirs shipped: {shipped or '(none)'}")

    # cuSolverMg needs peer-to-peer.  Check it directly rather than letting the
    # kernel fail with an opaque status code.
    lib = None
    for cand in ("libcudart.so.12", "libcudart.so"):
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError:
            continue
    if lib is None:
        found = glob.glob(
            os.path.join(
                sys.prefix,
                "lib",
                "python*",
                "site-packages",
                "nvidia",
                "cuda_runtime",
                "lib",
                "libcudart.so*",
            )
        )
        if found:
            lib = ctypes.CDLL(found[0])
    if lib is not None:
        can = ctypes.c_int()
        lib.cudaDeviceCanAccessPeer(ctypes.byref(can), 0, 1)
        if can.value == 0:
            _fail(
                "GPU 0 cannot peer-access GPU 1 (cudaDeviceCanAccessPeer = 0). "
                "cuSolverMg requires P2P.  NVIDIA disables it on GeForce, where "
                "`nvidia-smi topo -p2p r` reports CNS (Chipset Not Supported). "
                "Run this on data-center GPUs or a pair with NVLink."
            )
        print("peer-to-peer: available")


def bench(fn, repeats: int = 5) -> float:
    import jax

    jax.block_until_ready(fn())
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    preflight()

    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    jax.config.update("jax_enable_x64", True)
    import jaxmg

    devs = jax.devices()
    mesh = Mesh(np.array(devs), ("gpus",))
    spec = P("gpus", None)
    print(f"devices: {devs}\n")

    @jax.jit
    def ref_eigh(m):
        return jnp.linalg.eigh(m)[0]

    header = f"{'D':>3} {'chi':>4} {'n':>6} {'T_A':>5} {'1-GPU':>10} {'multi':>10} {'speedup':>8}  correctness"
    print(header)
    print("-" * len(header))

    worth_it = []
    for D, chi, n, t_a in CASES:
        rng = np.random.default_rng(0)
        a = rng.normal(size=(n, n))
        rho = np.asarray((a + a.T) / 2.0, dtype=np.float64)

        single = jnp.asarray(rho, device=devs[0])
        t_ref = bench(lambda: ref_eigh(single))
        w_ref = np.sort(np.asarray(ref_eigh(single)))

        sharded = jax.device_put(jnp.asarray(rho), NamedSharding(mesh, spec))
        call = lambda: jaxmg.syevd(  # noqa: E731
            sharded, t_a, mesh, spec, return_eigenvectors=False, return_status=True
        )
        try:
            out = call()
        except Exception as exc:  # pragma: no cover - environment dependent
            print(
                f"{D:>3} {chi:>4} {n:>6} {t_a:>5} {t_ref * 1e3:>9.1f}ms "
                f"{'FAILED':>10} {'':>8}  {type(exc).__name__}: {str(exc)[:60]}"
            )
            continue

        status = int(out[-1])
        w = np.sort(np.asarray(out[0]).ravel()[:n])
        scale = max(float(np.max(np.abs(w_ref))), 1e-300)
        err = float(np.max(np.abs(w - w_ref))) / scale

        # Validate BEFORE timing.  A failing kernel returns NaN fast and would
        # otherwise be reported as a speedup.
        if status != 0 or not np.isfinite(w).all() or err > 1e-8:
            detail = (
                f"status={status}"
                + (" all-NaN" if not np.isfinite(w).any() else "")
                + (f" max rel eig err {err:.2e}" if np.isfinite(err) else " err=nan")
            )
            print(
                f"{D:>3} {chi:>4} {n:>6} {t_a:>5} {t_ref * 1e3:>9.1f}ms "
                f"{'INVALID':>10} {'--':>8}  {detail}"
            )
            continue

        t_mg = bench(lambda: call()[0])
        speedup = t_ref / t_mg
        worth_it.append((n, speedup))
        print(
            f"{D:>3} {chi:>4} {n:>6} {t_a:>5} {t_ref * 1e3:>9.1f}ms "
            f"{t_mg * 1e3:>9.1f}ms {speedup:>7.2f}x  max rel eig err {err:.2e}"
        )

    print()
    if not worth_it:
        print("No validated multi-GPU timing was produced -- see the rows above.")
        print("#846 remains unquantified; this is 'cannot evaluate', not 'no benefit'.")
        return 1
    best_n, best = max(worth_it, key=lambda t: t[1])
    print(f"Best validated speedup: {best:.2f}x at n={best_n}.")
    print(
        "Weigh against: eigh is ~81-89% of a dense CTM sweep, so the sweep-level "
        f"gain is bounded by roughly 1/((1-0.85) + 0.85/{best:.2f})"
        f" = {1 / (0.15 + 0.85 / best):.2f}x."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
