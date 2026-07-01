"""#632 Increment 2 gate — chunked dense-CTM *backward* through the implicit-AD adjoint.

The forward chunking (Increment 1) avoids materializing the chi^2 * D^6 edge-absorption
intermediate via ``lax.map`` over the boundary-chi axis. The open question this gate
answers: does the implicit-AD *backward* (``_ctm_energy_ad.py``, custom_vjp + fixed-point
Neumann/GMRES adjoint) also stay bounded once ``ctm_chunk_size`` is threaded into
``jit_step_bwd`` — i.e. does ``jax.vjp`` through the chunked ``lax.map`` chunk the reverse
pass, or does XLA re-materialize the full chi^2 * D^6 intermediate (making the backward the
new wall)?

Gate sub-parts (mirrors the rung-2 backward-sharding gate structure):
  C1 correctness : value_and_grad grad parity, chunk-off vs chunk-on (recipe=1x1, dense).
  C2 memory      : 3-way per-device peak of full value_and_grad (fwd+bwd) —
                   (off, fwd-chunk-only[legacy], fwd+bwd-chunk) — isolates the BACKWARD.
  C3 reach       : does fwd+bwd-chunk run where off / fwd-only OOM?

Usage:
    # correctness (CPU, fast):
    JAX_PLATFORMS=cpu uv run python examples/spike_chunk_backward_gate.py --mode correctness --D 2 --chi 6
    # memory (single A100, one D per process for a clean peak):
    CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
        uv run python examples/spike_chunk_backward_gate.py --mode memory --D 8 --chi 32 --chunk 8
"""

from __future__ import annotations

import argparse
import warnings

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from tenax.algorithms._ctm_energy_ad import ctm_energy_implicit  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import SINGLE_SITE_NEIGHBORS  # noqa: E402
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402

_D_PHYS = 2


def _indices(D: int):
    sym = U1Symmetry()
    bc = np.zeros(D, dtype=np.int32)
    pc = np.zeros(_D_PHYS, dtype=np.int32)
    return (
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, bc.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pc.copy(), FlowDirection.IN, label="phys"),
    )


def _init_data(D: int, seed: int, well_conditioned: bool):
    key = jax.random.PRNGKey(seed)
    data = jax.random.normal(key, (D, D, D, D, _D_PHYS))
    if well_conditioned:
        data = 0.02 * data
        data = data.at[0, 0, 0, 0, :].add(1.0)
    return data / (jnp.linalg.norm(data) + 1e-10)


def energy_and_grad(
    *, D, chi, chunk, seed=0, well_conditioned=True, max_iter=50, gmres_maxiter=200,
    grad=True,
):
    """(value_and_)grad of the recipe=1x1 implicit-AD CTM energy, ctm_chunk_size=chunk.

    ``grad=False`` runs only the forward (energy, no backward) so the forward and
    backward peak contributions can be isolated.
    """
    idx = _indices(D)
    gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(2, 2, 2, 2)
    data0 = _init_data(D, seed, well_conditioned)

    def loss(data):
        A = DenseTensor(data, idx)
        return ctm_energy_implicit(
            {(0, 0): A},
            SINGLE_SITE_NEIGHBORS,
            gate,
            chi=chi,
            max_iter=max_iter,
            conv_tol=1e-10,
            forward_gauge="phase",
            adjoint_method="fixed_point",
            recipe="1x1",
            ctm_chunk_size=chunk,
            gmres_maxiter=gmres_maxiter,
        )

    if grad:
        e, g = jax.value_and_grad(loss)(data0)
        return float(e), np.asarray(g)
    # forward-only: call eagerly (the implicit-AD energy has host-side control
    # flow, so it cannot be wrapped in a single jax.jit). The forward CTM sweeps
    # are internally jitted; the peak is reached inside them regardless.
    e = loss(data0)
    jax.block_until_ready(e)
    return float(e), np.zeros_like(data0)


def peak_gb():
    try:
        return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        return float("nan")


def _correctness(args):
    # Capture warnings so we can DETECT a silent fallback (chunk requested but env
    # not dense -> monolith path, which would make the whole gate meaningless).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e_off, g_off = energy_and_grad(D=args.D, chi=args.chi, chunk=None)
        e_on, g_on = energy_and_grad(D=args.D, chi=args.chi, chunk=args.chunk)
    fell_back = any("not dense" in str(w.message) for w in caught)
    de = abs(e_off - e_on)
    dg = float(np.max(np.abs(g_off - g_on)))
    gmax = float(np.max(np.abs(g_off)))
    print(f"# devices={jax.devices()} x64={jax.config.jax_enable_x64}")
    print(
        f"[C1] D={args.D} chi={args.chi} chunk={args.chunk} recipe=1x1  "
        f"|dE|={de:.2e}  grad_max|delta|={dg:.2e}  gmax={gmax:.3e}  "
        f"chunk_fell_back_to_monolith={fell_back}"
    )
    ok = (de < 1e-9) and (dg < 1e-9) and (not fell_back)
    print(f"[C1] {'PASS' if ok else 'FAIL'} "
          f"(need |dE|<1e-9, grad delta<1e-9, chunk engaged)")
    return ok


def _memory(args):
    lbl = "chunk=OFF" if args.chunk is None else f"chunk={args.chunk}"
    # well_conditioned=True: the chi^2*D^6 intermediate is STRUCTURAL (same size
    # regardless of values), but a well-separated leading CTM eigenvalue makes the
    # adjoint fixed-point converge fast and stay on the fused path (no eager-GMRES
    # fallback), giving a clean structural peak in bounded wall time.
    which = "grad(fwd+bwd)" if args.grad else "fwd-only"
    try:
        e, g = energy_and_grad(
            D=args.D, chi=args.chi, chunk=args.chunk,
            well_conditioned=True, max_iter=args.max_iter,
            gmres_maxiter=args.gmres_maxiter, grad=args.grad,
        )
        print(
            f"[C2] D={args.D} chi={args.chi} {lbl} {which} "
            f"recipe=1x1  OK  E={e:.6f}  |g|={float(np.linalg.norm(g)):.3e}  "
            f"peak={peak_gb():.2f} GB"
        )
    except Exception as ex:  # noqa: BLE001
        print(
            f"[C2] D={args.D} chi={args.chi} {lbl} {which} "
            f"recipe=1x1  FAILED ({type(ex).__name__}: {str(ex)[:100]})"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["correctness", "memory"], default="correctness")
    ap.add_argument("--D", type=int, default=2)
    ap.add_argument("--chi", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=6)
    ap.add_argument("--gmres-maxiter", type=int, default=30)
    ap.add_argument("--no-grad", dest="grad", action="store_false",
                    help="forward-only peak (isolate fwd from bwd)")
    args = ap.parse_args()
    if args.chunk is not None and args.chunk <= 0:
        args.chunk = None  # allow --chunk 0 to mean OFF from a shell sweep
    if args.mode == "correctness":
        _correctness(args)
    else:
        _memory(args)


if __name__ == "__main__":
    main()
