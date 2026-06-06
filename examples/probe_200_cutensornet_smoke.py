"""P-B0 smoke: call a GPU tensor-contraction library from inside JAX (#200).

The highest-unknown for the cuTensorNet/cuTENSOR block-sparse backend is purely
plumbing: *can a JAX computation hand operands to an opaque CUDA contraction
kernel and get the result back as a JAX array on GPU, under jit?* This probe
answers that with the cheapest faithful bridge the handoff picked (callback
route): a trivial dense pairwise contraction ``ij,jk->ik`` routed through
cupy's cuTENSOR-backed contraction inside ``jax.pure_callback``, wrapped so the
whole contraction is ONE opaque XLA op.

It is forward-only and dense (no block-sparse, no VJP) on purpose: P-B0 isolates
the bridge from the kernel and the seam. Success = value matches ``jnp.einsum``
at f64 and c128, the op shows up as a single ``custom-call``/callback in the
jaxpr, and it runs under jit on the GPU.

Run:  JAX_PLATFORMS=cuda uv run python examples/probe_200_cutensornet_smoke.py
"""

from __future__ import annotations

import numpy as np


def _cutensor_contract_host(a_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    """Eager ``ij,jk->ik`` on the GPU via cupy's cuTENSOR-backed contraction.

    Receives host numpy (what ``pure_callback`` delivers), uploads to device,
    contracts with a real CUDA tensor library, returns host numpy. The upload/
    download is the host-roundtrip cost of the *callback* bridge (a real FFI
    handler would stay on-device); for P-B0 we only need to prove the call path
    and the value.
    """
    import cupy as cp

    a_d = cp.asarray(a_h)
    b_d = cp.asarray(b_h)

    path = "cupyx.cutensor"
    try:
        from cupyx import cutensor as cut

        da = cut.create_tensor_descriptor(a_d)
        db = cut.create_tensor_descriptor(b_d)
        out = cp.empty((a_d.shape[0], b_d.shape[1]), dtype=a_d.dtype)
        dc = cut.create_tensor_descriptor(out)
        # modes: a='ij', b='jk', c='ik'
        cut.contraction(
            1.0,
            a_d, da, ("i", "j"),
            b_d, db, ("j", "k"),
            0.0,
            out, dc, ("i", "k"),
        )
    except Exception:
        # Fall back to cp.einsum (still GPU, cuTENSOR/cuBLAS) so the *bridge*
        # is still exercised even if the low-level descriptor API differs by
        # cupy version. Probe reports which path ran.
        path = "cp.einsum"
        out = cp.einsum("ij,jk->ik", a_d, b_d)

    _cutensor_contract_host.path = path
    return cp.asnumpy(out)


def cutensor_matmul(a, b):
    """JAX-visible op: opaque GPU contraction via a single ``pure_callback``."""
    import jax

    out_shape = jax.ShapeDtypeStruct((a.shape[0], b.shape[1]), a.dtype)
    return jax.pure_callback(_cutensor_contract_host, out_shape, a, b)


def main() -> int:
    import jax
    import jax.numpy as jnp

    devs = jax.devices()
    print(f"devices: {devs}")
    on_gpu = any(d.platform == "gpu" for d in devs)
    print(f"x64 enabled: {jax.config.read('jax_enable_x64')}  on_gpu: {on_gpu}")

    rng = np.random.default_rng(0)
    ok = True
    for dt in (jnp.float64, jnp.complex128):
        a_np = rng.standard_normal((4, 5))
        b_np = rng.standard_normal((5, 3))
        if dt in (jnp.complex128,):
            a_np = a_np + 1j * rng.standard_normal((4, 5))
            b_np = b_np + 1j * rng.standard_normal((5, 3))
        a = jnp.asarray(a_np, dtype=dt)
        b = jnp.asarray(b_np, dtype=dt)

        f = jax.jit(cutensor_matmul)
        got = np.asarray(f(a, b))
        ref = np.asarray(jnp.einsum("ij,jk->ik", a, b))
        err = float(np.max(np.abs(got - ref)))
        passed = err < 1e-12
        ok = ok and passed
        kpath = getattr(_cutensor_contract_host, "path", "?")
        print(f"  {str(dt.__name__):11s} kernel={kpath:16s} max|Δ|={err:.2e}  "
              f"{'PASS' if passed else 'FAIL'}")

    # Confirm the contraction is ONE opaque op in the jaxpr (custom_call to the
    # callback), not lowered/fused by XLA — the property the backend relies on.
    jaxpr = jax.make_jaxpr(cutensor_matmul)(
        jnp.zeros((4, 5)), jnp.zeros((5, 3))
    )
    prims = [str(e.primitive) for e in jaxpr.jaxpr.eqns]
    n_callback = sum("callback" in p or "custom_call" in p for p in prims)
    print(f"  jaxpr primitives: {prims}")
    print(f"  opaque-callback ops in jaxpr: {n_callback} (expect >=1)")
    ok = ok and on_gpu and n_callback >= 1

    print(f"\nP-B0 smoke: {'PASS — JAX↔GPU contraction bridge works' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
