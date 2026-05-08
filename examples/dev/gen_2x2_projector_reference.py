"""Generate variPEPS 2x2 projector reference on a fixed-seed random tensor.

NOT FOR COMMIT (the .npz output). Outputs tests/_ctm_2x2_reference_data.npz
which is gitignored and used by downstream tier-2 sanity checks.

Run on GPU 1 with miniforge Python (variPEPS dep):
  CUDA_VISIBLE_DEVICES=1 /home/yjkao/miniforge3/bin/python \
      examples/dev/gen_2x2_projector_reference.py
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import varipeps
from varipeps.ctmrg import calc_ctmrg_env
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell


def main():
    D, chi, d = 2, 8, 2
    rng = np.random.default_rng(42)
    A = rng.standard_normal((D, D, d, D, D)) + 1j * rng.standard_normal((D, D, d, D, D))
    A = A / np.linalg.norm(A)
    A = jnp.asarray(A)

    pt = PEPS_Tensor.from_tensor(
        A, d=d, D=(D,) * 4, chi=chi, ctm_tensors_are_identities=True
    )
    uc = PEPS_Unit_Cell.from_tensor_list([pt], structure=((0,),))
    varipeps.varipeps_config.ctmrg_max_steps = 80
    varipeps.varipeps_config.ctmrg_convergence_eps = 1e-9
    arrs = [pt.tensor for pt in uc.get_unique_tensors()]
    res = calc_ctmrg_env(arrs, uc, eps=1e-9, enforce_elementwise_convergence=True)
    new_uc = res[0] if isinstance(res, tuple) else res
    pt_out = new_uc.get_unique_tensors()[0]

    out = Path("tests/_ctm_2x2_reference_data.npz")
    out.parent.mkdir(exist_ok=True)
    np.savez(
        out,
        A=np.asarray(A),
        C1=np.asarray(pt_out.C1),
        C2=np.asarray(pt_out.C2),
        C3=np.asarray(pt_out.C3),
        C4=np.asarray(pt_out.C4),
        T1=np.asarray(pt_out.T1),
        T2=np.asarray(pt_out.T2),
        T3=np.asarray(pt_out.T3),
        T4=np.asarray(pt_out.T4),
        D=D,
        chi=chi,
        d=d,
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
