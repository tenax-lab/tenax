# examples/_e_frag_610.py — record the unconstrained-truncation baseline energy.
import jax
from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax import compute_energy_ctm_tensor
from tenax.algorithms.ipeps import heisenberg_gate_u1sz, heisenberg_u1sz_init_pair

jax.config.update("jax_enable_x64", True)
A, _B = heisenberg_u1sz_init_pair(D=3, key=jax.random.PRNGKey(0))
env, _ = ctm_tensor(A, chi=12, max_iter=20, conv_tol=1e-7)
gate = heisenberg_gate_u1sz()
print("E_frag(D=3,chi=12) =", float(compute_energy_ctm_tensor(A, env, gate)))
