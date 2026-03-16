# Fermionic iPEPS (fPEPS)

Fermionic iPEPS extends the iPEPS algorithm to systems with fermionic
statistics using graded tensors that automatically handle anticommutation
(Koszul signs).

## Background

Standard tensor networks assume bosonic statistics — contracting two tensors
does not depend on the order of legs. For fermions, exchanging two legs picks
up a minus sign when both carry odd parity. Tenax's ``SymmetricTensor`` with
``FermionParity`` or ``FermionicU1`` symmetry tracks these signs automatically
through the ``bar()``, ``contract()``, and ``fuse_indices()`` operations.

Key properties:

- **Graded tensor formalism**: Koszul signs are computed from ``FlowDirection``
  and fermionic charge parity during contraction. No explicit Jordan-Wigner
  strings needed.
- **Spinless fermion gate** (``spinless_fermion_gate``): pre-built 2-site
  Hamiltonian for the t-V model with ``FermionParity`` symmetry.
- **Simple update**: imaginary-time evolution on the square lattice, identical
  to the bosonic path but using ``SymmetricTensor`` throughout.
- **CTM environment**: uses ``ctm_tensor`` (with automatic densify workaround
  for fermionic symmetries in the general 4-move path) or ``ctm_tensor_c4v``
  (single-move, no workaround needed).

## Configuration

```python
from tenax import FPEPSConfig

config = FPEPSConfig(
    D=2,                    # virtual bond dimension
    t=1.0,                  # hopping amplitude
    V=0.5,                  # nearest-neighbor interaction
    dt=0.05,                # imaginary time step
    num_imaginary_steps=200,
    ctm_chi=16,             # CTM bond dimension
    ctm_max_iter=50,
    ctm_conv_tol=1e-8,
)
```

## Example — spinless fermion t-V model

```python
from tenax import FPEPSConfig, fpeps, spinless_fermion_gate
import jax

config = FPEPSConfig(D=2, t=1.0, V=0.5, dt=0.01, num_imaginary_steps=500,
                     ctm_chi=16, ctm_max_iter=60)
H = spinless_fermion_gate(t=1.0, V=0.5)
energy, A_opt, env = fpeps(H, config, key=jax.random.PRNGKey(0))
print(f"E/site = {energy:.8f}")
```

## API

- ``fpeps(hamiltonian_gate, config, initial_tensor=None, key=None)`` — full
  pipeline: simple update + CTM + energy.
- ``spinless_fermion_gate(t, V, dt=None)`` — build the t-V model gate.
- ``FPEPSConfig`` — configuration dataclass.

## References

- Corboz et al., *Phys. Rev. B* **81**, 165104 (2010) — fermionic PEPS formalism.
- Barthel et al., *Phys. Rev. A* **80**, 042333 (2009) — graded tensor networks.
