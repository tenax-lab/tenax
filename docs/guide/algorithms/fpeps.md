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
from tenax import FPEPSConfig, fpeps, spinless_fermion_gate, sublattice_gap
import jax

config = FPEPSConfig(D=2, t=1.0, V=4.0, dt=0.05, num_imaginary_steps=200,
                     ctm_chi=8, ctm_max_iter=60)
H = spinless_fermion_gate(config)
energy, (A, B), (env_A, env_B) = fpeps(H, config, key=jax.random.PRNGKey(0))
print(f"E/site = {energy:.8f}")
print(f"CDW gap = {sublattice_gap(A, B, env_A, env_B):.4f}")
```

**The state and environment are pairs** (#878). The t-V ground state at finite
``V`` is a checkerboard charge-density wave, which no single tensor can
represent, and the 1-site ansatz that preceded this made ``A`` both ends of
every bond — its update kept only ``U`` from each SVD, so ``A`` received the
left/top half of every gate and never the right/bottom half, and the state
collapsed to a product state regardless of ``dt``.

``sublattice_gap(A, B, env_A, env_B)`` reports whether a run actually produced a
checkerboard: the trace distance between the two sublattices' one-site reduced
density matrices, traced out of the two-site RDM the energy already uses. It is
0 for a uniform state and 1 for the fully polarised occupied/empty checkerboard.
Do **not** use ``||A - B||`` or a ``T T†`` leg fingerprint instead — neither is
invariant under the bond gauge ``T -> G T``, so both measure the representation
rather than the state.

Two caveats. The sweep is **seed-dependent** (over seeds 0–4 at 600 steps, the
surviving fraction is 4/5 at D=2, 2/5 at D=3, 4/5 at D=4, 4/5 at D=6), and the
**absolute energy is not certified** (#392): ``H`` carries no chemical
potential, so both the empty state and the fully polarised checkerboard are
exact ``E = 0`` eigenstates and the sweep is observed to settle on them.

## API

- ``fpeps(hamiltonian_gate, config, initial_tensor=None, key=None)`` — full
  pipeline: simple update + CTM + energy. Returns
  ``(energy, (A, B), (env_A, env_B))``. ``initial_tensor`` takes either an
  ``(A, B)`` pair — the form this returns, so its own output restarts it — or a
  single tensor, which starts both sublattices from the same place.
- ``sublattice_gap(A, B, env_A, env_B)`` — checkerboard/CDW diagnostic, above.
- ``spinless_fermion_gate(config)`` — build the t-V model gate from an
  ``FPEPSConfig`` (it reads ``t`` and ``V``).
- ``FPEPSConfig`` — configuration dataclass.
- ``optimize_fpeps_ad(hamiltonian_gate, A_init, config, fpeps_config=None)`` —
  AD-based ground-state optimization (1-site). For a 2-site unit cell use
  ``optimize_gs_ad(gate, (A, B), config)`` with ``config.unit_cell="2site"``.

## Performance: AD compile cost on symmetric tensors

The AD path differentiates through the CTM fixed point. The backward is
traced and XLA-compiled **once per optimizer run** (then reused across
steps), but for block-sparse ``SymmetricTensor`` site tensors that single
trace+compile scales with the **number of charge blocks** — hence with the
symmetry's sector count and with ``D``/``chi``. This is *not* specific to
fermions: any charge-conserving iPEPS AD (U(1), Zₙ, FermionParity) is
affected; fermionic tensors simply always carry non-trivial parity sectors.

Practical consequence: the **first** gradient step can take from seconds (a
single block) to many minutes (large ``D``/``chi`` with many blocks). With
``gs_verbose=True`` a one-time notice is printed before step 1 so the wait is
not mistaken for a hang. Subsequent steps reuse the compiled backward and are
fast. If the first step seems stuck, it is almost always compiling, not
deadlocked. The underlying compile-time scaling — and the plan to fix it via
sweep-level block batching — is tracked in
[issue #566](https://github.com/tenax-lab/tenax/issues/566).

## References

- Corboz et al., *Phys. Rev. B* **81**, 165104 (2010) — fermionic PEPS formalism.
- Barthel et al., *Phys. Rev. A* **80**, 042333 (2009) — graded tensor networks.
