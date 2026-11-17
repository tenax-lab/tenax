# Gotchas

Common pitfalls when working with Tenax and JAX.

## Float64 precision and `JAX_ENABLE_X64`

Tenax defaults to `float64` for all tensors and algorithms. Importing
`tenax` automatically calls `jax.config.update("jax_enable_x64", True)`,
so 64-bit arithmetic is enabled out of the box.

If you use JAX directly *before* importing `tenax`, you may need to enable
x64 mode yourself:

```python
import jax
jax.config.update("jax_enable_x64", True)

import tenax  # also enables x64, but JAX was already imported above
```

Without x64 enabled:

- Computations run at `float32` precision (no errors raised, just warnings).
- DMRG/iDMRG energies may converge to fewer significant digits.
- Convergence tolerances below ~1e-7 may not be reachable.

## MPO index convention

The MPO W-tensor uses the convention `W[w_l, ket, bra, w_r]`:

- The two outer indices (`w_l`, `w_r`) are MPO bond dimensions.
- The two middle indices are physical: ket (top) and bra (bottom).

The DMRG effective-Hamiltonian matvec einsum is
`"abc,apqd,bpse,eqtf,dfg->cstg"` where `p,q` are ket and `s,t` are bra
physical indices.

## NumPy >= 2.0 dtype casting

Under NumPy 2.0+, adding a Python `complex` scalar (even `1+0j`) into a
`float64` array raises `UFuncOutputCastingError`. Extract the `.real` part
or use an explicit `complex128` dtype:

```python
# Bad — raises UFuncOutputCastingError with NumPy >= 2.0
arr = np.zeros(3, dtype=np.float64)
arr[0] = 1 + 0j  # fails

# Good
arr[0] = (1 + 0j).real
# or
arr = np.zeros(3, dtype=np.complex128)
arr[0] = 1 + 0j
```

## Odd-circumference cylinders

The `build_bulk_mpo_heisenberg_cylinder` function only accepts **even** `Ly`.
Wrapping a square lattice with odd circumference creates odd-length cycles in
the ring direction, breaking bipartiteness and frustrating antiferromagnetic
(Neel) order. This leads to poor iDMRG convergence and physically different
ground states.

## Local test failures on macOS x86_64

`uv run pytest` may fail on macOS x86_64 if jaxlib has no wheel for that
platform. On macOS/headless runs, force the CPU backend with
`JAX_PLATFORMS=cpu uv run pytest ...`.

## Contractions of `2**31` entries or more on CPU

On the CPU backend (jax 0.10.2), XLA returns zeros, and raises no error, when
it transposes some arrays of more than `2**31` entries (#1193). `jnp.einsum`
makes such a transpose of an operand before a dot. The raw vertical RDM builder
met it on the spin-1 kagome PESS state (d = 27, D = 2) at chi = 16, before
#1193 made its largest array smaller: the vertical bond gave 0, and
`compute_energy_ctm` gave -0.1771777483 per site instead of -0.3543554967.
Measured on that transpose, `2**31` entries are correct and `2**31 + 4096`
entries are zeros.

So on CPU, the raw RDM builders of `compute_energy_ctm` and
`compute_energy_ctm_2site`, and of the deprecated `compute_excitations`, raise
`ValueError` when an operand, an intermediate or the output of one of their
contractions has `2**31` entries or more. The largest array of a builder has
$\max(D^8 d^2, \chi^2 D^6, \chi^2 D^2 d^2, d^4)$ entries: the double-layer
site, a boundary row with its three $D^2$ legs open, a half network with one
site absorbed, and the RDM. So for d = 27 and D = 2 the builders refuse from
chi = 859 (the half network), and for d = 2 and D = 6 from chi = 215 (the
boundary row).

`ipeps()` computes its energy with `compute_energy_ctm_2site` as its last
step. So on CPU, with `compute_energy=True`, it computes this bound before its
simple update, from `config.ctm.chi`, `config.max_bond_dim` and the d of the
gate. If the bound is `2**31` or more, it raises `ValueError` at once, and no
time goes into a run that cannot give an energy. Use a smaller chi or D, or a
GPU. Or pass `compute_energy=False`: then `ipeps()` returns the simple-update
state, with no CTM, no energy and no size check.

The check does not cover:

- other contractions: the simple update (with `compute_energy=False` too), the
  CTM moves, `compute_energy_ctm_tensor` (the energy of the AD optimizers),
  `compute_spectrum` and `tenax.contraction`;
- the batched arrays under `jax.vmap`, because the check sees the per-example
  shapes. `compute_excitations` vmaps its builders over the $D^4 d$ basis
  vectors;
- under `jax.jit`, the device of the arguments: the check takes the default
  device.

## Symmetric tensors

Which legs may be contracted, the opt-in `TENAX_STRICT_CONTRACT=1` audit mode,
and the charge-grouped `bond_order="sector"` mode of the block-sparse
`svd`/`eigh` are covered in {doc}`symmetry`.

