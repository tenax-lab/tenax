# Symmetry System

Tenax's block-sparse `SymmetricTensor` stores only the charge sectors a
symmetry allows. The symmetry object defines how charges fuse and which blocks
are conserved; {doc}`core_concepts` introduces symmetries, indices and tensors,
and this page collects the rules that matter once you go beyond the basics:
fermionic signs, charge arithmetic for extension authors, which legs may be
contracted, and how the block-sparse decompositions order their output bond.

## Built-in symmetries

`U1Symmetry` (integer charges, fusion by addition), `ZnSymmetry(n)` (charges
mod `n`), `ProductSymmetry` (two factors, e.g. charge × S_z), and the fermionic
`FermionParity` / `FermionicU1`:

```python
from tenax import U1Symmetry, ZnSymmetry, ProductSymmetry, FermionParity
import numpy as np

# U(1): integer charges, fusion by addition
u1 = U1Symmetry()
charges = np.array([-1, 0, 1], dtype=np.int32)
print(u1.fuse(charges, charges))  # [-2, 0, 2]
print(u1.dual(charges))  # [1, 0, -1]

# Z_3: charges mod 3
z3 = ZnSymmetry(3)
print(
    z3.fuse(np.array([1, 2], dtype=np.int32), np.array([2, 2], dtype=np.int32))
)  # [0, 1]

# Product symmetry: combine two symmetries (e.g., charge × S_z)
sym = ProductSymmetry(U1Symmetry(), U1Symmetry())
packed = ProductSymmetry.encode_charges(
    np.array([0, 1, -1], dtype=np.int32),  # charge
    np.array([1, 0, -1], dtype=np.int32),  # S_z
)
q1, q2 = ProductSymmetry.decode_charges(packed)
```

## Fermionic swap gates

`SymmetricTensor.swap_gate(axes=(i, j))` multiplies each block by
`(-1)**(p_i * p_j)` — a minus sign exactly when *both* crossing legs carry
odd parity. This is the Corboz-style build-time encoding of fermionic
exchange statistics: place the sign where two fermionic lines cross in the
(fixed) network diagram, and the rest of the contraction needs no graded
logic. For an adjacent leg exchange it reproduces the Koszul sign of the
graded `transpose` exactly. The optional `grading=({charge: parity}, ...)`
override supplies the parity maps explicitly — needed by pipelines that
retype graded tensors onto bosonic symmetry objects, where `parity()` is
all-even by definition (see `docs/plans/2026-09-12-fermionic-ctm-ad-swap-gates-design.md`).

```python
import jax
import numpy as np
from tenax import FermionParity, FlowDirection, SymmetricTensor, TensorIndex

fp = FermionParity()
charges = np.array([0, 0, 1, 1], dtype=np.int32)  # both parities on each leg
idx = lambda flow, lbl: TensorIndex.from_charges(fp, charges, flow, label=lbl)
T = SymmetricTensor.random_normal(
    indices=(idx(FlowDirection.OUT, "a"), idx(FlowDirection.IN, "b")),
    key=jax.random.PRNGKey(0),
)

G = T.swap_gate((0, 1))  # odd-odd blocks flip sign, others unchanged

# involution: applying the same gate twice restores the tensor
assert np.allclose(np.asarray(G.swap_gate((0, 1))._data), np.asarray(T._data))

# adjacent-exchange identity: the graded transpose's Koszul sign IS the
# swap gate — transpose(T) block-equals sign-free-permute(swap_gate(T))
graded = T.transpose((1, 0))
```

(the-twist)=
## The twist (non-planar diagrams)

`Tensor.twist(axes)` multiplies each block by `(-1)**(sum of the parities of
that block's charges on `axes`)` — the categorical twist, matching TensorKit's
`twist(t, i)` and the `twist(F_west, 3)` PEPSKit applies when fusing a ket/bra
sandwich. It is the primitive #555 deferred when it removed the contractor's
automatic Koszul tracking:

> For planar networks — the only kind tenax's CTM/RDM/energy code uses — no
> signs are needed … For future non-planar applications an explicit `twist`
> primitive can be added.

**Planar networks do not need it.** Reach for it only where a diagram is *not*
planar, because there `FermionParity`'s R-symbol does contribute: a periodic
(torus) contraction wraps legs past one another, and the wrap crossings carry
signs `contract` does not apply. A periodic fermionic reference that skips them
is not ground truth — on the #995 adjudication the periodic and planar oracles
disagreed by up to 13x and reversed which CTM convention they favoured.

**Twisting *every* leg of a charge-conserving tensor is the identity**, since
the total parity is even. The operation can therefore only act through an
imbalance across a cut, which is what makes it safe to apply to one side of a
wrap bond — and why applying it to a whole closed diagram does nothing.

```python
import jax
import numpy as np
from tenax import FermionParity, FlowDirection, SymmetricTensor, TensorIndex

fp = FermionParity()
charges = np.array([0, 1], dtype=np.int32)
idx = lambda flow, lbl: TensorIndex.from_charges(fp, charges.copy(), flow, label=lbl)
T = SymmetricTensor.random_normal(
    indices=(idx(FlowDirection.OUT, "a"), idx(FlowDirection.IN, "b")),
    key=jax.random.PRNGKey(0),
)

W = T.twist((0,))  # blocks whose leg-0 charge is odd flip sign

# involution: twisting the same leg twice restores the tensor
assert all(
    np.allclose(np.asarray(W.twist((0,)).blocks[k]), np.asarray(v))
    for k, v in T.blocks.items()
)

# all legs at once is the identity -- parity is conserved
assert all(
    np.allclose(np.asarray(T.twist((0, 1)).blocks[k]), np.asarray(v))
    for k, v in T.blocks.items()
)
```

**This is the fermionic twist only.** The sign `(-1)**p` is the ribbon element
of a Z2-graded category and nothing more general. A bosonic symmetry — and any
`DenseTensor` — is returned unchanged, which is correct: with no grading the
twist *is* the identity. A symmetry declaring `BraidingStyle.ANYONIC` raises
`NotImplementedError` instead, because its `twist_phase()` is a general complex
phase that this sign cannot represent, and silently returning the tensor
unchanged there would be wrong rather than trivial. Supporting it would also
cost the two properties above: the twist would no longer be its own inverse
(the inverse is the conjugate), and the all-legs identity rests on Z2 parity
summing to even.

## Charge arithmetic

`BaseSymmetry` is the sanctioned boundary for every charge operation. Extension
authors should call these rather than hand-rolling the arithmetic — the
hand-rolled forms assume the group inverse is integer negation and the group
operation is integer addition, which is true for U(1), accidentally true for
`Z_n`, and false for the bit-packed charges of `ProductSymmetry`.

```python
from tenax import U1Symmetry
import numpy as np

sym = U1Symmetry()
charges = np.array([-1, 0, 2], dtype=np.int32)

# Weight a charge by its leg's flow: IN (+1) unchanged, OUT (-1) inverted.
# Use this instead of `int(flow) * charge`.
sym.flow_charge(-1, charges)            # [1, 0, -2]

# Reduce to the canonical representative (`% n` for Z_n, identity for U(1)).
sym.canonicalize_charges(charges)

# Evaluate a conservation law. A block is valid exactly when the net charge
# equals `identity()`. Use this instead of `sum(flow * q for ...)`.
sym.net_charge([1, 1], flows=[1, -1])   # 0
sym.is_conserved([1, 1], flows=[1, -1]) # True
```

**Charge width.** Charges are *stored* as `int32`. Intermediate arithmetic in
the conservation law uses `charge_accumulator_dtype`, which is `int64` for U(1)
and `FermionicU1` — whose charges are unbounded by definition — and `int32`
elsewhere, since `Z_n` reduces mod `n` and `ProductSymmetry`'s charges are
bounded by their packing. This puts the overflow ceiling at 2⁶³ rather than
2³¹; it does not remove it.

**Limitations:** `ProductSymmetry` combines exactly two factors by bit-packing two int16 charges into one int32. Nesting is not supported, so three-factor groups (e.g., U(1)×U(1)×Z₂) require a future `MultiProductSymmetry`. Each factor charge must fit in the int16 range [-32768, 32767].

(which-legs-may-be-contracted)=
## Which legs may be contracted

Two symmetric legs may be contracted when they have **opposite flows and
identical charges** — what `flip_flow()` on a `TensorIndex`, or `bar()` on a
tensor, produces. This is not the same as `is_dual_of()` / `dual()` / `dagger()`,
which negate the charges: block-sparse contraction pairs blocks by charge
*value* while dense contraction pairs by *position*, and negation permutes the
position→charge map, so the two representations then compute different sums.

```python
from tenax import FlowDirection, SymmetricTensor, TensorIndex, U1Symmetry, contract
import jax, numpy as np

sym = U1Symmetry()
charges = np.array([-1, 0, 1], dtype=np.int32)
free_a = TensorIndex.from_charges(sym, charges, FlowDirection.OUT, label="i")
free_b = TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="j")
shared = TensorIndex.from_charges(sym, charges, FlowDirection.IN, label="k")

A = SymmetricTensor.random_normal((free_a, shared.flip_flow()), jax.random.PRNGKey(0))
B = SymmetricTensor.random_normal((shared, free_b), jax.random.PRNGKey(1))
contract(A, B)          # `k` is OUT on A and IN on B, with identical charges
```

Mixing the conventions makes `contract()` return a representation-dependent
answer, silently (#834). Set `TENAX_STRICT_CONTRACT=1` to make it raise
`ValueError` instead — naming both legs — when the two representations would
disagree:

```bash
TENAX_STRICT_CONTRACT=1 python my_script.py
```

It is opt-in rather than the default because the checks are structural while the
disagreement depends on the blocks' values: the CTM initial environment contracts
non-dual bonds and discards products by the thousand, and is exact anyway because
those products are all zero. Turn it on when auditing a path, not in production.

While armed it also forces the reference per-block contraction, overriding the
accelerated block-sparse backends (`TENAX_BATCH_BLOCKSPARSE`,
`TENAX_STACK_BLOCKSPARSE`, `TENAX_USE_CUTENSOR_BLOCKSPARSE`) for the duration.
Those paths drop out-of-set output keys without consulting the check, so an
audit that left them enabled would report clean on the products it never
inspected — and a diagnostic whose silence is unreliable is worse than none.

(eigh-bond-order)=
## Bond ordering of a block-sparse `eigh`

`tenax.linalg.eigh` returns its eigenvalues **algebraically descending** by
default — largest first, so a negative eigenvalue sorts below every positive one
whatever its magnitude — and lays the output bond out in that order. On a
`SymmetricTensor` that ranking is a comparison *across* charge sectors, so it
reads the eigenvalues on the host, and that raises under `jax.jit`. It is why a
block-sparse `eigh` cannot appear in a traced computation.

Pass `bond_order="sector"` to get the bond charge-grouped instead:

```python
from tenax.linalg import eigh

V, w = eigh(m, ["row"], ["col"], new_bond_label="k", bond_order="sector")
```

`"sector"` is **not value-ordered at all**: sectors come in ascending charge
order and each keeps `jnp.linalg.eigh`'s own ascending output, so `w[0]` is not
the largest and the array is not monotone. On an indefinite operator with
sectors `{0: [-5, -3], 1: [2, 0.5]}` the default returns `[2, 0.5, -3, -5]` and
`"sector"` returns `[0.5, 2, -5, -3]`.

The two modes differ only by a permutation of the bond — `V` and `w` are permuted
together, and `V diag(w) V†` is unchanged — so nothing that pairs the two is
affected. Anything that reads `w[0]` as "the largest", or assumes the array is
sorted, is.

Two constraints:

- It is **rejected with `max_eigenvalues`**, because a truncation has to rank the
  sectors against each other; that is exactly the host read the option exists to
  avoid. Without a truncation the ranking decides nothing, which is what makes
  the option safe.
- It is **ignored on the dense path**, which has no sectors to group by and is
  traceable already.

The caller this exists for is `ipeps_bp_gauge._sqrt_pinv`, which factors a PSD
message and never truncates.

(svd-bond-order)=
## Bond ordering of a block-sparse `svd`

`tenax.linalg.svd` has the same pair of modes, for the same reason: the default
ranks the whole spectrum on the host, which raises under `jax.jit`, and under a
tracer the block-sparse path is silently rerouted to a static-allocation
variant whose per-sector SVD applies a subrank floor — real singular values
below `1e-12 · (s_max + 1e-30)` come back **exactly zero**, which on a 1×1
sector makes the `+1e-30` term an absolute ~1e-42 cutoff.

```python
from tenax.linalg import svd

U, s, Vh, s_full = svd(t, ["row"], ["col"], new_bond_label="k", bond_order="sector")
```

`"sector"` emits the bond charge-grouped — ascending by the bond charge each
sector carries, values **descending within** each sector — and takes the same
code path eager and traced, so no reroute and no floor: a 4.6e-43 singular
value comes back as itself. The array is not globally monotone, `s[0]` is not
the largest, and `s_full` **is** `s` (nothing was truncated). As with `eigh`,
the two modes differ only by a permutation of the bond, with `U`, `s`, `Vh`
permuted together.

Constraints, one more than `eigh`'s:

- It is **rejected with `max_singular_values` and with `max_truncation_err`** —
  both truncation knobs rank sectors against each other on the host.
- It is **ignored on the dense path**, which has no sectors to group by.
- Reverse-mode AD through sector mode uses the default SVD JVP, not the
  Lorentzian-regularized `truncated_svd_ad`; do not differentiate it at
  degenerate spectra.

The caller this exists for is `ipeps_bp_gauge._gauge_bond`, which re-gauges a
bond at full rank and never truncates.
