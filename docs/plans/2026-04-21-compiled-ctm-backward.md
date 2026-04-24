# Compiled CTM Backward Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce CTM backward JIT compilation from 30+ min to <2 min by compiling label-based Tensor contractions into raw-array einsum at Python level, so JAX traces a lean XLA graph.

**Architecture:** Add a `CompiledBlueprint` that extends `NetworkBlueprint` with a `launch_raw(arrays)` method operating on raw `jnp.ndarray` (no Tensor wrapping). Each CTM move is defined as a `.net`-style blueprint compiled once at Python level; the backward VJP traces only raw einsum calls. The GMRES loop runs in Python calling JIT'd matvec (no nested `while_loop` compilation).

**Tech Stack:** JAX, opt_einsum, existing NetworkBlueprint parser

---

## Background

The CTM backward currently JIT-compiles a monolithic function containing:
- VJP of 4 CTM moves (each with 5-6 label-based `contract()` calls + relabels + fuse)
- Each `contract()` re-resolves labels to einsum subscripts inside the JIT trace
- Inline phase fix adds `todense()` + `ravel()` + `argmax()` per tensor

This generates a huge XLA graph (30+ min compile at chi=16). variPEPS uses raw arrays and compiles the same backward in ~60s.

## Design

### Core: `CompiledBlueprint.launch_raw(*arrays) -> jnp.ndarray`

Extend `NetworkBlueprint` with a method that:
1. Accepts raw `jnp.ndarray` in declaration order (no Tensor protocol)
2. Executes pre-compiled einsum subscripts via `opt_einsum.contract` on raw arrays
3. Returns raw `jnp.ndarray`

This means the JIT trace sees only `jnp.einsum` / `jnp.tensordot` calls — no label resolution, no Tensor wrapping, no relabeling.

### CTM move compilation

Each CTM move is a sequence of sub-contractions. Instead of one giant einsum (which can be suboptimal), express each move as a sequence of `CompiledBlueprint` steps:

1. **Corner growth**: `C1·T1 → C1g`, `C4·T3 → C4g`  
2. **Edge growth**: `T4·a → T4g`
3. **Fuse legs**: reshape (not a contraction, just reshape + transpose)
4. **Projector**: SVD/eigh of C1g, C4g (unchanged — already raw-array internally)
5. **Apply projector**: `P·C1g → C1_new`, etc.
6. **Phase fix**: raw-array norm + argmax (no Tensor wrapping)

Steps 1, 2, 5 are contractions → compiled blueprints. Steps 3, 6 are reshapes/elementwise ops → direct raw-array code.

### Backward structure

```
f_bwd(residuals, g):
    # Python-level backward — each piece JIT'd separately
    dE_denv = _jit_dE_denv(...)       # small JIT: energy VJP
    
    def matvec(v):                     # called by GMRES
        return _jit_apply_Jt(...)      # JIT'd: VJP of compiled sweep
    
    lam = jax_gmres(matvec, dE_denv)   # Python-level GMRES
    
    return _jit_chain_rule(...)        # small JIT: direct + indirect
```

The `_jit_apply_Jt` function traces through `compiled_sweep_raw(arrays) -> arrays` which uses only raw einsum — lean XLA graph, fast compile.

---

### Task 1: Add `launch_raw` to NetworkBlueprint

**Files:**
- Modify: `src/tenax/network/netfile.py`
- Test: `tests/test_network.py`

**Step 1: Write the failing test**

```python
def test_blueprint_launch_raw():
    """launch_raw should produce the same result as launch but on raw arrays."""
    bp = NetworkBlueprint("""
    A: i, j
    B: j, k
    TOUT: i, k
    """)
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    
    # Tensor path
    from tenax.core import DenseTensor, TensorIndex, FlowDirection
    idx_i = TensorIndex.from_trivial(2, FlowDirection.OUT, "i")
    idx_j = TensorIndex.from_trivial(2, FlowDirection.IN, "j")
    idx_k = TensorIndex.from_trivial(2, FlowDirection.OUT, "k")
    tA = DenseTensor(A, (idx_i, idx_j))
    tB = DenseTensor(B, (idx_j, idx_k))
    bp.put_tensor("A", tA)
    bp.put_tensor("B", tB)
    result_tensor = bp.launch()
    
    # Raw path
    result_raw = bp.launch_raw(A, B)
    
    np.testing.assert_allclose(result_tensor.todense(), result_raw, atol=1e-12)
```

**Step 2: Implement `launch_raw`**

Add to `NetworkBlueprint`:

```python
def launch_raw(self, *arrays: jnp.ndarray, optimize: str = "auto") -> jnp.ndarray:
    """Contract using pre-computed subscripts on raw arrays (no Tensor protocol).
    
    Arrays must be passed in tensor declaration order. This produces a
    minimal XLA graph suitable for JIT tracing in AD backward passes.
    """
    if len(arrays) != len(self._node_order):
        raise ValueError(
            f"Expected {len(self._node_order)} arrays, got {len(arrays)}"
        )
    shapes = tuple(a.shape for a in arrays)
    path = _cached_contraction_path(self._subscripts, shapes, optimize)
    return opt_einsum.contract(self._subscripts, *arrays, optimize=path)
```

**Step 3: Run test, commit**

---

### Task 2: Compile CTM moves as blueprint sequences

**Files:**
- Create: `src/tenax/algorithms/_ctm_compiled_moves.py`
- Test: `tests/test_ctm_compiled.py`

**Step 1: Define CTM left-move blueprint**

Each CTM move is a function `(C1, C2, C3, C4, T1, T2, T3, T4, a, chi) -> (C1', C4', T4')` operating on raw arrays. The contractions use pre-compiled einsum subscripts.

```python
def compiled_ctm_move_left(env_self_arrays, env_nb_arrays, a_array, chi, 
                            projector_method, projector_backward):
    """Left CTM move on raw arrays.
    
    env_self_arrays: (C1, C2, C3, C4, T1, T2, T3, T4) raw ndarrays
    env_nb_arrays:   same, from neighbor site
    a_array:         double-layer tensor (D², D², D², D²) raw ndarray
    
    Returns: updated (C1, C2, C3, C4, T1, T2, T3, T4) tuple
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env_self_arrays
    _, _, _, _, T1_nb, _, T3_nb, _ = env_nb_arrays
    
    # Corner growth: C1(ab) · T1_nb(buc) -> C1g(a,u,c)
    C1g = jnp.einsum("ab,buc->auc", C1, T1_nb)
    # Fuse: (a,u) -> fused
    C1g = C1g.reshape(C1g.shape[0] * C1g.shape[1], C1g.shape[2])
    
    # ... (similar for C4g, T4g)
    # Projector computation (reuse existing _compute_projector_raw)
    # Apply projector
    # Phase fix normalize
    ...
```

The exact einsum subscripts depend on the leg convention. We extract them from the existing label-based moves by tracing `_labels_to_subscripts` once.

**Step 2: Write test comparing compiled vs label-based**

```python
def test_compiled_left_move_matches_tensor_move():
    """Compiled raw-array move must match label-based Tensor move."""
    # Random env + double-layer tensor
    # Run both paths, compare outputs
```

**Step 3: Implement all 4 directions, full sweep**

```python
def compiled_ctm_sweep(site_arrays, env_arrays, neighbors, chi, ...):
    """Full CTM sweep on raw arrays. Returns updated env_arrays."""
```

---

### Task 3: Raw-array backward with compiled sweep

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py`
- Test: `tests/test_ipeps.py` (existing 2-site tests)

**Step 1: Replace `_jit_apply_Jt` with compiled sweep**

The `gauge_fixed_sweep_from_env` function currently uses label-based `_ctm_tensor_sweep_multisite`. Replace the inner call with `compiled_ctm_sweep` operating on raw arrays:

```python
@jax.jit
def _jit_apply_Jt(params_arrays, env_arrays, v):
    def sweep_raw(env_flat):
        # unflatten env_flat -> per-site raw arrays
        env_out = compiled_ctm_sweep(params_arrays, env_dict, neighbors, chi, ...)
        env_out = phase_fix_raw(env_out)  # raw-array phase fix
        # flatten back
        return env_out_flat
    
    _, vjp_fn = jax.vjp(sweep_raw, env_arrays)
    jt_v = vjp_fn(v)[0]
    return tuple(vi - ji for vi, ji in zip(v, jt_v))
```

**Step 2: Test compilation time**

Run 2-site chi=16 and verify:
- Compile time < 2 min (vs current 30+ min)
- Gradients finite (JAX GMRES via Python loop)
- Energy variational

**Step 3: Benchmark vs variPEPS**

Compare step time and gradient quality against the variPEPS reference.

---

### Task 4: Integration and cleanup

**Files:**
- Modify: `src/tenax/algorithms/_ctm_energy_ad.py` (wire up)
- Modify: `src/tenax/algorithms/ipeps_optimize.py` (use new backward)

**Step 1: Make compiled backward the default for implicit AD**

**Step 2: Run full test suite**

```bash
uv run pytest -m core -x
```

**Step 3: Benchmark C4v and non-C4v at chi=8 and chi=16**

**Step 4: Commit and update PR**

---

## Key risks

1. **Leg ordering**: The raw einsum subscripts must match the exact axis ordering of env arrays. Off-by-one transpose → wrong gradients. Mitigated by Task 2's comparison test.

2. **Projector computation**: SVD/eigh projectors operate on raw arrays internally already (via `todense()`), but the input fused tensors need correct reshape. Must match the label-based fuse exactly.

3. **Symmetric tensor path**: The compiled moves work on dense arrays only. For SymmetricTensor envs, we densify at the boundary (already done in `ctm_energy_implicit`).
