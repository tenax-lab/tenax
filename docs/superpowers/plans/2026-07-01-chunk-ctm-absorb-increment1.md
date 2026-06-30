# Chunked dense-CTM absorb (Increment 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-OFF `chunk_size` knob that runs the expensive χ²·D⁶ edge absorption of the production dense 1×1 CTM moves via a chunked `lax.map` over the boundary-χ axis, composing with the existing GSPMD mesh so per-device peak memory drops ≈÷(N·K) at large D.

**Architecture:** The gate (`docs/superpowers/handoffs/2026-06-30-chunk-shard-ctm-move-findings.md`, STRONG GO) validated the lever on raw einsum (`_compiled_move_*`, which is *test-only dead code*). Production runs the label-based `Tensor.contract()` absorb path. So we add a small raw-array chunked core (`_ctm_chunked_absorb.py`) that faithfully reproduces `contract(edge, a)` + `_apply_projector_tensor`'s `T_new`, and call it from the four 1×1 `_ctm_tensor_move_*` functions **only when** `chunk_size` is set AND the env is a `DenseTensor` (the large-D multi-GPU use case; symmetric large-D is YASTN territory). Default-off path is byte-for-byte unchanged. The knob threads through `CTMConfig.ctm_chunk_size` → `ctm_converge_kwargs` → `ctm_energy_implicit`/`python_loop_ctm_converge` → `_make_jit_ctm_step` → `_ctm_tensor_sweep_multisite` → the 1×1 move, exactly mirroring how `device_mesh` is threaded today. We also wire the per-move re-shard (`_shard_a`) into the 1×1 sweep branch (currently 2×2-only) so chunk composes with the mesh on the 1×1 recipe the multi-GPU dense studies use.

**Tech Stack:** Python, JAX (`jax.lax.map(..., batch_size=...)`, `jnp.einsum`/`tensordot`), Tenax `DenseTensor`/`TensorIndex`, pytest (`-m core`).

**Scope:** Increment 1 = **forward** only, **dense** only, recipe **1×1** first. Out of scope: Increment 2 (chunked backward through the implicit-AD adjoint — gate-first, separate plan), 2×2 recipe, symmetric/fermionic envs, and the `optimize_gs_ad` multi-GPU benchmark.

---

## Background facts the executor needs (verified against source)

Production 1×1 moves live in `src/tenax/algorithms/_ctm_tensor_moves.py`:
`_ctm_tensor_move_left` (line 997), `_ctm_tensor_move_right` (1052), `_ctm_tensor_move_top` (1106), `_ctm_tensor_move_bottom` (1160).

Each has the same shape. For LEFT (the reference):
```python
# THE χ²·D⁶ PEAK:
T4_with_a = contract(env_self.T4, a)                       # (t4_d, t4_u, u2, d2, r2)
T4g = _fuse_pair_by_label(T4_with_a, "t4_d", "u2", "fl", IN)
T4g = _fuse_pair_by_label(T4g, "t4_u", "d2", "fr", OUT)    # (fl, fr, r2)
P_1, P_2, _eps_t = _compute_projector_tensor(C1g, C4g, chi, projector_method, base_charges, projector_backward)
C1_new, C4_new, T4_new = _apply_projector_with_reembed(P_1, P_2, C1g, C4g, T4g, "fl", "fr")
```
`_apply_projector_with_reembed` for `DenseTensor` delegates to `_apply_projector_tensor` (line 253), which computes:
```python
P1_bar = P_1.bar(); P2_bar = P_2.bar()
C1_new = contract(P1_bar, C1g)                  # cheap, no peak
C4_new = contract(P2_bar, C4g)                  # cheap, no peak
P_left = P1_bar.relabel("fused", fused_l); step = contract(P_left, Tg)      # uses Tg (peak)
P_right = P_2.relabels({"fused": fused_r, "chi_new": "chi_new_r"}); T_new = contract(step, P_right)
# T_new labels: (chi_new, <surviving D2>, chi_new_r)
```
So the chunked core must reproduce `T_new = conj(P_1)ᵀ · (edge·a) · P_2` (P_1 is barred = conj data + flipped flow; P_2 is **not** barred), chunked over the boundary-χ axis, **without** materializing `edge·a`. The corners `C1_new/C4_new` are cheap and identical whether chunked or not.

**Per-direction map** (canonical edge leg order; `a` canonical `(u2, d2, l2, r2)`; per-chunk einsum contracts the shared `a` leg; output of the full einsum is `(chunk_χ, other_χ, <3 free D² legs>)`):

| dir | edge (canonical) | shared `a` leg | full einsum | `fl`=(χ_chunk, D²) | `fr`=(χ_other, D²) | surviving D² | T_new relabels | flow flip |
|-----|------------------|----------------|-------------|--------------------|--------------------|--------------|----------------|-----------|
| left   | T4 `(t4_d, l2, t4_u)` | l2 | `"ijk,lmjn->iklmn"` | `(t4_d, u2)` | `(t4_u, d2)` | `r2` | `chi_new→t4_d, chi_new_r→t4_u, r2→l2` | flip `l2` |
| right  | T2 `(t2_u, r2, t2_d)` | r2 | `"ijk,lmnj->iklmn"` | `(t2_u, u2)` | `(t2_d, d2)` | `l2` | `chi_new→t2_u, chi_new_r→t2_d, l2→r2` | flip `r2` |
| top    | T1 `(t1_l, u2, t1_r)` | u2 | `"ijk,jlmn->iklmn"` | `(t1_l, l2)` | `(t1_r, r2)` | `d2` | `chi_new→t1_l, chi_new_r→t1_r, d2→u2` | flip `u2` |
| bottom | T3 `(t3_r, d2, t3_l)` | d2 | `"ijk,ljmn->iklmn"` | `(t3_r, l2)` | `(t3_l, r2)` | `u2` | `chi_new→t3_r, chi_new_r→t3_l, u2→d2` | flip `d2` |

The relabel + flow-flip + `_phase_fix_normalize_tensor` tail in each production move is **reused verbatim** in the chunked branch — only the `T*_new` *value* and its three `TensorIndex` objects are produced differently.

`DenseTensor(data, indices)` (`src/tenax/core/tensor.py:468`): `data` is a JAX array, `indices` a tuple of `TensorIndex`. `.todense()` returns the raw array; `.transpose(axes)` permutes; `.labels()`/`.indices` expose metadata; `TensorIndex.relabel(new)` and `.flip_flow()` exist.

---

## File Structure

- **Create** `src/tenax/algorithms/_ctm_chunked_absorb.py` — raw-array chunked edge-absorption core (one function per direction) + a small raw-extraction helper. One clear responsibility: compute the new edge `T_new` array via `lax.map` over boundary-χ.
- **Modify** `src/tenax/algorithms/_ctm_tensor_moves.py` — add `chunk_size` param + dense-gated chunked branch to the four 1×1 `_ctm_tensor_move_*`.
- **Modify** `src/tenax/algorithms/_ctm_tensor_convergence.py` — thread `chunk_size` into `_ctm_tensor_sweep_multisite` and the 1×1 dispatch loop; wire `_shard_a` into the 1×1 branch.
- **Modify** `src/tenax/algorithms/_ctm_python_loop.py` — add `ctm_chunk_size` to `_make_jit_ctm_step` (cache key + static arg) and `python_loop_ctm_converge`.
- **Modify** `src/tenax/algorithms/ipeps_config.py` — add `CTMConfig.ctm_chunk_size` field + docstring.
- **Modify** `src/tenax/algorithms/ipeps_ad_policy.py` — forward `ctm_chunk_size` in `ctm_converge_kwargs` and the `ctm_energy_implicit` call.
- **Create** `tests/test_ctm_chunked_absorb.py` — parity unit tests (core + full move) and an end-to-end converge parity test, marked `core`.

---

## Task 1: Chunked absorb core (the validated lever, raw arrays)

**Files:**
- Create: `src/tenax/algorithms/_ctm_chunked_absorb.py`
- Test: `tests/test_ctm_chunked_absorb.py`

- [ ] **Step 1: Write the failing test (core parity for the LEFT contraction)**

This test pins the chunked core against the monolithic einsum+sandwich (the gate's G4 oracle), at K=1 (chunk_size=χ) and K>1.

```python
# tests/test_ctm_chunked_absorb.py
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from tenax.algorithms._ctm_chunked_absorb import _chunked_T_new_left


def _monolith_left(T4, a, P1, P2, chi, D2):
    # Faithful reference: contract(T4, a) then conj(P1)^T · Tg · P2 (matches
    # _apply_projector_tensor for the LEFT move).
    T4a = jnp.einsum("ijk,lmjn->iklmn", T4, a)          # (t4_d, t4_u, u2, d2, r2)
    Tg = T4a.transpose(0, 2, 1, 3, 4).reshape(chi * D2, chi * D2, D2)  # (fl, fr, r2)
    step = jnp.tensordot(P1.conj(), Tg, axes=([0], [0]))  # (chi_new, fr, r2)
    return jnp.tensordot(step, P2, axes=([1], [0]))       # (chi_new, r2, chi_new_r)


@pytest.mark.parametrize("batch", [16, 8, 5])  # K=1, K=2, ragged
def test_chunked_left_matches_monolith(batch):
    D, chi = 3, 16
    D2 = D * D
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    T4 = jax.random.normal(k[0], (chi, D2, chi))
    a = jax.random.normal(k[1], (D2, D2, D2, D2))
    P1 = jax.random.normal(k[2], (chi * D2, chi))
    P2 = jax.random.normal(k[3], (chi * D2, chi))
    ref = _monolith_left(T4, a, P1, P2, chi, D2)
    got = _chunked_T_new_left(T4, a, P1, P2, chi, D2, batch)
    rel = float(jnp.max(jnp.abs(got - ref)) / (jnp.max(jnp.abs(ref)) + 1e-30))
    assert rel <= 1e-12, rel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py::test_chunked_left_matches_monolith -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tenax.algorithms._ctm_chunked_absorb'`.

- [ ] **Step 3: Write the core module**

```python
# src/tenax/algorithms/_ctm_chunked_absorb.py
"""Chunked dense-CTM edge absorption (raw arrays).

Reproduces the expensive ``contract(edge, a)`` + projector sandwich of the
1x1 ``_ctm_tensor_move_*`` functions, but chunks the boundary-chi axis with
``lax.map`` so the chi^2 * D^6 intermediate is never materialized in full.
Numerically faithful to ``_apply_projector_tensor`` (T_new = conj(P1)^T . (edge.a) . P2;
P1 is barred -> conj data, P2 is not). Dense path only. See the gate findings
docs/superpowers/handoffs/2026-06-30-chunk-shard-ctm-move-findings.md (STRONG GO).
"""
from __future__ import annotations

import jax.numpy as jnp
from jax import lax


def _chunked_T_new_left(T4, a, P1, P2, chi, D2, batch):
    """LEFT: edge T4 (t4_d, l2, t4_u), a (u2, d2, l2, r2).

    P1 raw (fl=(t4_d, u2), chi_new); P2 raw (fr=(t4_u, d2), chi_new).
    Returns T_new (chi_new, r2, chi_new_r).
    """
    P1r = P1.reshape(chi, D2, -1)                       # (t4_d, u2, chi_new)

    def per_i(args):
        T4_i, P1_i = args                               # (l2, t4_u), (u2, chi_new)
        T4a_i = jnp.einsum("jk,lmjn->klmn", T4_i, a)    # (t4_u, u2, d2, r2)
        Tg_i = T4a_i.transpose(1, 0, 2, 3).reshape(D2, chi * D2, D2)  # (u2, (t4_u,d2), r2)
        step = jnp.tensordot(P1_i.conj(), Tg_i, axes=([0], [0]))      # (chi_new, fr, r2)
        return jnp.tensordot(step, P2, axes=([1], [0]))              # (chi_new, r2, chi_new_r)

    return lax.map(per_i, (T4, P1r), batch_size=batch).sum(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py::test_chunked_left_matches_monolith -v`
Expected: PASS for all three `batch` values.

- [ ] **Step 5: Add the other three directions + their parity tests**

Append to the test file (one monolith reference + parametrized test per direction). Each monolith mirrors `_monolith_left` with that direction's einsum, transpose-to-`(fl_D2, fr, surv)`, identical sandwich. Use the per-direction table in Background. Example for RIGHT:

```python
def _monolith_right(T2, a, P1, P2, chi, D2):
    T2a = jnp.einsum("ijk,lmnj->iklmn", T2, a)          # (t2_u, t2_d, u2, d2, l2)
    Tg = T2a.transpose(0, 2, 1, 3, 4).reshape(chi * D2, chi * D2, D2)  # (fl=(t2_u,u2), fr=(t2_d,d2), l2)
    step = jnp.tensordot(P1.conj(), Tg, axes=([0], [0]))
    return jnp.tensordot(step, P2, axes=([1], [0]))     # (chi_new, l2, chi_new_r)
```

Add the matching core functions to `_ctm_chunked_absorb.py`:

```python
def _chunked_T_new_right(T2, a, P1, P2, chi, D2, batch):
    """RIGHT: edge T2 (t2_u, r2, t2_d). P1 fl=(t2_u,u2), P2 fr=(t2_d,d2). T_new (chi_new, l2, chi_new_r)."""
    P1r = P1.reshape(chi, D2, -1)                       # (t2_u, u2, chi_new)

    def per_i(args):
        T2_i, P1_i = args                               # (r2, t2_d), (u2, chi_new)
        T2a_i = jnp.einsum("jk,lmnj->klmn", T2_i, a)    # (t2_d, u2, d2, l2)
        Tg_i = T2a_i.transpose(1, 0, 2, 3).reshape(D2, chi * D2, D2)  # (u2, (t2_d,d2), l2)
        step = jnp.tensordot(P1_i.conj(), Tg_i, axes=([0], [0]))
        return jnp.tensordot(step, P2, axes=([1], [0]))

    return lax.map(per_i, (T2, P1r), batch_size=batch).sum(0)


def _chunked_T_new_top(T1, a, P1, P2, chi, D2, batch):
    """TOP: edge T1 (t1_l, u2, t1_r). P1 fl=(t1_l,l2), P2 fr=(t1_r,r2). T_new (chi_new, d2, chi_new_r)."""
    P1r = P1.reshape(chi, D2, -1)                       # (t1_l, l2, chi_new)

    def per_i(args):
        T1_i, P1_i = args                               # (u2, t1_r), (l2, chi_new)
        T1a_i = jnp.einsum("jk,jlmn->klmn", T1_i, a)    # (t1_r, d2, l2, r2)
        # need (fl_D2=l2, fr=(t1_r,r2), surv=d2): order axes (l2, t1_r, r2, d2)
        Tg_i = T1a_i.transpose(2, 0, 3, 1).reshape(D2, chi * D2, D2)  # (l2, (t1_r,r2), d2)
        step = jnp.tensordot(P1_i.conj(), Tg_i, axes=([0], [0]))      # (chi_new, fr, d2)
        return jnp.tensordot(step, P2, axes=([1], [0]))              # (chi_new, d2, chi_new_r)

    return lax.map(per_i, (T1, P1r), batch_size=batch).sum(0)


def _chunked_T_new_bottom(T3, a, P1, P2, chi, D2, batch):
    """BOTTOM: edge T3 (t3_r, d2, t3_l). P1 fl=(t3_r,l2), P2 fr=(t3_l,r2). T_new (chi_new, u2, chi_new_r)."""
    P1r = P1.reshape(chi, D2, -1)                       # (t3_r, l2, chi_new)

    def per_i(args):
        T3_i, P1_i = args                               # (d2, t3_l), (l2, chi_new)
        T3a_i = jnp.einsum("jk,ljmn->klmn", T3_i, a)    # (t3_l, u2, l2, r2)
        # need (fl_D2=l2, fr=(t3_l,r2), surv=u2): order axes (l2, t3_l, r2, u2)
        Tg_i = T3a_i.transpose(2, 0, 3, 1).reshape(D2, chi * D2, D2)  # (l2, (t3_l,r2), u2)
        step = jnp.tensordot(P1_i.conj(), Tg_i, axes=([0], [0]))      # (chi_new, fr, u2)
        return jnp.tensordot(step, P2, axes=([1], [0]))              # (chi_new, u2, chi_new_r)

    return lax.map(per_i, (T3, P1r), batch_size=batch).sum(0)
```

> NOTE for top/bottom: the per-chunk einsum output is `(other_χ, A, B, C)` where the free `a` legs land in `a`'s declaration order minus the shared leg. The `transpose(2, 0, 3, 1)` puts `(fl_D2, other_χ, fr_D2, surv)` so the reshape yields `(fl_D2, fr, surv)`. The parametrized parity test is the oracle — if a transpose is wrong it fails loudly. Verify each direction's einsum output ordering against the per-direction table before trusting the transpose.

- [ ] **Step 6: Run all core parity tests**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py -v -k "matches_monolith"`
Expected: PASS for all 4 directions × 3 batch sizes.

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/_ctm_chunked_absorb.py tests/test_ctm_chunked_absorb.py
git commit -m "feat(#632): chunked dense-CTM edge-absorption core (raw arrays)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Wire chunked core into the 1×1 LEFT move (dense-gated, default-off)

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py:997-1049` (`_ctm_tensor_move_left`)
- Test: `tests/test_ctm_chunked_absorb.py`

- [ ] **Step 1: Write the failing test (full LEFT move, chunked == non-chunked)**

```python
# tests/test_ctm_chunked_absorb.py  (append)
import numpy as np
from tenax.algorithms._ctm_tensor_init import initialize_ctm_tensor_env, _build_double_layer_tensor
from tenax.algorithms._ctm_tensor_moves import _ctm_tensor_move_left
from tenax.core.tensor import DenseTensor


def _random_dense_site(D=3, phys=2, seed=0):
    # Build a DenseTensor site A[u,d,l,r,phys] with the labels the CTM expects.
    from tenax.core.index import TensorIndex
    from tenax.algorithms._ctm_tensor_init import IN, OUT
    rng = np.random.default_rng(seed)
    arr = jnp.asarray(rng.standard_normal((D, D, D, D, phys)))
    labels = ["u", "d", "l", "r", "phys"]
    flows = [IN, OUT, IN, OUT, OUT]
    idx = tuple(TensorIndex.from_dim(s, f, label=l) for s, f, l in zip(arr.shape, flows, labels))
    return DenseTensor(arr, idx)


def test_full_left_move_chunked_matches_default():
    chi = 12
    A = _random_dense_site()
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)
    base = _ctm_tensor_move_left(env, env, a, chi)[0]
    chunked = _ctm_tensor_move_left(env, env, a, chi, chunk_size=4)[0]
    for field in ("C1", "C4", "T4"):
        b = getattr(base, field).transpose(tuple(range(getattr(base, field).ndim))).todense()
        c = getattr(chunked, field).todense()
        # align leg order via labels before comparing
        bl = list(getattr(base, field).labels()); cl = list(getattr(chunked, field).labels())
        perm = [cl.index(x) for x in bl]
        c = jnp.transpose(getattr(chunked, field).todense(), perm)
        rel = float(jnp.max(jnp.abs(c - getattr(base, field).todense())) / (jnp.max(jnp.abs(b)) + 1e-30))
        assert rel <= 1e-12, (field, rel)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py::test_full_left_move_chunked_matches_default -v`
Expected: FAIL — `TypeError: _ctm_tensor_move_left() got an unexpected keyword argument 'chunk_size'`.

- [ ] **Step 3: Add a raw-extraction helper to the core module**

```python
# src/tenax/algorithms/_ctm_chunked_absorb.py  (append)
def _raw_in_label_order(T, order):
    """Return T's raw array transposed to the given label order (DenseTensor only)."""
    cur = list(T.labels())
    perm = [cur.index(lbl) for lbl in order]
    return T.transpose(tuple(perm)).todense()
```

- [ ] **Step 4: Add the chunked branch to `_ctm_tensor_move_left`**

Add `chunk_size: int | None = None` to the signature (after `projector_backward`). Replace the block from `# T4(self) · a(neighbor)` through the `_apply_projector_with_reembed(...)` call with:

```python
    # Native projector (needs only the grown corners, not the grown edge)
    P_1, P_2, _eps_t = _compute_projector_tensor(
        C1g, C4g, chi, projector_method, base_charges, projector_backward
    )

    if chunk_size is not None and isinstance(env_self.T4, DenseTensor):
        from tenax.algorithms._ctm_chunked_absorb import (
            _chunked_T_new_left,
            _raw_in_label_order,
        )

        chi_dim = env_self.T4.indices[env_self.T4.labels().index("t4_d")].dim
        D2 = a.indices[a.labels().index("r2")].dim
        T4_raw = _raw_in_label_order(env_self.T4, ["t4_d", "l2", "t4_u"])
        a_raw = _raw_in_label_order(a, ["u2", "d2", "l2", "r2"])
        P1_bar = P_1.bar()
        P1_raw = _raw_in_label_order(P1_bar, ["fused", "chi_new"]).conj()  # un-bar data: bar conjugated it
        P2_raw = _raw_in_label_order(P_2, ["fused", "chi_new"])
        T4_arr = _chunked_T_new_left(T4_raw, a_raw, P1_raw, P2_raw, chi_dim, D2, chunk_size)

        # Cheap corners (identical to _apply_projector_tensor)
        C1_new = contract(P1_bar, C1g)
        C4_new = contract(P_2.bar(), C4g)
        # Rewrap T_new with the production index objects: first leg = P1_bar.chi_new,
        # middle = a's surviving (r2) leg, third = P_2.chi_new relabeled chi_new_r.
        chi_new_idx = P1_bar.indices[P1_bar.labels().index("chi_new")]
        surv_idx = a.indices[a.labels().index("r2")]
        chi_new_r_idx = P_2.indices[P_2.labels().index("chi_new")].relabel("chi_new_r")
        T4_new = DenseTensor(T4_arr, (chi_new_idx, surv_idx, chi_new_r_idx))
    else:
        # T4(self) · a(neighbor)  — the χ²·D⁶ peak (default path, unchanged)
        T4_with_a = contract(env_self.T4, a)
        T4g = _fuse_pair_by_label(T4_with_a, "t4_d", "u2", "fl", IN)
        T4g = _fuse_pair_by_label(T4g, "t4_u", "d2", "fr", OUT)
        C1_new, C4_new, T4_new = _apply_projector_with_reembed(
            P_1, P_2, C1g, C4g, T4g, "fl", "fr"
        )
```

The existing relabel / `_flip_leg_flow` / `_phase_fix_normalize_tensor` tail (lines 1040-1048) stays unchanged and runs for both branches.

> CRITICAL re P1 conjugation: `_chunked_T_new_left` applies `P1_i.conj()` internally (matching `_apply_projector_tensor`'s `P1_bar = P_1.bar()`, which conjugates). We extract `P1_bar` (already conj'd by `.bar()`) and `.conj()` it back to recover the un-conjugated `P_1` data, because the core conjugates again. Net data passed to the core = `P_1` (un-barred), so the core's `.conj()` reproduces `P_1.bar()`'s data. The parity test verifies this; if it fails by a complex conjugate, drop the `.conj()` on `P1_raw` and extract from `P_1` directly. (For real f64 inputs the conj is a no-op and the test passes either way — keep it correct for the complex path.)

- [ ] **Step 5: Run the full-move parity test**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py::test_full_left_move_chunked_matches_default -v`
Expected: PASS (rel ≤ 1e-12 on C1/C4/T4).

- [ ] **Step 6: Run the existing CTM move tests to confirm default-off is untouched**

Run: `uv run pytest tests/test_ctm_compiled.py -m core -q && uv run pytest -k "ctm" -m core -q`
Expected: PASS (no regressions; default path byte-identical).

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/_ctm_chunked_absorb.py src/tenax/algorithms/_ctm_tensor_moves.py tests/test_ctm_chunked_absorb.py
git commit -m "feat(#632): chunked branch in 1x1 left CTM move (dense, default-off)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Generalize to RIGHT / TOP / BOTTOM 1×1 moves

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py` (`_ctm_tensor_move_right` 1052, `_ctm_tensor_move_top` 1106, `_ctm_tensor_move_bottom` 1160)
- Test: `tests/test_ctm_chunked_absorb.py`

- [ ] **Step 1: Write the failing tests (full move parity for each direction)**

Add `test_full_right_move_chunked_matches_default`, `..._top...`, `..._bottom...`, each identical in shape to `test_full_left_move_chunked_matches_default` but calling the matching move and comparing the fields that move updates (right: C2/C3/T2; top: C1/C2/T1; bottom: C4/C3/T3).

```python
# tests/test_ctm_chunked_absorb.py  (append; one per direction)
from tenax.algorithms._ctm_tensor_moves import (
    _ctm_tensor_move_right, _ctm_tensor_move_top, _ctm_tensor_move_bottom,
)

@pytest.mark.parametrize("move_fn,fields", [
    (_ctm_tensor_move_right, ("C2", "C3", "T2")),
    (_ctm_tensor_move_top, ("C1", "C2", "T1")),
    (_ctm_tensor_move_bottom, ("C4", "C3", "T3")),
])
def test_full_move_chunked_matches_default(move_fn, fields):
    chi = 12
    A = _random_dense_site()
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)
    base = move_fn(env, env, a, chi)[0]
    chunked = move_fn(env, env, a, chi, chunk_size=4)[0]
    for field in fields:
        bl = list(getattr(base, field).labels()); cl = list(getattr(chunked, field).labels())
        perm = [cl.index(x) for x in bl]
        c = jnp.transpose(getattr(chunked, field).todense(), perm)
        b = getattr(base, field).todense()
        rel = float(jnp.max(jnp.abs(c - b)) / (jnp.max(jnp.abs(b)) + 1e-30))
        assert rel <= 1e-12, (field, rel)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py::test_full_move_chunked_matches_default -v`
Expected: FAIL — `TypeError: ... unexpected keyword argument 'chunk_size'`.

- [ ] **Step 3: Add the chunked branch to each of the three moves**

For each move, mirror Task 2 Step 4 using that direction's row in the per-direction table:
- Add `chunk_size: int | None = None` to the signature.
- Move the `_compute_projector_tensor(...)` call above the branch (it needs only the grown corners).
- In the chunked branch: extract the edge raw via `_raw_in_label_order(edge, <canonical order>)`, `a` raw via `["u2","d2","l2","r2"]`, `P1_raw` from `P_1.bar()` then `.conj()`, `P2_raw` from `P_2`; call the matching `_chunked_T_new_<dir>`; build `*_new` corners via `contract(P1_bar, Cg)` / `contract(P_2.bar(), Cg)`; rewrap `T*_new = DenseTensor(arr, (P1_bar.chi_new, a.<surviving>, P_2.chi_new→chi_new_r))`.
  - RIGHT surviving = `l2`; edge canonical `["t2_u","r2","t2_d"]`; corners C2g (P_1 side), C3g (P_2 side).
  - TOP surviving = `d2`; edge canonical `["t1_l","u2","t1_r"]`; corners C1g, C2g.
  - BOTTOM surviving = `u2`; edge canonical `["t3_r","d2","t3_l"]`; corners C4g, C3g.
- Leave the per-direction relabel / `_flip_leg_flow` / `_phase_fix_normalize_tensor` tail unchanged for both branches.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py::test_full_move_chunked_matches_default -v`
Expected: PASS for all three directions.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_moves.py tests/test_ctm_chunked_absorb.py
git commit -m "feat(#632): chunked branch in 1x1 right/top/bottom CTM moves

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Thread `chunk_size` through sweep → jit step → converge → config; wire `_shard_a` into 1×1

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_convergence.py:274-340` (`_ctm_tensor_sweep_multisite`, 1×1 branch)
- Modify: `src/tenax/algorithms/_ctm_python_loop.py` (`_make_jit_ctm_step` cache key + static arg; `python_loop_ctm_converge` + `_python_loop_chi_ramp`)
- Modify: `src/tenax/algorithms/ipeps_config.py` (`CTMConfig.ctm_chunk_size`)
- Modify: `src/tenax/algorithms/ipeps_ad_policy.py` (`ctm_converge_kwargs` + `ctm_energy_implicit` call)
- Test: `tests/test_ctm_chunked_absorb.py`

- [ ] **Step 1: Write the failing end-to-end test (CTM converge, chunk on == off)**

```python
# tests/test_ctm_chunked_absorb.py  (append)
from tenax.algorithms._ctm_python_loop import python_loop_ctm_converge


def test_converge_1x1_chunk_matches_default():
    chi = 12
    A = _random_dense_site(D=2)
    site_tensors = {(0, 0): A}
    neighbors = {(0, 0): {d: (0, 0) for d in ("left", "right", "top", "bottom")}}
    kw = dict(chi=chi, max_iter=8, min_iter=8, recipe="1x1", conv_tol=0.0)
    envs_off, _ = python_loop_ctm_converge(site_tensors, neighbors, **kw)
    envs_on, _ = python_loop_ctm_converge(site_tensors, neighbors, ctm_chunk_size=3, **kw)
    e_off, e_on = envs_off[(0, 0)], envs_on[(0, 0)]
    for field in e_off._fields:
        b = getattr(e_off, field); c = getattr(e_on, field)
        perm = [list(c.labels()).index(x) for x in list(b.labels())]
        cc = jnp.transpose(c.todense(), perm)
        rel = float(jnp.max(jnp.abs(cc - b.todense())) / (jnp.max(jnp.abs(b.todense())) + 1e-30))
        assert rel <= 1e-10, (field, rel)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py::test_converge_1x1_chunk_matches_default -v`
Expected: FAIL — `TypeError: python_loop_ctm_converge() got an unexpected keyword argument 'ctm_chunk_size'`.

- [ ] **Step 3: Thread `chunk_size` into the sweep's 1×1 branch + wire `_shard_a` there**

In `_ctm_tensor_sweep_multisite` (`_ctm_tensor_convergence.py`):
- Add `chunk_size: int | None = None` to the signature (after `device_mesh`).
- Hoist the `_shard_a` definition (currently inside the `recipe == "2x2"` branch, lines 348-358) so it is defined before the `if recipe == "1x1":` block, available to both.
- In the 1×1 loop (lines 328-340), pass the re-sharded `a` and the chunk size:

```python
    if recipe == "1x1":
        for direction, move_fn in _DIRECTION_MOVES:
            for coord in _sort_coords_for_direction(all_coords, direction):
                nb = neighbors[coord][direction]
                envs[coord], eps_t = move_fn(
                    envs[coord],
                    envs[nb],
                    _shard_a(double_layers[nb], direction),
                    chi,
                    projector_method,
                    base_charges=base_charges,
                    projector_backward=projector_backward,
                    chunk_size=chunk_size,
                )
                max_eps = jnp.maximum(max_eps, jnp.asarray(eps_t))
```

> The four 1×1 `move_fn` already accept `chunk_size` (Tasks 2-3). `_shard_a` is a no-op when `device_mesh is None`, so the chunk-only path (no mesh) is unaffected.

- [ ] **Step 4: Thread `chunk_size` through `_make_jit_ctm_step`**

In `_ctm_python_loop.py`:
- Add `ctm_chunk_size=None` param to `_make_jit_ctm_step`; include it in `cache_key` (4-tuple: `(id(neighbors), recipe, device_mesh, ctm_chunk_size)`) and add the type to `_JIT_STEP_CACHE`'s annotation.
- Add `"chunk_size"` to `_step`'s `static_argnames` and pass `chunk_size=ctm_chunk_size` into the `_ctm_tensor_sweep_multisite(...)` call. (`chunk_size` is a Python int/None, so it must be static — it changes shapes inside `lax.map`.)

```python
def _make_jit_ctm_step(neighbors, recipe="2x2", device_mesh=None, ctm_chunk_size=None):
    cache_key = (id(neighbors), recipe, device_mesh, ctm_chunk_size)
    ...
    @partial(jax.jit, static_argnames=("chi", "projector_method", "renormalize",
                                       "projector_backward", "chunk_size"))
    def _step(site_tensors, envs, *, chi, projector_method="svd", renormalize=False,
              projector_backward="auto", chunk_size=None):
        double_layers = {c: _build_double_layer_tensor(A) for c, A in site_tensors.items()}
        return _ctm_tensor_sweep_multisite(
            envs, double_layers, neighbors, chi, renormalize, projector_method,
            projector_backward=projector_backward, recipe=recipe,
            device_mesh=device_mesh, chunk_size=chunk_size,
        )
```

Then in `python_loop_ctm_converge`: add `ctm_chunk_size: int | None = None` param; pass `ctm_chunk_size=ctm_chunk_size` to `_make_jit_ctm_step`; and pass `chunk_size=ctm_chunk_size` in every `jit_step(...)` call (the QR warm-up loop and via `_run_ctm_loop_with_bump` — check whether the bump helper forwards extra kwargs; if not, the warm-up + main loop both need `chunk_size=ctm_chunk_size` threaded). Also add `ctm_chunk_size` to `_python_loop_chi_ramp` and forward it in its recursive `python_loop_ctm_converge` call.

> Inspect `src/tenax/algorithms/_ctm_loop_core.py::_run_ctm_loop_with_bump` first: it calls `jit_step` internally, so `chunk_size` must reach it. Either (a) bind it via `functools.partial(jit_step, chunk_size=ctm_chunk_size)` before passing `jit_step` into the helper, or (b) add a `chunk_size` passthrough param. Option (a) is the smaller change and avoids touching the bump helper's signature — prefer it. Apply the same `partial` to the warm-up loop's `jit_step`.

- [ ] **Step 5: Run the end-to-end parity test**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py::test_converge_1x1_chunk_matches_default -v`
Expected: PASS (rel ≤ 1e-10 across all env fields after 8 fixed sweeps).

- [ ] **Step 6: Add the `CTMConfig` field + policy threading**

In `ipeps_config.py`, add to `CTMConfig` (after `device_mesh`, line 250) — appended last to preserve positional ABI:

```python
    ctm_chunk_size: int | None = None
    """Chunk the χ²·D⁶ edge absorption over the boundary-χ axis via
    ``lax.map(batch_size=ctm_chunk_size)`` (1×1 recipe, dense envs only).
    Lowers per-device peak memory ≈÷K at large D and composes with
    ``device_mesh`` (≈÷(N·K)). ``None`` (default) → single monolithic
    contraction (byte-for-byte unchanged). See the #632 chunk×shard gate."""
```

In `ipeps_ad_policy.py`: add `"ctm_chunk_size": ctm_cfg.ctm_chunk_size` to the `ctm_converge_kwargs` dict (next to `"device_mesh"`, ~line 189), and `ctm_chunk_size=ctm_cfg.ctm_chunk_size` to the `ctm_energy_implicit(...)` call (next to `device_mesh=`, ~line 373). Then thread `ctm_chunk_size` through `ctm_energy_implicit` → `_run_forward` → `_sigma_gauged_ctm_converge`/`python_loop_ctm_converge` exactly as `device_mesh` is threaded today (grep `device_mesh` in `_ctm_energy_ad.py` and add a sibling `ctm_chunk_size` param at each forward call site; leave the backward step builder at `_ctm_energy_ad.py:1007` as-is — backward chunking is Increment 2).

- [ ] **Step 7: Run config + policy + CTM test buckets**

Run: `uv run pytest tests/test_ctm_chunked_absorb.py -m core -q && uv run pytest -k "config or ipeps_ad or ctm" -m core -q`
Expected: PASS, no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_convergence.py src/tenax/algorithms/_ctm_python_loop.py src/tenax/algorithms/ipeps_config.py src/tenax/algorithms/ipeps_ad_policy.py tests/test_ctm_chunked_absorb.py
git commit -m "feat(#632): thread ctm_chunk_size config through forward 1x1 CTM + wire _shard_a into 1x1

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: CI marker, docs, and full-suite check

**Files:**
- Modify: `tests/test_ctm_chunked_absorb.py` (ensure `core` marker)
- Modify: `docs/superpowers/handoffs/` (new short Increment-1 build note)

- [ ] **Step 1: Confirm the test file is `core`-marked and runs in the required CI bucket**

`conftest.py` auto-marks by filename. Confirm `test_ctm_chunked_absorb.py` lands in `core` (not `slow`):
Run: `uv run pytest tests/test_ctm_chunked_absorb.py -m core --collect-only -q`
Expected: all tests collected under `-m core`. If any are mis-bucketed, add an explicit `@pytest.mark.core` or rename per `conftest.py` rules.

- [ ] **Step 2: Run the full core bucket (the CI required check)**

Run: `uv run pytest -m core -q`
Expected: PASS (this is what `Tests (Python 3.11/3.12)` gate on).

- [ ] **Step 3: Write a short build-findings note**

Create `docs/superpowers/handoffs/2026-07-01-chunk-ctm-absorb-increment1-build.md` summarizing: what shipped (forward 1×1 dense chunked absorb + `ctm_chunk_size`), the parity guarantees (core/full-move/converge rel bounds), the default-off contract, and the explicit deferral of Increment 2 (backward gate) and 2×2/symmetric. Note that `device_mesh` is still dropped at the backward (`_ctm_energy_ad.py:1007`) — that's Increment 2's problem.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ctm_chunked_absorb.py docs/superpowers/handoffs/2026-07-01-chunk-ctm-absorb-increment1-build.md
git commit -m "docs/test(#632): Increment-1 chunked-absorb build note + core CI marker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**1. Spec coverage** (against the gate findings "Build (recommended)" + the two confirmed decisions):
- "opt-in `n_chunks`/`batch` knob on the edge contraction" → `ctm_chunk_size` (Tasks 1-4). ✓
- "composed with the `ctm_sharding` mesh" → `_shard_a` wired into the 1×1 branch (Task 4 Step 3); chunk keeps `a` sharded inside `lax.map` (gate G3a). ✓
- "all 4 moves + corner accumulation" → 4 moves chunked (Tasks 2-3); corners use the cheap `contract(P_bar, Cg)` path. ✓
- "default-OFF (bit-identical when off) + parity CI" → default `None`; core/full-move/converge parity tests, `core`-marked (Tasks 1-5). ✓
- Decision: chunk the production Tensor absorb path, not the dead `_compiled_move_*` → done via `_ctm_chunked_absorb.py` called from `_ctm_tensor_move_*`. ✓
- Decision: 1×1 recipe first → only the 1×1 path is touched; 2×2 untouched (deferred). ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Per-direction differences are given as an explicit table + per-direction core functions. The two "inspect first" notes (Task 4 Step 4 on `_run_ctm_loop_with_bump`; Task 1 Step 5 transpose verification) are guarded by the parity oracle, not hand-waves.

**3. Type/name consistency:** `chunk_size` (move-level param) vs `ctm_chunk_size` (config/converge-level) is intentional and consistent: `ctm_chunk_size` is the public config name; it maps to the `chunk_size` kwarg at `_make_jit_ctm_step`/move boundary. Core functions `_chunked_T_new_{left,right,top,bottom}` and `_raw_in_label_order` are referenced with matching signatures across tasks. `DenseTensor(data, indices)`, `.bar()`, `.todense()`, `TensorIndex.relabel`/`.flip_flow`/`.dim` all match `src/tenax/core/tensor.py`.

---

## Out of scope (do NOT build here)

- **Increment 2** — chunked **backward** through the implicit-AD adjoint (`_ctm_energy_ad.py`, custom_vjp + fixed-point). This is gate-first: prove correct + bounded-memory grads through the chunked move BEFORE wiring `optimize_gs_ad`. The forward `device_mesh`/`ctm_chunk_size` are deliberately NOT threaded into the backward step (`:1007`).
- **2×2 recipe** chunking (more absorb call sites; the 2×2 absorb already has per-move sharding).
- **Symmetric / fermionic** envs (knob is a no-op there; large-D symmetric → YASTN).
- The **D=10–12 / large-χ multi-GPU `optimize_gs_ad` benchmark** (depends on Increment 2).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-01-chunk-ctm-absorb-increment1.md`.
