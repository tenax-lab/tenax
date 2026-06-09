# Drop-in QR projector for the 2×2 Fishman AD path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two non-truncating M1/M2 half-system SVDs in the symmetric 2×2 Fishman projector with gauge-fixed block-sparse QR (cheap backward, no `1/(sᵢ²−sⱼ²)` F-matrix), keeping the M′ truncation SVD, opt-in behind `projector_method="qr"`, to cut the χ-scaling CTM-AD backward cost.

**Architecture:** A new gauge-fixed block-sparse QR helper feeds a `decomp="svd"|"qr"` branch inside `_compute_2x2_projector_symmetric`. With `decomp="svd"` the function is byte-identical to today. `projector_method="qr"` is threaded from the move wrappers (which currently discard it) into this branch. QR changes the fixed point slightly, so validation is *physical agreement* (energy/gradient/biorthogonality), not byte-parity. Two spikes gate the build: a QR-VJP stability check and a per-SVD cost-attribution measurement.

**Tech Stack:** JAX (`jnp.linalg.qr`, `jax.custom_vjp`, `jax.test_util.check_grads`), Tenax `SymmetricTensor` / `tenax.linalg.{svd,qr}`, pytest (`-m core`/`-m algorithm`).

**Spec:** `docs/superpowers/specs/2026-06-09-qr-projector-2x2-ad-drop-in-570.md`

---

## File structure

- **Modify** `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` — add `_gauge_fix_symmetric_qr`; add `decomp` param + QR branch to `_compute_2x2_projector_symmetric`; thread `decomp` through `_compute_2x2_projector`.
- **Modify** `src/tenax/algorithms/_ctm_tensor_moves.py` — stop discarding `projector_method` on the 2×2 path (lines ~918/1027/1116); map `"qr"`→`decomp="qr"`.
- **Modify** `src/tenax/algorithms/_ad_primitives.py` — add `regularized_qr` *only if* Task 1 shows the raw QR VJP is unstable.
- **Create** `examples/probe_qr_vjp_stability_570.py` — Task 1 spike.
- **Create** `examples/probe_svd_split_attribution_570.py` — Task 2 spike (extends `probe_bwd_subop_attribution_570.py`).
- **Create** `tests/test_qr_projector_2x2.py` — gauge-fix, biorthogonality, energy/gradient agreement, regression.
- **Reuse** `examples/profile_570_sweepvjp_compile.py` — Task 8 perf measurement.

Run fast tests with `uv run pytest -m core`; algorithm tests with `uv run pytest tests/test_qr_projector_2x2.py -v`.

---

## Task 1: SPIKE — block-sparse QR VJP stability (gates everything)

Decides whether a `regularized_qr` custom-VJP is needed before any projector work. No `regularized_qr` is added speculatively.

**Files:**
- Create: `examples/probe_qr_vjp_stability_570.py`

- [ ] **Step 1: Write the probe**

```python
"""SPIKE (#570): is the block-sparse QR backward stable enough to drop into the
2x2 projector? Checks grads on well-conditioned AND near-rank-deficient sectors.

Run: JAX_PLATFORMS=cpu uv run python examples/probe_qr_vjp_stability_570.py
"""
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)


def _qr_reduce(M):
    # Mirror the per-sector op the projector differentiates: QR then use Q only
    # (the isometry), since the projector consumes Q and folds R forward.
    Q, R = jnp.linalg.qr(M)
    # diag(R) >= 0 gauge fix (mirrors _ctm_projector.py:1064-1071)
    d = jnp.diag(R)
    phase = jnp.where(jnp.abs(d) > 0, d / jnp.where(jnp.abs(d) > 0, jnp.abs(d), 1.0), 1.0)
    Q = Q * phase[None, :]
    return Q


def main():
    key = jax.random.PRNGKey(0)
    for label, M in [
        ("well-conditioned 12x12", jax.random.normal(key, (12, 12))),
        ("tall 16x8", jax.random.normal(key, (16, 8))),
        ("near-rank-deficient 12x12", _make_rank_deficient(key, 12, drop=4)),
    ]:
        try:
            check_grads(_qr_reduce, (M,), order=1, modes=["rev"], atol=1e-4, rtol=1e-4)
            print(f"PASS  {label}")
        except Exception as e:  # noqa: BLE001 — spike, report and continue
            print(f"FAIL  {label}: {type(e).__name__}: {str(e)[:120]}")


def _make_rank_deficient(key, n, drop):
    A = jax.random.normal(key, (n, n))
    U, s, Vh = jnp.linalg.svd(A)
    s = s.at[n - drop:].set(0.0)
    return (U * s) @ Vh


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `JAX_PLATFORMS=cpu uv run python examples/probe_qr_vjp_stability_570.py`
Expected: PASS on well-conditioned + tall. The near-rank-deficient case reveals whether `regularized_qr` is needed.

- [ ] **Step 3: Record the verdict in the spec**

Append a `## Task 1 result` block to `docs/superpowers/specs/2026-06-09-qr-projector-2x2-ad-drop-in-570.md` stating: which cases passed, and the decision — **"raw QR VJP sufficient"** (skip `regularized_qr`) or **"add `regularized_qr`"** (with the failing case). Subsequent tasks assume the raw VJP unless this says otherwise.

- [ ] **Step 4: Commit**

```bash
git add examples/probe_qr_vjp_stability_570.py docs/superpowers/specs/2026-06-09-qr-projector-2x2-ad-drop-in-570.md
git commit -m "spike(#570): block-sparse QR VJP stability probe + verdict"
```

- [ ] **Step 5 (CONDITIONAL): add `regularized_qr` if Step 3 says so**

Only if the near-rank-deficient case failed. In `src/tenax/algorithms/_ad_primitives.py`, after `regularized_svd` (line ~357), add a `jax.custom_vjp` `regularized_qr(M) -> (Q, R)` whose backward clips the `R⁻¹` in the standard QR adjoint (`dM = (dQ + Q·copyltu(...)) R⁻ᵀ`-style) with a `where`-guarded reciprocal mirroring `regularized_svd`'s floor. Add a `check_grads` test for it to `tests/test_qr_projector_2x2.py` and re-run Step 2's probe routed through it. Commit separately: `feat(#570): regularized_qr custom-VJP for near-rank-deficient sectors`.

---

## Task 2: SPIKE — per-SVD cost attribution (go/no-go number)

Confirms the M1/M2 SVDs are a large-enough slice of the backward to justify the build. Explicit off-ramp.

**Files:**
- Create: `examples/probe_svd_split_attribution_570.py`

- [ ] **Step 1: Write the probe**

Extend the source-attribution approach of `examples/probe_bwd_subop_attribution_570.py` (which buckets ops by emitting function via `eqn.source_info.traceback`). Add a finer split that distinguishes the three `tensor_svd` call sites inside `_compute_2x2_projector_symmetric` by their `new_bond_label` (`"m1_bond"`, `"m2_bond"`, `"chi_new"`) — match on the traceback frame line numbers of the three `tensor_svd(...)` calls (Stage 2 M1 ≈ line 879, M2 ≈ line 889, Stage 4 M′ ≈ line 943/962/971).

```python
"""SPIKE (#570): split the 2x2 backward SVD-VJP cost across M1 / M2 / M_prime.

Reuses the jaxpr source-attribution machinery of probe_bwd_subop_attribution_570.
Reports per-call op counts so we know how much of svd_vjp QR can remove
(M1+M2) vs how much stays (M_prime).

Run: JAX_PLATFORMS=cpu uv run python examples/probe_svd_split_attribution_570.py \
        --D 2 --chi 4 8 12 --full
"""
# Import and call the existing probe's tracing helper, then re-bucket the
# svd_vjp category by source line of the originating tensor_svd call.
```

- [ ] **Step 2: Run it**

Run: `JAX_PLATFORMS=cpu uv run python examples/probe_svd_split_attribution_570.py --D 2 --chi 4 8 12 --full`
Expected: a table `chi | M1_ops | M2_ops | Mprime_ops | total_svd_vjp`.

- [ ] **Step 3: Record verdict + decide**

Append `## Task 2 result` to the spec with the table and the **expected backward-op reduction** = `(M1+M2) / total_backward × (1 − 1/2.6)`. **Go** if `(M1+M2)` is ≳ the M′ share (i.e. drop-in removes a meaningful slice); **No-go** → stop, write up, do not build the projector branch. Get explicit user confirmation on the go/no-go before Task 4.

- [ ] **Step 4: Commit**

```bash
git add examples/probe_svd_split_attribution_570.py docs/superpowers/specs/2026-06-09-qr-projector-2x2-ad-drop-in-570.md
git commit -m "spike(#570): per-SVD (M1/M2/M_prime) cost attribution + go/no-go"
```

---

## Task 3: `_gauge_fix_symmetric_qr` + its test

Per-sector phase-fix so `diag(R)` is real-nonnegative, mirroring `_gauge_fix_symmetric_svd` (same file) and the dense QR fix (`_ctm_projector.py:1064-1071`). Vectorized per bond-charge sector from the start.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py` (add after `_gauge_fix_symmetric_svd`, ~line 135)
- Test: `tests/test_qr_projector_2x2.py`

- [ ] **Step 1: Write the failing test**

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.core.index import FlowDirection, TensorIndex
from tenax.core.tensor import SymmetricTensor
from tenax.core.symmetry import U1Symmetry
from tenax.linalg import qr as tensor_qr
from tenax.algorithms._ctm_tensor_projector_2x2 import _gauge_fix_symmetric_qr


def _random_symmetric_matrix(seed, charges):
    """A 2-leg U(1) SymmetricTensor with the given charge list on both legs."""
    sym = U1Symmetry()
    left = TensorIndex(charges=charges, flow=FlowDirection.OUT, symmetry=sym, label="l")
    right = TensorIndex(charges=charges, flow=FlowDirection.IN, symmetry=sym, label="r")
    key = jax.random.PRNGKey(seed)
    return SymmetricTensor.random(indices=(left, right), key=key)


def test_gauge_fix_qr_makes_diag_R_real_nonnegative():
    M = _random_symmetric_matrix(0, [0, 0, 1, 1])
    Q, R = tensor_qr(M, left_labels=("l",), right_labels=("r",), new_bond_label="b")
    Q_fixed, R_fixed = _gauge_fix_symmetric_qr(Q, R)
    # Reconstruction preserved: Q_fixed @ R_fixed == Q @ R == M
    from tenax.contraction.contractor import contract
    recon = contract(Q_fixed, R_fixed)
    orig = contract(Q, R)
    np.testing.assert_allclose(recon.todense(), orig.todense(), atol=1e-10)
    # diag(R_fixed) >= 0 per sector
    for key, block in R_fixed.blocks.items():
        d = jnp.diag(jnp.reshape(block, (block.shape[0], -1))[:, : block.shape[0]])
        assert jnp.all(jnp.real(d) >= -1e-12)
        assert jnp.allclose(jnp.imag(d), 0.0, atol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_qr_projector_2x2.py::test_gauge_fix_qr_makes_diag_R_real_nonnegative -v`
Expected: FAIL — `ImportError: cannot import name '_gauge_fix_symmetric_qr'`.

- [ ] **Step 3: Implement `_gauge_fix_symmetric_qr`**

In `_ctm_tensor_projector_2x2.py`, after `_gauge_fix_symmetric_svd`:

```python
def _gauge_fix_symmetric_qr(
    Q_T: SymmetricTensor, R_T: SymmetricTensor
) -> tuple[SymmetricTensor, SymmetricTensor]:
    """Per-sector QR gauge fix: rephase columns of Q and rows of R so each
    sector's ``diag(R)`` is real-nonnegative, preserving ``Q @ R``.

    Mirrors the dense QR sign-fix (``_ctm_projector.py``) and the per-sector
    style of :func:`_gauge_fix_symmetric_svd`.  Without it, ``jnp.linalg.qr``'s
    sign choice depends on the input values, so tiny CTM-iteration perturbations
    flip column signs and make the AD gradient correspond to the wrong branch.

    ``Q`` legs: (left..., bond) — last axis is the QR bond.
    ``R`` legs: (bond, right...) — first axis is the QR bond.
    Phase on column j of Q is ``conj(phase_j)``; on row j of R is ``phase_j``,
    where ``phase_j`` is the unit phase of ``R``'s sector diagonal entry j
    (phase 1 when that diagonal entry is zero — gauge unconstrained).
    """
    bond_idx = Q_T.indices[-1]
    bond_charges = np.asarray(bond_idx.charges, dtype=np.int32)

    r_keys_by_q: dict[int, list] = {}
    for key in R_T.blocks:
        r_keys_by_q.setdefault(int(key[0]), []).append(key)
    q_keys_by_q: dict[int, list] = {}
    for key in Q_T.blocks:
        q_keys_by_q.setdefault(int(key[-1]), []).append(key)

    new_q_blocks = dict(Q_T.blocks)
    new_r_blocks = dict(R_T.blocks)
    sample = next(iter(R_T.blocks.values()))
    is_complex = jnp.issubdtype(sample.dtype, jnp.complexfloating)

    for q in np.unique(bond_charges):
        q_int = int(q)
        r_keys = r_keys_by_q.get(q_int, [])
        if not r_keys:
            continue
        # The sector's square R-diagonal: stack the bond-by-bond diagonal of the
        # R blocks for this sector.  n_q = bond multiplicity for sector q.
        n_q = new_r_blocks[r_keys[0]].shape[0]
        # Concatenate this sector's R blocks along their non-bond axes, reshape
        # to (n_q, -1), take the leading n_q x n_q diagonal.
        R_q = jnp.concatenate(
            [jnp.reshape(new_r_blocks[k], (n_q, -1)) for k in r_keys], axis=1
        )
        diag = R_q[jnp.arange(n_q), jnp.arange(n_q)]  # (n_q,)
        absd = jnp.abs(diag)
        phase = jnp.where(
            absd > 0,
            diag / jnp.maximum(absd, jnp.asarray(1e-30, dtype=absd.dtype)),
            jnp.ones_like(diag),
        )
        conj_phase = jnp.conj(phase) if is_complex else jnp.real(phase)
        bare_phase = phase if is_complex else jnp.real(phase)

        for k in q_keys_by_q.get(q_int, []):
            new_q_blocks[k] = new_q_blocks[k] * conj_phase  # columns (last axis)
        for k in r_keys:
            blk = new_r_blocks[k]
            bcast = jnp.reshape(bare_phase, (n_q,) + (1,) * (blk.ndim - 1))
            new_r_blocks[k] = blk * bcast  # rows (first axis)

    Q_out = SymmetricTensor._from_blocks_unchecked(new_q_blocks, Q_T.indices)
    R_out = SymmetricTensor._from_blocks_unchecked(new_r_blocks, R_T.indices)
    return Q_out, R_out
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_qr_projector_2x2.py::test_gauge_fix_qr_makes_diag_R_real_nonnegative -v`
Expected: PASS.

- [ ] **Step 5: Add a smoothness (no sign-flip) test**

```python
def test_gauge_fix_qr_is_smooth_under_perturbation():
    M = _random_symmetric_matrix(1, [0, 0, 1, 1])
    eps = 1e-7
    dM = _random_symmetric_matrix(2, [0, 0, 1, 1])

    def q_of(scale):
        Mp = M.add(dM.scale(scale)) if hasattr(M, "add") else M  # use real API
        Q, R = tensor_qr(Mp, left_labels=("l",), right_labels=("r",), new_bond_label="b")
        Qf, _ = _gauge_fix_symmetric_qr(Q, R)
        return Qf

    Q0 = q_of(0.0).todense()
    Q1 = q_of(eps).todense()
    # Gauge-fixed Q is continuous: no O(1) sign flips for an O(eps) perturbation.
    assert jnp.max(jnp.abs(Q1 - Q0)) < 1e-3
```

Run: `uv run pytest tests/test_qr_projector_2x2.py -k gauge_fix_qr -v`
Expected: PASS (adjust the `M.add(dM.scale(...))` line to the actual SymmetricTensor arithmetic API — check `SymmetricTensor.__add__`/`scale` in `core/tensor.py`).

- [ ] **Step 6: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_qr_projector_2x2.py
git commit -m "feat(#570): _gauge_fix_symmetric_qr (per-sector diag(R)>=0) + tests"
```

---

## Task 4: QR branch in `_compute_2x2_projector_symmetric` (prototype-driven against biorthogonality)

Add `decomp="svd"|"qr"`. `decomp="svd"` is byte-identical to today. `decomp="qr"` replaces the M1/M2 SVDs with QR; M′ stays SVD. The exact seam-contraction for the QR cross-projector is **validated by the biorthogonality test (Step 1)**, which is the oracle — iterate the contraction until it passes.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_projector_2x2.py:804` (`_compute_2x2_projector_symmetric` signature + Stage 2/4/5 branch)
- Test: `tests/test_qr_projector_2x2.py`

- [ ] **Step 1: Write the failing biorthogonality test (the oracle)**

```python
from tenax.algorithms._ctm_tensor_projector_2x2 import _compute_2x2_projector_symmetric

def _enlarged_corners(seed, chi, D):
    """Build four 4-leg enlarged-corner SymmetricTensors for a U(1) model.
    Reuse the helper already used by tests/test_block_sparse_ctm_ad.py — import
    that fixture rather than re-deriving the leg structure here."""
    ...  # import from the existing CTM-AD test module's helpers

@pytest.mark.parametrize("direction", ["left", "right", "top", "bottom"])
def test_qr_projector_biorthogonality(direction):
    Q_TL, Q_TR, Q_BL, Q_BR = _enlarged_corners(seed=0, chi=6, D=2)
    P_top, P_bot, eps_T = _compute_2x2_projector_symmetric(
        Q_TL, Q_TR, Q_BL, Q_BR, chi=6, direction=direction, decomp="qr",
    )
    # Biorthogonality: contracting P_top with P_bot over the fused (chi_outer,
    # fused_D2) legs yields identity on the new chi bonds.
    from tenax.contraction.contractor import contract
    ident = contract(P_top, P_bot)  # relabel chi_new_top/bot to a shared bond
    dense = ident.todense()
    np.testing.assert_allclose(dense, np.eye(dense.shape[0]), atol=1e-8)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_qr_projector_2x2.py -k biorthogonality -v`
Expected: FAIL — `TypeError: _compute_2x2_projector_symmetric() got an unexpected keyword argument 'decomp'`.

- [ ] **Step 3: Add the `decomp` param + QR branch**

Change the signature (line 804) to add `decomp: str = "svd"` (keyword-only, after `base_charges`). At the top, `if decomp not in ("svd", "qr"): raise ValueError(...)`.

In **Stage 2**, branch the M1/M2 decomposition. SVD branch is the existing code. QR branch (the M1/M2 → QR replacement):

```python
    from tenax.linalg import qr as tensor_qr

    # ---- Stage 2: decompose M1, M2 (SVD or QR) with per-sector gauge fix. ----
    if decomp == "svd":
        U_M1_T, M1_S, Vh_M1_T, _ = tensor_svd(
            M1_T, left_labels=m1_left_labels, right_labels=m1_right_labels,
            new_bond_label="m1_bond", max_singular_values=None,
        )
        U_M1_T, Vh_M1_T = _gauge_fix_symmetric_svd(U_M1_T, Vh_M1_T)
        M1_S = _fishman_truncate_S(M1_S, eps=1e-12)
        # ... existing M2 SVD ...
    else:  # decomp == "qr"
        # M1 isometry spans the half used by `first_half` (m1_left for
        # left/bottom, m1_right for right/top — same orientation table as the
        # SVD path's choice of U_M1 vs Vh_M1).  R carries the weight into M'.
        if direction in ("left", "bottom"):
            m1_iso_labels, m1_wt_labels = m1_left_labels, m1_right_labels
            m2_iso_labels, m2_wt_labels = m2_right_labels, m2_left_labels
        else:  # "right", "top"
            m1_iso_labels, m1_wt_labels = m1_right_labels, m1_left_labels
            m2_iso_labels, m2_wt_labels = m2_left_labels, m2_right_labels
        Q_M1_T, R_M1_T = tensor_qr(
            M1_T, left_labels=m1_iso_labels, right_labels=m1_wt_labels,
            new_bond_label="m1_bond",
        )
        Q_M1_T, R_M1_T = _gauge_fix_symmetric_qr(Q_M1_T, R_M1_T)
        Q_M2_T, R_M2_T = tensor_qr(
            M2_T, left_labels=m2_iso_labels, right_labels=m2_wt_labels,
            new_bond_label="m2_bond",
        )
        Q_M2_T, R_M2_T = _gauge_fix_symmetric_qr(Q_M2_T, R_M2_T)
```

In **Stage 3/4**, the QR branch sets `first_half = Q_M1_T`, `second_half = Q_M2_T` (already isometric — no √S scaling, no `first_norm`/`second_norm` normalization needed since Q is orthonormal), and forms `M_prime` from the **R factors** contracted over the cut seam plus the seam bond, so that `M_prime` carries exactly the cross-correlation the SVD path's `M_prime` did. **This contraction is what Step 1 validates** — start from: contract `R_M1_T` and `R_M2_T` over the shared cut-seam labels (the `m1_iso`/`m2_iso` legs are held by the Q's; the seam is closed through the Q isometries), producing `M_prime_T` with legs `(m2_bond, m1_bond)` for `second_first`. Then the existing Stage-4 `tensor_svd(M_prime_T, ... new_bond_label="chi_new", max_singular_values=chi)` and `_gauge_fix_symmetric_svd` are reused unchanged.

In **Stage 5**, the cross-projectors become `P_first = contract(Q_M1_T, Vh_Mp_T.bar())`, `P_second = contract(U_Mp_T.bar(), Q_M2_T)` (the `first_half`/`second_half` in the existing Stage-5 code are now the Q's) scaled by `S_inv_sqrt` exactly as today.

> **Implementation note:** keep the existing `direction`/`prime_order`/`first_outer_labels` bookkeeping. The only changes are (a) Q replaces U·√S / √S·Vh, (b) M′ is built from R-factors, (c) no half-normalization. Iterate Step 4 below until biorthogonality holds for all four directions; the seam-contraction indices are the thing to get right.

- [ ] **Step 4: Run the oracle until green**

Run: `uv run pytest tests/test_qr_projector_2x2.py -k biorthogonality -v`
Expected: PASS for all four directions. If not, fix the Stage-4 R-factor seam contraction (the held vs contracted legs) — this is the core derivation; the test is authoritative.

- [ ] **Step 5: Assert `decomp="svd"` is byte-identical (regression guard)**

```python
def test_decomp_svd_is_unchanged():
    corners = _enlarged_corners(seed=3, chi=6, D=2)
    out_default = _compute_2x2_projector_symmetric(*corners, chi=6, direction="left")
    out_svd = _compute_2x2_projector_symmetric(*corners, chi=6, direction="left", decomp="svd")
    for a, b in zip(out_default[:2], out_svd[:2]):
        np.testing.assert_array_equal(a.todense(), b.todense())
```

Run: `uv run pytest tests/test_qr_projector_2x2.py -k decomp_svd_is_unchanged -v`
Expected: PASS.

- [ ] **Step 6: Thread `decomp` through the dense dispatcher**

In `_compute_2x2_projector` (line 302), add `decomp: str = "svd"` to the signature and pass it to the `_compute_2x2_projector_symmetric(...)` call (line ~397).

- [ ] **Step 7: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_projector_2x2.py tests/test_qr_projector_2x2.py
git commit -m "feat(#570): QR decomp branch in 2x2 Fishman projector (biorthogonality-validated)"
```

---

## Task 5: Wire `projector_method="qr"` from the move wrappers

Stop discarding `projector_method` on the 2×2 path; map `"qr"` → `decomp="qr"`.

**Files:**
- Modify: `src/tenax/algorithms/_ctm_tensor_moves.py` (the `del projector_method` sites ~918, 1027, 1116, and their `_compute_2x2_projector(...)` calls ~937, 1044, 1133, 1222)

- [ ] **Step 1: Write the failing wiring test**

```python
def test_projector_method_qr_routes_to_qr_decomp(monkeypatch):
    import tenax.algorithms._ctm_tensor_projector_2x2 as proj
    seen = {}
    orig = proj._compute_2x2_projector_symmetric
    def spy(*a, **k):
        seen["decomp"] = k.get("decomp", "svd")
        return orig(*a, **k)
    monkeypatch.setattr(proj, "_compute_2x2_projector_symmetric", spy)
    # Drive one 2x2 move with projector_method="qr" via the move wrapper used by
    # the CTM-AD path (pick the wrapper exercised by test_block_sparse_ctm_ad).
    ...  # call the move wrapper with projector_method="qr"
    assert seen["decomp"] == "qr"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_qr_projector_2x2.py -k routes_to_qr -v`
Expected: FAIL — `decomp` is `"svd"` because the wrapper still does `del projector_method`.

- [ ] **Step 3: Implement the mapping**

At each of the three move wrappers, replace `del projector_method  # 2x2 projector uses Fishman SVD unconditionally` with:

```python
    _decomp = "qr" if projector_method == "qr" else "svd"
```

and add `decomp=_decomp` to the corresponding `_compute_2x2_projector(...)` call. Update each wrapper's docstring line that says *"projector_method: Currently ignored"* to *"projector_method: 'qr' routes the 2x2 projector to the QR decomposition (M1/M2 via QR); otherwise Fishman SVD."*

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/test_qr_projector_2x2.py -k routes_to_qr -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/tenax/algorithms/_ctm_tensor_moves.py tests/test_qr_projector_2x2.py
git commit -m "feat(#570): route projector_method='qr' to the QR 2x2 decomp"
```

---

## Task 6: Forward energy + gradient agreement (the physics gate)

QR changes the fixed point, so assert *agreement within tolerance*, not byte-parity.

**Files:**
- Test: `tests/test_qr_projector_2x2.py`

- [ ] **Step 1: Write the energy-agreement test**

```python
@pytest.mark.algorithm
@pytest.mark.parametrize("chi", [6, 10])
def test_qr_vs_svd_energy_agreement_heisenberg_D2(chi):
    """Converged CTM energy agrees between projector_method svd and qr, and the
    gap shrinks as chi grows (both -> same value)."""
    e_svd = _heisenberg_d2_ground_energy(chi=chi, projector_method="svd")
    e_qr = _heisenberg_d2_ground_energy(chi=chi, projector_method="qr")
    assert abs(e_qr - e_svd) < 5e-4  # different fixed point, loosened vs eps
```

Implement `_heisenberg_d2_ground_energy` by reusing the existing 2D Heisenberg D=2 fixture already present in the iPEPS/CTM-AD tests (import it; do not hand-roll the model). Add an analogous `..._D3` case at `chi in [8, 12]` with tol `1e-3`.

- [ ] **Step 2: Run**

Run: `uv run pytest tests/test_qr_projector_2x2.py -k energy_agreement -v`
Expected: PASS. If the gap exceeds tol, check warm-up (`qr_warmup_steps`) is active and the gauge fix is applied — a too-large gap usually means the QR subspace wasn't warmed into the rank-χ regime.

- [ ] **Step 3: Write the gradient-agreement test**

```python
@pytest.mark.algorithm
def test_qr_gradient_matches_fd_and_svd():
    # (a) AD gradient of the QR-path energy vs central finite difference.
    g_ad = _energy_grad(projector_method="qr")           # jax.grad
    g_fd = _energy_grad_fd(projector_method="qr", eps=1e-5)
    np.testing.assert_allclose(g_ad, g_fd, atol=1e-4, rtol=1e-3)
    # (b) QR AD gradient close to SVD AD gradient on the same small model.
    g_svd = _energy_grad(projector_method="svd")
    np.testing.assert_allclose(g_ad, g_svd, atol=1e-3, rtol=1e-2)
```

- [ ] **Step 4: Run**

Run: `uv run pytest tests/test_qr_projector_2x2.py -k gradient_matches -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_qr_projector_2x2.py
git commit -m "test(#570): QR-vs-SVD energy + gradient agreement (Heisenberg D2/D3)"
```

---

## Task 7: Regression — existing CTM-AD suite under `projector_method="qr"`

**Files:**
- Test: `tests/test_qr_projector_2x2.py` (or parametrize the existing suite)

- [ ] **Step 1: Add a multi-block + fermionic regression test**

Parametrize a multi-sector (non-trivial-charge) U(1) CTM-AD step and a FermionParity smoke case (the #565/#566 surfacing model) with `projector_method="qr"`, asserting: a finite gradient (no NaN), and energy within `5e-4` of the `"svd"` run. Reuse the corner/model fixtures from `tests/test_block_sparse_ctm_ad.py`.

- [ ] **Step 2: Run the relevant existing suite with QR**

Run: `uv run pytest tests/test_block_sparse_ctm_ad.py tests/test_qr_projector_2x2.py -v`
Expected: PASS (no NaN, energies agree). Investigate any failure before proceeding.

- [ ] **Step 3: Commit**

```bash
git add tests/test_qr_projector_2x2.py
git commit -m "test(#570): multi-block + fermionic CTM-AD regression under QR projector"
```

---

## Task 8: A100 perf measurement (the deliverable, not pass/fail)

**Files:**
- Reuse: `examples/profile_570_sweepvjp_compile.py`

- [ ] **Step 1: Confirm the rig honors `projector_method`**

Read `examples/profile_570_sweepvjp_compile.py`; ensure it accepts a projector-method/`decomp` switch. If not, add a `--projector {svd,qr}` flag that threads into the CTM config.

- [ ] **Step 2: Run SVD vs QR on A100**

Per the A100 env (CUDA13, `uv sync --extra cuda13`, `JAX_PLATFORMS=cuda,cpu`, x64), run D=4, χ∈{8,12,16} for both `--projector svd` and `--projector qr`, capturing **backward HLO instruction count** (deterministic), total compile time, and warm-step runtime.

Run (example):
```bash
JAX_PLATFORMS=cuda,cpu uv run python examples/profile_570_sweepvjp_compile.py \
    --D 4 --chi 8 12 16 --full --projector qr --x64
```

- [ ] **Step 3: Record the comparison**

Append a `## Task 8 result` table (χ | svd HLO | qr HLO | ratio | svd compile | qr compile | svd warm-step | qr warm-step) to the spec. Confirm the measured backward-op reduction matches Task 2's prediction (~25%).

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-06-09-qr-projector-2x2-ad-drop-in-570.md examples/profile_570_sweepvjp_compile.py
git commit -m "bench(#570): A100 QR-vs-SVD 2x2 projector backward HLO/compile/runtime"
```

---

## Task 9: Docs + PR

- [ ] **Step 1: Update user-facing docs**

In `README.md` and the iPEPS/CTM docs (`docs/guide/algorithms/ctm.md`), note that `projector_method="qr"` now runs a real QR projector on the 2×2 AD path (M1/M2 via QR; M′ truncation via SVD), opt-in, with the measured backward speedup. Update `ipeps_config.py`'s `projector_method` comment.

- [ ] **Step 2: Run the core suite**

Run: `uv run pytest -m core`
Expected: PASS.

- [ ] **Step 3: Open the PR**

```bash
git push -u origin feat/qr-projector-2x2-570
gh pr create --title "feat(#570): drop-in QR projector for the 2x2 Fishman AD path" \
  --body "$(cat <<'EOF'
Replaces the two non-truncating M1/M2 half-SVDs in the symmetric 2x2 Fishman
projector with gauge-fixed block-sparse QR (no 1/(s_i^2-s_j^2) F-matrix),
keeping the M' truncation SVD. Opt-in via projector_method="qr" (previously an
eigh misnomer on the standard path). Validated by biorthogonality + Heisenberg
D2/D3 energy/gradient agreement (QR changes the fixed point, so not byte-parity).
Includes the per-SVD cost-attribution go/no-go and the A100 HLO/compile/runtime
table. Faithful reduced-corner QR-CTMRG deferred (logged follow-up).

Spec: docs/superpowers/specs/2026-06-09-qr-projector-2x2-ad-drop-in-570.md
Plan: docs/superpowers/plans/2026-06-09-qr-projector-2x2-ad-drop-in.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Merge after CI**

```bash
gh pr merge --squash --delete-branch --auto
```

---

## Self-review notes

- **Spec coverage:** C1→Tasks 1,3; C2→Task 4; C3→Task 5; C4→default in Task 4 sig; C5→Tasks 2,8; testing T1→Task1, T2→Task2, T3→Task4, T4/T5→Task6, T6→Task7, T7→Task8. All covered.
- **Gating:** Task 1 (QR-VJP stability) and Task 2 (cost go/no-go) precede the build (Task 4); Task 2 Step 3 requires explicit user confirmation before Task 4.
- **Known prototype point:** Task 4 Stage-4 R-factor seam contraction is derived against the biorthogonality oracle (Step 1) rather than asserted blind — the one genuinely research-y step, correctly TDD-gated.
- **Fixtures:** Tasks 4/6/7 reuse existing model/corner fixtures from `tests/test_block_sparse_ctm_ad.py` and the iPEPS tests rather than re-deriving leg structures; locate and import them during execution.
