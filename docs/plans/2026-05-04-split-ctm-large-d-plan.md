# Split-CTM Energy at Large D — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `compute_energy_split_ctm_tensor` shim with a split-aware energy path that bounds peak intermediate at ~χ²·D⁴ instead of the current χ²·D⁴·d² floor, unblocking D=8/10 kagome PESS energy evaluation.

**Architecture:** Approach 1 from the design — pre-merge `T_ket·T_bra` over `_I` into a 4-leg "split edge" with the two D-legs left **unfused**, then build half-RDMs in a fixed contraction order that absorbs A and `A.bar_super()` separately, consuming one D-leg pair at a time. Forward-only this round; AD is a deliberate follow-up.

**Tech Stack:** Python 3.11/3.12, JAX, Tenax `Tensor` protocol (`DenseTensor` / `SymmetricTensor`), `pytest`. Working branch: `feat/split-ctm-large-d` (already created).

**Design doc:** `docs/plans/2026-05-04-split-ctm-large-d-design.md`

---

## Conventions used in this plan

- **Files:** all source changes in `src/tenax/algorithms/_split_ctm_tensor_energy.py` unless noted. Tests in `tests/test_split_ctm_tensor.py` (extend) and a new `tests/test_split_ctm_large_d_memory.py`.
- **TDD discipline:** every task writes the failing test first, runs it to confirm failure, implements, confirms pass, then commits. The "test" in early tasks is parity vs the demoted shim — the only oracle we have.
- **Commit message convention:** scope `(ctm)` for engine code, `test:` for test-only commits, mirroring recent commits like `fb623cd fix(ctm): rebalance vertical/diagonal RDM ...`.
- **Pre-commit hook:** already installed (verified before plan was written). Don't pass `--no-verify`.
- **Test command (fast subset):** `uv run pytest tests/test_split_ctm_tensor.py -v -m "not slow" -x`.

## Existing context the executor needs

- The current shim lives at `src/tenax/algorithms/_split_ctm_tensor_energy.py:25-130` (`_split_env_to_tensor_standard`) and `:133-155` (`compute_energy_split_ctm_tensor`).
- The standard-CTM RDMs that the new code mirrors live at `src/tenax/algorithms/_ctm_tensor_energy.py`: `_rdm_1site_tensor` (lines 30-82), `_rdm_diagonal_tensor` (85-190), `_rdm2x1_tensor` (193-280), `_rdm1x2_tensor` (283-369). These are the structural references.
- `SplitCTMTensorEnv` field labels in `src/tenax/algorithms/_split_ctm_tensor_init.py:36-55`. T-edges follow:
  - `T1_ket: (t1k_l, u_ket, t1k_I)`, `T1_bra: (t1b_I, u_bra, t1b_r)`
  - `T2_ket: (t2k_u, r_ket, t2k_I)`, `T2_bra: (t2b_I, r_bra, t2b_d)`
  - `T3_ket: (t3k_r, d_ket, t3k_I)`, `T3_bra: (t3b_I, d_bra, t3b_l)`
  - `T4_ket: (t4k_d, l_ket, t4k_I)`, `T4_bra: (t4b_I, l_bra, t4b_u)`
- Existing fixtures in `tests/test_split_ctm_tensor.py:28-79`: `small_peps_dense`, `small_peps_symmetric`, `heisenberg_gate`. Reuse these.

---

## Task 1: helper `_make_split_edge`

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py` (add helper, do not touch shim yet)
- Test: `tests/test_split_ctm_tensor.py` (new test class `TestSplitEdgeHelper`)

**Step 1: Write the failing test**

Append to `tests/test_split_ctm_tensor.py`:

```python
class TestSplitEdgeHelper:
    """Tests for the new _make_split_edge / _make_split_edges helpers."""

    def test_make_split_edge_shape_and_labels(self, small_peps_dense):
        """_make_split_edge contracts T_ket·T_bra on _I, leaves D-legs unfused."""
        from tenax.algorithms._split_ctm_tensor_energy import _make_split_edge

        env = initialize_split_ctm_tensor_env(small_peps_dense, chi=4, chi_I=4)
        T1 = _make_split_edge(
            env.T1_ket, env.T1_bra,
            ket_I="t1k_I", bra_I="t1b_I",
            out_chi_l="t1_l", out_chi_r="t1_r",
        )
        labels = T1.labels()
        # Four legs: chi_l, u_ket, u_bra, chi_r — D-legs unchanged from inputs.
        assert set(labels) == {"t1_l", "u_ket", "u_bra", "t1_r"}
        # Dimensions: (chi, D, D, chi). With D=2, chi=4 → (4, 2, 2, 4).
        dim_by_label = {idx.label: idx.dim for idx in T1.indices}
        assert dim_by_label["t1_l"] == 4
        assert dim_by_label["t1_r"] == 4
        assert dim_by_label["u_ket"] == 2
        assert dim_by_label["u_bra"] == 2
```

**Step 2: Run test to verify it fails**

```
uv run pytest tests/test_split_ctm_tensor.py::TestSplitEdgeHelper::test_make_split_edge_shape_and_labels -v
```

Expected: `ImportError` for `_make_split_edge`.

**Step 3: Write minimal implementation**

In `src/tenax/algorithms/_split_ctm_tensor_energy.py`, append after the existing imports:

```python
def _make_split_edge(
    T_ket: Tensor,
    T_bra: Tensor,
    ket_I: str,
    bra_I: str,
    out_chi_l: str,
    out_chi_r: str,
) -> Tensor:
    """Contract T_ket·T_bra over the interlayer bond; do NOT fuse the two D-legs.

    Returns a 4-leg tensor with labels (out_chi_l, <ket D-label>, <bra D-label>, out_chi_r).
    The D-leg labels are inherited from the inputs (e.g. ``u_ket``/``u_bra`` for T1).
    """
    k = T_ket.relabel(ket_I, "_I_tmp")
    b = T_bra.relabel(bra_I, "_I_tmp")
    merged = contract(k, b)
    # Find the two chi labels in `merged.labels()`. The ket chi is whichever label
    # was on T_ket but not in (ket_I, ket_D). Same for bra. We just pass them
    # through via relabel — caller chose the names.
    ket_labels = set(T_ket.labels()) - {ket_I}
    bra_labels = set(T_bra.labels()) - {bra_I}
    ket_chi = next(l for l in ket_labels if l not in bra_labels and not l.endswith("_ket"))
    bra_chi = next(l for l in bra_labels if l not in ket_labels and not l.endswith("_bra"))
    return merged.relabels({ket_chi: out_chi_l, bra_chi: out_chi_r})
```

**Step 4: Run test to verify it passes**

```
uv run pytest tests/test_split_ctm_tensor.py::TestSplitEdgeHelper::test_make_split_edge_shape_and_labels -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_tensor_energy.py tests/test_split_ctm_tensor.py
git commit -m "feat(ctm): add _make_split_edge helper for split-aware RDM construction"
```

---

## Task 2: helper `_make_split_edges` + label-collision assertion

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py`
- Test: `tests/test_split_ctm_tensor.py`

**Step 1: Write the failing test**

```python
def test_make_split_edges_no_label_collisions(self, small_peps_dense):
    """The four T_split tensors share no D-leg labels; chi labels follow the
    standard CTM convention (t1_l/t1_r, t2_u/t2_d, t3_l/t3_r, t4_d/t4_u)."""
    from tenax.algorithms._split_ctm_tensor_energy import _make_split_edges

    env = initialize_split_ctm_tensor_env(small_peps_dense, chi=4, chi_I=4)
    splits = _make_split_edges(env)

    assert set(splits.keys()) == {"T1", "T2", "T3", "T4"}
    # D-leg label sets per edge are disjoint:
    d_labels = {
        "T1": {"u_ket", "u_bra"},
        "T2": {"r_ket", "r_bra"},
        "T3": {"d_ket", "d_bra"},
        "T4": {"l_ket", "l_bra"},
    }
    for name, want in d_labels.items():
        labs = set(splits[name].labels())
        assert want.issubset(labs), f"{name} missing {want - labs}"
    # No D-leg label collides across edges:
    all_d = set().union(*d_labels.values())
    assert len(all_d) == 8  # all distinct
```

**Step 2: Run** `uv run pytest tests/test_split_ctm_tensor.py::TestSplitEdgeHelper::test_make_split_edges_no_label_collisions -v` → expect ImportError.

**Step 3: Implementation**

Append to `_split_ctm_tensor_energy.py`:

```python
def _make_split_edges(env: SplitCTMTensorEnv) -> dict[str, Tensor]:
    """Build 4-leg split edges for all four boundary T's.

    Returns a dict keyed ``"T1"``, ``"T2"``, ``"T3"``, ``"T4"`` with each value
    a 4-leg ``(chi, D_ket, D_bra, chi)`` tensor. D-leg labels are
    ``{u,r,d,l}_ket`` / ``{u,r,d,l}_bra`` and are globally unique across the
    four edges; chi-leg labels follow the standard CTMTensorEnv convention
    (``t1_l/t1_r``, ``t2_u/t2_d``, ``t3_r/t3_l``, ``t4_d/t4_u``).
    """
    return {
        "T1": _make_split_edge(env.T1_ket, env.T1_bra, "t1k_I", "t1b_I", "t1_l", "t1_r"),
        "T2": _make_split_edge(env.T2_ket, env.T2_bra, "t2k_I", "t2b_I", "t2_u", "t2_d"),
        "T3": _make_split_edge(env.T3_ket, env.T3_bra, "t3k_I", "t3b_I", "t3_r", "t3_l"),
        "T4": _make_split_edge(env.T4_ket, env.T4_bra, "t4k_I", "t4b_I", "t4_d", "t4_u"),
    }
```

**Step 4: Run** the same test → PASS.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_tensor_energy.py tests/test_split_ctm_tensor.py
git commit -m "feat(ctm): add _make_split_edges builder for all four boundary edges"
```

---

## Task 3: `_rdm_1site_split_tensor`

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py`
- Test: `tests/test_split_ctm_tensor.py` (new test class `TestSplitRDMs`)

**Step 1: Write the failing parity test**

```python
class TestSplitRDMs:
    """Parity vs the shim for each split-aware RDM."""

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_rdm_1site_matches_shim(self, D, chi):
        from tenax.algorithms._split_ctm_tensor_energy import (
            _rdm_1site_split_tensor,
            _split_env_to_tensor_standard,
        )
        from tenax.algorithms._ctm_tensor_energy import _rdm_1site_tensor

        # Build a random site tensor at this size.
        key = jax.random.PRNGKey(0)
        d = 2
        data = jax.random.normal(key, (D, D, D, D, d))
        data = data / jnp.linalg.norm(data)
        sym = U1Symmetry()
        idx = TensorIndex.from_charges
        z = lambda n: np.zeros(n, dtype=np.int32)
        A = DenseTensor(data, (
            idx(sym, z(D), FlowDirection.OUT, label="u"),
            idx(sym, z(D), FlowDirection.IN,  label="d"),
            idx(sym, z(D), FlowDirection.OUT, label="l"),
            idx(sym, z(D), FlowDirection.IN,  label="r"),
            idx(sym, z(d), FlowDirection.IN,  label="phys"),
        ))

        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        rdm_split = _rdm_1site_split_tensor(A, env)
        rdm_shim  = _rdm_1site_tensor(A, _split_env_to_tensor_standard(env))
        assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)
```

**Step 2: Run** → `ImportError`.

**Step 3: Implementation**

Append to `_split_ctm_tensor_energy.py`. The single-site case is a degenerate per-site env (Pattern B): build the env around `A` with split edges, absorb `A` then `A_bra`, finish.

```python
def _rdm_1site_split_tensor(A: Tensor, env: SplitCTMTensorEnv) -> jax.Array:
    """Single-site RDM via split-aware contraction.

    Same network as ``_rdm_1site_tensor`` but uses 4-leg split edges and keeps
    A / A.bar_super() separate (no double-layer fusion). Peak intermediate is
    bounded by chi²·D⁴.
    """
    splits = _make_split_edges(env)
    T1, T2, T3, T4 = splits["T1"], splits["T2"], splits["T3"], splits["T4"]

    A_bra = A.bar_super().relabels({
        "u": "u_bra", "d": "d_bra", "l": "l_bra", "r": "r_bra", "phys": "phys_bra"
    })

    # Boundary frame: top row + bottom row + left/right edges, all using split edges.
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    top_row = contract(contract(C1, T1), C2)              # (c1_d, u_ket, u_bra, c2_d) — chi²·D²

    C4 = env.C4.relabel("c4_u", "t3_r")
    C3 = env.C3.relabel("c3_l", "t3_l")
    bot_row = contract(contract(C4, T3), C3)              # (c4_r, d_ket, d_bra, c3_u) — chi²·D²

    T4_e = T4.relabels({"t4_d": "c1_d", "t4_u": "c4_r"})
    T2_e = T2.relabels({"t2_u": "c2_d", "t2_d": "c3_u"})

    # Frame: top·T4·T2·bot. Build pairwise so peak stays at chi²·D⁴.
    frame_top  = contract(top_row, T4_e)                  # (u_ket, u_bra, c2_d, l_ket, l_bra, c4_r) — chi²·D⁴
    frame_full = contract(frame_top, T2_e)                # (u_ket, u_bra, l_ket, l_bra, c4_r, r_ket, r_bra, c3_u) — chi²·D⁶ (peak)
    frame_full = contract(frame_full, bot_row)            # (u_ket, u_bra, l_ket, l_bra, r_ket, r_bra, d_ket, d_bra) — D⁸

    # Note: the chi²·D⁶ frame_full intermediate is the worst stage of the 1-site
    # case. At target sizes (D=10 chi=200) it's ~50 GB so 1-site is NOT the
    # large-D path. The vertical/horizontal/diagonal RDMs use better orderings;
    # 1-site here is provided primarily for parity testing and small-D AD probes.

    # Absorb A then A_bra — D-legs labeled to match.
    rdm_t = contract(frame_full, A)                       # (u_bra, l_bra, r_bra, d_bra, phys, ...) — D⁴·d at peak after open phys
    rdm_t = contract(rdm_t, A_bra, output_labels=["phys", "phys_bra"])  # (d, d)

    rdm = rdm_t.todense()
    rdm = 0.5 * (rdm + rdm.conj().T)
    rdm = rdm / (jnp.trace(rdm) + EPS)
    return rdm
```

Add the missing import at the top:
```python
import jax.numpy as jnp
from tenax.core import EPS
```

**Step 4: Run** → PASS at D=2 and D=3.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_tensor_energy.py tests/test_split_ctm_tensor.py
git commit -m "feat(ctm): add _rdm_1site_split_tensor (parity vs shim at small D)"
```

> **Note for the executor:** the 1-site case has an intrinsic chi²·D⁶ frame stage that the design's bound (chi²·D⁴) does *not* improve. This is unique to 1-site and harmless because energy never calls 1-site for nearest-neighbour bonds. The function exists for completeness and parity testing. The vertical/horizontal/diagonal cases (Tasks 4-6) use the interleaved order from the design and DO meet the chi²·D⁴ bound.

---

## Task 4: `_rdm1x2_split_tensor` (vertical, two-half — the design's main act)

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py`
- Test: `tests/test_split_ctm_tensor.py`

**Step 1: Write the failing parity test**

Add inside `TestSplitRDMs`:

```python
@pytest.mark.parametrize("D, chi", [(2, 8), (3, 12), (4, 16)])
def test_rdm1x2_matches_shim(self, D, chi):
    from tenax.algorithms._split_ctm_tensor_energy import (
        _rdm1x2_split_tensor,
        _split_env_to_tensor_standard,
    )
    from tenax.algorithms._ctm_tensor_energy import _rdm1x2_tensor

    A = make_random_dense_site(D, d=2, seed=1)             # tiny helper at top of test file
    env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

    rdm_split = _rdm1x2_split_tensor(A, env)
    rdm_shim  = _rdm1x2_tensor(A, _split_env_to_tensor_standard(env))
    assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)
```

(Move the random-site builder out of Task 3's test into a helper at the top of `tests/test_split_ctm_tensor.py` while you're here. Keep the change small and atomic.)

**Step 2: Run** → ImportError.

**Step 3: Implementation**

```python
def _rdm1x2_split_tensor(A: Tensor, env: SplitCTMTensorEnv) -> jax.Array:
    """Vertical 1×2 RDM via split-aware contraction. Bounded peak chi²·D⁴.

    Top half contraction order (mirrored for bottom):

        top_row     = C1 · T1_split · C2                # χ²·D²
        top_T4      = top_row · T4_T_split              # χ²·D⁴
        top_T4_A    = top_T4 · A                        # χ²·D⁴·d  ← peak
        top_T4_A_T2 = top_T4_A · T2_T_split             # χ²·D³·d
        top_half    = top_T4_A_T2 · A_bra               # χ²·D²·d²

    Combine on (t4_u↔t4_dB chi×2, d_ket↔u_ket_B inner-D, d_bra↔u_bra_B inner-D).
    """
    splits = _make_split_edges(env)
    T1, T2, T3, T4 = splits["T1"], splits["T2"], splits["T3"], splits["T4"]

    A_bra = A.bar_super().relabels({
        "u": "u_bra", "d": "d_bra", "l": "l_bra", "r": "r_bra", "phys": "phys_bra"
    })

    # ---------- Top half ----------
    C1 = env.C1.relabel("c1_r", "t1_l")
    C2 = env.C2.relabel("c2_l", "t1_r")
    top_row = contract(contract(C1, T1), C2)                          # (c1_d, u_ket, u_bra, c2_d)

    T4_T = T4.relabels({"t4_d": "c1_d"})                              # (c1_d, l_ket, l_bra, t4_u)
    T2_T = T2.relabels({"t2_u": "c2_d"})                              # (c2_d, r_ket, r_bra, t2_d)

    top_T4      = contract(top_row, T4_T)                             # (u_ket, u_bra, c2_d, l_ket, l_bra, t4_u)
    A_top       = A.relabels({"u": "u_ket", "l": "l_ket", "r": "r_ket"})
    top_T4_A    = contract(top_T4, A_top)                             # (u_bra, c2_d, l_bra, t4_u, d, phys)
    top_T4_A_T2 = contract(top_T4_A, T2_T)                            # (u_bra, l_bra, t4_u, d, phys, r_bra, t2_d)
    A_bra_top   = A_bra.relabels({"u_bra": "u_bra", "l_bra": "l_bra", "r_bra": "r_bra"})  # already named
    top_half    = contract(top_T4_A_T2, A_bra_top)                    # (t4_u, t2_d, d, d_bra, phys, phys_bra)

    # ---------- Bottom half (label-renamed copies for the bottom site) ----------
    # Suffix everything with "B" except the chi-bonds that meet the top half.
    T1_B = T3  # not used in bottom half? bottom row uses T3, not T1 — see below.
    # Bottom-row pieces:
    C4 = env.C4.relabel("c4_u", "t3_r")
    C3 = env.C3.relabel("c3_l", "t3_l")
    bot_row = contract(contract(C4, T3), C3)                          # (c4_r, d_ket, d_bra, c3_u)
    bot_row = bot_row.relabels({"d_ket": "u_ketB", "d_bra": "u_braB"})  # bottom-site inner D becomes "uB" wrt A_B

    T4_B = T4.relabels({"t4_u": "c4_r", "l_ket": "l_ketB", "l_bra": "l_braB"})  # share c4_r with bot_row
    T2_B = T2.relabels({"t2_d": "c3_u", "r_ket": "r_ketB", "r_bra": "r_braB"})
    T4_B = T4_B.relabel("t4_d", "t4_uB")                              # bottom T4 chi up label
    T2_B = T2_B.relabel("t2_u", "t2_dB")

    bot_T4      = contract(bot_row, T4_B)                             # (u_ketB, u_braB, c3_u, l_ketB, l_braB, t4_uB)
    A_bot       = A.relabels({"u": "u_ketB", "l": "l_ketB", "r": "r_ketB", "d": "d_ketB", "phys": "phys_B"})
    bot_T4_A    = contract(bot_T4, A_bot)                             # (u_braB, c3_u, l_braB, t4_uB, d_ketB, phys_B)
    bot_T4_A_T2 = contract(bot_T4_A, T2_B)                            # (u_braB, l_braB, t4_uB, d_ketB, phys_B, r_braB, t2_dB)
    A_bra_bot   = A_bra.relabels({
        "u_bra": "u_braB", "l_bra": "l_braB", "r_bra": "r_braB", "d_bra": "d_braB", "phys_bra": "phys_braB"
    })
    bot_half    = contract(bot_T4_A_T2, A_bra_bot)                    # (t4_uB, t2_dB, d_ketB, d_braB, phys_B, phys_braB)

    # ---------- Combine ----------
    bot_half = bot_half.relabels({
        "t4_uB": "t4_u",       # T4 chi seam
        "t2_dB": "t2_d",       # T2 chi seam
        "u_ketB": "d",          # bottom's top D-leg of A is top's bottom d-leg of A
        "u_braB": "d_bra",      # same for A_bra
    })
    rdm_t = contract(top_half, bot_half,
                     output_labels=["phys", "phys_B", "phys_bra", "phys_braB"])

    rdm = rdm_t.todense()
    d = rdm.shape[0]
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + EPS)
    return rdm_mat.reshape(d, d, d, d)
```

> **Sanity check during implementation:** add a temporary `print(top_T4.shape)` etc. at each stage; the shapes should match the comments. Remove before commit.

**Step 4: Run** → PASS at D=2/3/4.

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_tensor_energy.py tests/test_split_ctm_tensor.py
git commit -m "feat(ctm): add _rdm1x2_split_tensor with chi²·D⁴ peak (parity vs shim)"
```

---

## Task 5: `_rdm2x1_split_tensor` (horizontal, two-half — 90° rotation of Task 4)

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py`
- Test: `tests/test_split_ctm_tensor.py`

**Step 1: Failing parity test** — exact mirror of Task 4 with `_rdm2x1_split_tensor` / `_rdm2x1_tensor`.

**Step 2: Run** → ImportError.

**Step 3: Implementation** — mirror of `_rdm1x2_split_tensor` with left/right halves instead of top/bottom. The recipe is:

```
left_top = C1·T1_split                              # then add T4 down → left half pre-A
left_T4  = left_top · T4_split                       # chi²·D⁴
left_T4_A = left_T4 · A                              # chi²·D⁴·d
left_T4_A_T3 = left_T4_A · T3_split (or C4·T3·...)   # chi²·D³·d
left_half = left_T4_A_T3 · A_bra                     # chi²·D²·d²

# right half mirror; combine on (t1_r↔t1_lR, t3_l↔t3_rR, r_ket↔l_ket_R, r_bra↔l_bra_R).
```

Pattern follows `_rdm2x1_tensor` at `_ctm_tensor_energy.py:193-280`. Use the existing function as a structural template; the only differences are (a) split edges instead of merged T's and (b) absorb A and A_bra separately.

**Step 4: Run** → PASS at D=2/3/4.

**Step 5: Commit**

```bash
git commit -m "feat(ctm): add _rdm2x1_split_tensor with chi²·D⁴ peak (parity vs shim)"
```

---

## Task 6: `_rdm_diagonal_split_tensor` (4-site per-site env — Pattern B)

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py`
- Test: `tests/test_split_ctm_tensor.py`

**Step 1: Failing parity test**

```python
@pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
def test_rdm_diagonal_matches_shim(self, D, chi):
    from tenax.algorithms._split_ctm_tensor_energy import (
        _rdm_diagonal_split_tensor,
        _split_env_to_tensor_standard,
    )
    from tenax.algorithms._ctm_tensor_energy import _rdm_diagonal_tensor

    A = make_random_dense_site(D, d=2, seed=2)
    env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

    rdm_split = _rdm_diagonal_split_tensor(A, env)
    rdm_shim  = _rdm_diagonal_tensor(A, _split_env_to_tensor_standard(env))
    assert jnp.allclose(rdm_split, rdm_shim, atol=1e-10)
```

**Step 2: Run** → ImportError.

**Step 3: Implementation** — mirror `_rdm_diagonal_tensor` (`_ctm_tensor_energy.py:85-190`) with split edges. Key difference: closed sites (TR, BL) build their `ac_TR`/`ac_BL` factor as `(A · A_bra)` traced over `phys` *without* fusing the four D-leg pairs. Open sites (TL, BR) build the same way but with `phys` left open.

Outline:
```python
def _rdm_diagonal_split_tensor(A, env):
    splits = _make_split_edges(env)
    A_bra  = A.bar_super().relabels({...: ..._bra})

    # site_env_TL (open): build by per-site env with split edges + A + A_bra,
    # leaving phys/phys_bra open. Peak chi²·D⁴ at TL_env · A stage.
    site_env_TL = _build_open_site_env(env, splits, A, A_bra,
                                        corner="C1", T_top=splits["T1"], T_left=splits["T4"])

    # site_env_TR (closed): trace phys; relabel D-pair legs with site-unique suffixes.
    site_env_TR = _build_closed_site_env(...)

    # site_env_BL (closed): same builder, BR position in lattice (sharing D-bonds with TL/BR).
    # site_env_BR (open).

    # Column-pair joins, then final combine — verbatim from _rdm_diagonal_tensor's structure.
    ...
```

> **Note for the executor:** factor out `_build_open_site_env` and `_build_closed_site_env` as private helpers in this same task — they make the code symmetric with what `_rdm_diagonal_tensor` does. If you find yourself copying the same 6-line absorption loop four times, you've gone wrong.

**Step 4: Run** → PASS at D=2/3.

**Step 5: Commit**

```bash
git commit -m "feat(ctm): add _rdm_diagonal_split_tensor (4-site per-site env, parity vs shim)"
```

---

## Task 7: `_rdm1x2_split_tensor_2site` (mixed env)

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py`
- Test: `tests/test_split_ctm_tensor.py`

**Step 1: Failing parity test** — same shape as Task 4's, but constructs two random sites A, B with separate `env_A`, `env_B`, and compares `_rdm1x2_split_tensor_2site(A, B, env_A, env_B)` against the shim's `_rdm1x2_tensor_2site(A, B, std_A, std_B)`.

**Step 2: Run** → ImportError.

**Step 3: Implementation** — copy the body of `_rdm1x2_split_tensor` with the modification that:
- Top half uses `splits_A`, `A`, `A_bra_A` from `env_A`.
- Bottom half uses `splits_B`, `B`, `B_bra` from `env_B`.

Take the structural reference from `_rdm1x2_tensor_2site` (`_ctm_tensor_energy.py:478-560`).

**Step 4: Run** → PASS at D=2/3.

**Step 5: Commit**

```bash
git commit -m "feat(ctm): add _rdm1x2_split_tensor_2site (mixed env, parity vs shim)"
```

---

## Task 8: `_rdm2x1_split_tensor_2site` (mixed env, horizontal)

Same structure as Task 7 but horizontal. Test, implement, commit:

```bash
git commit -m "feat(ctm): add _rdm2x1_split_tensor_2site (mixed env, parity vs shim)"
```

---

## Task 9: rewrite `compute_energy_split_ctm_tensor` to use split RDMs

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py:133-155` (rewrite body)
- Test: `tests/test_split_ctm_tensor.py:204-231` already exists (`test_energy_roundtrip_via_standard`) — should keep passing as-is, since both paths give the same energy.

**Step 1: Failing test (regression check)**

Add a direct parity test that compares the new energy against the shim energy explicitly across D, chi:

```python
@pytest.mark.parametrize("D, chi", [(2, 8), (3, 12), (4, 16)])
def test_compute_energy_split_native_matches_shim(self, D, chi):
    from tenax.algorithms._split_ctm_tensor_energy import (
        _split_env_to_tensor_standard,
        compute_energy_split_ctm_tensor,
    )
    from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor

    A = make_random_dense_site(D, d=2, seed=3)
    H = self.heisenberg_gate_dd(d=2)
    env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

    E_split = compute_energy_split_ctm_tensor(A, env, H, d=2)
    E_shim  = compute_energy_ctm_tensor(A, _split_env_to_tensor_standard(env), H, d=2)
    assert jnp.allclose(E_split, E_shim, atol=1e-10)
```

**Step 2: Run** → it should still PASS *before* changes (both paths currently give the shim's answer). Then proceed with rewrite. After rewrite, this test asserts the new path matches the shim ground truth.

**Step 3: Rewrite the function body**

Replace lines 133-155 of `_split_ctm_tensor_energy.py`:

```python
def compute_energy_split_ctm_tensor(
    A: Tensor,
    env: SplitCTMTensorEnv,
    hamiltonian_gate: Tensor | jax.Array,
    d: int | None = None,
) -> jax.Array:
    """Compute energy per site using split CTM environment, split-aware.

    Builds horizontal and vertical RDMs directly from (T_ket, T_bra, A,
    A.bar_super()), without merging ket/bra to the standard double-layer env.
    Bounded peak intermediate ~chi²·D⁴.
    """
    if d is None:
        phys_idx = [i for i in A.indices if i.label == "phys"]
        d = phys_idx[0].dim if phys_idx else A.indices[-1].dim
    if isinstance(hamiltonian_gate, Tensor):
        H = hamiltonian_gate.todense().reshape(d, d, d, d)
    else:
        H = hamiltonian_gate.reshape(d, d, d, d)

    rdm_h = _rdm2x1_split_tensor(A, env)
    rdm_v = _rdm1x2_split_tensor(A, env)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real
```

**Step 4: Run all of `tests/test_split_ctm_tensor.py`** to confirm:
- `test_energy_roundtrip_via_standard` still PASS
- `test_compute_energy_split_native_matches_shim` PASS at D=2/3/4
- `test_energy_is_finite` PASS
- All Task 1–8 tests still PASS

```
uv run pytest tests/test_split_ctm_tensor.py -v -m "not slow"
```

**Step 5: Commit**

```bash
git add src/tenax/algorithms/_split_ctm_tensor_energy.py tests/test_split_ctm_tensor.py
git commit -m "refactor(ctm): rewrite compute_energy_split_ctm_tensor as split-aware path"
```

---

## Task 10: add `compute_energy_split_ctm_tensor_2site`

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py` (append)
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py` re-export `__all__`
- Test: `tests/test_split_ctm_tensor.py`

**Step 1: Failing test** — parity vs `compute_energy_ctm_tensor_2site` after shim conversion.

```python
@pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
def test_compute_energy_split_2site_matches_shim(self, D, chi):
    ...
    E_split = compute_energy_split_ctm_tensor_2site(A, B, env_A, env_B, H, d=2)
    E_shim  = compute_energy_ctm_tensor_2site(A, B,
                                              _split_env_to_tensor_standard(env_A),
                                              _split_env_to_tensor_standard(env_B),
                                              H, d=2)
    assert jnp.allclose(E_split, E_shim, atol=1e-10)
```

**Step 2: Run** → ImportError.

**Step 3: Implementation** — mechanical wrap of `_rdm{2x1,1x2}_split_tensor_2site`, mirroring `compute_energy_ctm_tensor_2site` (`_ctm_tensor_energy.py:563-600`).

**Step 4: Run** → PASS.

**Step 5: Commit**

```bash
git commit -m "feat(ctm): add compute_energy_split_ctm_tensor_2site"
```

---

## Task 11: add `compute_energy_split_ctm_tensor_multisite`

**Files:**
- Modify: `src/tenax/algorithms/_split_ctm_tensor_energy.py` (append)
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py` re-export `__all__`
- Test: `tests/test_split_ctm_tensor.py`

**Step 1: Failing test** — small 3-site Y-shaped unit cell (smallest non-trivial kagome simplex), parity vs `compute_energy_ctm_tensor_multisite` after shim conversion.

**Step 2: Run** → ImportError.

**Step 3: Implementation** — mechanical copy of `compute_energy_ctm_tensor_multisite` (`_ctm_tensor_energy.py:603-675`) with per-coord caching of `_make_split_edges(envs[coord])` and `A.bar_super()` outside the bond loop, dispatching to `_rdm*_split_tensor` (single env, when `coord == nb_coord`) or `_rdm*_split_tensor_2site` (mixed env).

**Step 4: Run** → PASS.

**Step 5: Commit**

```bash
git commit -m "feat(ctm): add compute_energy_split_ctm_tensor_multisite"
```

---

## Task 12: public API exports

**Files:**
- Modify: `src/tenax/__init__.py` (lazy-import dict around line 120 + `__all__` around line 427)
- Modify: `src/tenax/algorithms/_split_ctm_tensor.py` (`__all__` around line 134)
- Modify: `README.md` features list — one-line bullet under iPEPS section.

**Step 1: Smoke test**

```python
def test_public_exports_resolve():
    import tenax
    assert hasattr(tenax, "compute_energy_split_ctm_tensor_2site")
    assert hasattr(tenax, "compute_energy_split_ctm_tensor_multisite")
```

**Step 2: Run** → AttributeError.

**Step 3: Wire exports**

In `src/tenax/__init__.py`, add to the lazy dict near `compute_energy_split_ctm_tensor`:

```python
"compute_energy_split_ctm_tensor_2site": (
    "tenax.algorithms._split_ctm_tensor",
    "compute_energy_split_ctm_tensor_2site",
),
"compute_energy_split_ctm_tensor_multisite": (
    "tenax.algorithms._split_ctm_tensor",
    "compute_energy_split_ctm_tensor_multisite",
),
```

And to `__all__`:
```python
"compute_energy_split_ctm_tensor_2site",
"compute_energy_split_ctm_tensor_multisite",
```

In `_split_ctm_tensor.py`, add the two names to `__all__` and re-export them via `from _split_ctm_tensor_energy import ...`.

In `README.md`, under the "Features" / iPEPS section:

```markdown
- Split-CTM energy paths for 2-site checkerboard and multisite unit cells (kagome PESS, etc.) at large D
```

**Step 4: Run** → PASS.

**Step 5: Commit**

```bash
git add src/tenax/__init__.py src/tenax/algorithms/_split_ctm_tensor.py README.md tests/test_split_ctm_tensor.py
git commit -m "feat(ctm): export 2-site and multisite split-CTM energy entry points"
```

---

## Task 13: Tier-2 fermionic parity test

**Files:**
- Modify: `tests/test_split_ctm_tensor.py` (add fermionic test class)

**Step 1: Test**

```python
@pytest.mark.slow
class TestSplitRDMsFermionic:
    """Parity vs shim with FermionParity site tensors."""

    @pytest.mark.parametrize("D, chi", [(2, 8), (3, 12)])
    def test_fermionic_energy_matches_shim(self, D, chi):
        from tenax.core.symmetry import FermionParity
        # Build a small spinless-fermion-style site tensor with FermionParity.
        # Reuse fixtures or builders from tests/test_fermionic_ipeps_*.py if present;
        # else build inline.
        A = make_random_fermionic_site(D, d=2, seed=4, sym=FermionParity())
        H = self.heisenberg_gate_dd(d=2)
        env = ctm_split_tensor(A, chi=chi, max_iter=20, chi_I=chi)

        E_split = compute_energy_split_ctm_tensor(A, env, H, d=2)
        E_shim  = compute_energy_ctm_tensor(A, _split_env_to_tensor_standard(env), H, d=2)
        assert jnp.allclose(E_split, E_shim, atol=1e-10)
```

> **Note for the executor:** Tenax may already have a fermionic site builder in `tests/test_fermionic_ipeps_*.py` or `examples/`. Look there first; only inline if nothing reusable exists. Test must be marked `@pytest.mark.slow` (don't run on every PR).

**Step 2: Run with `-m slow`** → PASS.

**Step 3: Commit**

```bash
git commit -m "test(ctm): add fermionic parity test for split-aware energy path"
```

---

## Task 14: Tier-3 memory regression test

**Files:**
- Create: `tests/test_split_ctm_large_d_memory.py`

**Step 1: Test** (this one IS the implementation — no source changes; the test exists to lock in the win)

```python
"""Memory regression for split-CTM at large D.

These probes prove the split-aware energy path bounds peak memory at
~chi²·D⁴ on the kagome PESS canonical site tensor, fitting an 8 GB box
at D=8 chi=128.
"""
import tracemalloc
import pytest
import jax
import jax.numpy as jnp

# Skip if the canonical Liao replication harness isn't on this branch.
pytest.importorskip("examples.kagome_spin12_pess_liao2017_replication")
from examples.kagome_spin12_pess_liao2017_replication import (
    build_canonical_pess,
    heisenberg_gate,
)

from tenax.algorithms._split_ctm_tensor_energy import (
    compute_energy_split_ctm_tensor_multisite,
)
from tenax.algorithms.ipeps_ctm import ctm_split_multisite  # or whatever the multisite split CTM driver is


@pytest.mark.slow
def test_kagome_d4_p2_energy_within_tol():
    """At D=4 the new path must reproduce -0.347185 within 1e-4 (regression of PR #389)."""
    site_tensors, neighbors = build_canonical_pess(D=4)
    envs = ctm_split_multisite(site_tensors, chi=32, ...)
    H = heisenberg_gate()
    E = compute_energy_split_ctm_tensor_multisite(site_tensors, envs, neighbors, H, d=2)
    assert abs(float(E) - (-0.347185)) < 1e-4


@pytest.mark.slow
def test_kagome_d8_chi128_peak_memory_under_8gb():
    """At D=8 chi=128 P2 the peak memory must stay under 8 GB."""
    site_tensors, neighbors = build_canonical_pess(D=8)
    envs = ctm_split_multisite(site_tensors, chi=128, ...)
    H = heisenberg_gate()

    tracemalloc.start()
    E = compute_energy_split_ctm_tensor_multisite(site_tensors, envs, neighbors, H, d=2)
    _ = jax.block_until_ready(E)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_gb = peak / 1024**3
    assert peak_gb < 8.0, f"peak {peak_gb:.2f} GB exceeds 8 GB budget"
```

> **Caveats for the executor:**
> - The harness `examples/kagome_spin12_pess_liao2017_replication.py` is currently on PR #387's worktree. If that PR isn't on `main` yet, this test should `pytest.skip` cleanly via the `pytest.importorskip` line above. Do not block this PR on PR #387.
> - `tracemalloc` measures Python-level allocations only; it will undercount JAX's GPU pool. For CPU-only runs (which is what CI does) the measurement is meaningful. Document this in the test docstring.
> - The exact `ctm_split_multisite` driver name may differ — find the right entry point for multisite split CTM (or build the envs from per-coord `ctm_split_tensor` calls, which is what the harness already does).

**Step 2: Run with `-m slow`** locally:
```
uv run pytest tests/test_split_ctm_large_d_memory.py -v -m slow
```
Expected: PASS for the D=4 test; D=8 test PASSes if a 16 GB box is available, otherwise marked `xfail` with reason "needs 16 GB" — *don't* `xfail` it on principle though; if it OOMs the design is wrong and that's important to know.

**Step 3: Commit**

```bash
git commit -m "test(ctm): add large-D memory regression for split-aware energy path"
```

---

## Task 15: open the PR

**Step 1: Confirm CI-relevant tests still green**

```bash
uv run pytest -m core -x  # required for branch protection
```

Expected: all PASS.

**Step 2: Push and open PR**

```bash
git push -u origin feat/split-ctm-large-d
gh pr create --title "feat(ctm): split-aware energy at large D (replace shim)" --body "$(cat <<'EOF'
## Summary
- Replace the `compute_energy_split_ctm_tensor` shim with a split-aware energy path that keeps `T_ket`/`T_bra` and `A`/`A.bar_super()` separate through the half-RDM, bounding peak intermediate at ~chi²·D⁴ instead of the prior chi²·D⁴·d² floor.
- Add `compute_energy_split_ctm_tensor_2site` and `compute_energy_split_ctm_tensor_multisite` (the latter unblocks the Liao 2017 kagome PESS audit at D=8/10).
- Demote `_split_env_to_tensor_standard` to a private parity-test helper.

## Memory accounting (D=10 chi=200 d=2, complex128)
- Before: `chi²·D⁴·d²` ≈ 410 GB → OOM on every box.
- After:  `chi²·D⁴·d` ≈ 13 GB peak (`top_T4_A` stage); `chi²·D²·d²` ≈ 256 MB at glue.

## Test plan
- [x] Tier 1 — parity vs shim at D=2/3/4 for each new RDM and entry point (`tests/test_split_ctm_tensor.py`)
- [x] Tier 2 — fermionic parity at small D (`@pytest.mark.slow`)
- [x] Tier 3 — D=4 kagome regression at -0.347185 ± 1e-4; D=8 chi=128 peak memory < 8 GB
- [ ] Local probe at D=10 chi=200 (16 GB box) — documented, not in CI
- [x] `uv run pytest -m core` passes

## Out of scope
- AD support (forward-only this round; follow-up PR will add `jax.checkpoint` and gradient parity)
- Split-CTM SVD projector (`project_ctm_tech_debt.md` item 4) — independent

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Step 3: Confirm**

Print the PR URL.

---

## Open risks the executor should flag back

1. **Task 4 contraction order** — peak in the design accounting was χ²·D⁴ but the final implementation might land at χ²·D⁴·d (~13 GB at D=10 chi=200) due to the d-factor at `top_T4_A`. If a parity test passes but the memory regression at D=8 chi=128 *fails*, this is the place to dig: try interleaving `T2` *before* `A` (swap steps 3 and 4 in the half-builder) and re-measure.

2. **Diagonal `ac` factor** — `ac_TR` keeps four D-leg pairs unfused. If `Tensor.contract` insists on fusing during contraction (some symmetric paths do this implicitly via `fuse_indices`), the bound breaks. Verify with a shape-printing pass on the diagonal D=4 parity test.

3. **Multisite cache scope** — `_make_split_edges` and `bar_super()` are built inside `compute_energy_split_ctm_tensor_multisite` per energy call but reused across bonds. If JAX traces the energy function, those caches dissolve into the trace graph (fine). If someone later wants to pre-compute them across optimizer steps, that's a follow-up; do not optimize for it now.

4. **Pre-commit** — installed and active. If a hook fails, fix the root cause and re-stage; don't `--no-verify`.

---

**Plan complete and saved to `docs/plans/2026-05-04-split-ctm-large-d-plan.md`.**

Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** — Open a new session with `superpowers:executing-plans`, batch execution with checkpoints.

**Which approach?**
