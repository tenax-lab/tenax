# Chunk × Shard CTM-Move Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a four-variant probe that measures whether composing chunking (single-GPU, peak ÷ K) with GSPMD sharding (multi-GPU, peak ÷ N) on the real dense-CTM edge contraction cuts per-device peak ÷(N·K) and lets the sharded contraction dodge the giant-gemm autotuner wall.

**Architecture:** A standalone throwaway spike (`examples/spike_chunk_shard_ctm_move.py`) extending `spike_chunked_ctm_move.py` with a `jax.sharding` mesh. It shards `a`'s `r2=n` axis (a free D² output leg = `ctm_sharding`'s left-move surviving axis), orthogonal to the chunk axis (boundary χ). Four variants, one per process (`peak_bytes_in_use` is cumulative): `full` / `chunked` / `sharded` / `chunkshard`. No `src/` change. Then a (D,χ) scan + HLO check + findings doc.

**Tech Stack:** JAX (x64, CUDA13 extra), GSPMD `NamedSharding`, `jax.lax.map`, 2× A100 (devices 0,2).

**Spec:** `docs/superpowers/specs/2026-06-30-chunk-shard-ctm-move-gate-design.md`

---

### Task 1: Create the four-variant chunk×shard harness

**Files:**
- Create: `examples/spike_chunk_shard_ctm_move.py`
- Reference (do not modify): `examples/spike_chunked_ctm_move.py`, `src/tenax/algorithms/ctm_sharding.py:23` (`build_ctm_mesh`), `src/tenax/algorithms/_ctm_compiled_moves.py` (`_apply_projector_raw`)

- [ ] **Step 1: Write the harness**

```python
"""Gate: chunk x shard the REAL dense-CTM edge move.

Extends spike_chunked_ctm_move.py with a GSPMD mesh. Shards a's r2=n (a free D2
output leg = ctm_sharding's left-move surviving axis), orthogonal to the chunk
axis (boundary chi). Four variants, ONE per process (peak_bytes is cumulative):
  full       : replicated monolith            (peak ~ chi^2 D^6)
  chunked    : 1-device, chunked over i        (peak / K)
  sharded    : monolith, a sharded on n        (peak / N)
  chunkshard : chunked over i, a sharded on n   (peak / (N*K))

  CUDA_VISIBLE_DEVICES=0,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --extra cuda13 python examples/spike_chunk_shard_ctm_move.py \
      --D 10 --chi 64 --batch 16 --mesh-n 2 --variant chunkshard
"""
import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.sharding import NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from tenax.algorithms._ctm_compiled_moves import _apply_projector_raw  # noqa: E402
from tenax.algorithms.ctm_sharding import build_ctm_mesh  # noqa: E402


def _inputs(D, chi, seed):
    D2 = D * D
    k = jax.random.split(jax.random.PRNGKey(seed), 5)
    T4 = jax.random.normal(k[0], (chi, D2, chi)) / D2  # (t4_d=i, l2=j, t4_u=k)
    a = jax.random.normal(k[1], (D2, D2, D2, D2)) / D2  # (u2=l, d2=m, l2=j, r2=n)
    P1 = jax.random.normal(k[2], (chi * D2, chi)) / (chi * D2)
    P2 = jax.random.normal(k[3], (chi * D2, chi)) / (chi * D2)
    C1g = jax.random.normal(k[4], (chi * D2, chi)) / (chi * D2)
    C4g = jnp.zeros((chi * D2, chi))
    return T4, a, P1, P2, C1g, C4g


def full_move(T4, a, P1, P2, C1g, C4g, chi, D2):
    T4_a = jnp.einsum("ijk,lmjn->iklmn", T4, a)  # (chi,chi,D2,D2,D2)  PEAK
    T4_a = T4_a.transpose(0, 2, 1, 3, 4)
    T4g = T4_a.reshape(chi * D2, chi * D2, D2)
    _c1, _c4, T_new = _apply_projector_raw(P1, P2, C1g, C4g, T4g)
    return T_new  # (k_out, D2, k_out)


def chunked_move(T4, a, P1, P2, C1g, C4g, chi, D2, batch):
    P1r = P1.reshape(chi, D2, -1)  # (i, l, k_out)

    def per_i(args):
        T4_i, P1_i = args  # (D2[j], chi[k]) , (D2[l], k_out)
        T4a_i = jnp.einsum("jk,lmjn->klmn", T4_i, a)  # (k, l, m, n)=(chi,D2,D2,D2)
        T4g_i = T4a_i.transpose(1, 0, 2, 3).reshape(D2, chi * D2, D2)  # (l, fr, n)
        step = jnp.tensordot(P1_i.conj(), T4g_i, axes=([0], [0]))  # (k_out, fr, n)
        return jnp.tensordot(step, P2, axes=([1], [0]))  # (k_out, D2, k_out)

    contribs = lax.map(per_i, (T4, P1r), batch_size=batch)  # (chi, k_out, D2, k_out)
    return contribs.sum(0)


def _commit(mesh, tensors, shard_a):
    rep = NamedSharding(mesh, P())
    a_sh = NamedSharding(mesh, P(None, None, None, "d")) if shard_a else rep
    T4, a, P1, P2, C1g, C4g = tensors
    return (
        jax.device_put(T4, rep),
        jax.device_put(a, a_sh),
        jax.device_put(P1, rep),
        jax.device_put(P2, rep),
        jax.device_put(C1g, rep),
        jax.device_put(C4g, rep),
    )


def _peak_gb(devs):
    vals = []
    for d in devs:
        try:
            vals.append(d.memory_stats()["peak_bytes_in_use"] / 1e9)
        except Exception:  # noqa: BLE001
            pass
    return max(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, required=True)
    ap.add_argument("--chi", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mesh-n", type=int, default=1)
    ap.add_argument(
        "--variant",
        default="chunkshard",
        choices=["full", "chunked", "sharded", "chunkshard", "parity"],
    )
    ap.add_argument("--hlo", action="store_true", help="print HLO collective/temp summary, no run")
    args = ap.parse_args()

    D2 = args.D * args.D
    devs = jax.devices()[: args.mesh_n]
    mesh = build_ctm_mesh(devs)
    shard = args.variant in ("sharded", "chunkshard")
    if shard and D2 % args.mesh_n != 0:
        raise SystemExit(f"D2={D2} not divisible by mesh_n={args.mesh_n}")
    tens = _inputs(args.D, args.chi, 0)
    print(
        f"# chunk-shard  D={args.D} chi={args.chi} D2={D2} batch={args.batch} "
        f"({args.chi // args.batch} chunks) mesh_n={args.mesh_n} variant={args.variant}"
    )

    if args.variant == "parity":
        T4, a, P1, P2, C1g, C4g = _commit(mesh, tens, shard_a=False)
        full = jax.jit(lambda: full_move(T4, a, P1, P2, C1g, C4g, args.chi, D2))()
        T4s, as_, P1s, P2s, C1s, C4s = _commit(mesh, tens, shard_a=True)
        cs = jax.jit(
            lambda: chunked_move(T4s, as_, P1s, P2s, C1s, C4s, args.chi, D2, args.batch)
        )()
        full = jax.block_until_ready(full)
        cs = jax.block_until_ready(cs)
        d = float(jnp.max(jnp.abs(full - cs)))
        rel = d / (float(jnp.max(jnp.abs(full))) + 1e-30)
        print(f"PARITY chunkshard vs full: max|d|={d:.2e} rel={rel:.2e}")
        return

    T4, a, P1, P2, C1g, C4g = _commit(mesh, tens, shard_a=shard)
    if args.variant in ("chunked", "chunkshard"):
        fn = lambda: chunked_move(T4, a, P1, P2, C1g, C4g, args.chi, D2, args.batch)
    else:
        fn = lambda: full_move(T4, a, P1, P2, C1g, C4g, args.chi, D2)
    jfn = jax.jit(fn)

    if args.hlo:
        try:
            compiled = jfn.lower().compile()
            txt = compiled.as_text()
            ag = txt.count("all-gather")
            full_tok = f"[{D2}]"  # the sharded n=D2 axis, gathered would show full D2
            ag_fulln = sum(
                1 for ln in txt.splitlines() if "all-gather" in ln and full_tok in ln
            )
            temp = compiled.memory_analysis().temp_size_in_bytes / 1e9
            print(f"HLO all-gather={ag} (touching full n={D2}: {ag_fulln}) temp={temp:.3f} GB")
        except Exception as ex:  # noqa: BLE001
            print(f"HLO FAILED({type(ex).__name__}: {str(ex)[:90]})")
        return

    try:
        t0 = time.perf_counter()
        jax.block_until_ready(jfn())
        dt = time.perf_counter() - t0
        print(f"RUN variant={args.variant} peak={_peak_gb(devs):.2f} GB wall={dt:.3f}s")
    except Exception as ex:  # noqa: BLE001
        print(f"RUN variant={args.variant} FAILED({type(ex).__name__}: {str(ex)[:90]})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add examples/spike_chunk_shard_ctm_move.py
git commit -m "spike(#632): chunk x shard CTM-move four-variant gate harness

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: G4 — parity (correctness first)

**Files:** Run-only (`examples/spike_chunk_shard_ctm_move.py`).

- [ ] **Step 1: Run parity on the 2-GPU mesh**

Run:
```bash
CUDA_VISIBLE_DEVICES=0,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --extra cuda13 python examples/spike_chunk_shard_ctm_move.py \
    --D 6 --chi 16 --batch 8 --mesh-n 2 --variant parity
```
Expected: `PARITY chunkshard vs full: max|d|=…e-1X rel=…e-1X` with **rel ≤ 1e-14**.

- [ ] **Step 2: Decision gate**

If `rel > 1e-14` → STOP: the sharded+chunked move is not bit-faithful; debug `_commit`/sharding axis before measuring peaks. If `rel ≤ 1e-14` → G4 GO, proceed.

---

### Task 3: G1 (composition) + G2 (reach) — the (D,χ) scan

**Files:** Run-only. Each invocation is one variant/process (peak is cumulative).

- [ ] **Step 1: G1 composition at a fixed config that all four can run**

Run (four processes):
```bash
for V in full chunked sharded chunkshard; do
  CUDA_VISIBLE_DEVICES=0,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --extra cuda13 python examples/spike_chunk_shard_ctm_move.py \
      --D 8 --chi 48 --batch 12 --mesh-n 2 --variant $V
done
```
Expected: four `RUN … peak=… GB` lines. **G1 GO** if `chunkshard.peak ≈ chunked.peak / 2` (within ~1.5×) and `≈ full.peak / (2·4)` (N=2, K=chi/batch=4).

- [ ] **Step 2: G2 reach — push D until variants fail**

Run (scan D for the two single-levers vs chunkshard):
```bash
for D in 10 12 14; do for V in chunked sharded chunkshard; do
  CUDA_VISIBLE_DEVICES=0,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --extra cuda13 python examples/spike_chunk_shard_ctm_move.py \
      --D $D --chi 64 --batch 16 --mesh-n 2 --variant $V
done; done
```
Expected: a D where `chunked` and `sharded` print `FAILED(...)` (OOM / autotuner) but `chunkshard` prints `RUN … peak=…`. **G2 GO** if such a D exists.

- [ ] **Step 3: Record the raw lines** into a scratch file `/tmp/chunkshard_scan.txt` (paste stdout) for the findings doc.

---

### Task 4: G3 — layout + autotuner-dodge (the deep question)

**Files:** Run-only (`--hlo`).

- [ ] **Step 1: Confirm the sharded axis stays sharded (no full-n gather)**

Run:
```bash
CUDA_VISIBLE_DEVICES=0,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  uv run --extra cuda13 python examples/spike_chunk_shard_ctm_move.py \
    --D 8 --chi 48 --batch 12 --mesh-n 2 --variant chunkshard --hlo
```
Expected: `HLO all-gather=… (touching full n=64: 0) temp=… GB`. **G3a GO** if the full-n gather count is **0** (the `n=D²` axis stays sharded inside `lax.map`).

- [ ] **Step 2: Autotuner-dodge — `sharded` vs `chunkshard` at a large config**

Run both at a config where the monolithic gemm is stressed:
```bash
for V in sharded chunkshard; do
  CUDA_VISIBLE_DEVICES=0,2 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    uv run --extra cuda13 python examples/spike_chunk_shard_ctm_move.py \
      --D 12 --chi 128 --batch 16 --mesh-n 2 --variant $V --hlo
done
```
Expected (strong-GO signal): `sharded` prints `HLO FAILED(... autotun...)` or a large `temp`, while `chunkshard` compiles with small per-device `temp`. Record whichever happens.

---

### Task 5: Findings handoff + GO/NO-GO

**Files:**
- Create: `docs/superpowers/handoffs/2026-06-30-chunk-shard-ctm-move-findings.md`

- [ ] **Step 1: Write the findings doc**

Fill this template with the measured numbers from Tasks 2–4:

```markdown
# Chunk × Shard dense-CTM move — gate findings

**Date:** 2026-06-30
**Hardware:** 2× A100-80GB (devices 0,2), f64.
**Spec:** docs/superpowers/specs/2026-06-30-chunk-shard-ctm-move-gate-design.md
**Verdict: GATE = <GO | NO-GO>**

## G4 parity
chunkshard vs full: rel = <…>  (pass ⟺ ≤1e-14)

## G1 composition (D=8 χ=48 batch=12, N=2 K=4)
| variant | per-device peak (GB) |
|---|---|
| full | <…> |
| chunked | <…> |
| sharded | <…> |
| chunkshard | <…> |
chunkshard ≈ chunked/N ?  <yes/no, factor>  ;  ≈ full/(N·K) ? <…>

## G2 reach (χ=64 batch=16, N=2)
| D | chunked | sharded | chunkshard |
|---|---|---|---|
| 10 | <peak/FAIL> | <…> | <…> |
| 12 | <…> | <…> | <…> |
| 14 | <…> | <…> | <…> |
chunkshard runs where both single levers fail ?  <yes/no, at D=…>

## G3 layout + autotuner
- full-n all-gathers in chunkshard HLO: <0 ?>
- sharded vs chunkshard at D=12 χ=128: <sharded autotuner-FAIL while chunkshard runs ? = strong GO>

## Verdict
GO ⟺ G1 composes (÷N·K within ~1.5×) AND G2 reach past both single levers AND G4 parity exact.
Strong GO additionally if G3 autotuner-dodge. <state which, and the build recommendation or the
NO-GO close-out>.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/handoffs/2026-06-30-chunk-shard-ctm-move-findings.md
git commit -m "docs(#632): chunk x shard CTM-move gate findings (<GO|NO-GO>)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:** G1 (Task 3.1), G2 (Task 3.2), G3a + G3b autotuner-dodge (Task 4), G4 parity (Task 2), four-variant harness with `a` sharded on `r2=n` (Task 1), findings + GO/NO-GO (Task 5), no `src/` change (only `examples/` + `docs/`). All spec measurements map to a task.

**Placeholder scan:** the findings doc has `<…>` fill-ins by design (measured values unknown until run) — these are data slots, not unspecified logic. All code and commands are complete and runnable.

**Type/name consistency:** `full_move`/`chunked_move`/`_commit`/`_peak_gb` signatures are used consistently across tasks; `build_ctm_mesh`, `_apply_projector_raw`, `P(None,None,None,"d")` match `ctm_sharding.py` and the parent spike. Chunk axis = `i` (boundary χ), shard axis = `a` axis 3 (`r2=n`) — orthogonal, consistent throughout.

## Build (only if GO — out of scope here)

Add an opt-in `n_chunks`/`batch` knob to the four `_compiled_move_*` edge/corner contractions, composed with the `ctm_sharding` mesh; gradient/AD through the chunked-sharded move; D=10–12 / large-χ multi-GPU `optimize_gs_ad` benchmark vs shard-only.
