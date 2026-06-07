# #570 finding — the CTM-AD compile wall is the dense **SVD projector VJP**, and it is `projector_method`-invariant

**Date:** 2026-06-07 (A100) · **Follows:** [`2026-06-07-cutensornet-pb4-finding-nogo.md`](2026-06-07-cutensornet-pb4-finding-nogo.md) (#200 NO-GO) · **Issue:** #570

## TL;DR

P-B4 (#200) localized the minutes-long fermionic CTM-AD compile to the jitted
fixed-point backward `_jit_fused_fixed_point_bwd` and proved it **backend-invariant**
(perblock → stacked → cuTensorNet does not move it). The open #570 question was *which
sub-op inside that backward dominates the compile*. Answer, with two parts:

1. **It is the decomposition VJP — specifically a dense SVD.** The dominant repeated
   compile unit (`apply_Jt`, one gauge-fixed sweep VJP) carries **24 dense `svd`
   primitives** (D=2/χ=12 trace). Its cold XLA compile grows **super-linearly in χ**
   (~χ^1.3): a *single* sweep-VJP unit goes **50.7 s → 522.9 s** as χ goes 6 → 36 (10.3×).
   This is the χ-driver behind the ~549 s full-backward wall.

2. **But it is `projector_method`-invariant — the production path is SVD-only.** The
   default `recipe="2x2"` plaquette projector (`_ctm_tensor_projector_2x2.py`) contains
   **no** `qr`/`eigh`/`projector_method` code — it is hardwired to `jnp.linalg.svd`.
   `svd`/`qr`/`eigh` all trace to a **byte-identical** backward graph (TOTAL=34927; 24
   `svd`, 0 `qr`, 0 `eigh` in every case). `projector_method` only affects the alternate
   1×1 recipe (`_compute_projector_tensor`), which the production fermionic path does not use.

**Consequence:** #570's lever-1 ("swap SVD → QR projector") is a **no-op as a config
flip** on the production path. It requires real implementation, or pivot to lever-2
(**truncated backprop**), which is projector-agnostic.

## Method

`examples/profile_570_sweepvjp_compile.py` (new) isolates `apply_Jt` — the VJP of one
gauge-fixed CTM sweep w.r.t. the environment, documented as the dominant *repeated*
compile unit of `_jit_fused_fixed_point_bwd` — `jax.jit`s it, and times the **cold
`lower().compile()`** (no execution; pure compile attribution), per `projector_method`,
swept over χ. `--full` also compiles the params-sweep VJP. Cold compiles use a fresh
persistent-cache dir + `clear_caches()` (the #584/#585 lesson). This avoids the ~1000 s
full `value_and_grad` compile while measuring the actual repeated unit.

Cross-checked against `examples/probe_backward_jaxpr_566.py` (trace-only op histogram):
its `decomp` bucket counts only the literal `svd`/`eigh` *primitives* (flat at 48,
χ-blind) and **cannot** see the VJP expansion — which is why compile-time attribution was
needed.

## Results

### (1) χ-scaling of the SVD sweep-VJP compile (A100, D=2 fermionic, 16 blocks, x64)

| χ | env_compile (s) | env HLO instrs | total env+par (s) | ×(vs χ=6) |
|---:|---:|---:|---:|---:|
| 6  | 23.3  | 52,842  | 50.7  | 1.00× |
| 12 | 49.3  | 111,934 | 104.1 | 2.05× |
| 18 | 86.9  | 188,976 | 178.5 | 3.52× |
| 24 | 146.4 | 307,580 | 296.8 | 5.86× |
| 36 | 259.5 | 302,517 | 522.9 | 10.32× |

Compile time tracks HLO instruction count almost exactly; both grow ~linearly-to-super-
linearly in χ over 6→24 (instr 5.8×, compile 5.9× for a 4× χ). Past χ=24 the **instr
count saturates** (302k) because at D=2 the corner truncation `k = min(χ, available)`
caps out at the limited bond dim — so χ≥24 at D=2 is the saturated regime; the genuine
χ-scaling is the 6→24 window. (A D=4 sweep would push the saturation point out; not
needed to establish the fingerprint.) Raw: `570_results/profile_570_sweepvjp_a100.json`.

### (2) `projector_method`-invariance of the backward graph (trace-only, D=2 χ=12)

| method | apply_Jt TOTAL ops | `svd` | `qr` | `eigh` | `dot_general` |
|---|---:|---:|---:|---:|---:|
| svd  | 34927 | 24 | 0 | 0 | 1072 |
| qr   | 34927 | 24 | 0 | 0 | 1072 |
| eigh | 34927 | 24 | 0 | 0 | 1072 |

Byte-identical. Confirmed at the compile level too: svd and qr gave identical HLO instr
counts (52,842 / 111,934 at χ=6 / 12) in the smoke run.

## Why (mechanistic)

- The production fermionic CTM-AD loss (`make_ctm_energy_fn`, `iPEPSConfig` defaults,
  `gs_implicit_ad=True`, `adjoint_method="fixed_point"`) runs the symmetric tensor sweep
  `_ctm_tensor_sweep_multisite` with the default `recipe="2x2"`, whose projector is
  `_compute_2x2_projector` / `_compute_plaquette_projector_pair`. That file
  (`_ctm_tensor_projector_2x2.py`) is **SVD-only** (`_gauge_fixed_svd` → `jnp.linalg.svd`);
  it never reads `projector_method`.
- `projector_method` (`svd`/`qr`/`eigh`) only branches inside the **1×1** recipe's
  `_compute_projector_tensor` (`_ctm_projector.py`), and even there the **AD-traced**
  path deliberately densifies to an SVD VJP (the "Task 2.2" design decision, lines
  ~946–962: backward keeps a dense fallback rather than a block-sparse AD-traced SVD).
- So under AD, the decomposition the backward differentiates is **always a dense SVD on
  the χ-sized corner matrix**, per projector. Its VJP (the SVD-gradient F-matrix dense
  algebra) is what grows with χ and dominates compile.

## Implication for #570 — the levers, re-scoped

1. **Lever-1 "QR projector instead of SVD" is NOT a config flip.** The default 2×2 path
   has no QR. Realizing it means *implementing* a QR-based 2×2 plaquette projector with a
   stable AD backward (QR backward is rank-deficiency-sensitive — note `_ctm_projector.py`
   already routes its QR *tracer* fallback through a `regularized_svd`, i.e. back to SVD,
   precisely to dodge QR-backward instability). High effort, unproven energy/variational
   parity. **Do not start here.**

2. **Lever-2 "truncated backprop" is the recommended first move.** It differentiates
   through a *truncated* number of CTM sweeps instead of the full fixed point, shrinking
   the backward jaxpr **independent of the decomposition** — so it cuts the SVD-VJP count
   directly and is orthogonal to projector choice. Lower risk; the main validation is
   gradient/energy parity vs the exact fixed-point adjoint.

3. **Lever-3 (alternative): block-sparse AD-traced SVD VJP.** Replace the dense-corner SVD
   fallback with a per-sector truncated SVD VJP (smaller decomps). The design note flags
   the static-shape-per-sector (`k_q` must be a Python int) complexity. Medium effort,
   directly attacks the measured cost.

## Next experiment (whichever lever)

`profile_570_sweepvjp_compile.py` is the measurement rig. To validate a lever, A/B its
sweep-VJP `total_compile_s` and `env_hlo_instr` vs the SVD baseline above:
- **truncated backprop:** compile the N-sweep unrolled backward for N ∈ {1,2,4} and show
  compile scales with N while energy/grad track the full adjoint.
- **QR-in-2×2 / block-sparse SVD:** once implemented, expect `total_compile_s` to drop and
  flatten in χ relative to the SVD curve above.

## What would change the conclusion

Only if the production path stopped using the SVD-only 2×2 projector (e.g. a config or
code change routing the fermionic AD path through a method-aware projector). As of this
commit it does not — the SVD-only finding is at the code level, not just empirical.

## Artifacts

- `examples/profile_570_sweepvjp_compile.py` — compile-attribution rig (new)
- `570_results/profile_570_sweepvjp_a100.json` — raw χ-sweep data
- `examples/probe_backward_jaxpr_566.py` — trace-only op histogram (existing; cross-check)
