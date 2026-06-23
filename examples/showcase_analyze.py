"""Robust post-hoc analysis of the Heisenberg scaling-showcase results.

The per-cell worker stores the raw per-optimizer-step wall times. Here we derive
a robust steady-state per-step cost and assemble the scaling tables used in the
findings doc.

Per-step cost = ``median(step_times[1:])``:
  - ``[1:]`` drops step 0 (the initial XLA compile).
  - the median is robust to (a) the 1-2 extra recompile spikes XLA emits while
    re-autotuning at chi>=48, and (b) single async-dispatch fast outliers (JAX
    is async; an individual step occasionally reports dispatch rather than
    compute time).

A few cells are REPRODUCIBLY non-monotonic — per-step well below a smaller-chi
cell in the same (D, n_devices) row (e.g. D3 chi64 = 3.7s vs chi48 = 54s, two
independent runs agree). This is NOT noise: on the reduced-corner CTM path the
per-step cost is non-monotonic in NOMINAL chi because XLA's autotuner selects
different-speed kernels and/or the effective SVD rank at large chi falls below
nominal. We flag these ``*`` (so the D^6 D-scaling fit isn't skewed) but keep
them — the non-monotonicity is itself a finding. Run after the sweep:
``uv run python examples/showcase_analyze.py``.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import statistics
import sys
from pathlib import Path

RESULTS = Path("examples/showcase_results")
REF_E = -0.669437  # Sandvik QMC, square-lattice spin-1/2 Heisenberg AFM


def _load_showcase_module():
    """Path-load the main showcase module to reuse make_plots / table / CSV."""
    p = Path(__file__).resolve().parent / "showcase_heisenberg_scaling.py"
    spec = importlib.util.spec_from_file_location("showcase_heisenberg_scaling", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def per_step_ms(d):
    st = d.get("step_times")
    if not st:
        return None
    warm = st[1:] if len(st) > 1 else st
    return 1000.0 * statistics.median(warm)


def load_cells():
    cells = []
    for f in sorted(glob.glob(str(RESULTS / "*.json"))):
        with open(f) as fh:
            cells.append(json.load(fh))
    return cells


def _row(cells, D, n, anchor=False):
    """(chi-sorted) cells for one (D, n_devices, kind) row."""
    return sorted(
        [c for c in cells if c["D"] == D and c["n_devices"] == n
         and bool(c.get("is_anchor")) == anchor],
        key=lambda c: c["chi"],
    )


def flag_artifacts(row):
    """Mark cells whose per-step is reproducibly below a smaller-chi cell in the
    same row (non-monotonic in nominal chi: XLA kernel selection / effective
    rank). Kept in the data, excluded only from the D-scaling fit."""
    out, prev = [], None
    for c in row:
        ms = per_step_ms(c)
        bad = ms is not None and prev is not None and ms < 0.6 * prev
        out.append((c, ms, bad))
        if ms is not None and not bad:
            prev = ms
    return out


def main():
    cells = load_cells()
    print("# Heisenberg scaling showcase — robust per-step (median of warm steps)\n")
    print("'*' = reproducibly non-monotonic in chi (XLA kernel / effective rank);"
          " 'ERR' = timeout/error\n")

    for n in (1, 4):
        print(f"\n## {n}-GPU metrics")
        print("| D | chi | per-step (s) | peak GB | note |")
        print("|---|-----|--------------|---------|------|")
        for D in (2, 3, 4):
            for c, ms, bad in flag_artifacts(_row(cells, D, n)):
                if ms is None:
                    note = f"ERR {str(c.get('error'))[:22]}"
                    print(f"| {D} | {c['chi']} | — | — | {note} |")
                else:
                    star = " *" if bad else ""
                    print(f"| {D} | {c['chi']} | {ms / 1000:.1f}{star} | "
                          f"{c.get('peak_gb') or 0:.3f} | {'async-artifact' if bad else ''} |")

    # D-scaling at the smallest shared chi (16): the D^6 jump.
    print("\n## D-scaling at chi=16 (1-GPU), per-step")
    base = {}
    for D in (2, 3, 4):
        row = _row(cells, D, 1)
        c16 = next((c for c in row if c["chi"] == 16), None)
        if c16:
            base[D] = per_step_ms(c16) / 1000
    for D in (2, 3, 4):
        if D in base:
            ratio = f" ({base[D] / base[2]:.1f}x D2)" if 2 in base else ""
            print(f"  D{D} chi16: {base[D]:.1f}s{ratio}")

    # 1-GPU vs 4-GPU at matched (D, chi).
    print("\n## 1-GPU vs 4-GPU (matched cells), per-step")
    print("| D | chi | 1-GPU s | 4-GPU s | speedup |")
    print("|---|-----|---------|---------|---------|")
    for D in (2, 3, 4):
        r1 = {c["chi"]: per_step_ms(c) for c in _row(cells, D, 1)}
        r4 = {c["chi"]: per_step_ms(c) for c in _row(cells, D, 4)}
        for chi in sorted(set(r1) & set(r4)):
            a, b = r1[chi], r4[chi]
            if a and b:
                print(f"| {D} | {chi} | {a / 1000:.1f} | {b / 1000:.1f} | {a / b:.2f}x |")

    # Anchors (best-effort converged energy vs QMC).
    print("\n## Anchors (best-effort converged energy)")
    print("| D | chi | E/site | dE vs QMC | converged |")
    print("|---|-----|--------|-----------|-----------|")
    for c in [c for c in cells if c.get("is_anchor")]:
        e = c.get("E_site")
        if e is None:
            print(f"| {c['D']} | {c['chi']} | — | — | ERR {str(c.get('error'))[:18]} |")
        else:
            print(f"| {c['D']} | {c['chi']} | {e:.6f} | {e - REF_E:+.4f} | "
                  f"{c.get('converged')} |")

    if "--write-outputs" in sys.argv:
        _write_outputs(cells)


def _write_outputs(cells):
    """Regenerate the table / CSV / plots with the ROBUST median per-step
    (reproducibly non-monotonic cells set to None so they neither skew the plots
    nor the fit), reusing the tested make_plots / reporting code."""
    mod = _load_showcase_module()
    disp = []
    for n in (1, 4):
        for D in (2, 3, 4):
            for c, ms, bad in flag_artifacts(_row(cells, D, n)):
                d = dict(c)
                d["ms_per_step"] = None if (ms is None or bad) else ms
                disp.append(d)
    disp += [dict(c) for c in cells if c.get("is_anchor")]
    (RESULTS / "scaling_table.md").write_text(mod.results_to_markdown(disp))
    import csv
    rows = mod.results_to_csv_rows(disp)
    if rows:
        with open(RESULTS / "scaling_results.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    written = mod.make_plots(disp, str(RESULTS))
    print("\n[write] scaling_table.md + scaling_results.csv + "
          + ", ".join(Path(p).name for p in written))


if __name__ == "__main__":
    main()
