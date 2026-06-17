# examples/census_u1sz_block_shapes_566.py
"""S1 of the #566 U(1)-Sz spike: block-shape fragmentation census.

Grouping collapse ceiling = n_blocks / n_distinct_shapes per tensor. Even-D
FermionParity ~ 16 (all blocks one shape); general U(1) is the open question.
Pure static metadata (block keys/shapes) over real CTM env tensors. No grad,
short CTM, cheap. Run on CPU.

    JAX_PLATFORMS=cpu uv run python examples/census_u1sz_block_shapes_566.py \
        --D 2 3 --chi-factor 4 --json census_u1sz.json
"""
import argparse
import json

import jax

from tenax.algorithms._ctm_tensor import ctm_tensor
from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair


def census_one(D: int, chi: int) -> dict:
    A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=chi, max_iter=4, conv_tol=1e-4)
    rows = []
    targets = [("site", A)] + [(n, getattr(env, n)) for n in env._fields]
    for name, t in targets:
        shapes = list(getattr(t, "_block_shapes", ()))
        n_blocks = len(shapes)
        n_shapes = len(set(shapes))
        collapse = (n_blocks / n_shapes) if n_shapes else 0.0
        rows.append({
            "tensor": name, "n_blocks": n_blocks,
            "n_distinct_shapes": n_shapes, "collapse_ceiling": round(collapse, 3),
        })
    return {"D": D, "chi": chi, "rows": rows}


def predict_sectors_under_keep(block_keys, chi_axes, keep) -> int:
    """Count blocks that survive restricting the chi-axis charges to ``keep``.

    Static Gate-A proxy for the post-candidate sector count: a block survives
    iff every chi-axis charge component lies in ``keep`` (the sector-dropping
    candidate, e.g. keep={-1,0,1} drops |Sz|=2 chi sectors).
    """
    keep = set(keep)
    return sum(
        all(k[a] in keep for a in chi_axes)
        for k in block_keys
    )


def _chi_axes_for(tensor) -> tuple:
    """Indices of the truncated CTM (chi) bond legs.

    DISCRIMINATOR — chosen by introspecting real D=3 env tensors (see #610):
    the candidate drops |Sz|>1 on the *chi* environment bonds, NOT on the
    physical D^2 leg. Observed leg schema (fixed by ``_ctm_tensor_init.py``):

        corners  C1..C4 : 2 legs, labels 'c{i}_{dir}'  (both chi bonds)
        edges    T1..T4 : 3 legs, labels 't{i}_{dir}', '{dir}2', 't{i}_{dir}'
                          -> axes 0 & 2 are chi bonds; axis 1 is the D^2 leg
                          labeled 'u2'/'r2'/'d2'/'l2' (dim == D^2).

    The naive label-prefix test ('chi'...) returns () here (labels are
    'c1_d'/'t1_l', not 'chi*') -> bogus 1.0x reduction. A dim-based test
    (dim == chi) is ALSO unsafe: at chi==D^2 (e.g. chi=9, D=3) the D^2 leg
    is indistinguishable by dimension, and at chi<D^2 it picks the wrong leg.

    Robust, chi-independent discriminator: a chi bond's label contains an
    underscore ('c1_d', 't1_r'), while the D^2 leg label never does
    ('u2','r2','d2','l2'). We keep an explicit 'chi'-prefix branch first as a
    forward-compatible fallback for any path that does emit chi-labeled bonds.
    """
    labels = [str(ix.label) for ix in tensor.indices]
    pref = tuple(i for i, lab in enumerate(labels) if lab.lower().startswith("chi"))
    if pref:
        return pref
    # CTM env legs: chi bonds carry an underscore ('c#_dir'/'t#_dir');
    # the D^2 physical leg ('u2'/'r2'/'d2'/'l2') does not.
    return tuple(i for i, lab in enumerate(labels) if "_" in lab)


def candidate_report(D: int, chi: int, keep) -> dict:
    """For each env tensor, baseline n_blocks vs predicted-kept under ``keep``."""
    A, _B = heisenberg_u1sz_init_pair(D=D, key=jax.random.PRNGKey(0))
    env, _ = ctm_tensor(A, chi=chi, max_iter=4, conv_tol=1e-4)
    rows = []
    for name in env._fields:
        t = getattr(env, name)
        keys = list(getattr(t, "_block_keys", ()))
        chi_axes = _chi_axes_for(t)
        n0 = len(keys)
        n1 = predict_sectors_under_keep(keys, chi_axes, keep) if chi_axes else n0
        rows.append({
            "tensor": name, "n_blocks": n0, "kept": n1,
            "reduction": round(n0 / n1, 3) if n1 else float("inf"),
        })
    return {"D": D, "chi": chi, "keep": sorted(keep), "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--chi-factor", type=int, default=4)
    ap.add_argument("--candidate-keep", type=int, nargs="+", default=None,
                    help="Sector-keep set on chi bonds, e.g. -1 0 1. When given, "
                         "print the static Gate-A candidate report instead of "
                         "the fragmentation census.")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

    if args.candidate_keep is not None:
        keep = set(args.candidate_keep)
        out = []
        for D in args.D:
            chi = args.chi_factor * D
            res = candidate_report(D, chi, keep)
            out.append(res)
            print(f"\n=== U(1)-Sz candidate report D={D} chi={chi} "
                  f"keep={sorted(keep)} ===")
            print(f"{'tensor':>6} {'n_blocks':>9} {'kept':>6} {'reduction':>10}")
            for r in res["rows"]:
                print(f"{r['tensor']:>6} {r['n_blocks']:>9} "
                      f"{r['kept']:>6} {r['reduction']:>10}")
            reductions = sorted(r["reduction"] for r in res["rows"])
            median = reductions[len(reductions) // 2]
            res["median_reduction"] = median
            print(f"  median env-tensor reduction = {median:.3f}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\nwrote {args.json}")
        return

    out = []
    for D in args.D:
        res = census_one(D, args.chi_factor * D)
        out.append(res)
        print(f"\n=== U(1)-Sz census D={D} chi={args.chi_factor * D} ===")
        print(f"{'tensor':>6} {'n_blocks':>9} {'n_shapes':>9} {'collapse':>9}")
        for r in res["rows"]:
            print(f"{r['tensor']:>6} {r['n_blocks']:>9} "
                  f"{r['n_distinct_shapes']:>9} {r['collapse_ceiling']:>9}")
        ceilings = [r["collapse_ceiling"] for r in res["rows"]]
        print(f"  median collapse ceiling = {sorted(ceilings)[len(ceilings)//2]:.2f}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
