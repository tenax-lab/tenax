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

from tenax.algorithms.ipeps import heisenberg_u1sz_init_pair
from tenax.algorithms._ctm_tensor import ctm_tensor


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--chi-factor", type=int, default=4)
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()

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
