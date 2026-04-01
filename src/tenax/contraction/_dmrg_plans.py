"""Hard-coded DMRG contraction plans.

Replaces ``opt_einsum.contract_path`` + ``build_blas_plan`` for the two
fixed DMRG einsum patterns with direct TTGT parameter computation.
This eliminates thousands of calls to ``opt_einsum.contract_path``
(~3.6 s per DMRG run at large bond dimension).

The two patterns:

- **2-site**: ``"abc,apqd,bpse,eqtf,dfg->cstg"``
- **1-site**: ``"abc,apd,bpxe,def->cxf"``

For any other subscript string the dispatcher :func:`get_dmrg_plan` falls
back to the generic ``get_cached_blas_plan``.
"""

from __future__ import annotations

import functools

from tenax.contraction._blas_plan import BlasExecPlan, GemmStep

# ------------------------------------------------------------------ #
# Two-site plan                                                        #
# ------------------------------------------------------------------ #

_TWO_SITE_SUBSCRIPTS = "abc,apqd,bpse,eqtf,dfg->cstg"


def build_two_site_plan(
    shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
) -> BlasExecPlan:
    """Build a BLAS plan for the 2-site DMRG matvec contraction.

    Subscripts: ``"abc,apqd,bpse,eqtf,dfg->cstg"``

    Fixed contraction order::

        Step 0: L(abc) @ theta(apqd) -> I0(bcpqd)   [contract a]
        Step 1: I0     @ W1(bpse)    -> I1(cqdse)    [contract b,p]
        Step 2: I1     @ W2(eqtf)    -> I2(cdstf)    [contract e,q]
        Step 3: I2     @ R(dfg)      -> out(cstg)    [contract d,f]

    Buffer layout: 0=L, 1=theta, 2=W1, 3=W2, 4=R, 5=I0, 6=I1, 7=I2, 8=final.
    """
    # Unpack dimension sizes from shapes
    sa, sb, sc = shapes[0]  # L: (a, b, c)
    _a, sp, sq, sd = shapes[1]  # theta: (a, p, q, d)
    _b, _p, ss, se = shapes[2]  # W1: (b, p, s, e)
    _e, _q, st, sf = shapes[3]  # W2: (e, q, t, f)
    _d, _f, sg = shapes[4]  # R: (d, f, g)

    # ---- Step 0: L(abc) @ theta(apqd) -> I0(bcpqd) [contract a] ----
    # left = L(abc), needs perm to (b, c, a) => (1, 2, 0)
    # right = theta(apqd), already (a, p, q, d) => identity
    step0 = GemmStep(
        left_idx=0,
        right_idx=1,
        out_idx=5,
        trans_a=False,
        trans_b=False,
        m=sb * sc,
        n=sp * sq * sd,
        k=sa,
        left_perm=(1, 2, 0),
        right_perm=(),
        out_shape=(sb, sc, sp, sq, sd),
    )

    # ---- Step 1: I0(bcpqd) @ W1(bpse) -> I1(cqdse) [contract b,p] ----
    # left = I0(bcpqd), needs perm to (c, q, d, b, p) => (1, 3, 4, 0, 2)
    # right = W1(bpse), already (b, p, s, e) => identity
    step1 = GemmStep(
        left_idx=5,
        right_idx=2,
        out_idx=6,
        trans_a=False,
        trans_b=False,
        m=sc * sq * sd,
        n=ss * se,
        k=sb * sp,
        left_perm=(1, 3, 4, 0, 2),
        right_perm=(),
        out_shape=(sc, sq, sd, ss, se),
    )

    # ---- Step 2: I1(cqdse) @ W2(eqtf) -> I2(cdstf) [contract e,q] ----
    # left = I1(cqdse), needs perm to (c, d, s, q, e) => (0, 2, 3, 1, 4)
    # right = W2(eqtf), needs perm to (q, e, t, f) => (1, 0, 2, 3)
    step2 = GemmStep(
        left_idx=6,
        right_idx=3,
        out_idx=7,
        trans_a=False,
        trans_b=False,
        m=sc * sd * ss,
        n=st * sf,
        k=sq * se,
        left_perm=(0, 2, 3, 1, 4),
        right_perm=(1, 0, 2, 3),
        out_shape=(sc, sd, ss, st, sf),
    )

    # ---- Step 3: I2(cdstf) @ R(dfg) -> out(cstg) [contract d,f] ----
    # left = I2(cdstf), needs perm to (c, s, t, d, f) => (0, 2, 3, 1, 4)
    # right = R(dfg), already (d, f, g) => identity
    step3 = GemmStep(
        left_idx=7,
        right_idx=4,
        out_idx=8,
        trans_a=False,
        trans_b=False,
        m=sc * ss * st,
        n=sg,
        k=sd * sf,
        left_perm=(0, 2, 3, 1, 4),
        right_perm=(),
        out_shape=(sc, ss, st, sg),
    )

    # Result subscript is "cstg" which matches the target => no output perm.
    return BlasExecPlan(
        steps=(step0, step1, step2, step3),
        n_buffers=9,
        n_inputs=5,
        output_perm=(),
    )


# ------------------------------------------------------------------ #
# One-site plan                                                        #
# ------------------------------------------------------------------ #

_ONE_SITE_SUBSCRIPTS = "abc,apd,bpxe,def->cxf"


def build_one_site_plan(
    shapes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...],
) -> BlasExecPlan:
    """Build a BLAS plan for the 1-site DMRG matvec contraction.

    Subscripts: ``"abc,apd,bpxe,def->cxf"``

    Fixed contraction order::

        Step 0: L(abc) @ site(apd) -> I0(bcpd)   [contract a]
        Step 1: I0     @ W(bpxe)   -> I1(cdxe)    [contract b,p]
        Step 2: I1     @ R(def)    -> out(cxf)     [contract d,e]

    Buffer layout: 0=L, 1=site, 2=W, 3=R, 4=I0, 5=I1, 6=final.
    """
    # Unpack dimension sizes from shapes
    sa, sb, sc = shapes[0]  # L: (a, b, c)
    _a, sp, sd = shapes[1]  # site: (a, p, d)
    _b, _p, sx, se = shapes[2]  # W: (b, p, x, e)
    _d, _e, sf = shapes[3]  # R: (d, e, f)

    # ---- Step 0: L(abc) @ site(apd) -> I0(bcpd) [contract a] ----
    # left = L(abc), needs perm to (b, c, a) => (1, 2, 0)
    # right = site(apd), already (a, p, d) => identity
    step0 = GemmStep(
        left_idx=0,
        right_idx=1,
        out_idx=4,
        trans_a=False,
        trans_b=False,
        m=sb * sc,
        n=sp * sd,
        k=sa,
        left_perm=(1, 2, 0),
        right_perm=(),
        out_shape=(sb, sc, sp, sd),
    )

    # ---- Step 1: I0(bcpd) @ W(bpxe) -> I1(cdxe) [contract b,p] ----
    # left = I0(bcpd), needs perm to (c, d, b, p) => (1, 3, 0, 2)
    # right = W(bpxe), already (b, p, x, e) => identity
    step1 = GemmStep(
        left_idx=4,
        right_idx=2,
        out_idx=5,
        trans_a=False,
        trans_b=False,
        m=sc * sd,
        n=sx * se,
        k=sb * sp,
        left_perm=(1, 3, 0, 2),
        right_perm=(),
        out_shape=(sc, sd, sx, se),
    )

    # ---- Step 2: I1(cdxe) @ R(def) -> out(cxf) [contract d,e] ----
    # left = I1(cdxe), needs perm to (c, x, d, e) => (0, 2, 1, 3)
    # right = R(def), already (d, e, f) => identity
    step2 = GemmStep(
        left_idx=5,
        right_idx=3,
        out_idx=6,
        trans_a=False,
        trans_b=False,
        m=sc * sx,
        n=sf,
        k=sd * se,
        left_perm=(0, 2, 1, 3),
        right_perm=(),
        out_shape=(sc, sx, sf),
    )

    # Result subscript is "cxf" which matches the target => no output perm.
    return BlasExecPlan(
        steps=(step0, step1, step2),
        n_buffers=7,
        n_inputs=4,
        output_perm=(),
    )


# ------------------------------------------------------------------ #
# Cached dispatcher                                                    #
# ------------------------------------------------------------------ #


@functools.lru_cache(maxsize=8192)
def get_dmrg_plan(
    subscripts: str,
    shapes: tuple[tuple[int, ...], ...],
) -> BlasExecPlan:
    """Return a cached BLAS plan, using hard-coded plans for DMRG patterns.

    For the two known DMRG subscripts the plan is built directly without
    calling ``opt_einsum.contract_path``.  Any other subscript falls back
    to :func:`tenax.contraction._blas_plan.get_cached_blas_plan`.

    Parameters
    ----------
    subscripts : str
        Einsum subscripts string.
    shapes : tuple of tuple[int, ...]
        Shapes of input tensors (must be a tuple for hashability).
    """
    if subscripts == _TWO_SITE_SUBSCRIPTS:
        return build_two_site_plan(shapes)
    if subscripts == _ONE_SITE_SUBSCRIPTS:
        return build_one_site_plan(shapes)
    # Fallback to generic plan builder
    from tenax.contraction._blas_plan import get_cached_blas_plan

    return get_cached_blas_plan(subscripts, shapes)
