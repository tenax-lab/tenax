"""#785 re-measurement harness: does any diagnostic predict root-implicit
gradient error inside the clamped set?

#785 asks which quantity orders ``grad_err`` across states where the rank clamp
(``_rank_capped_spectrum``) has fired.  Two candidate laws have already been
withdrawn there after looking monotone over a handful of states.  This harness
exists because a third refutation is worth less than fixing the measurement the
refutations rest on.

Three problems with that measurement, each addressed below.

**1. The issue is not reproducible.**  Its probe scripts were never committed,
and the only squash knob in the test suite -- ``_site_tensor(D, d, seed, eps)``
-- is demonstrably not the state the issue used: at ``eps=1e-3`` it gives
``|grad| = 3.3e-09`` where #785's own reproduction block records
``analytic g.v = 1.196767600118e+00``, nine orders apart, and the ranks
disagree too.  ``_site_tensor`` scales the *entire* random part, so as
``eps -> 0`` the state limits to a product state and E and dE/dA collapse
together.  ``squashed()`` below is the reconstruction that matches what the
issue describes (near-rank-deficiency in the *virtual bond*).

**2. The FD reference was validated on the wrong axis.**  The control in #785
varies sweep depth (10/20/40/80/160) and finds a plateau, which rules out
*truncation* of the explicit map.  It never varies ``h``, which governs the
*floating-point* floor -- a central difference of an O(1) energy has an
absolute noise floor around ``eps*|E|/h ~ 1e-11`` at ``h=1e-5``.  Different
confounds.  ``fd_with_h_scan`` scans ``h`` and picks it by self-consistency
between consecutive refinements, **never** by agreement with the analytic value
under test; choosing the ``h`` that best matches the quantity being validated
biases ``grad_err`` downward and measures the harness instead of the gradient.
Every row carries ``|g.v|`` and the reference's own uncertainty ``fd_unc``, and
rows with ``grad_err <= 10*fd_unc`` are reported as NOT measurable rather than
silently included.

**3. Synthetic clamp knobs confound "clamp fired" with "gradient vanished".**
Both families collapse ``|grad|`` by 12+ orders as the clamp bites harder,
because both approach a product state, so in synthetic probes the two are
collinear by construction and a *relative* ``grad_err`` cannot separate "the
clamp damaged the gradient" from "there is no gradient left to measure".
``su_D2`` is the only state found so far with a healthy gradient (2.8e-02) AND
a fired clamp (rank 3 of chi).  Breaking that confound -- an SU
imaginary-time family that clamps without collapsing -- is the open work.

Candidates measured per row, all free (every quantity is already computed
inside ``asym_root_implicit_energy_and_grad`` and discarded):

    A_cancel  ||vjp_p(F_bar)|| / ||grad||     REFUTED: pinned at 0.500
    B_adj     ||F_bar|| / ||y_bar||           REFUTED: pinned at 1.0000
    L_diag    L_abs / ||S_bar||               REFUTED: rho +0.429, inverts
    L_grad    L_abs / ||grad||                REFUTED: rho +0.429, inverts
    L_abs     ||S_bar[i,i]|| over clamped i   OPEN: rho +0.886 on n=3 states

A_cancel's refutation is structural and worth keeping: on every clamped row
``||direct|| + ||adj|| = ||grad||`` to four digits, and ``|x - y| = |x| + |y|``
only when x and y are exactly anti-parallel, so the two terms *reinforce*.
There is no catastrophic cancellation in that subtraction.  With ``B_adj``
pinned at 1.0 and ``adjoint_residual ~ 1e-15``, the pair rules out the whole
"conditioning of the linear algebra" family: the error enters through neither
the solve nor the assembly.

Usage:
    uv run python docs/plans/reference/785-remeasure.py
    uv run python docs/plans/reference/785-remeasure.py --check-clamp

``--check-clamp`` runs the cross-check described on ``clamped_indices``; run it
whenever that detector is touched.
"""

from __future__ import annotations

import argparse
import warnings

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import tenax.algorithms._ctm_root_implicit_asym as M  # noqa: E402
from tenax.algorithms._ctm_c4v_root_implicit import _solve_root_adjoint  # noqa: E402
from tenax.algorithms._ctm_tensor_init import (  # noqa: E402
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.core.index import FlowDirection, TensorIndex  # noqa: E402
from tenax.core.symmetry import U1Symmetry  # noqa: E402
from tenax.core.tensor import DenseTensor  # noqa: E402
from tests._su_fixtures import physical_su_d2  # noqa: E402

H_SCAN = [1e-3, 1e-4, 1e-5, 1e-6]


# --------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------


def _site_tensor(D=2, d=2, seed=42, eps=1.0):
    """The knob that exists in the test suite.  NOT #785's "squashed" state.

    ``eps`` scales the whole random part, so the state limits to a product
    state and the gradient collapses with the energy.  Kept because it is what
    a reader will reach for, and because the reconstruction below starts here.
    """
    rng = np.random.RandomState(seed)
    data = eps * jnp.array(rng.standard_normal((D, D, D, D, d)))
    data = data.at[0, 0, 0, 0, 0].set(1.0)
    data = data / (jnp.linalg.norm(data) + 1e-10)
    sym = U1Symmetry()
    ch = np.zeros(D, dtype=np.int32)
    pch = np.zeros(d, dtype=np.int32)
    idx = (
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="u"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="d"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.OUT, label="l"),
        TensorIndex.from_charges(sym, ch.copy(), FlowDirection.IN, label="r"),
        TensorIndex.from_charges(sym, pch.copy(), FlowDirection.IN, label="phys"),
    )
    return DenseTensor(data, idx)


def squashed(eps, D=2, d=2, seed=42):
    """Squash the VIRTUAL bond spectrum, leaving the dominant block alone.

        A'[u,d,l,r,p] = A[u,d,l,r,p] * w[u] w[d] w[l] w[r],   w = (1, eps, ...)

    The ``(0,0,0,0,.)`` block stays O(1), so E and dE/dA stay O(1), while the
    half-infinite environment picks up singular directions at O(eps^2) and
    O(eps^4) -- which is what makes the rank clamp fire.  A knob on the *cut*,
    which is what #785 is about, rather than on the whole state.

    It still does not fully break the confound: |grad| falls with eps here too,
    just far more slowly than under ``_site_tensor``.
    """
    A = _site_tensor(D=D, d=d, seed=seed, eps=1.0)
    data = np.asarray(A.todense(), dtype=float)
    w = np.ones(D)
    w[1:] = eps
    data = (
        data
        * w[:, None, None, None, None]
        * w[None, :, None, None, None]
        * w[None, None, :, None, None]
        * w[None, None, None, :, None]
    )
    return DenseTensor(jnp.asarray(data / np.linalg.norm(data)), A.indices)


STATES = [
    ("random", lambda: _site_tensor(D=2, seed=42, eps=1.0)),
    ("bsq1e-1", lambda: squashed(1e-1)),
    ("bsq3e-2", lambda: squashed(3e-2)),
    ("bsq1e-2", lambda: squashed(1e-2)),
    ("bsq3e-3", lambda: squashed(3e-3)),
    ("bsq1e-3", lambda: squashed(1e-3)),
    ("bsq1e-4", lambda: squashed(1e-4)),
    # The only row so far with a healthy gradient AND a fired clamp.
    ("su_D2", physical_su_d2),
]


def _gate(delta=1.0):
    Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    H = delta * jnp.kron(Sz, Sz) + 0.5 * jnp.kron(Sp, Sm) + 0.5 * jnp.kron(Sm, Sp)
    return H.reshape(2, 2, 2, 2)


def _a_array(A):
    at = _build_double_layer_tensor(A)
    lab = list(at.labels())
    perm = tuple(lab.index(x) for x in ("u2", "d2", "l2", "r2"))
    return jnp.asarray(at.transpose(perm).todense())


def _nrm(tree):
    return float(jnp.sqrt(sum(jnp.sum(jnp.abs(x) ** 2) for x in jax.tree.leaves(tree))))


# --------------------------------------------------------------------------
# Clamped-direction detection
# --------------------------------------------------------------------------


def clamped_indices(S_k, *, rel_floor=None, tol=1e-9):
    """Indices of the diagonal sitting AT the clamp level.

    Test against the clamp level, not the tied minimum.  ``jnp.maximum`` sets
    every clamped entry to the same value, which makes "the tied minimum" look
    like the right detector -- but on a *full-rank* cut the minimum is just the
    smallest genuine singular value, so the tied-min rule reports one false
    positive per direction.  Measured: ``random`` at chi=4 expects ``[0,0,0,0]``
    clamped directions and the tied-min rule returns ``[1,1,1,1]``, inflating
    ``L_abs`` by five orders on one row and producing a confident, wrong
    refutation of the L family.

    ``S_keep = diag(s_capped / ||s_capped||)`` and the clamp is
    ``cut = rel_floor * s[0]``, so in these units the floor sits at
    ``rel_floor * d[0]``.

    Cross-check this against ``chi - usable_rank`` (``--check-clamp``) whenever
    it is touched.
    """
    d = np.asarray(jnp.diag(S_k).real)
    if rel_floor is None:
        eps = jnp.finfo(S_k.dtype).eps
        eps = float(eps.real if jnp.iscomplexobj(S_k) else eps)
        rel_floor = eps ** (1.0 / 3.0)
    thresh = rel_floor * float(d[0]) * (1.0 + tol)
    return np.flatnonzero(d <= thresh)


# --------------------------------------------------------------------------
# Instrumented gradient
# --------------------------------------------------------------------------


def instrumented(A, gate, chi, *, max_iter=300, conv_tol=1e-13):
    """Replica of ``asym_root_implicit_energy_and_grad`` that keeps the
    intermediates the production path computes and discards."""
    A_const = DenseTensor(jax.lax.stop_gradient(A.todense()), A.indices)
    env, a_arr, _meta, fwd_projs = M.converge(
        A_const,
        chi,
        max_iter=max_iter,
        conv_tol=conv_tol,
        min_iter=4,
        return_projectors=True,
        rel_floor=None,
    )
    rank_report = M.retained_rank_report(env, a_arr, chi, None)
    root, root_residual, usable_rank = M.asym_root_parametrize(
        env,
        a_arr,
        chi,
        prev_projs=fwd_projs,
        polish_steps=40,
        polish_tol=1e-10,
        rel_floor=None,
        return_usable_rank=True,
    )
    root_cov = M.asym_root_to_covariant_convention(root)
    S_star = root_cov.s
    tilde = M.remove_inverse_roots(root_cov.env, S_star)
    y_star = (tilde, root_cov.u, S_star, root_cov.v)

    template = initialize_ctm_tensor_env(A_const, chi)
    A_data = jnp.asarray(A.todense())

    def energy_of(a_data, env_tilde, S_all):
        A_live = DenseTensor(a_data, A.indices)
        return M.asym_energy(
            A_live, M.absorb_inverse_roots(env_tilde, S_all), template, gate
        )

    energy, vjp_energy = jax.vjp(energy_of, A_data, tilde, S_star)
    grad_direct, tilde_bar, S_bar = vjp_energy(jnp.ones((), dtype=energy.dtype))
    y_bar = (
        tilde_bar,
        tuple(jnp.zeros_like(x) for x in root_cov.u),
        S_bar,
        tuple(jnp.zeros_like(x) for x in root_cov.v),
    )

    def F_of_y(y):
        return M.asym_characteristic_residual_covariant(y, a_arr, root_cov, chi)

    _F, vjp_y = jax.vjp(F_of_y, y_star)
    F_bar, solve_resid = _solve_root_adjoint(
        lambda v: vjp_y(v)[0], y_bar, tol=1e-8, maxiter=400, restart=30
    )

    def F_of_p(a_data):
        return M.asym_characteristic_residual_covariant(
            y_star, _a_array(DenseTensor(a_data, A.indices)), root_cov, chi
        )

    _, vjp_p = jax.vjp(F_of_p, A_data)
    adj_term = vjp_p(F_bar)[0]
    grad = grad_direct - adj_term

    # Leverage of the energy cotangent on the clamped directions.  This asks
    # the question r_next failed to ask -- not how much weight the clamp
    # destroys, but whether the objective cares about the destroyed directions.
    lev_sq, n_clamped, per_dir = 0.0, 0, []
    for k in range(4):
        idx = clamped_indices(S_star[k])
        d_bar = np.asarray(jnp.diag(S_bar[k]))
        contrib = float(np.sum(np.abs(d_bar[idx]) ** 2)) if len(idx) else 0.0
        lev_sq += contrib
        n_clamped += int(len(idx))
        per_dir.append((int(len(idx)), float(np.sqrt(contrib))))

    L_abs = float(np.sqrt(lev_sq))
    n_Sbar, n_grad = _nrm(S_bar), _nrm(grad)
    return {
        "grad": grad,
        "S_star": S_star,
        "env": env,
        "a_arr": a_arr,
        "usable_rank": int(rank_report["usable_rank"]),
        "root_residual": float(root_residual),
        "adjoint_residual": float(solve_resid),
        "n_direct": _nrm(grad_direct),
        "n_adj": _nrm(adj_term),
        "n_grad": n_grad,
        "A_cancel": _nrm(adj_term) / (n_grad + 1e-300),
        "B_adj": _nrm(F_bar) / (_nrm(y_bar) + 1e-300),
        "L_abs": L_abs,
        "L_diag": L_abs / (n_Sbar + 1e-300),
        "L_grad": L_abs / (n_grad + 1e-300),
        "n_clamped": n_clamped,
        "per_dir": per_dir,
    }


# --------------------------------------------------------------------------
# Finite-difference reference, with the validity gate
# --------------------------------------------------------------------------


def fd_with_h_scan(A, gate, chi, grad, *, sweeps=30, seed=1):
    """Directional FD of the SWEEP MAP FROM A FIXED START, scanned over h.

    Never FD a re-converged CTM energy or a root: both are gauge/branch
    discontinuous in the parameter, so the difference *diverges* as h shrinks
    (measured -0.105 -> -1.63 across h=1e-3..1e-6 on a re-converging energy).
    Freezing ``env0`` and applying a fixed number of sweeps makes the map
    smooth in the parameter.

    Returns ``(fd_star, analytic, fd_unc, table)``.  ``fd_unc`` is the relative
    disagreement between the two consecutive h that agree best -- the
    reference's own uncertainty.  ``grad_err`` at or below it is the
    reference's error, not the gradient's.
    """
    A_data = jnp.asarray(A.todense())
    A_const = DenseTensor(jax.lax.stop_gradient(A_data), A.indices)
    env0, _a, _m = M.converge(A_const, chi, max_iter=300, conv_tol=1e-13)
    template = initialize_ctm_tensor_env(A_const, chi)

    def energy_explicit(pdata):
        A_live = DenseTensor(pdata, A.indices)
        a_live = _a_array(A_live)
        env, projs = env0, None
        for _ in range(sweeps):
            env, projs = M.sweep(env, a_live, chi, projs)
        return M.asym_energy(A_live, env, template, gate)

    rng = np.random.RandomState(seed)
    V = jnp.asarray(rng.standard_normal(A_data.shape))
    V = V / jnp.linalg.norm(V)
    analytic = float(jnp.real(jnp.sum(grad * V)))

    table = []
    for h in H_SCAN:
        fd = float(
            (energy_explicit(A_data + h * V) - energy_explicit(A_data - h * V))
            / (2 * h)
        )
        table.append((h, fd, abs(fd - analytic) / max(abs(analytic), 1e-300)))

    # Pick h WITHOUT looking at `analytic`.
    pairs = [
        (i, abs(table[i][1] - table[i + 1][1]) / max(abs(table[i][1]), 1e-300))
        for i in range(len(table) - 1)
    ]
    i_best, fd_unc = min(pairs, key=lambda t: t[1])
    return table[i_best][1], analytic, fd_unc, table


# --------------------------------------------------------------------------
# Cross-check for the clamped-direction detector
# --------------------------------------------------------------------------


def check_clamp_detector(chis=(4, 6)):
    """``max`` of the detected per-direction counts must equal
    ``chi - usable_rank`` from the independent half-infinite SVD.

    ``max``, not per-direction equality.  ``retained_rank_report`` reduces with
    ``rank = min(rank, int(usable))`` over the four directions, so
    ``chi - usable_rank`` is ``max_k(clamped_k)`` -- a single worst-case scalar,
    not a per-direction count.  Directions genuinely differ: ``su_D2`` at chi=4
    detects ``[1, 0, 0, 1]`` and ``bsq1e-1`` detects ``[1, 1, 2, 2]``.

    Asserting per-direction equality passes anyway on the bond-squashed family,
    because squashing every virtual leg by the same ``w`` makes the four
    directions equivalent -- so a subset of symmetric states cannot tell the two
    invariants apart.  That is how the wrong one got adopted here first.
    """
    gate = _gate(1.0)
    ok = True
    for name, mk in STATES:
        for chi in chis:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = instrumented(mk(), gate, chi)
            detected = [n for n, _ in res["per_dir"]]
            expected = chi - res["usable_rank"]
            match = max(detected) == expected
            ok &= match
            print(
                f"  {name:9s} chi={chi}  detected={detected}  "
                f"max={max(detected)}  chi-usable_rank={expected}  "
                f"{'ok' if match else 'MISMATCH'}"
            )
    print(f"\nALL MATCH = {ok}")
    return ok


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-clamp", action="store_true")
    ap.add_argument("--chi", type=int, nargs="+", default=[4, 6])
    args = ap.parse_args()

    if args.check_clamp:
        print("clamped-index detector vs chi - usable_rank:\n")
        raise SystemExit(0 if check_clamp_detector(tuple(args.chi)) else 1)

    gate = _gate(1.0)
    rows = []
    for name, mk in STATES:
        for chi in args.chi:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    A = mk()
                    res = instrumented(A, gate, chi)
                    fd, analytic, fd_unc, table = fd_with_h_scan(
                        A, gate, chi, res["grad"]
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"{name:9s} chi={chi} FAILED {type(exc).__name__}: {exc}")
                    continue
            ge = abs(fd - analytic) / max(abs(analytic), 1e-300)
            res.update(
                state=name,
                chi=chi,
                grad_err=ge,
                analytic=analytic,
                fd_unc=fd_unc,
                measurable=ge > 10 * fd_unc,
                table=table,
            )
            rows.append(res)
            print(
                f"{name:9s} chi={chi} rank={res['usable_rank']}/{chi} "
                f"|grad|={res['n_grad']:.3e} g.v={analytic:+.4e} "
                f"grad_err={ge:.3e} fd_unc={fd_unc:.2e} "
                f"MEASURABLE={str(res['measurable']):5s} "
                f"L_abs={res['L_abs']:.3e} A_c={res['A_cancel']:.3e} "
                f"B={res['B_adj']:.3e}",
                flush=True,
            )
            for h, f, r in table:
                print(f"            h={h:.0e} fd={f:+.10e} rel={r:.3e}")

    good = [r for r in rows if r["measurable"]]
    clamped = [r for r in good if r["usable_rank"] < r["chi"]]
    print(
        f"\n=== ORDERING (rows past the validity gate: {len(good)}/{len(rows)}; "
        f"clamped subset: {len(clamped)}) ==="
    )
    print(
        "Rows below the gate are NOT evidence either way -- they are states "
        "whose gradient sits under the FD floor."
    )
    for label, subset in (("VALID", good), ("VALID+CLAMPED", clamped)):
        print(f"\n-- {label} (n={len(subset)}) --")
        if len(subset) < 3:
            print("   too few rows to order")
            continue
        for key in ("L_abs", "L_diag", "L_grad", "A_cancel", "B_adj"):
            s = sorted(subset, key=lambda r: r[key])
            errs = [r["grad_err"] for r in s]
            rho = float(
                np.corrcoef(np.argsort(np.argsort(errs)), np.arange(len(errs)))[0, 1]
            )
            order = [f"{r['state']}/{r['chi']}" for r in s]
            errs_s = [f"{e:.1e}" for e in errs]
            print(f"{key:9s} rho={rho:+.3f} errs={errs_s}")
            print(f"{'':9s} order={order}")


if __name__ == "__main__":
    main()
