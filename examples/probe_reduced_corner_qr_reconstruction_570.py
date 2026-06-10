#!/usr/bin/env python3
"""SPIKE (#570): reduced-corner QR-CTMRG projector reconstruction.

Reconstruct + energy-validate the reduced-corner QR projector of
Yang/Zhang/Corboz (arXiv:2505.00494) inside Tenax's single-site (1x1)
dense CTM, on the spin-1/2 2D Heisenberg model at D=2.

Idea: in the 1x1 left move, the enlarged corners ``C1g`` (labels
``(fused, t1_r)``) and ``C4g`` (labels ``(fused, t3_l)``) have a *cut* leg
(``t1_r`` / ``t3_l``) that is ALREADY dimension chi.  Viewing each corner as
a ``(fused | cut)`` matrix, an *unpivoted* QR yields ``Q : (fused, chi)`` —
a chi-isometry on the fused leg with NO truncation SVD.  The reduced-corner
projector then combines ``Q1`` and ``Q4``.

This probe substitutes candidate dense projectors into the existing 1x1 CTM
(via monkeypatching ``_compute_projector_tensor``) after an eigh warm-up, and
compares the converged energy to the eigh oracle at chi in {8, 16, 24}.

The deliverable is the VALIDATED construction + a VERDICT — see the spec
``docs/superpowers/specs/2026-06-10-reduced-corner-qr-ctmrg-phase1-dense-570.md``.

Run::

    JAX_PLATFORMS=cpu uv run python examples/probe_reduced_corner_qr_reconstruction_570.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import tenax.algorithms._ctm_projector as _proj_mod
import tenax.algorithms._ctm_tensor_moves as _moves_mod
from tenax import CTMConfig, heisenberg_gate, ipeps, iPEPSConfig
from tenax.algorithms._ctm_projector import (
    _make_chi_new_index,
    _tensor_matrix_data,
    _wrap_dense_projector,
)
from tenax.algorithms._ctm_tensor_convergence import (
    _corner_singular_values,
    _ctm_sv_diff,
    _ctm_tensor_sweep,
)
from tenax.algorithms._ctm_tensor_energy import compute_energy_ctm_tensor
from tenax.algorithms._ctm_tensor_init import (
    _build_double_layer_tensor,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ipeps import sublattice_rotate_gate, symmetrize_c4v
from tenax.core.tensor import DenseTensor

# Keep a handle on the genuine projector for the eigh oracle / warm-up.
_TRUE_COMPUTE_PROJECTOR = _proj_mod._compute_projector_tensor


# --------------------------------------------------------------------------- #
# Gauge-fixed QR helper                                                        #
# --------------------------------------------------------------------------- #
def _qr_gauge_fixed(M: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Unpivoted thin QR with a ``diag(R) >= 0`` gauge fix.

    Returns ``(Q, R)`` with ``M = Q @ R``, ``Q`` column-orthonormal, and
    ``diag(R)`` real and non-negative.  For a zero diagonal entry the gauge
    is unconstrained, so phase = 1 (leave that column of Q untouched).
    """
    Q, R = jnp.linalg.qr(M)
    diag_R = jnp.diag(R)
    abs_diag = jnp.abs(diag_R)
    phase = jnp.where(
        abs_diag > 0, diag_R / jnp.where(abs_diag > 0, abs_diag, 1.0), 1.0
    ).astype(R.dtype)
    Q = Q * phase[None, :]
    R = R * jnp.conj(phase)[:, None]
    return Q, R


def _wrap(P_dense: jax.Array, C1g, base_charges) -> object:
    """Wrap a dense ``(fused, chi_new)`` matrix as a Tensor projector."""
    fused_idx = C1g.indices[C1g.labels().index("fused")]
    k = P_dense.shape[1]
    chi_new_idx = _make_chi_new_index(fused_idx, k, base_charges)
    from tenax.core.tensor import SymmetricTensor

    return _wrap_dense_projector(
        P_dense, fused_idx, chi_new_idx, as_symmetric=isinstance(C1g, SymmetricTensor)
    )


# --------------------------------------------------------------------------- #
# Candidate projectors:  proj(C1g, C4g, chi, base_charges) -> P                #
#                                                                              #
# Each returns a dense isometry P with labels (fused, chi_new), flows (IN,OUT).#
# --------------------------------------------------------------------------- #
def candidate_A(C1g, C4g, chi: int, base_charges):
    """A (primary): pure reduced corner, NO large SVD.

    Q1 = qr(C1g), Q4 = qr(C4g) (each gauge-fixed).  The cut legs t1_r / t3_l
    are already chi, so Q1, Q4 : (fused, chi) directly.  Form the chi×chi
    overlap O = Q1† Q4, align Q4 onto Q1's column space via its QR (polar)
    factor W (unitary part of O), and set P = Q1.

    P_1 applied to C1g side = Q1; P_2 applied to C4g side = Q4 @ W† so that
    P_1† P_2 = Q1† Q4 W† = O W† = (orthogonal alignment).  For a symmetric
    single-isometry CTM (eigh-style) we return P = Q1 for both sides — the
    half-system isometry — which is the faithful truncation-free target.
    """
    C1 = _tensor_matrix_data(C1g)
    # Candidate A deliberately uses ONLY C1g (single-corner reduced isometry);
    # C4g is unused on purpose — that is the whole point of the pure form.
    del C4g
    Q1, _ = _qr_gauge_fixed(C1)  # (fused, chi)
    # Symmetric single-isometry projector: P = Q1 (the reduced-corner
    # isometry on the fused leg).  No truncation, no large SVD.
    k = min(chi, Q1.shape[1])
    return _wrap(Q1[:, :k], C1g, base_charges)


def candidate_B(C1g, C4g, chi: int, base_charges):
    """B: reduced corner + SMALL chi×chi overlap SVD.

    Q1 = qr(C1g), Q4 = qr(C4g).  Overlap O = Q1† Q4 (chi×chi) → SVD U s Vh.
    P = Q1 @ U[:, :chi].  Half-system QR projector; still no large (chiD^2)
    SVD — only the tiny chi×chi alignment SVD.
    """
    C1 = _tensor_matrix_data(C1g)
    C4 = _tensor_matrix_data(C4g)
    Q1, _ = _qr_gauge_fixed(C1)  # (fused, chi)
    Q4, _ = _qr_gauge_fixed(C4)  # (fused, chi)
    overlap = Q1.conj().T @ Q4  # (chi, chi)
    U_o, _s_o, _Vh_o = jnp.linalg.svd(overlap, full_matrices=False)
    k = min(chi, U_o.shape[1])
    P_dense = Q1 @ U_o[:, :k]  # (fused, k)
    return _wrap(P_dense, C1g, base_charges)


def candidate_C(C1g, C4g, chi: int, base_charges):
    """C (DIAGNOSTIC ONLY — not the faithful target).

    Concatenate [C1g | C4g] (fused × 2*chi), QR, then eigh the small
    R R† object and keep the top-chi eigenvectors:  P = Q @ V.
    This is the existing dense ``qr`` projector path in _ctm_projector.py.
    It uses a larger (2chi-column) object than the pure reduced corner.
    Recorded only to confirm the energy target is reachable.  If ONLY C
    matches => STOP signal.
    """
    C1 = _tensor_matrix_data(C1g)
    C4 = _tensor_matrix_data(C4g)
    M = jnp.concatenate([C1, C4], axis=1)  # (fused, 2*chi)
    Q, R = _qr_gauge_fixed(M)
    rho_small = R @ R.conj().T
    rho_small = 0.5 * (rho_small + rho_small.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(rho_small)
    k = min(chi, len(eigvals))
    V = eigvecs[:, -k:][:, ::-1]
    P_dense = Q @ V  # (fused, k)
    return _wrap(P_dense, C1g, base_charges)


_CANDIDATES = {"A": candidate_A, "B": candidate_B, "C": candidate_C}


def _make_patched_projector(candidate_fn):
    """Return a _compute_projector_tensor replacement using ``candidate_fn``.

    Mirrors the eigh/qr return contract: (P, P, eps_T) with P_1 = P_2 = P.
    """

    def _patched(
        C1g,
        C4g,
        chi,
        projector_method="svd",
        base_charges=None,
        projector_backward="auto",
    ):
        # Warm-up sweeps drive projector_method="eigh"; defer to the oracle.
        if projector_method == "eigh":
            return _TRUE_COMPUTE_PROJECTOR(
                C1g, C4g, chi, "eigh", base_charges, projector_backward
            )
        P = candidate_fn(C1g, C4g, chi, base_charges)
        return P, P, jnp.asarray(0.0)

    return _patched


# --------------------------------------------------------------------------- #
# Harness                                                                      #
# --------------------------------------------------------------------------- #
def build_physical_state():
    """Make a physical D=2 single-site Heisenberg tensor + the rotated gate.

    Uses sublattice rotation so the Neel AFM ground state is a *uniform*
    single-site iPEPS, then runs simple update to get a converged physical
    ``A``.  Returns ``(A, gate_rot)`` for the single-site CTM + energy.
    """
    gate = heisenberg_gate()
    gate_rot = sublattice_rotate_gate(gate)
    # Simple update with the rotated gate -> uniform state; take site A.
    config = iPEPSConfig(
        max_bond_dim=2,
        num_imaginary_steps=400,
        dt=0.05,
        ctm=CTMConfig(chi=16, max_iter=80, projector_method="eigh"),
    )
    _E_su, (A, _B), _envs = ipeps(gate_rot, initial_peps=None, config=config)
    # C4v-symmetrize the site tensor.  The sublattice-rotated AFM ground state
    # is C4v-symmetric; enforcing it makes the four directional 1x1 moves
    # equivalent so the single-site eigh CTM reaches a genuine fixed point
    # (otherwise the directional sweep limit-cycles at sv_diff~1e-4, the
    # documented #425/#426 plateau, and the eigh oracle is untrustworthy).
    A = DenseTensor(symmetrize_c4v(A._data), A.indices)
    A = A * (1.0 / float(A.norm()))
    return A, gate_rot


class _Info:
    """Lightweight convergence info."""

    def __init__(self, converged, iterations, sv_diff):
        self.converged = converged
        self.iterations = iterations
        self.sv_diff = sv_diff


def _run_1x1_ctm(A, chi, candidate_fn, warmup, max_iter, conv_tol):
    """Self-contained single-site (1x1) CTM convergence loop.

    Drives ``_ctm_tensor_sweep`` (the canonical single-site sweep that calls
    ``_ctm_tensor_move_{left,top,right,bottom}`` with ``env, env`` self-
    neighbors) so the 1x1 enlarged-corner projector ``_compute_projector_tensor``
    in ``_ctm_tensor_moves`` is actually exercised.  This is the path the task
    spec points at (moves at _ctm_tensor_moves.py:659-711).

    When ``candidate_fn`` is given, the projector is monkeypatched to the
    candidate after ``warmup`` eigh sweeps; otherwise pure eigh is used (the
    oracle).
    """
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)
    patched = (
        _make_patched_projector(candidate_fn) if candidate_fn is not None else None
    )

    prev_sv = None
    converged = False
    sv_diff = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        use_candidate = candidate_fn is not None and it > warmup
        method = "eigh" if (candidate_fn is None or not use_candidate) else "candidate"
        _moves_mod._compute_projector_tensor = (
            patched if use_candidate else _TRUE_COMPUTE_PROJECTOR
        )
        try:
            env, _eps = _ctm_tensor_sweep(env, a, chi, False, method)
        finally:
            _moves_mod._compute_projector_tensor = _TRUE_COMPUTE_PROJECTOR

        sv = _corner_singular_values(env.C1)
        if prev_sv is not None and sv.shape == prev_sv.shape:
            sv_diff = float(_ctm_sv_diff(sv, prev_sv))
            if it >= 10 and sv_diff < conv_tol:
                converged = True
                break
        prev_sv = sv

    return env, _Info(converged, it, sv_diff)


def run_ctm_energy(A, gate, chi, candidate_fn=None, warmup=6, max_iter=200):
    """Converge 1x1 CTM (eigh, or eigh-warmup then candidate) and return E."""
    env, info = _run_1x1_ctm(
        A, chi, candidate_fn, warmup, max_iter, conv_tol=1e-10
    )
    E = float(compute_energy_ctm_tensor(A, env, gate))
    return E, info


def corner_diagnostic(A, chi=8, warmup=30):
    """Inspect a converged-ish 1x1 left move: corner shapes + subspace overlaps.

    Confirms the reduced-corner property: the cut leg of ``C1g`` is already
    ``chi``, so ``QR(C1g)`` truncates nothing.  Also reports how the
    reduced-corner isometry ``Q1`` relates to the eigh density-matrix
    subspace and to ``Q4``.
    """
    a = _build_double_layer_tensor(A)
    env = initialize_ctm_tensor_env(A, chi)
    for _ in range(warmup):
        env, _ = _ctm_tensor_sweep(env, a, chi, False, "eigh")

    cap = {}

    def spy(C1g, C4g, c, m="svd", bc=None, pb="auto"):
        if "done" not in cap:
            c1 = _tensor_matrix_data(C1g)
            c4 = _tensor_matrix_data(C4g)
            Q1, _ = _qr_gauge_fixed(c1)
            Q4, _ = _qr_gauge_fixed(c4)
            rho = c1 @ c1.conj().T + c4 @ c4.conj().T
            rho = 0.5 * (rho + rho.conj().T)
            _w, v = jnp.linalg.eigh(rho)
            Peig = v[:, -c:][:, ::-1]

            def P(M):
                return M @ M.conj().T

            cap["C1g_shape"] = tuple(c1.shape)
            cap["C4g_shape"] = tuple(c4.shape)
            cap["||C1g-C4g||"] = float(jnp.linalg.norm(c1 - c4))
            cap["||span(Q1)-span(eigh)||"] = float(jnp.linalg.norm(P(Q1) - P(Peig)))
            cap["||span(Q1)-span(Q4)||"] = float(jnp.linalg.norm(P(Q1) - P(Q4)))
            cap["done"] = 1
        return _TRUE_COMPUTE_PROJECTOR(C1g, C4g, c, m, bc, pb)

    _moves_mod._compute_projector_tensor = spy
    try:
        _ctm_tensor_sweep(env, a, chi, False, "eigh")
    finally:
        _moves_mod._compute_projector_tensor = _TRUE_COMPUTE_PROJECTOR
    cap.pop("done", None)
    return cap


def main():
    print("=" * 72)
    print("SPIKE #570: reduced-corner QR-CTMRG projector reconstruction")
    print("Model: spin-1/2 2D Heisenberg, D=2, single-site (1x1) CTM")
    print("=" * 72)

    A, gate_rot = build_physical_state()
    chis = [8, 16, 24]

    print("\nCorner diagnostic (chi=8 left move, near fixed point):")
    diag = corner_diagnostic(A, chi=8)
    for k, v in diag.items():
        print(f"  {k} = {v}")

    print("\nReference eigh CTM energies:")
    E_eigh = {}
    for chi in chis:
        E, info = run_ctm_energy(A, gate_rot, chi)
        E_eigh[chi] = E
        print(f"  chi={chi:>3}: E_eigh = {E:.10f}  (conv={info.converged}, "
              f"iters={info.iterations})")

    results = {}  # cand -> {chi: E}
    for name, fn in _CANDIDATES.items():
        results[name] = {}
        print(f"\nCandidate {name}:")
        for chi in chis:
            try:
                E, info = run_ctm_energy(A, gate_rot, chi, candidate_fn=fn)
            except Exception as exc:  # noqa: BLE001
                E = float("nan")
                info = None
                print(f"  chi={chi:>3}: FAILED ({type(exc).__name__}: {exc})")
                continue
            results[name][chi] = E
            dE = abs(E - E_eigh[chi])
            conv = info.converged if info else False
            print(f"  chi={chi:>3}: E = {E:.10f}  |dE| = {dE:.3e}  "
                  f"(conv={conv}, iters={info.iterations if info else '-'})")

    # ----- summary table -----
    print("\n" + "=" * 72)
    print("RESULT TABLE (|dE| = |E_cand - E_eigh|)")
    print("=" * 72)
    header = f"{'candidate':<10}" + "".join(f"chi={c:<14}" for c in chis) \
        + f"{'max|dE|':<14}"
    print(header)
    print("-" * len(header))
    print(f"{'eigh':<10}" + "".join(f"{E_eigh[c]:<18.10f}" for c in chis)
          + f"{'(oracle)':<14}")
    cand_dE = {}
    for name in _CANDIDATES:
        row = f"{name:<10}"
        max_dE = 0.0
        for c in chis:
            E = results[name].get(c, float("nan"))
            row += f"{E:<18.10f}"
            if not np.isnan(E):
                max_dE = max(max_dE, abs(E - E_eigh[c]))
        cand_dE[name] = max_dE
        row += f"{max_dE:<14.3e}"
        print(row)

    # ----- |dE| vs chi table (for shrink check) -----
    print("\n|dE| vs chi (shrink check):")
    dE_hdr = f"{'candidate':<10}" + "".join(f"chi={c:<14}" for c in chis)
    print(dE_hdr)
    print("-" * len(dE_hdr))
    for name in _CANDIDATES:
        row = f"{name:<10}"
        for c in chis:
            E = results[name].get(c, float("nan"))
            dE = abs(E - E_eigh[c]) if not np.isnan(E) else float("nan")
            row += f"{dE:<18.3e}"
        print(row)

    # ----- verdict -----
    def matches(name):
        r = results[name]
        if 8 not in r or np.isnan(r[8]):
            return False
        ok8 = abs(r[8] - E_eigh[8]) < 1e-3
        # shrinks (or stays at floor) as chi grows
        dEs = [abs(r[c] - E_eigh[c]) for c in chis if c in r and not np.isnan(r[c])]
        shrinks = len(dEs) >= 2 and dEs[-1] <= dEs[0] + 1e-9
        return ok8 and shrinks

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    a_ok = matches("A")
    b_ok = matches("B")
    c_ok = matches("C")
    print(f"  A matches (faithful, no large SVD): {a_ok}")
    print(f"  B matches (reduced + chi×chi SVD):  {b_ok}")
    print(f"  C matches (diagnostic, larger obj): {c_ok}")

    if a_ok:
        print("\n  WINNER: Candidate A — pure reduced corner, NO large SVD.")
        print("  Faithful truncation-free target REACHED.")
    elif b_ok:
        print("\n  WINNER: Candidate B — reduced corner + small chi×chi SVD.")
        print("  No large (chiD^2) SVD; faithful-ish target reached.")
    elif c_ok:
        print("\n  STOP: only Candidate C (larger object) matches.")
        print("  Pure reduced corner does NOT reproduce the energy — rethink.")
    else:
        print("\n  STOP: no candidate matches eigh within 1e-3 at chi=8.")

    return E_eigh, results, (a_ok, b_ok, c_ok)


if __name__ == "__main__":
    main()
