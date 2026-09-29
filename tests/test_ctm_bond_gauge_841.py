"""Per-bond (per-chi-index) sign/phase gauge for the CTM forward (#841).

``forward_gauge="phase"`` removes one global phase per env tensor, so it
cannot absorb the per-chi-index +-1 signs the 2x2 projector SVD re-draws each
sweep.  At D=3 (fermionic t-V, chi=12) those signs form an exact period-2
map and the forward never reaches an element-wise fixed point, so the
implicit adjoint has nothing to linearize around.  ``"bond_phase"`` aligns
every chi index of every bond family to the previous environment.

Properties pinned here:

1. The bond families follow the energy contraction, including the chi bonds
   the mixed 2-site RDMs share ACROSS coords (a per-coord gauge is not exact
   there -- asserted, so the family construction cannot silently shrink).
2. The gauge is an exact gauge transform: energy and every NN RDM unchanged
   to ~1e-13, dense and fermionic, 2-site.
3. It recovers a known random per-index gauge element-wise (real +-1 and
   complex unit phases), whatever the per-tensor global signs.
4. Mechanism: a forward whose step re-draws per-index signs every sweep
   reaches an element-wise fixed point under ``"bond_phase"`` and not under
   ``"phase"`` (the signals are mocked, not a convergence benchmark).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tenax.algorithms._ctm_python_loop import _make_jit_ctm_step  # noqa: E402
from tenax.algorithms._ctm_tensor_convergence import (  # noqa: E402
    CHECKERBOARD_NEIGHBORS,
    SINGLE_SITE_NEIGHBORS,
    _max_env_leaf_diff,
)
from tenax.algorithms._ctm_tensor_energy import (  # noqa: E402
    _rdm1x2_tensor_2site,
    _rdm2x1_tensor_2site,
    compute_energy_ctm_tensor_multisite,
)
from tenax.algorithms._ctm_tensor_init import (  # noqa: E402
    CTMTensorEnv,
    initialize_ctm_tensor_env,
)
from tenax.algorithms.ad_utils import (  # noqa: E402
    _BOND_CHI_LEGS,
    _BOND_OUT_LABELS,
    _bond_gauge_families,
    _bond_phase_fix_envs,
    _phase_fix_ctm_tensor,
    _wrap_tensor,
)
from tenax.algorithms.fermionic_ipeps import (  # noqa: E402
    FPEPSConfig,
    _initialize_fpeps,
    spinless_fermion_gate,
)
from tenax.algorithms.ipeps_optimize import _wrap_as_dense_tensor  # noqa: E402

CHI = 6
FIELDS = tuple(_BOND_CHI_LEGS)


def _heisenberg_gate():
    sz = np.diag([0.5, -0.5])
    sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    h = np.kron(sz, sz) + 0.5 * (np.kron(sp, sp.T) + np.kron(sp.T, sp))
    return jnp.asarray(h.reshape(2, 2, 2, 2))


def _sweep(sites, neighbors, envs, n):
    step = _make_jit_ctm_step(neighbors, "2x2")
    for _ in range(n):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            envs, _e, _s = step(
                sites,
                envs,
                chi=CHI,
                projector_method="svd",
                renormalize=True,
                projector_backward="lorentzian",
            )
        envs = {c: _phase_fix_ctm_tensor(e) for c, e in envs.items()}
    return envs


@pytest.fixture(scope="module")
def dense_2site():
    keys = jax.random.split(jax.random.PRNGKey(841), 2)
    sites = {}
    for c, k in zip(((0, 0), (1, 0)), keys):
        A = jax.random.normal(k, (2, 2, 2, 2, 2))
        sites[c] = _wrap_as_dense_tensor(A / jnp.linalg.norm(A))
    nb = CHECKERBOARD_NEIGHBORS
    envs = {c: initialize_ctm_tensor_env(A, CHI) for c, A in sites.items()}
    envs = _sweep(sites, nb, envs, 12)
    return sites, nb, envs, _heisenberg_gate()


@pytest.fixture(scope="module")
def ferm_2site():
    cfg = FPEPSConfig(D=2, t=1.0, V=1.0, dt=0.05)
    gate = spinless_fermion_gate(cfg)
    sites = {
        (0, 0): _initialize_fpeps(cfg, jax.random.PRNGKey(3)),
        (1, 0): _initialize_fpeps(cfg, jax.random.PRNGKey(4)),
    }
    nb = CHECKERBOARD_NEIGHBORS
    envs = {c: initialize_ctm_tensor_env(A, CHI) for c, A in sites.items()}
    envs = _sweep(sites, nb, envs, 12)
    return sites, nb, envs, gate


def _random_family_gauge(envs, families, rng, *, complex_=False, tensor_signs=True):
    """Apply a random per-index family gauge (w on out-legs, conj(w) on
    in-legs -- an exact transform) plus a random global sign per tensor."""
    w_leg = {}
    for fam in families:
        c0, lab0 = fam[0]
        f0 = next(f for f, p in _BOND_CHI_LEGS.items() if lab0 in p)
        n = next(i.dim for i in getattr(envs[c0], f0).indices if i.label == lab0)
        if complex_:
            w = np.exp(1j * rng.uniform(0, 2 * np.pi, n))
        else:
            w = rng.choice([-1.0, 1.0], n)
        for leg in fam:
            w_leg[leg] = w
    out = {}
    for c, env in envs.items():
        tensors = {}
        for f in FIELDS:
            t = getattr(env, f)
            X = np.asarray(t.todense())
            if complex_:
                X = X.astype(complex)
            for ax, idx in enumerate(t.indices):
                if (c, idx.label) not in w_leg:
                    continue
                w = w_leg[(c, idx.label)]
                w = w if idx.label in _BOND_OUT_LABELS else np.conj(w)
                shape = [1] * X.ndim
                shape[ax] = -1
                X = X * w.reshape(shape)
            if tensor_signs:
                X = X * rng.choice([-1.0, 1.0])
            tensors[f] = _wrap_tensor(jnp.asarray(X), t)
        out[c] = CTMTensorEnv(**tensors)
    return out


def _nn_rdms(sites, envs):
    A, B = sites[(0, 0)], sites[(1, 0)]
    eA, eB = envs[(0, 0)], envs[(1, 0)]
    return [
        np.asarray(_rdm2x1_tensor_2site(A, B, eA, eB)),
        np.asarray(_rdm2x1_tensor_2site(B, A, eB, eA)),
        np.asarray(_rdm1x2_tensor_2site(A, B, eA, eB)),
        np.asarray(_rdm1x2_tensor_2site(B, A, eB, eA)),
    ]


def _rdm_dist(rdms0, rdms1):
    return [float(np.max(np.abs(r1 - r0))) for r0, r1 in zip(rdms0, rdms1)]


def _env_dist(a, b):
    return max(_max_env_leaf_diff(a[c], b[c]) for c in a)


# --------------------------------------------------------------------------
# 1. Families follow the energy contraction
# --------------------------------------------------------------------------


def test_checkerboard_families_span_both_coords():
    fams = _bond_gauge_families(((0, 0), (1, 0)), CHECKERBOARD_NEIGHBORS)
    assert len(fams) == 8
    assert all(len(f) == 4 for f in fams)
    # every family touches both coords: the mixed RDMs contract T1[A].t1_r
    # with T1[B].t1_l, so a per-coord gauge is not a gauge there.
    assert all({c for c, _ in f} == {(0, 0), (1, 0)} for f in fams)
    top = next(f for f in fams if ((0, 0), "c1_r") in f)
    assert set(top) == {
        ((0, 0), "c1_r"),
        ((0, 0), "t1_l"),
        ((1, 0), "t1_r"),
        ((1, 0), "c2_l"),
    }


def test_single_site_families_are_the_four_sigma_bonds():
    fams = _bond_gauge_families(((0, 0),), SINGLE_SITE_NEIGHBORS)
    labels = sorted(tuple(sorted(lab for _, lab in f)) for f in fams)
    assert labels == sorted(
        [
            ("c1_r", "c2_l", "t1_l", "t1_r"),
            ("c2_d", "c3_u", "t2_d", "t2_u"),
            ("c3_l", "c4_u", "t3_l", "t3_r"),
            ("c1_d", "c4_r", "t4_d", "t4_u"),
        ]
    )


# --------------------------------------------------------------------------
# 2. Exact gauge transform
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["dense", "fermionic"])
def test_family_gauge_is_exact_and_per_coord_gauge_is_not(
    which, dense_2site, ferm_2site
):
    sites, nb, envs, gate = dense_2site if which == "dense" else ferm_2site
    fams = _bond_gauge_families(tuple(sites), nb)
    rng = np.random.default_rng(1)
    gauged = _random_family_gauge(envs, fams, rng, tensor_signs=False)
    # Every NN RDM of the cell (all four bonds the energy sums) is unchanged,
    # so the energy is too.
    rdms0 = _nn_rdms(sites, envs)
    assert max(_rdm_dist(rdms0, _nn_rdms(sites, gauged))) < 1e-13
    # The regime: splitting the families per coord (gauging each coord's
    # legs independently) is NOT a gauge -- which is why they are joined.
    per_coord = [tuple(leg for leg in f if leg[0] == c) for f in fams for c in sites]
    broken = _random_family_gauge(envs, per_coord, np.random.default_rng(2))
    assert max(_rdm_dist(rdms0, _nn_rdms(sites, broken))) > 1e-6


@pytest.mark.parametrize("which", ["dense", "fermionic"])
def test_bond_phase_fix_is_an_exact_gauge_transform(which, dense_2site, ferm_2site):
    """The fix itself, applied to a step output against an unrelated
    reference, changes no RDM and no energy."""
    sites, nb, envs, gate = dense_2site if which == "dense" else ferm_2site
    fams = _bond_gauge_families(tuple(sites), nb)
    new = _sweep(sites, nb, envs, 1)
    ref = _random_family_gauge(envs, fams, np.random.default_rng(3))
    fixed = _bond_phase_fix_envs(new, ref, fams)
    # Regime: the fix really moved the env (it is not the identity here).
    assert _env_dist(new, fixed) > 0.1
    assert max(_rdm_dist(_nn_rdms(sites, new), _nn_rdms(sites, fixed))) < 1e-13
    E0 = float(compute_energy_ctm_tensor_multisite(sites, new, nb, gate))
    E1 = float(compute_energy_ctm_tensor_multisite(sites, fixed, nb, gate))
    assert abs(E1 - E0) < 1e-13, (E0, E1)


# --------------------------------------------------------------------------
# 3. Recovery of a known gauge
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "which,complex_",
    [("dense", False), ("dense", True), ("fermionic", False)],
    ids=["dense-real", "dense-complex", "fermionic-real"],
)
def test_a_random_per_index_gauge_is_undone_elementwise(
    which, complex_, seed, dense_2site, ferm_2site
):
    sites, nb, envs, _gate = dense_2site if which == "dense" else ferm_2site
    fams = _bond_gauge_families(tuple(sites), nb)
    if complex_:
        envs = {
            c: CTMTensorEnv(
                **{
                    f: _wrap_tensor(
                        getattr(e, f).todense().astype(complex), getattr(e, f)
                    )
                    for f in FIELDS
                }
            )
            for c, e in envs.items()
        }
    rng = np.random.default_rng(100 + seed)
    scrambled = _random_family_gauge(envs, fams, rng, complex_=complex_)
    # Regime: the scramble is O(1) element-wise, not a no-op.
    assert _env_dist(envs, scrambled) > 0.1
    fixed = _bond_phase_fix_envs(scrambled, envs, fams)
    assert _env_dist(envs, fixed) < 1e-12


# --------------------------------------------------------------------------
# 4. Mechanism: a sign-redrawing step reaches an element-wise fixed point
# --------------------------------------------------------------------------


def test_a_sign_redrawing_forward_converges_elementwise_only_under_bond_phase(
    dense_2site,
):
    """Mock the #841 signal: every sweep's output carries a fresh random
    per-index +-1 family gauge (what the projector SVD's sign map does at
    D=3).  ``"bond_phase"`` must reach an element-wise fixed point; the
    per-tensor ``"phase"`` gauge must not (the regime)."""
    sites, nb, envs0, _gate = dense_2site
    fams = _bond_gauge_families(tuple(sites), nb)
    step = _make_jit_ctm_step(nb, "2x2")
    kw = dict(
        chi=CHI,
        projector_method="svd",
        renormalize=True,
        projector_backward="lorentzian",
    )
    # Converge the underlying (unscrambled) map first so only gauge motion is left.
    envs0 = _sweep(sites, nb, envs0, 40)

    def run(gauge):
        rng = np.random.default_rng(7)
        envs = envs0
        diffs = []
        for _ in range(6):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                new, _e, _s = step(sites, envs, **kw)
            new = _random_family_gauge(new, fams, rng)
            if gauge == "phase":
                fixed = {c: _phase_fix_ctm_tensor(new[c]) for c in new}
            else:
                fixed = _bond_phase_fix_envs(new, envs, fams)
            diffs.append(_env_dist(envs, fixed))
            envs = fixed
        return diffs

    phase = run("phase")
    bond = run("bond_phase")
    assert min(phase[2:]) > 0.1, phase
    assert max(bond[2:]) < 1e-8, bond


# --------------------------------------------------------------------------
# 5. Wiring: the implicit-AD path accepts the gauge; unsupported paths refuse
# --------------------------------------------------------------------------


def test_implicit_ad_with_bond_phase_matches_phase_where_both_converge(dense_2site):
    """On a 2-site dense state where the ``"phase"`` forward already reaches
    an element-wise fixed point, ``"bond_phase"`` must give the same energy
    and gradient (the backward linearizes the bond-gauged step, whose extra
    factors are +-1 constants there)."""
    import tenax.algorithms._ctm_energy_ad as ead

    sites, nb, _envs, gate = dense_2site
    coords = tuple(sites)

    def run(gauge):
        def loss(p):
            return ead.ctm_energy_implicit(
                dict(zip(coords, p)),
                nb,
                gate,
                chi=CHI,
                max_iter=80,
                conv_tol=1e-10,
                conv_method="elementwise",
                forward_gauge=gauge,
                gmres_tol=1e-10,
            )

        ead._F3_LAST_DIAGNOSTICS.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            E, g = jax.value_and_grad(loss)(tuple(sites[c] for c in coords))
        diag = ead.get_last_implicit_ad_diagnostics()
        return float(E), [np.asarray(gi.todense()) for gi in g], diag

    E_p, g_p, d_p = run("phase")
    E_b, g_b, d_b = run("bond_phase")
    # Regime: the phase forward is stationary here, so the two must agree.
    assert d_p["forward_stationarity_residual"] < 1e-7
    assert d_b["forward_stationarity_residual"] < 1e-7
    assert abs(E_b - E_p) < 1e-12
    for a, b in zip(g_p, g_b):
        assert np.all(np.isfinite(b))
        assert np.max(np.abs(a - b)) < 1e-7 * max(1.0, np.max(np.abs(a)))


def test_bond_phase_refuses_chi_ramp(dense_2site):
    import tenax.algorithms._ctm_energy_ad as ead

    sites, nb, _envs, gate = dense_2site
    with pytest.raises(NotImplementedError, match="chi_ramp"):
        ead.ctm_energy_implicit(
            sites, nb, gate, chi=CHI, chi_ramp=[(4, 5)], forward_gauge="bond_phase"
        )


def test_policy_accepts_bond_phase_and_the_packed_config_keeps_it():
    from tenax.algorithms.ad_utils import _config_from_tuple, _config_to_tuple
    from tenax.algorithms.ipeps_ad_policy import validate_ctm_for_implicit_ad
    from tenax.algorithms.ipeps_config import CTMConfig

    cfg = CTMConfig(chi=4, forward_gauge="bond_phase", ctm_conv_method="elementwise")
    validate_ctm_for_implicit_ad(cfg)
    # The packed config must not silently turn "bond_phase" into "qr".
    assert _config_from_tuple(_config_to_tuple(cfg)).forward_gauge == "bond_phase"


def test_complex_gradient_is_finite_with_a_dead_chi_index(dense_2site):
    """Complex envs differentiate the phases, and symmetric/rank-deficient
    envs carry chi indices with exactly zero weight (v_i = 0 in the power
    iteration).  ``abs`` has a 0/0 VJP at 0, which a ``where`` on the output
    does not fence -- the backward must still be finite there."""
    sites, nb, envs, _gate = dense_2site
    fams = _bond_gauge_families(tuple(sites), nb)
    dead = 2  # zero chi index 2 on every leg of every tensor

    def kill(t):
        X = t.todense().astype(complex)
        for ax, idx in enumerate(t.indices):
            if idx.label in _BOND_OUT_LABELS or any(
                idx.label == p[1] for p in _BOND_CHI_LEGS.values()
            ):
                sl = [slice(None)] * X.ndim
                sl[ax] = dead
                X = X.at[tuple(sl)].set(0.0)
        return _wrap_tensor(X, t)

    ref = {
        c: CTMTensorEnv(**{f: kill(getattr(e, f)) for f in FIELDS})
        for c, e in envs.items()
    }
    new = _random_family_gauge(ref, fams, np.random.default_rng(5), complex_=True)
    leaves, treedef = jax.tree.flatten(new)

    def f(ls):
        fixed = _bond_phase_fix_envs(jax.tree.unflatten(treedef, ls), ref, fams)
        return sum(jnp.sum(jnp.real(x)) for x in jax.tree.leaves(fixed))

    g = jax.grad(f)(leaves)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in g)


# --------------------------------------------------------------------------
# 6. Codex P1s on #1057: block-sparse, and refused on split CTM
# --------------------------------------------------------------------------


def test_symmetric_path_never_densifies(ferm_2site, monkeypatch):
    """The gauge works on the charge blocks of a SymmetricTensor env: the
    per-index factors are diagonal on each sector, so no ``todense`` is
    needed (CLAUDE.md: avoid densifying on the symmetric path).  The only
    dense objects are the chi x chi overlap/Gram matrices of each bond."""
    from tenax.core.tensor import SymmetricTensor

    sites, nb, envs, _gate = ferm_2site
    assert isinstance(envs[(0, 0)].T1, SymmetricTensor)  # regime
    fams = _bond_gauge_families(tuple(sites), nb)
    scrambled = _random_family_gauge(envs, fams, np.random.default_rng(11))
    assert _env_dist(envs, scrambled) > 0.1  # regime: not a no-op

    def _no_todense(self):
        raise AssertionError("bond_phase densified a SymmetricTensor")

    monkeypatch.setattr(SymmetricTensor, "todense", _no_todense)
    fixed = _bond_phase_fix_envs(scrambled, envs, fams)
    monkeypatch.undo()
    assert _env_dist(envs, fixed) < 1e-12


def test_bond_phase_is_refused_on_split_ctm():
    """The split forwards never read ``forward_gauge``; accepting
    ``bond_phase`` there would silently run another gauge."""
    from tenax.algorithms.ipeps_ad_policy import validate_split_ctm_config
    from tenax.algorithms.ipeps_config import CTMConfig

    cfg = CTMConfig(chi=4, fuse_virtual_legs=False, forward_gauge="bond_phase")
    for recipe in ("1x1", "2x2"):
        with pytest.raises(NotImplementedError, match="bond_phase"):
            validate_split_ctm_config(cfg, recipe)
    # Regime: the same split config with the default gauge is accepted.
    validate_split_ctm_config(
        CTMConfig(chi=4, fuse_virtual_legs=False, forward_gauge="phase"), "2x2"
    )
