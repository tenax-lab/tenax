"""#976: the legacy ``ctm`` / ``ctm_2site`` / ``ctm_split`` ignored
``CTMConfig.min_iter`` and could declare convergence after 2 sweeps against a
``min_iter=30``.  #925: ``ctm`` excluded QR warm-up sweeps from ``n_iter``,
violating ``CTMConvergenceInfo``'s own invariant (``n_iter == max_iter`` when
not converged).

These are mechanism tests, not convergence tests: they run on a 1x1 product
state whose corner spectrum is stationary after the first sweep, so ``diff`` is
deterministically below any tolerance and the only thing under test is the
*stopping rule* -- when ``min_iter`` releases the loop and how the warm-up is
counted.  No physical convergence is asserted (that is for benchmarks).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tenax import CTMConfig, ctm, ctm_2site, ctm_split

# A 1x1 product state: chi=1 corners are stationary from the second sweep, so
# ``diff < conv_tol`` holds immediately and the stopping rule is all that moves.
A = jnp.ones((1, 1, 1, 1, 2))


def _info_ctm(cfg):
    return ctm(A, cfg, return_meta=True)[1]


def _info_ctm_2site(cfg):
    return ctm_2site(A, A, cfg, return_meta=True)[2]


def _info_ctm_split(cfg):
    return ctm_split(A, cfg, return_meta=True)[1]


_ENTRY_POINTS = [
    pytest.param(_info_ctm, id="ctm"),
    pytest.param(_info_ctm_2site, id="ctm_2site"),
    pytest.param(_info_ctm_split, id="ctm_split"),
]


@pytest.mark.parametrize("info_fn", _ENTRY_POINTS)
@pytest.mark.parametrize("min_iter", [2, 10, 30])
def test_convergence_waits_for_min_iter(info_fn, min_iter):
    """#976: an already-stationary state is not certified converged until the
    minimum sweep count is reached -- so ``n_iter`` equals ``min_iter`` exactly,
    not 2.
    """
    info = info_fn(CTMConfig(chi=1, max_iter=40, min_iter=min_iter))
    assert bool(info.converged)
    assert int(info.n_iter) == min_iter


@pytest.mark.parametrize("info_fn", _ENTRY_POINTS)
def test_min_iter_is_capped_at_max_iter(info_fn):
    """The cap: ``min_iter`` above ``max_iter`` must not make a state
    unconvergeable -- it converges at ``max_iter`` instead of flipping to
    ``converged=False`` (the alternative rejected in #976's review)."""
    info = info_fn(CTMConfig(chi=1, max_iter=5, min_iter=30))
    assert bool(info.converged)
    assert int(info.n_iter) == 5  # min(30, 5)


def test_ctm_counts_qr_warmup_in_n_iter():
    """#925: warm-up sweeps count toward ``n_iter``.  With ``qr_warmup_steps=3,
    max_iter=3`` the post-warm-up loop runs zero sweeps, but three real sweeps
    happened -- ``n_iter`` must be 3 (was 0), satisfying ``n_iter == max_iter``
    when not converged.
    """
    info = _info_ctm(
        CTMConfig(
            chi=1, max_iter=3, min_iter=1, projector_method="qr", qr_warmup_steps=3
        )
    )
    assert int(info.n_iter) == 3
    assert not bool(info.converged)


def test_ctm_warmup_counts_toward_min_iter():
    """#925/#976 together: with a warm-up, ``min_iter`` is measured from the
    first sweep (warm-up included), so ``qr_warmup_steps=6`` + 2 post-warm-up
    sweeps reports ``n_iter=8`` and stops there rather than doing 10 more.
    """
    info = _info_ctm(
        CTMConfig(
            chi=1,
            max_iter=40,
            min_iter=8,
            projector_method="qr",
            qr_warmup_steps=6,
        )
    )
    assert bool(info.converged)
    assert int(info.n_iter) == 8
