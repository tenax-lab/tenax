import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest
from _frontier_grad_probe import frontier_energy_and_grad


@pytest.mark.parametrize("path", ["dense", "split"])
def test_frontier_probe_finite(path):
    e, g = frontier_energy_and_grad(path=path, D=2, chi=6, max_iter=15)
    assert np.isfinite(e), e
    assert np.all(np.isfinite(g)), path
    assert g.shape == (2, 2, 2, 2, 2), g.shape


def test_frontier_split_rejects_mesh():
    with pytest.raises(ValueError):
        frontier_energy_and_grad(path="split", D=2, chi=6, device_mesh=object())


def test_frontier_split_rejects_chunk():
    with pytest.raises(ValueError):
        frontier_energy_and_grad(path="split", D=2, chi=6, ctm_chunk_size=4)


def test_frontier_unknown_path():
    with pytest.raises(ValueError):
        frontier_energy_and_grad(path="nope", D=2, chi=6)
