from tenax.algorithms._ctm_tensor_convergence import (
    SINGLE_SITE_NEIGHBORS,
    make_neighbors,
)


def test_make_neighbors_1x1():
    neighbors = make_neighbors(1, 1)
    assert neighbors == SINGLE_SITE_NEIGHBORS


def test_make_neighbors_2x1():
    neighbors = make_neighbors(2, 1)
    assert set(neighbors.keys()) == {(0, 0), (1, 0)}
    assert neighbors[(0, 0)]["right"] == (1, 0)
    assert neighbors[(1, 0)]["right"] == (0, 0)
    # In a 2x1 grid, top/bottom wrap to self (ny=1)
    assert neighbors[(0, 0)]["top"] == (0, 0)
    assert neighbors[(0, 0)]["bottom"] == (0, 0)


def test_make_neighbors_2x2():
    neighbors = make_neighbors(2, 2)
    assert set(neighbors.keys()) == {(0, 0), (1, 0), (0, 1), (1, 1)}
    assert neighbors[(0, 0)]["right"] == (1, 0)
    assert neighbors[(0, 0)]["bottom"] == (0, 1)
    assert neighbors[(1, 1)]["right"] == (0, 1)
    assert neighbors[(1, 1)]["bottom"] == (1, 0)


def test_make_neighbors_3x3():
    neighbors = make_neighbors(3, 3)
    assert len(neighbors) == 9
    assert neighbors[(2, 0)]["right"] == (0, 0)
    assert neighbors[(0, 2)]["bottom"] == (0, 0)
