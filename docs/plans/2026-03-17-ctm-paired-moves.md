# CTM Paired-Move Sweep Design

**Date**: 2026-03-17
**Status**: Draft

## Problem

The general 4-move CTM sweep (`_ctm_tensor_sweep`) computes independent
projectors for each direction (left, right, top, bottom). For
SymmetricTensor with non-trivial charges, each projector independently
splits chi across charge sectors, leading to incompatible block sizes
between environment tensors from different moves after 2-3 sweeps.

Attempted fixes (SVD corner, flow flip) address symptoms but not the
root cause: **chi legs from different projectors have different charge
distributions**.

## Solution: Paired Horizontal/Vertical Moves

Following YASTN's approach (`_env_ctm.py`, `moves='hv'`):

Replace 4 independent moves with 2 paired moves:
- **Horizontal move**: updates T4 (left edge), T2 (right edge), and
  all 4 corners, using ONE projector pair
- **Vertical move**: updates T1 (top edge), T3 (bottom edge), and
  all 4 corners, using ONE projector pair

Each paired move ensures all updated chi legs come from the SAME
projector pair, eliminating charge-distribution divergence.

## YASTN's Algorithm (Reference)

### Step 1: Build 2×2 enlarged corners

For a horizontal move, build 4 enlarged corners from the current
environment:

```
cor_tl = T_left · C_tl · T_top · ψ     (top-left 2×2 cluster)
cor_bl = T_bot  · C_bl · T_left · ψ     (bottom-left 2×2 cluster)
cor_tr = T_top  · C_tr · T_right · ψ    (top-right 2×2 cluster)
cor_br = T_right · C_br · T_bot · ψ     (bottom-right 2×2 cluster)
```

Each enlarged corner is a rank-2 tensor after fusing pairs of legs.

### Step 2: Compute paired projectors

For the LEFT side: merge cor_tl and cor_bl, SVD to get projectors.

```python
rr = cor_tl @ cor_bl.T     # half-system density matrix
U, S, V = svd(rr)          # one SVD for both projectors
p_top = cor_bl @ (S^{-1/2} V)†   # projector for top
p_bot = cor_tl @ (U S^{-1/2})†   # projector for bottom
```

For the RIGHT side: same with cor_tr and cor_br.

Key: the left and right projector PAIRS share the same chi charge
distribution because they come from the same SVD.

### Step 3: Update environment

Apply projectors to update edges and corners:

```python
# Left edge update: T4_new = p_top† · T4 · ψ · p_bot
# Right edge update: T2_new = p_top† · T2 · ψ · p_bot
# Corner updates: C1_new = p† · (C1 · T1), etc.
```

### Step 4: Vertical move (analogous)

Same structure but for top/bottom directions.

## Mapping to Tenax

### New function: `_ctm_tensor_move_horizontal`

```python
def _ctm_tensor_move_horizontal(
    env_self: CTMTensorEnv,
    env_left_nb: CTMTensorEnv,
    env_right_nb: CTMTensorEnv,
    a_left: Tensor,      # double-layer for left neighbor
    a_right: Tensor,     # double-layer for right neighbor
    chi: int,
    projector_method: str = "eigh",
) -> CTMTensorEnv:
```

Steps:
1. Build 4 enlarged corners (2×2 clusters):
   - `cor_tl`: `env.T4 · env.C1 · env_nb.T1 · a`
   - `cor_bl`: `env_nb.T3 · env.C4 · env.T4 · a`
   - `cor_tr`: `env_nb.T1 · env.C2 · env.T2 · a`
   - `cor_br`: `env.T2 · env.C3 · env_nb.T3 · a`
2. Fuse each to rank-2: `(fused_top, fused_bot)`
3. Compute left projectors: `proj_corners(cor_tl, cor_bl)` → `(P_left_top, P_left_bot)`
4. Compute right projectors: `proj_corners(cor_tr, cor_br)` → `(P_right_top, P_right_bot)`
5. Update T4: sandwich with left projectors + a_left
6. Update T2: sandwich with right projectors + a_right
7. Update corners C1, C2, C3, C4 using appropriate projectors
8. Return updated env

### New function: `_ctm_tensor_move_vertical`

Analogous for top/bottom.

### Modified sweep

```python
def _ctm_tensor_sweep_paired(env, a, chi, renormalize, projector_method):
    """One sweep using paired horizontal + vertical moves."""
    env = _ctm_tensor_move_horizontal(env, env, env, a, a, chi, projector_method)
    env = _ctm_tensor_move_vertical(env, env, env, a, a, chi, projector_method)
    if renormalize:
        env = _renormalize_tensor_env(env)
    return env
```

For multisite:
```python
def _ctm_tensor_sweep_multisite_paired(envs, double_layers, neighbors, chi, ...):
    # Horizontal: for each site, get left and right neighbors
    for coord in sorted(envs.keys()):
        nb_left = neighbors[coord]["left"]
        nb_right = neighbors[coord]["right"]
        envs[coord] = _ctm_tensor_move_horizontal(
            envs[coord], envs[nb_left], envs[nb_right],
            double_layers[nb_left], double_layers[nb_right], chi, ...
        )
    # Vertical: analogous
    for coord in sorted(envs.keys()):
        nb_top = neighbors[coord]["top"]
        nb_bottom = neighbors[coord]["bottom"]
        envs[coord] = _ctm_tensor_move_vertical(
            envs[coord], envs[nb_top], envs[nb_bottom],
            double_layers[nb_top], double_layers[nb_bottom], chi, ...
        )
    return envs
```

### proj_corners implementation

```python
def _proj_corners_tensor(
    cor_top: Tensor,   # (fused_top, col)
    cor_bot: Tensor,   # (fused_bot, col)  — "col" shared
    chi: int,
    projector_method: str = "eigh",
) -> tuple[Tensor, Tensor]:
    """Compute paired projectors from two half-system corners.

    Returns (P_top, P_bot) where both have the same chi charge
    distribution on their output leg.
    """
    # Contract: rr = cor_top @ cor_bot.T on "col" leg
    # SVD: rr = U S V†
    # P_top = cor_bot @ (S^{-1/2} V)†
    # P_bot = cor_top @ (U S^{-1/2})†
```

## Corner Updates

The critical change: corners are updated using projectors from the
SAME paired computation, not from independent moves.

For the horizontal move:
- `C1_new = P_left_bot† · (C1 · T1_nb)` — fused, then projected
- `C4_new = P_left_top† · (T3_nb · C4)` — fused, then projected
- `C2_new = P_right_bot† · (T1_nb · C2)` — fused, then projected
- `C3_new = P_right_top† · (C3 · T3_nb)` — fused, then projected

Note: C1 and C4 share left projectors; C2 and C3 share right
projectors. The projectors' chi legs are all from the same SVDs.

## File Changes

| File | Change |
|------|--------|
| Create `_ctm_tensor_paired_moves.py` | Paired horizontal/vertical moves + proj_corners |
| Modify `_ctm_tensor_convergence.py` | Add `_ctm_tensor_sweep_paired`, wire into `ctm_tensor()` for SymmetricTensor |
| Create `tests/test_ctm_paired.py` | Tests for paired moves |
| Keep `_ctm_tensor_moves.py` | Unchanged — existing 4-move sweep still used for DenseTensor (backward compat) |

## Testing

1. Dense: paired sweep energy matches 4-move sweep energy
2. U(1) SymmetricTensor: converges without charge divergence
3. FermionParity: converges without densify workaround
4. 2-site checkerboard: works with SymmetricTensor
5. Energy matches: paired SymmetricTensor ≈ DenseTensor result

## Implementation Order

1. `_proj_corners_tensor` — paired projector computation
2. `_ctm_tensor_move_horizontal` — horizontal paired move
3. `_ctm_tensor_move_vertical` — vertical paired move
4. `_ctm_tensor_sweep_paired` — new sweep function
5. Wire into `ctm_tensor()` — use paired sweep for SymmetricTensor
6. Remove fermionic densify workaround
7. Tests
