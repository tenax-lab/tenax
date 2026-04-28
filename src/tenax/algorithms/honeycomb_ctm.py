"""Public honeycomb iPEPS CTM API — explicit named re-exports.

Internal code should import from the concrete modules:

* ``_ctm_honeycomb_ad`` — :func:`honeycomb_ctm_energy_implicit`
* ``_ctm_honeycomb_forward`` — :func:`honeycomb_ctm_run`
* ``_ctm_honeycomb_env`` — :class:`HoneycombCTMEnv`
* ``_ctm_honeycomb_init`` — :func:`initialize_honeycomb_env`,
  :func:`_double_layer_honeycomb`
* ``_ctm_honeycomb_topology`` — :data:`HONEYCOMB_NEIGHBORS`,
  :data:`HONEYCOMB_DIRECTIONS`
* ``_ctm_honeycomb_energy`` — :func:`compute_honeycomb_energy`,
  :func:`compute_honeycomb_triangle_energy`

This shim re-exports for downstream / notebook usage so callers can write
``from tenax.algorithms.honeycomb_ctm import honeycomb_ctm_energy_implicit``.
"""

from tenax.algorithms._ctm_honeycomb_ad import (
    honeycomb_ctm_energy_implicit as honeycomb_ctm_energy_implicit,
)
from tenax.algorithms._ctm_honeycomb_energy import (
    compute_honeycomb_energy as compute_honeycomb_energy,
)
from tenax.algorithms._ctm_honeycomb_energy import (
    compute_honeycomb_triangle_energy as compute_honeycomb_triangle_energy,
)
from tenax.algorithms._ctm_honeycomb_env import (
    HoneycombCTMEnv as HoneycombCTMEnv,
)
from tenax.algorithms._ctm_honeycomb_forward import (
    HoneycombConvergeInfo as HoneycombConvergeInfo,
)
from tenax.algorithms._ctm_honeycomb_forward import (
    honeycomb_ctm_run as honeycomb_ctm_run,
)
from tenax.algorithms._ctm_honeycomb_init import (
    initialize_honeycomb_env as initialize_honeycomb_env,
)
from tenax.algorithms._ctm_honeycomb_topology import (
    HONEYCOMB_DIRECTIONS as HONEYCOMB_DIRECTIONS,
)
from tenax.algorithms._ctm_honeycomb_topology import (
    HONEYCOMB_NEIGHBORS as HONEYCOMB_NEIGHBORS,
)
