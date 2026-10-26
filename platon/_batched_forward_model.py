"""Bounded-memory batches that preserve the tuned scalar forward fusion."""
import jax
from jax import lax

from . import _forward_model as fm


def make_forward(cfg, dual=True, with_vmrs=False):
    """Map scalar-shaped limbs, reusing opacity/tau storage between lanes.

    Only small packed inputs and binned outputs scale with batch size. A vmap
    changes the opacity fusion and dot reductions, so retain the scalar core:
    the fused cloudy/clear one for partial clouds (dual), otherwise the
    single-terminator one, as TransitDepthCalculator.compute_depths chooses.
    with_vmrs maps (packed, vmrs) lanes, for free VMRs.
    """
    core = fm._transit_depths_dual if dual else fm._transit_depths_only
    if with_vmrs:
        return jax.jit(lambda data, packed, vmrs: lax.map(
            lambda lane: core(cfg, data, fm.ForwardInputs(lane[0], lane[1])),
            (packed, vmrs)))
    return jax.jit(lambda data, packed: lax.map(
        lambda p: core(cfg, data, fm.ForwardInputs(p)), packed))
