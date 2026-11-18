from typing import TypeVar

import equinox as eqx
import jax

Params = TypeVar("Params")


@eqx.filter_jit
def genetic_soft_update(
    rl_params: Params, elite_params: Params, tau: float
) -> Params:
    """Blend elite parameters into rl_params: tau * elite + (1 - tau) * rl."""
    rl_arrays, rl_static = eqx.partition(rl_params, eqx.is_array)
    elite_arrays, _ = eqx.partition(elite_params, eqx.is_array)
    merged = jax.tree.map(
        lambda rl, el: tau * el + (1.0 - tau) * rl, rl_arrays, elite_arrays
    )
    return eqx.combine(merged, rl_static)
