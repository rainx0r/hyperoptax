from functools import partial

import distrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


@partial(jax.jit, static_argnames=("exploration_bias",))
def expected_improvement(
    mu: Float[Array, "batch num_dims"],
    var: Float[Array, "batch num_dims"],
    best_mu: Float[Array, "batch num_dims"],
    exploration_bias: float = 0.5,
) -> Float[Array, "batch num_dims"]:
    prior = distrax.Normal(0, 1)
    sigma = jnp.sqrt(var)

    imp = mu - best_mu - exploration_bias
    z = imp / (sigma + 1e-8)

    ei = imp * prior.cdf(z) + sigma * prior.prob(z)
    ei[sigma == 0] = 0.0
    return ei


@partial(jax.jit, static_argnames=("exploration_bias",))
def probability_of_improvement(
    mu: Float[Array, "batch num_dims"],
    var: Float[Array, "batch num_dims"],
    best_mu: Float[Array, "batch num_dims"],
    exploration_bias: float = 0.5,
) -> Float[Array, "batch num_dims"]:
    prior = distrax.Normal(0, 1)
    sigma = jnp.sqrt(var)

    imp = mu - best_mu - exploration_bias
    z = imp / (sigma + 1e-8)

    return jnp.array(prior.cdf(z), dtype=jnp.float32)
