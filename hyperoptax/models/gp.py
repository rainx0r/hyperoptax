# Based on https://github.com/jax-ml/jax/blob/main/examples/gaussian_process_regression.py

from collections.abc import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray
from kernels import Kernel

from hyperoptax.types import LogDict


def _cov_map(
    kernel: Kernel,
    x1s: Float[Array, "num_x1_pts num_dims"],
    x2s: Float[Array, "num_x2_pts num_dims"] | None = None,
) -> Float[Array, "num_x1_pts num_x2_pts"]:
    if x2s is None:
        return jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(x1s))(x1s)
    else:
        return jax.vmap(lambda x: jax.vmap(lambda y: kernel(x, y))(x1s))(x2s).T


class GaussianProcess(nn.Module):
    kernel: Kernel

    def setup(self):
        self.amp = self.param("amp", nn.zeros_init(), (1, 1))

        # TODO: Init of this should be -5 (?)
        # CARBS has 1.0e-4 for jitter and lengthscale gets init according to the search centre
        self.noise = self.param("noise", nn.zeros_init(), (1, 1))

    def _forward_pass(
        self,
        amp: Float[Array, "#num_pts #num_pts"],
        x: Float[Array, "num_pts num_dims"],
        y: Float[Array, " num_pts"],
    ) -> tuple[tuple[Float[Array, "num_pts num_pts"], bool], Float[Array, " num_pts"]]:
        y -= y.mean()

        noise = jax.nn.softplus(self.noise)

        # TODO: is this as fast as doing one big vmap for all pairs?
        cov = amp * _cov_map(self.kernel, x) + jnp.eye(x.shape[0]) * (noise + 1e-6)
        chol = jax.scipy.linalg.cho_factor(cov, lower=True)
        kinvy = jax.scipy.linalg.cho_solve(chol, y)

        return chol, kinvy

    def __call__(
        self, x: Float[Array, "num_pts num_dims"], y: Float[Array, " num_pts"]
    ) -> Float[Array, " num_pts1"]:
        """Get the marginal likelihood of the function output y at point x"""
        num_pts = x.shape[0]
        amp = jax.nn.softplus(self.amp)

        (chol, _), kinvy = self._forward_pass(amp, x, y)
        log2pi = jnp.log(2.0 * jnp.pi)

        ml = jnp.sum(
            -0.5 * jnp.dot(y.T, kinvy)
            - jnp.sum(jnp.log(jnp.diag(chol)))
            - (num_pts / 2.0) * log2pi
        )
        # Lognormal prior
        ml -= jnp.sum(-0.5 * log2pi - jnp.log(amp) - 0.5 * jnp.log(amp) ** 2)

        return -ml

    def predict(
        self,
        x: Float[Array, "num_pts num_dims"],
        y: Float[Array, "num_pts num_dims"],
        x_new: Float[Array, "num_new_pts num_dims"],
    ) -> distrax.Distribution:
        amp = jax.nn.softplus(self.amp)

        chol, kinvy = self._forward_pass(amp, x, y)

        cross_cov = amp * _cov_map(self.kernel, x, x_new)
        mu = jnp.dot(cross_cov.T, kinvy) + y.mean()

        # TODO: numpyro's example has cross_cov.T @ v, not v.T @ v
        # check which is correct
        v = jax.scipy.linalg.cho_solve(chol, cross_cov)
        var = amp * _cov_map(self.kernel, x_new) - jnp.dot(v.T, v)

        return distrax.MultivariateNormalDiag(
            loc=mu, scale_diag=jnp.sqrt(jnp.diag(var))
        )


class GPTrainState(TrainState):
    predict_fn: Callable[[FrozenDict, jax.Array, jax.Array, jax.Array], jax.Array]


@jax.jit
def _update(
    model: TrainState,
    x: Float[Array, "num_pts num_dims"],
    y: Float[Array, "num_pts num_dims"],
) -> tuple[TrainState, LogDict]:
    def loss_fn(params: FrozenDict, x: jax.Array, y: jax.Array) -> Float[Array, ""]:
        return model.apply_fn(params, x, y)

    loss, grads = jax.value_and_grad(loss_fn)(model.params, x, y)
    model = model.apply_gradients(grads=grads)

    return model, {"gp_loss": loss}


def gaussian_process_regression(
    kernel: Kernel,
    x: Float[Array, "num_pts num_dims"],
    y: Float[Array, "num_pts num_dims"],
    key: PRNGKeyArray,
    train_steps: int = 1000,
    learning_rate: float = 0.01,
):
    model_obj = GaussianProcess(kernel)

    gp_init_key, key = jax.random.split(key)

    model = GPTrainState.create(
        apply_fn=model_obj.apply,
        predict_fn=model_obj.predict,
        params=model_obj.init(gp_init_key, x, y),
        tx=optax.adam(learning_rate=learning_rate),
    )

    for step in range(train_steps):
        model, logs = _update(model, x, y)

        if step % 50 == 0:
            print("Step: %d, logs: %s" % (step, str(logs)))

    return model
