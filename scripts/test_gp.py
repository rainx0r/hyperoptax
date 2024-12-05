from hyperoptax.models.gp import gaussian_process_regression
from hyperoptax.models.kernels import Matern, Linear

import jax.numpy as jnp
import jax


def main() -> None:
    lr = 0.01  # Learning rate
    num_x = 7

    key, fun_key = jax.random.split(jax.random.PRNGKey(42))
    key, x_key = jax.random.split(key)

    def y_fun(x):
        return jnp.sin(x) + 0.1 * jax.random.normal(fun_key, shape=(x.shape[0], 1))

    x = (jax.random.uniform(x_key, shape=(7, 1)) * 4.0) + 1
    y = y_fun(x)
    xtest = jnp.linspace(0, 6.0, 200)[:, None]

    key, gp_train_key = jax.random.split(key)
    gp = gaussian_process_regression(
        kernel=lambda x: Matern(parent=x) + Linear(parent=x),
        x=x,
        y=y,
        train_steps=1000,
        learning_rate=lr,
        key=gp_train_key,
    )

    print(gp.params)
    # mu, var = gp.predict_fn(gp.params, x, y, xtest)


if __name__ == "__main__":
    main()
