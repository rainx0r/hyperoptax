# Kernels based on Pyro https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/gp/kernels
import abc

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

KernelOperand = Float[Array, "num_dims"]
KernelOutput = Float[Array, ""]


class Kernel(abc.ABC, nn.Module):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    @nn.compact
    def __call__(self, x1: KernelOperand, x2: KernelOperand) -> KernelOutput: ...

    def __add__(self, other: "Kernel") -> "Kernel":
        class KernelSum(Kernel):
            @nn.compact
            def __call__(_self, x1: KernelOperand, x2: KernelOperand) -> KernelOutput:
                return self(x1, x2) + other(x1, x2)

            def __repr__(_self) -> str:
                return f"{self.__repr__()} + {other.__repr__()}"

        return KernelSum()

    def __sub__(self, other: "Kernel") -> "Kernel":
        class KernelDifference(Kernel):
            @nn.compact
            def __call__(_self, x1: KernelOperand, x2: KernelOperand) -> KernelOutput:
                return self(x1, x2) - other(x1, x2)

            def __repr__(_self) -> str:
                return f"{self.__repr__()} - {other.__repr__()}"

        return KernelDifference()

    def __mul__(self, other: "Kernel") -> "Kernel":
        class KernelProduct(Kernel):
            @nn.compact
            def __call__(_self, x1: KernelOperand, x2: KernelOperand) -> KernelOutput:
                return self(x1, x2) * other(x1, x2)

            def __repr__(_self) -> str:
                return f"{self.__repr__()} * {other.__repr__()}"

        return KernelProduct()

    def __truediv__(self, other: "Kernel") -> "Kernel":
        class KernelDivision(Kernel):
            @nn.compact
            def __call__(_self, x1: KernelOperand, x2: KernelOperand) -> KernelOutput:
                return self(x1, x2) / (other(x1, x2) + 1e-8)

            def __repr__(_self) -> str:
                return f"{self.__repr__()} / {other.__repr__()}"

        return KernelDivision()


class Matern(Kernel):
    """A Matern v=3/2 Kernel"""

    # TODO: Implement Matern v=1/2 and v=5/2 kernels

    isotropic: bool = True
    init_variance: jax.nn.initializers.Initializer = jax.nn.initializers.constant(1.0)
    init_lengthscale: jax.nn.initializers.Initializer = jax.nn.initializers.constant(
        0.0
    )

    @nn.compact
    def __call__(self, x1: KernelOperand, x2: KernelOperand) -> KernelOutput:
        chex.assert_equal_size(x1, x2)

        param_shape = (x1.shape[-1],) if self.isotropic else (1,)

        ls = jax.nn.softplus(
            self.param("lengthscale", self.init_lengthscale, param_shape)
        )
        var = jax.nn.softplus(self.param("variance", self.init_variance, (1,)))

        x1 /= ls + 1e-8
        x2 /= ls + 1e-8

        # ||x - y||^2 = ||x||^2 - 2 <x,y> + ||y||^2
        dist = (
            jnp.sum(x1**2, axis=-1)
            - 2 * jnp.einsum("...n,...n->...", x1, x2)
            + jnp.sum(x2**2, axis=-1)
        )

        sqrt3_d = jnp.sqrt(3) * dist
        return var * (1 + sqrt3_d) * jnp.exp(-sqrt3_d)


class Linear(Kernel):
    variance: Float[Array, " #num_dims"]

    @nn.compact
    def __call__(self, x1: KernelOperand, x2: KernelOperand) -> KernelOutput:
        chex.assert_equal_size(x1, x2)

        var = self.param("variance", self.init_variance, (1,))

        return var * jnp.einsum("...n,...n->...", x1, x2)
