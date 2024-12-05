# Kernels based on Pyro https://github.com/pyro-ppl/pyro/tree/dev/pyro/contrib/gp/kernels
import abc

import jax.numpy as jnp
from jaxtyping import Float, Array
import chex

KernelOperand = Float[Array, "num_dims"]
CovarianceMatrix = Float[Array, "num_dims num_dims"]


class Kernel(abc.ABC):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def __call__(self, x1: KernelOperand, x2: KernelOperand) -> CovarianceMatrix: ...

    def __add__(self, other: "Kernel") -> "Kernel":
        class KernelSum(Kernel):
            def __call__(_self, x1: KernelOperand, x2: KernelOperand) -> CovarianceMatrix:
                return self(x1, x2) + other(x1, x2)

            def __repr__(_self) -> str:
                return f"{self.__repr__()} + {other.__repr__()}"

        return KernelSum()

    def __sub__(self, other: "Kernel") -> "Kernel":
        class KernelDifference(Kernel):
            def __call__(_self, x1: KernelOperand, x2: KernelOperand) -> CovarianceMatrix:
                return self(x1, x2) - other(x1, x2)

            def __repr__(_self) -> str:
                return f"{self.__repr__()} - {other.__repr__()}"

        return KernelDifference()

    def __mul__(self, other: "Kernel") -> "Kernel":
        class KernelProduct(Kernel):
            def __call__(_self, x1: KernelOperand, x2: KernelOperand) -> CovarianceMatrix:
                return self(x1, x2) * other(x1, x2)

            def __repr__(_self) -> str:
                return f"{self.__repr__()} * {other.__repr__()}"

        return KernelProduct()

    def __truediv__(self, other: "Kernel") -> "Kernel":
        class KernelDivision(Kernel):
            def __call__(_self, x1: KernelOperand, x2: KernelOperand) -> CovarianceMatrix:
                return self(x1, x2) / (other(x1, x2) + 1e-8)

            def __repr__(_self) -> str:
                return f"{self.__repr__()} / {other.__repr__()}"

        return KernelDivision()


class Matern(Kernel):
    """A Matern v=3/2 Kernel"""

    # TODO: Implement Matern v=1/2 and v=5/2 kernels

    variance: Float[Array, " #num_dims"]
    lengthscale: Float[Array, " #num_dims"]

    def __init__(
        self,
        variance: Float[Array, " #num_dims"] | None = None,
        lengthscale: Float[Array, " #num_dims"] | None = None,
    ) -> None:
        self.variance = variance or jnp.array([1.0])
        self.lengthscale = lengthscale or jnp.array([1.0])

    def __call__(self, x1: KernelOperand, x2: KernelOperand) -> CovarianceMatrix:
        chex.assert_equal_size(x1, x2)

        dist = jnp.sqrt((x1[:, None] - x2) ** 2)
        sqrt3_d_rho = (jnp.sqrt(3) * dist) / (self.lengthscale + 1e-8)
        return self.variance * (1 + sqrt3_d_rho) * jnp.exp(-sqrt3_d_rho)


class Linear(Kernel):
    variance: Float[Array, " #num_dims"]

    def __init__(
        self,
        variance: Float[Array, " #num_dims"] | None = None,
    ) -> None:
        self.variance = variance or jnp.array([1.0])

    def __call__(self, x1: KernelOperand, x2: KernelOperand) -> CovarianceMatrix:
        chex.assert_equal_size(x1, x2)

        return self.variance * (x1[:, None] @ x2[:, None].T)
