import abc
from collections.abc import Callable
from typing import Any

from flax import struct

Observation = Any
Result = Any


class Optimizer(abc.ABC, struct.PyTreeNode):
    @staticmethod
    @abc.abstractmethod
    def initialize(initial_data: list[tuple[Observation, Result]]) -> "Optimizer": ...

    @abc.abstractmethod
    def suggest(self) -> Observation: ...

    @abc.abstractmethod
    def observe(self, observation: Observation, result: Result): ...

    @abc.abstractmethod
    def recommend(self) -> Observation: ...


class Tuner(abc.ABC):
    optimizer: Optimizer

    @abc.abstractmethod
    def __init__(self): ...

    @abc.abstractmethod
    def tune(self, objective_fn: Callable[[Observation], float]) -> Observation: ...
