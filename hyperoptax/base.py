import abc
from collections.abc import Callable
from typing import Any

from flax import struct

Suggestion = Any
Result = Any


class Optimizer(abc.ABC, struct.PyTreeNode):
    @staticmethod
    @abc.abstractmethod
    def initialize(initial_data: list[tuple[Suggestion, Result]]) -> "Optimizer": ...

    @abc.abstractmethod
    def suggest(self) -> Suggestion: ...

    @abc.abstractmethod
    def observe(self, suggestion: Suggestion, result: Result) -> "Optimizer": ...

    @abc.abstractmethod
    def recommend(self) -> Suggestion: ...


class Tuner(abc.ABC):
    optimizer: Optimizer

    @abc.abstractmethod
    def __init__(self): ...

    @abc.abstractmethod
    def tune(self, objective_fn: Callable[[Suggestion], float]) -> Suggestion: ...
