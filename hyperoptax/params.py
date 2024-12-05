# Heavily based on https://github.com/imbue-ai/carbs/blob/main/carbs/utils.py

import abc
import math
from dataclasses import dataclass
from functools import cached_property
from typing import Generic, TypeVar, override

import numpy as np
import numpy.typing as npt

ParamType = TypeVar("ParamType", int, float)


@dataclass(frozen=True)
class Space(abc.ABC, Generic[ParamType]):
    @abc.abstractmethod
    def to_raw(self, value: ParamType) -> float: ...

    @abc.abstractmethod
    def from_raw(self, value: float) -> ParamType: ...


@dataclass(frozen=True)
class Numerical(Space[int | float]):
    min: float = float("-inf")
    max: float = float("+inf")

    scale: float = 1.0

    integer: bool = False
    """Whether or not to use Integer numbers."""
    rounding_factor: int = 1
    """Unused if integer=False."""

    @cached_property
    def is_bounded(self) -> bool:
        return not math.isinf(self.min) or not math.isinf(self.max)

    @abc.abstractmethod
    def round(self, value: npt.NDArray) -> npt.NDArray: ...

    def __post_init__(self):
        if self.integer:
            object.__setattr__(self, "min", self.min - 0.1)
            object.__setattr__(self, "max", self.max + 0.1)


@dataclass(frozen=True)
class Linear(Numerical):
    @override
    def to_raw(self, value: float | int) -> float:
        return value / self.scale

    @override
    def from_raw(self, value: float, rounded: bool = True) -> float | int:
        value *= self.scale
        if self.integer and rounded:
            value = round(value / self.rounding_factor) * self.rounding_factor
        return value

    @override
    def round(self, value: npt.NDArray) -> npt.NDArray:
        if not self.integer:
            return value

        return (
            np.round(value * self.scale / self.rounding_factor)
            * self.rounding_factor
            / self.scale
        )


@dataclass(frozen=True)
class Log(Numerical):
    min: float = 0.0
    base: int = 2

    @override
    def to_raw(self, value: int | float) -> float:
        if value == 0.0:
            return float("-inf")
        return math.log(value, self.base) / self.scale

    @override
    def from_raw(self, value: float, rounded: bool = True) -> int | float:
        value = self.base ** (value * self.scale)
        if self.integer and rounded:
            value = round(value / self.rounding_factor) * self.rounding_factor
        return value

    @override
    def round(self, value: npt.NDArray) -> npt.NDArray:
        if not self.integer:
            return value

        rounded_value = (
            np.round(self.base ** (value * self.scale) / self.rounding_factor)
            * self.rounding_factor
        )
        return np.log(rounded_value) / self.scale / math.log(self.base)


@dataclass(frozen=True)
class Logit(Numerical):
    min: float = 0.0
    max: float = 1.0

    @override
    def to_raw(self, value: float) -> float:
        if value == 0.0:
            return float("-inf")
        if value == 1.0:
            return float("+inf")
        return math.log10(value / (1 - value) / self.scale)

    @override
    def from_raw(self, value: float) -> float:
        value = 1 / (10 ** (-value * self.scale) + 1)
        return value


@dataclass(frozen=True)
class Param(Generic[ParamType]):
    name: str
    space: Space[ParamType]
    search_center: ParamType
