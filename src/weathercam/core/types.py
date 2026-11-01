import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        values = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"Box coordinates must be finite, got {values}")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Box requires x2 >= x1 and y2 >= y1, got {values}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    label: int
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.label < 0:
            raise ValueError(f"Detection label must be >= 0, got {self.label}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Detection score must be in [0, 1], got {self.score}")
