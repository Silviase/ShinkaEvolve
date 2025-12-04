from __future__ import annotations

import math
from typing import List, Sequence


def non_negative(val: float) -> float:
    return float(val) if val > 0 else 0.0


def sinusoidal_series(
    horizon: int, base: float, amplitude: float, phase: float = 0.0
) -> List[float]:
    return [non_negative(base + amplitude * math.sin(phase + t * 0.8)) for t in range(horizon)]


def zero_matrix(h: int, w: int, fill: float = 0.0) -> List[List[float]]:
    return [[fill for _ in range(w)] for _ in range(h)]


def bool_matrix(h: int, w: int, value: bool = True) -> List[List[bool]]:
    return [[value for _ in range(w)] for _ in range(h)]


def normalize_shipments(shipments: Sequence[float], edge_count: int) -> List[float]:
    data = list(shipments)
    if len(data) < edge_count:
        data.extend([0.0] * (edge_count - len(data)))
    return [max(0.0, float(x)) for x in data[:edge_count]]
