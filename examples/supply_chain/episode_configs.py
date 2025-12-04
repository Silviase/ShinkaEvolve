from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class EpisodeConfig:
    """Configuration for a single evaluation episode."""

    level: int
    horizon: int
    num_nodes: int
    edges: List[Tuple[int, int]]
    supply_nodes: List[int]
    demand_nodes: List[int]
    initial_inventory: List[float]
    demand_series: List[List[float]]
    edge_capacity: List[float]
    edge_cost: List[float]
    availability_series: List[List[bool]]
    risk_series: List[List[float]]
    loss_penalty: List[float]
    supply_reward_weight: float = 1.0
    cost_weight: float = 0.1
    risk_weight: float = 0.0
    cvar_lambda: float = 0.0
    cvar_tau: float = 0.2
    supply_injection: Optional[List[List[float]]] = None


__all__ = ["EpisodeConfig"]
