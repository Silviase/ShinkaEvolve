from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

Shipments = List[float]


@dataclass
class LogisticsState:
    """State visible to the evolved logistics policy at each time step."""

    # Time
    t: int # Current time step
    T: int # Total time horizon

    # Nodes and edges
    num_nodes: int
    edges: List[Tuple[int, int]]

    # Node attributes
    inventory: List[float]
    demand: List[float]
    is_supply_node: List[bool]
    is_demand_node: List[bool]

    # Edge attributes
    edge_capacity: List[float]
    edge_cost: List[float]
    edge_available: List[bool]
    edge_risk: List[float]

    # Global weights
    cost_weight: float
    risk_weight: float
    supply_reward_weight: float
