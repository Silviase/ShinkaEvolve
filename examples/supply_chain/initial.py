from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.supply_chain.episode_configs import EpisodeConfig
from examples.supply_chain.logistics_state import LogisticsState, Shipments
from examples.supply_chain.config_loader import (
    generate_episode_config,
    generate_episode_config_from_file,
)
from examples.supply_chain.simulation import simulate_episode
# Helpers exposed to the evolved policy author:
# - generate_episode_config(level, seed) -> EpisodeConfig
# - simulate_episode(config, policy_fn, seed=0) -> metrics dict


# EVOLVE-BLOCK-START
def logistics_policy(state: LogisticsState) -> Shipments:
    """
    A simple heuristic logistics policy.

    The policy tries to satisfy immediate demand while building a small buffer
    at downstream nodes when there is time remaining. Edges are prioritized by
    (demand / (1 + cost + risk)) so lower-cost, safer routes are chosen first.
    """

    num_edges = len(state.edges)
    if num_edges == 0:
        return []

    shipments: Shipments = [0.0 for _ in range(num_edges)]
    working_inventory = [float(x) for x in state.inventory]

    demand_nodes = [i for i, flag in enumerate(state.is_demand_node) if flag]
    avg_demand = (
        sum(state.demand[i] for i in demand_nodes) / max(1, len(demand_nodes))
    )
    desired_buffer = avg_demand * 0.25 if state.t < state.T - 1 else 0.0

    dest_need = [0.0 for _ in range(state.num_nodes)]
    for idx in range(state.num_nodes):
        base = max(0.0, state.demand[idx])
        if base == 0.0 and not state.is_demand_node[idx]:
            base = max(0.0, desired_buffer - state.inventory[idx] * 0.5)
        dest_need[idx] = base

    edge_priority: List[Tuple[float, int]] = []
    for e_idx, (src, dst) in enumerate(state.edges):
        if not state.edge_available[e_idx]:
            edge_priority.append((0.0, e_idx))
            continue
        need = dest_need[dst]
        penalty = 1.0 + state.edge_cost[e_idx] * state.cost_weight
        penalty += state.edge_risk[e_idx] * max(0.0, state.risk_weight)
        weight = need / penalty if penalty > 0 else need
        edge_priority.append((weight, e_idx))

    edge_priority.sort(key=lambda x: x[0], reverse=True)

    allocated_to_dest = [0.0 for _ in range(state.num_nodes)]
    for _, e_idx in edge_priority:
        if not state.edge_available[e_idx]:
            continue
        src, dst = state.edges[e_idx]

        if working_inventory[src] <= 0:
            continue

        target_need = max(dest_need[dst] - allocated_to_dest[dst], 0.0)
        if target_need <= 0 and not state.is_demand_node[dst]:
            target_need = desired_buffer * 0.5

        if target_need <= 0:
            continue

        max_cap = state.edge_capacity[e_idx]
        amount = min(working_inventory[src], max_cap, target_need)
        if amount <= 0:
            continue

        risk_factor = 1.0 + state.edge_risk[e_idx] * state.risk_weight
        amount = amount / risk_factor

        amount = max(0.0, amount)
        shipments[e_idx] = amount
        working_inventory[src] -= amount
        allocated_to_dest[dst] += amount * (1.0 - state.edge_risk[e_idx])

    return shipments


# EVOLVE-BLOCK-END


def run_experiment(
    level: int = 1,
    seed: int = 0,
    config_path: str | None = None,
) -> Dict[str, Any]:
    """
    Entry point used by evaluate.py.

    Args:
        level: Curriculum level (used when config_path is not provided).
        seed: Random seed.
        config_path: Optional path to a YAML/JSON config under configs/tests for custom scenarios.
    """
    if config_path:
        config: EpisodeConfig = generate_episode_config_from_file(Path(config_path), seed)
    else:
        config = generate_episode_config(level, seed)

    result = simulate_episode(config, logistics_policy, seed=seed)
    result["config_path"] = str(config_path) if config_path else f"level{level}"
    return result
