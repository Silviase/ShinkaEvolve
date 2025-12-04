from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Sequence

from examples.supply_chain.episode_configs import EpisodeConfig
from examples.supply_chain.logistics_state import LogisticsState
from examples.supply_chain.utils import normalize_shipments, zero_matrix


def validate_config(config: EpisodeConfig) -> None:
    edge_count = len(config.edges)
    horizon = config.horizon
    if len(config.edge_capacity) != edge_count:
        raise ValueError("edge_capacity length mismatch")
    if len(config.edge_cost) != edge_count:
        raise ValueError("edge_cost length mismatch")
    if len(config.loss_penalty) != edge_count:
        raise ValueError("loss_penalty length mismatch")
    if len(config.demand_series) != horizon:
        raise ValueError("demand_series horizon mismatch")
    if len(config.availability_series) != horizon:
        raise ValueError("availability_series horizon mismatch")
    if len(config.risk_series) != horizon:
        raise ValueError("risk_series horizon mismatch")
    if config.supply_injection and len(config.supply_injection) != horizon:
        raise ValueError("supply_injection horizon mismatch")
    for t in range(horizon):
        if len(config.demand_series[t]) != config.num_nodes:
            raise ValueError("demand vector length mismatch")
        if len(config.availability_series[t]) != edge_count:
            raise ValueError("availability length mismatch")
        if len(config.risk_series[t]) != edge_count:
            raise ValueError("risk length mismatch")
        if config.supply_injection and len(config.supply_injection[t]) != config.num_nodes:
            raise ValueError("supply injection node length mismatch")


def simulate_episode(
    config: EpisodeConfig,
    policy_fn: Callable[[LogisticsState], Sequence[float]],
    seed: int = 0,
    return_history: bool = False,
) -> Dict[str, Any]:
    """Runs a single logistics episode and returns detailed metrics."""
    validate_config(config)
    rng = random.Random(seed)
    inventory = [float(x) for x in config.initial_inventory]
    horizon = config.horizon
    edge_count = len(config.edges)

    supply_injection = (
        config.supply_injection
        if config.supply_injection is not None
        else zero_matrix(horizon, config.num_nodes)
    )

    total_fulfilled = 0.0
    total_cost = 0.0
    total_loss = 0.0
    total_demand = 0.0
    ideal_supply_reward = 0.0

    per_step_fulfilled: List[float] = []
    per_step_cost: List[float] = []
    per_step_loss: List[float] = []
    history: List[Dict[str, Any]] = []

    for t in range(horizon):
        for node_idx in range(config.num_nodes):
            inventory[node_idx] += supply_injection[t][node_idx]

        inventory_start = list(inventory)

        state = LogisticsState(
            t=t,
            T=horizon,
            num_nodes=config.num_nodes,
            edges=config.edges,
            inventory=list(inventory),
            demand=list(config.demand_series[t]),
            is_supply_node=[i in config.supply_nodes for i in range(config.num_nodes)],
            is_demand_node=[i in config.demand_nodes for i in range(config.num_nodes)],
            edge_capacity=list(config.edge_capacity),
            edge_cost=list(config.edge_cost),
            edge_available=list(config.availability_series[t]),
            edge_risk=list(config.risk_series[t]),
            cost_weight=config.cost_weight,
            risk_weight=config.risk_weight,
            supply_reward_weight=config.supply_reward_weight,
        )

        proposed_shipments = policy_fn(state)
        shipments = normalize_shipments(proposed_shipments, edge_count)

        arrivals = [0.0 for _ in range(config.num_nodes)]
        step_cost = 0.0
        step_loss = 0.0
        feasible_shipments = [0.0 for _ in range(edge_count)]
        lost_per_edge = [0.0 for _ in range(edge_count)]

        for e_idx, amount in enumerate(shipments):
            src, dst = config.edges[e_idx]
            if not config.availability_series[t][e_idx]:
                continue
            feasible_amt = min(
                amount,
                config.edge_capacity[e_idx],
                max(0.0, inventory[src]),
            )
            feasible_amt = max(0.0, feasible_amt)
            inventory[src] -= feasible_amt
            step_cost += feasible_amt * config.edge_cost[e_idx]
            feasible_shipments[e_idx] = feasible_amt

            risk = config.risk_series[t][e_idx]
            if risk > 0 and rng.random() < risk:
                lost = feasible_amt
                arrived = 0.0
            else:
                lost = 0.0
                arrived = feasible_amt

            step_loss += lost * config.loss_penalty[e_idx]
            lost_per_edge[e_idx] = lost
            arrivals[dst] += arrived

        demand_vec = config.demand_series[t]
        fulfilled_this_step = 0.0
        fulfilled_per_node = [0.0 for _ in range(config.num_nodes)]
        for node_idx in range(config.num_nodes):
            total_available = inventory[node_idx] + arrivals[node_idx]
            demand_here = demand_vec[node_idx]
            served = min(demand_here, total_available)
            fulfilled_this_step += served
            total_demand += demand_here
            ideal_supply_reward += config.supply_reward_weight * demand_here
            inventory[node_idx] = total_available - served
            fulfilled_per_node[node_idx] = served

        total_fulfilled += fulfilled_this_step
        total_cost += step_cost
        total_loss += step_loss

        per_step_fulfilled.append(fulfilled_this_step)
        per_step_cost.append(step_cost)
        per_step_loss.append(step_loss)

        if return_history:
            history.append(
                {
                    "t": t,
                    "inventory_start": inventory_start,
                    "inventory_end": list(inventory),
                    "demand": list(demand_vec),
                    "shipments": feasible_shipments,
                    "arrivals": arrivals,
                    "edge_available": list(config.availability_series[t]),
                    "edge_risk": list(config.risk_series[t]),
                    "fulfilled": fulfilled_per_node,
                    "lost": lost_per_edge,
                }
            )

    score = (
        config.supply_reward_weight * total_fulfilled
        - config.cost_weight * total_cost
        - config.risk_weight * total_loss
    )

    z_value = config.cost_weight * total_cost + config.risk_weight * total_loss
    fulfillment_rate = total_fulfilled / max(1e-6, total_demand)
    ideal_supply_reward = max(ideal_supply_reward, 1e-6)
    normalized_score = max(
        0.0,
        fulfillment_rate - (config.cost_weight * total_cost + config.risk_weight * total_loss) / ideal_supply_reward,
    )
    normalized_z_value = z_value / ideal_supply_reward

    result = {
        "level": config.level,
        "seed": seed,
        "score": float(score),
        "normalized_score": float(normalized_score),
        "fulfilled": float(total_fulfilled),
        "cost": float(total_cost),
        "loss": float(total_loss),
        "fulfillment_rate": float(fulfillment_rate),
        "z_value": float(z_value),
        "normalized_z_value": float(normalized_z_value),
        "cvar_lambda": float(config.cvar_lambda),
        "cvar_tau": float(config.cvar_tau),
        "total_demand": float(total_demand),
        "per_step_fulfilled": per_step_fulfilled,
        "per_step_cost": per_step_cost,
        "per_step_loss": per_step_loss,
        "final_inventory": inventory,
    }
    if return_history:
        result["history"] = history
    return result
