from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

# Optional dependency; fall back to JSON if missing.
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

# Visualization deps are imported lazily to keep the generator usable without them.
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    plt = None
    nx = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FILE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = FILE_DIR / "configs" / "generated"
DEFAULT_PREVIEW_DIR = FILE_DIR / "configs" / "generated_previews"

from examples.supply_chain.config_loader import generate_episode_config_from_file


@dataclass
class CaseInfo:
    path: Path
    preview_path: Path | None
    summary: str


def _write_yaml(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if yaml is None:
        path.write_text(json.dumps(data, indent=2))
        return
    dumped = yaml.safe_dump(
        data,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    )
    path.write_text(dumped)


def _float_list(rng: random.Random, length: int, lo: float, hi: float, digits: int = 1) -> List[float]:
    return [round(rng.uniform(lo, hi), digits) for _ in range(length)]


def _weights(rng: random.Random, cost_range: Tuple[float, float], risk_range: Tuple[float, float]) -> Dict[str, float]:
    return {
        "supply_reward_weight": 1.0,
        "cost_weight": round(rng.uniform(*cost_range), 3),
        "risk_weight": round(rng.uniform(*risk_range), 3),
    }


def _loss_penalty(rng: random.Random, base: Tuple[float, float]) -> float:
    return round(rng.uniform(*base), 2)


@dataclass
class BuildOptions:
    """User overrides for topologyとコスト周り。Noneならレベル側のデフォルトを使う。"""

    supply_count: int | None = None
    depth_range: Tuple[int, int] | None = None
    width_range: Tuple[int, int] | None = None
    allow_cross_edges: bool | None = None
    direct_supply_shortcuts: bool | None = None
    shortcut_fraction: float | None = None
    cross_edge_factor: float | None = None
    allow_demand_edges: bool | None = None
    demand_edge_fraction: float | None = None
    allow_multi_edges: bool | None = None
    multi_edge_prob: float | None = None
    cost_base_range: Tuple[float, float] | None = None
    cost_shortcut_mode: str = "premium"  # premium / discount / mixed
    cost_shortcut_premium: Tuple[float, float] = (0.2, 0.8)  # 1 + rand
    cost_shortcut_discount: Tuple[float, float] = (0.1, 0.35)  # 1 - rand
    cost_lateral_multiplier: Tuple[float, float] = (0.9, 1.1)


def _layout_positions(graph, seed: int):
    if plt is None or nx is None:
        return None
    try:
        # Kamada-Kawai tends to spread nodes more evenly; seed for reproducibility.
        return nx.kamada_kawai_layout(graph, weight=None, scale=1.0, seed=seed)
    except Exception:
        pass
    n = max(1, graph.number_of_nodes())
    k = 1.5 / math.sqrt(n)
    return nx.spring_layout(graph, seed=seed, k=k, iterations=200, scale=1.0)


def _scaled_int(rng: random.Random, base_min: int, base_max: int, factor: float) -> int:
    high = max(base_min, int(math.ceil(base_max * factor)))
    return rng.randint(base_min, high)


def _parse_range(text: str, cast_fn) -> Tuple:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Range must be 'lo,hi', got: {text}")
    lo, hi = cast_fn(parts[0]), cast_fn(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _make_cost_config(opts: BuildOptions) -> Dict[str, Any]:
    return {
        "shortcut_mode": opts.cost_shortcut_mode,
        "shortcut_premium": opts.cost_shortcut_premium,
        "shortcut_discount": opts.cost_shortcut_discount,
        "lateral_multiplier": opts.cost_lateral_multiplier,
    }


def _assign_capacity_and_cost(
    rng: random.Random,
    edges: List[Tuple[int, int]],
    node_layers: Dict[int, int],
    cap_range: Tuple[float, float],
    cost_base_range: Tuple[float, float],
    cost_cfg: Mapping[str, Any],
) -> Tuple[List[float], List[float]]:
    capacities: List[float] = []
    costs: List[float] = []

    shortcut_mode = cost_cfg.get("shortcut_mode", "premium")
    shortcut_premium = cost_cfg.get("shortcut_premium", (0.2, 0.8))
    shortcut_discount = cost_cfg.get("shortcut_discount", (0.1, 0.35))
    lateral_multiplier = cost_cfg.get("lateral_multiplier", (0.9, 1.1))

    for src, dst in edges:
        cap = round(rng.uniform(*cap_range), 1)
        capacities.append(cap)

        base = rng.uniform(*cost_base_range)
        src_layer = node_layers.get(src, 0)
        dst_layer = node_layers.get(dst, src_layer + 1)
        layer_gap = max(1, dst_layer - src_layer)

        edge_cost = base * layer_gap

        if dst_layer == src_layer:
            edge_cost *= rng.uniform(*lateral_multiplier)
        elif layer_gap > 1:
            mode = shortcut_mode
            if mode == "mixed":
                mode = rng.choice(["premium", "discount"])
            if mode == "premium":
                edge_cost *= 1.0 + rng.uniform(*shortcut_premium)
            else:
                edge_cost *= max(0.2, 1.0 - rng.uniform(*shortcut_discount))

        costs.append(round(edge_cost, 2))

    return capacities, costs


def _build_layered_graph(
    rng: random.Random,
    supply_count: int,
    depth_range: Tuple[int, int],
    width_range: Tuple[int, int],
    size_factor: float,
    direct_supply_shortcuts: bool = True,
    allow_cross_edges: bool = True,
    shortcut_fraction: float | None = None,
    cross_edge_factor: float | None = None,
    allow_demand_edges: bool = False,
    demand_edge_fraction: float | None = None,
    allow_multi_edges: bool = False,
    multi_edge_prob: float | None = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int], int, Dict[int, int]]:
    """Create a multi-layer directed graph with optional shortcuts for richer topologies."""
    min_depth, max_depth = depth_range
    min_width, max_width = width_range
    depth = rng.randint(min_depth, max_depth)

    supply_nodes = list(range(supply_count))
    layers: List[List[int]] = []
    node_layers: Dict[int, int] = {n: 0 for n in supply_nodes}
    edges: List[Tuple[int, int]] = []
    next_node = supply_count
    prev_layer = supply_nodes
    outgoing_count: Dict[int, int] = {n: 0 for n in supply_nodes}

    for layer_idx in range(1, depth + 1):
        width_raw = rng.randint(min_width, max_width)
        width = max(1, int(math.ceil(width_raw * size_factor)))
        layer_nodes: List[int] = []
        for _ in range(width):
            node_id = next_node
            next_node += 1
            layer_nodes.append(node_id)
            node_layers[node_id] = layer_idx
            parent_count = rng.randint(1, min(2, len(prev_layer))) if prev_layer else 0
            parents = rng.sample(prev_layer, k=parent_count) if parent_count else []
            if not parents and prev_layer:
                parents = [rng.choice(prev_layer)]
            for parent in parents:
                edges.append((parent, node_id))
                outgoing_count[parent] = outgoing_count.get(parent, 0) + 1
        layers.append(layer_nodes)

        # Ensure every node in prev_layer has at least one outgoing edge to avoid orphan non-demand leaves.
        for parent in prev_layer:
            if outgoing_count.get(parent, 0) == 0 and layer_nodes:
                child = rng.choice(layer_nodes)
                edges.append((parent, child))
                outgoing_count[parent] = 1

        prev_layer = layer_nodes
        for node in prev_layer:
            outgoing_count.setdefault(node, 0)

    demand_nodes = layers[-1] if layers else []

    if direct_supply_shortcuts and demand_nodes:
        fraction = shortcut_fraction if shortcut_fraction is not None else 0.3
        shortcut_count = int(len(demand_nodes) * fraction)
        if shortcut_count > 0:
            for node in rng.sample(demand_nodes, k=min(shortcut_count, len(demand_nodes))):
                edges.append((rng.choice(supply_nodes), node))

    if allow_cross_edges and len(layers) >= 2:
        base_cross_edges = rng.randint(1, max(1, int(len(edges) * 0.2)))
        cross_edges = base_cross_edges
        if cross_edge_factor is not None:
            cross_edges = max(0, int(round(base_cross_edges * cross_edge_factor)))
        for _ in range(cross_edges):
            upper_idx = rng.randint(0, len(layers) - 2)
            lower_idx = rng.randint(upper_idx + 1, len(layers) - 1)
            src = rng.choice(layers[upper_idx])
            dst = rng.choice(layers[lower_idx])
            if src != dst:
                edges.append((src, dst))

    # Optionally inject multi-edges; otherwise deduplicate.
    if allow_multi_edges:
        prob = multi_edge_prob if multi_edge_prob is not None else 0.15
        extras: List[Tuple[int, int]] = []
        for e in list(edges):
            if rng.random() < prob:
                extras.append(e)
        edges.extend(extras)
        unique_edges = list(edges)
    else:
        unique_edges = list({(src, dst) for src, dst in edges})

    # Optional demand-to-demand lateral edges to allow rebalancing among sinks.
    if allow_demand_edges and demand_nodes:
        fraction = demand_edge_fraction if demand_edge_fraction is not None else 0.2
        dd_count = max(1, int(len(demand_nodes) * fraction))
        added = 0
        tried = 0
        while added < dd_count and tried < dd_count * 3:
            tried += 1
            src = rng.choice(demand_nodes)
            dst = rng.choice(demand_nodes)
            if src == dst:
                continue
            edge = (src, dst)
            if edge in unique_edges:
                continue
            unique_edges.append(edge)
            added += 1

    return unique_edges, supply_nodes, demand_nodes, next_node, node_layers


def _build_level1_case(rng: random.Random, size_factor: float, opts: BuildOptions) -> Dict:
    demand_count = _scaled_int(rng, 2, 5, size_factor)
    num_nodes = demand_count + 1
    horizon = _scaled_int(rng, 1, 4, size_factor)

    demand_uniform = []
    total_base = 0.0
    for node in range(1, num_nodes):
        low = round(rng.uniform(4.0, 10.0), 1)
        high = round(low + rng.uniform(2.0, 6.0), 1)
        total_base += high
        demand_uniform.append({"node": node, "low": low, "high": high})

    min_inventory = max(25.0, total_base * rng.uniform(0.7, 1.0))
    initial_inventory = {"min": round(min_inventory, 1), "scale": round(rng.uniform(1.0, 1.25), 2)}

    # Multi-edge per demand: cheap/低容量の主経路 + 高コスト/高容量のバックアップを混在させる
    base_cap = min_inventory / max(1, demand_count)
    cost_base_range = opts.cost_base_range or (0.9, 1.5)
    edges: List[Tuple[int, int]] = []
    capacities: List[float] = []
    costs: List[float] = []
    for node in range(1, num_nodes):
        # main edge: cheaper, smaller capacity
        edges.append((0, node))
        cap_main = round(base_cap * rng.uniform(0.35, 0.7), 1)
        cost_main = round(rng.uniform(*cost_base_range) * rng.uniform(0.7, 0.95), 2)
        capacities.append(cap_main)
        costs.append(cost_main)

        # backup edge: optional, more expensive but larger capacity
        if rng.random() < 0.65:
            edges.append((0, node))
            cap_backup = round(base_cap * rng.uniform(0.7, 1.3), 1)
            cost_backup = round(rng.uniform(*cost_base_range) * rng.uniform(1.35, 1.9), 2)
            capacities.append(cap_backup)
            costs.append(cost_backup)

    return {
        "level": 1,
        "horizon": horizon,
        "num_nodes": num_nodes,
        "edges": edges,
        "supply_nodes": [0],
        "demand_nodes": list(range(1, num_nodes)),
        "demand_uniform": demand_uniform,
        "initial_inventory": initial_inventory,
        "edge_capacity": capacities,
        "edge_cost": costs,
        "availability_default": True,
        "risk_default": 0.0,
        "loss_penalty": 1.0,
        "weights": _weights(rng, (0.05, 0.12), (0.0, 0.02)),
        "cvar": {"lambda": 0.0, "tau": 0.2},
    }


def _build_level2_case(rng: random.Random, size_factor: float, opts: BuildOptions) -> Dict:
    supply_count = opts.supply_count or 1
    depth_range = opts.depth_range or (2, 3)
    width_range = opts.width_range or (2, 4)
    allow_cross_edges = opts.allow_cross_edges if opts.allow_cross_edges is not None else True
    direct_supply_shortcuts = opts.direct_supply_shortcuts if opts.direct_supply_shortcuts is not None else True
    allow_demand_edges = opts.allow_demand_edges if opts.allow_demand_edges is not None else False
    allow_multi_edges = opts.allow_multi_edges if opts.allow_multi_edges is not None else False

    edges, supply_nodes, demand_nodes, num_nodes, node_layers = _build_layered_graph(
        rng,
        supply_count=supply_count,
        depth_range=depth_range,
        width_range=width_range,
        size_factor=size_factor,
        direct_supply_shortcuts=direct_supply_shortcuts,
        allow_cross_edges=allow_cross_edges,
        shortcut_fraction=opts.shortcut_fraction,
        cross_edge_factor=opts.cross_edge_factor,
        allow_demand_edges=allow_demand_edges,
        demand_edge_fraction=opts.demand_edge_fraction,
        allow_multi_edges=allow_multi_edges,
        multi_edge_prob=opts.multi_edge_prob,
    )
    horizon = _scaled_int(rng, 6, 10, size_factor)

    demand_waves = []
    for node in demand_nodes:
        base = round(rng.uniform(6.0, 11.0), 1)
        amplitude = round(rng.uniform(0.0, base * 0.4), 2)
        freq = round(rng.uniform(0.35, 0.85), 2)
        func = rng.choice(["sin", "cos"])
        demand_waves.append({"node": node, "base": base, "amplitude": amplitude, "freq": freq, "func": func})

    cap_range = (10.0 * size_factor, 20.0 * size_factor)
    cost_base_range = opts.cost_base_range or (0.9, 1.3)
    capacities, costs = _assign_capacity_and_cost(
        rng,
        edges,
        node_layers,
        cap_range,
        cost_base_range,
        _make_cost_config(opts),
    )
    initial_inventory = [round(rng.uniform(60.0, 90.0) * size_factor, 1)] + _float_list(rng, num_nodes - 1, 0.0, 10.0 * size_factor)

    return {
        "level": 2,
        "horizon": horizon,
        "num_nodes": num_nodes,
        "edges": edges,
        "supply_nodes": supply_nodes,
        "demand_nodes": demand_nodes,
        "demand_waves": demand_waves,
        "edge_capacity": capacities,
        "edge_cost": costs,
        "availability_default": True,
        "risk_default": 0.0,
        "loss_penalty": _loss_penalty(rng, (1.0, 1.6)),
        "initial_inventory": initial_inventory,
        "supply_injection": {"per_period": [{"node": supply_nodes[0], "amount": round(rng.uniform(5.0, 9.0), 2)}]},
        "weights": _weights(rng, (0.05, 0.11), (0.0, 0.05)),
        "cvar": {"lambda": 0.0, "tau": 0.2},
    }


def _build_level3_case(rng: random.Random, size_factor: float, opts: BuildOptions) -> Dict:
    supply_count = opts.supply_count or 1
    depth_range = opts.depth_range or (3, 4)
    width_range = opts.width_range or (2, 4)
    allow_cross_edges = opts.allow_cross_edges if opts.allow_cross_edges is not None else True
    direct_supply_shortcuts = opts.direct_supply_shortcuts if opts.direct_supply_shortcuts is not None else True
    allow_demand_edges = opts.allow_demand_edges if opts.allow_demand_edges is not None else False
    allow_multi_edges = opts.allow_multi_edges if opts.allow_multi_edges is not None else False

    edges, supply_nodes, demand_nodes, num_nodes, node_layers = _build_layered_graph(
        rng,
        supply_count=supply_count,
        depth_range=depth_range,
        width_range=width_range,
        size_factor=size_factor,
        allow_cross_edges=allow_cross_edges,
        direct_supply_shortcuts=direct_supply_shortcuts,
        shortcut_fraction=opts.shortcut_fraction,
        cross_edge_factor=opts.cross_edge_factor,
        allow_demand_edges=allow_demand_edges,
        demand_edge_fraction=opts.demand_edge_fraction,
        allow_multi_edges=allow_multi_edges,
        multi_edge_prob=opts.multi_edge_prob,
    )
    horizon = _scaled_int(rng, 10, 16, size_factor)
    noise_std = round(rng.uniform(0.8, 2.2), 2)

    demand_waves = []
    for node in demand_nodes:
        base = round(rng.uniform(7.0, 12.5), 1)
        amplitude = round(rng.uniform(0.0, base * 0.45), 2)
        freq = round(rng.uniform(0.3, 0.9), 2)
        phase = round(rng.uniform(0.0, math.pi), 2)
        func = rng.choice(["sin", "cos"])
        demand_waves.append({"node": node, "base": base, "amplitude": amplitude, "freq": freq, "phase": phase, "func": func})

    cap_range = (12.0 * size_factor, 22.0 * size_factor)
    cost_base_range = opts.cost_base_range or (0.9, 1.25)
    capacities, costs = _assign_capacity_and_cost(
        rng,
        edges,
        node_layers,
        cap_range,
        cost_base_range,
        _make_cost_config(opts),
    )
    initial_inventory = [round(rng.uniform(70.0, 110.0) * size_factor, 1)] + _float_list(rng, num_nodes - 1, 0.0, 12.0 * size_factor)

    return {
        "level": 3,
        "horizon": horizon,
        "num_nodes": num_nodes,
        "edges": edges,
        "supply_nodes": supply_nodes,
        "demand_nodes": demand_nodes,
        "demand_waves": demand_waves,
        "noise_std": noise_std,
        "edge_capacity": capacities,
        "edge_cost": costs,
        "availability_default": True,
        "risk_default": 0.0,
        "loss_penalty": _loss_penalty(rng, (1.0, 1.7)),
        "initial_inventory": initial_inventory,
        "supply_injection": {"per_period": [{"node": supply_nodes[0], "amount": round(rng.uniform(5.0, 11.0), 2)}]},
        "weights": _weights(rng, (0.05, 0.12), (0.0, 0.08)),
        "cvar": {"lambda": 0.0, "tau": 0.2},
    }


def _build_level4_case(rng: random.Random, size_factor: float, opts: BuildOptions) -> Dict:
    supply_count = opts.supply_count or 1
    depth_range = opts.depth_range or (3, 5)
    width_range = opts.width_range or (2, 4)
    allow_cross_edges = opts.allow_cross_edges if opts.allow_cross_edges is not None else True
    direct_supply_shortcuts = opts.direct_supply_shortcuts if opts.direct_supply_shortcuts is not None else True
    allow_demand_edges = opts.allow_demand_edges if opts.allow_demand_edges is not None else False
    allow_multi_edges = opts.allow_multi_edges if opts.allow_multi_edges is not None else False

    edges, supply_nodes, demand_nodes, num_nodes, node_layers = _build_layered_graph(
        rng,
        supply_count=supply_count,
        depth_range=depth_range,
        width_range=width_range,
        size_factor=size_factor,
        allow_cross_edges=allow_cross_edges,
        direct_supply_shortcuts=direct_supply_shortcuts,
        shortcut_fraction=opts.shortcut_fraction,
        cross_edge_factor=opts.cross_edge_factor,
        allow_demand_edges=allow_demand_edges,
        demand_edge_fraction=opts.demand_edge_fraction,
        allow_multi_edges=allow_multi_edges,
        multi_edge_prob=opts.multi_edge_prob,
    )

    # Chance to add a direct long edge to make alternative paths.
    if rng.random() < 0.4 and len(edges) > 0:
        src = rng.choice(list(node_layers.keys()))
        dst = rng.choice(demand_nodes)
        if node_layers.get(src, 0) < node_layers.get(dst, 0):
            edges.append((src, dst))

    horizon = _scaled_int(rng, 12, 18, size_factor)

    demand_sinusoidal = []
    for node in demand_nodes:
        base = round(rng.uniform(6.0, 11.0), 1)
        amplitude = round(rng.uniform(1.0, 4.0), 1)
        phase = round(rng.uniform(0.0, math.pi), 2)
        demand_sinusoidal.append({"node": node, "base": base, "amplitude": amplitude, "phase": phase})

    open_prob_default = round(rng.uniform(0.72, 0.9), 2)
    per_edge = [round(max(0.5, min(0.98, rng.gauss(open_prob_default, 0.07))), 2) for _ in edges]

    risk_ranges = []
    for _ in edges:
        lo = round(rng.uniform(0.01, 0.08), 3)
        hi = round(lo + rng.uniform(0.04, 0.12), 3)
        risk_ranges.append([lo, hi])

    cap_range = (10.0 * size_factor, 20.0 * size_factor)
    cost_base_range = opts.cost_base_range or (0.9, 1.35)
    capacities, costs = _assign_capacity_and_cost(
        rng,
        edges,
        node_layers,
        cap_range,
        cost_base_range,
        _make_cost_config(opts),
    )
    initial_inventory = [round(rng.uniform(90.0, 130.0) * size_factor, 1)] + _float_list(rng, num_nodes - 1, 0.0, 10.0 * size_factor)

    return {
        "level": 4,
        "horizon": horizon,
        "num_nodes": num_nodes,
        "edges": edges,
        "supply_nodes": supply_nodes,
        "demand_nodes": demand_nodes,
        "demand_sinusoidal": demand_sinusoidal,
        "availability": {"open_probability": open_prob_default, "per_edge": per_edge},
        "risk": {"per_edge_range": risk_ranges},
        "loss_penalty": _loss_penalty(rng, (1.1, 1.8)),
        "edge_capacity": capacities,
        "edge_cost": costs,
        "initial_inventory": initial_inventory,
        "supply_injection": {"per_period": [{"node": supply_nodes[0] if supply_nodes else 0, "amount": round(rng.uniform(6.0, 10.0), 2)}]},
        "weights": _weights(rng, (0.05, 0.11), (0.1, 0.35)),
        "cvar": {"lambda": 0.0, "tau": 0.2},
    }


def _build_level5_case(rng: random.Random, size_factor: float, opts: BuildOptions) -> Dict:
    # Richer graph with dual depots, cross links, and deep demand nodes.
    supply_count = opts.supply_count or 2
    depth_range = opts.depth_range or (4, 5)
    width_range = opts.width_range or (2, 4)
    allow_cross_edges = opts.allow_cross_edges if opts.allow_cross_edges is not None else True
    direct_supply_shortcuts = opts.direct_supply_shortcuts if opts.direct_supply_shortcuts is not None else True
    allow_demand_edges = opts.allow_demand_edges if opts.allow_demand_edges is not None else False
    allow_multi_edges = opts.allow_multi_edges if opts.allow_multi_edges is not None else False

    edges, supply_nodes, demand_nodes, num_nodes, node_layers = _build_layered_graph(
        rng,
        supply_count=supply_count,
        depth_range=depth_range,
        width_range=width_range,
        size_factor=size_factor,
        allow_cross_edges=allow_cross_edges,
        direct_supply_shortcuts=direct_supply_shortcuts,
        shortcut_fraction=opts.shortcut_fraction,
        cross_edge_factor=opts.cross_edge_factor,
        allow_demand_edges=allow_demand_edges,
        demand_edge_fraction=opts.demand_edge_fraction,
        allow_multi_edges=allow_multi_edges,
        multi_edge_prob=opts.multi_edge_prob,
    )

    # Add extra cross-links and occasional direct supply-to-leaf to mix path lengths.
    extra_cross = rng.randint(1, 2)
    for _ in range(extra_cross):
        src = rng.choice(list(node_layers.keys()))
        dst = rng.choice(demand_nodes)
        if src != dst and node_layers.get(src, 0) < node_layers.get(dst, 0):
            edges.append((src, dst))

    horizon = _scaled_int(rng, 18, 24, size_factor)

    base_demands = {node: round(rng.uniform(6.5, 11.5), 1) for node in demand_nodes}
    phase1 = horizon // 3
    phase2 = 2 * horizon // 3
    regimes = [
        {"until": phase1, "scale": round(rng.uniform(0.85, 1.1), 2), "volatility": round(rng.uniform(0.6, 1.4), 2), "risk_base": round(rng.uniform(0.05, 0.12), 3)},
        {"until": phase2, "scale": round(rng.uniform(1.2, 1.6), 2), "volatility": round(rng.uniform(1.1, 2.0), 2), "risk_base": round(rng.uniform(0.12, 0.22), 3)},
        {"until": None, "scale": round(rng.uniform(1.0, 1.3), 2), "volatility": round(rng.uniform(0.8, 1.6), 2), "risk_base": round(rng.uniform(0.1, 0.18), 3)},
    ]

    primary_edges = rng.randint(max(1, min(5, len(edges))), len(edges))
    availability = {
        "primary": round(rng.uniform(0.78, 0.9), 2),
        "secondary": round(rng.uniform(0.68, 0.82), 2),
        "primary_edges": primary_edges,
    }

    risk_base_shift_per_edge = _float_list(rng, len(edges), 0.0, 0.08, digits=3)
    cap_range = (12.0 * size_factor, 24.0 * size_factor)
    cost_base_range = opts.cost_base_range or (0.85, 1.25)
    capacities, costs = _assign_capacity_and_cost(
        rng,
        edges,
        node_layers,
        cap_range,
        cost_base_range,
        _make_cost_config(opts),
    )

    supply_injection = {
        "per_period": [
            {"node": node, "amount": round(rng.uniform(7.0, 11.0), 2)} for node in supply_nodes
        ]
    }

    initial_inventory = [
        round(rng.uniform(140.0, 200.0) * size_factor, 1),
        round(rng.uniform(130.0, 190.0) * size_factor, 1),
        round(rng.uniform(25.0, 40.0) * size_factor, 1),
        round(rng.uniform(25.0, 40.0) * size_factor, 1),
        round(rng.uniform(8.0, 16.0) * size_factor, 1),
        round(rng.uniform(8.0, 16.0) * size_factor, 1),
    ] + [0.0 for _ in range(num_nodes - 6)]

    return {
        "level": 5,
        "horizon": horizon,
        "num_nodes": num_nodes,
        "edges": edges,
        "supply_nodes": supply_nodes,
        "demand_nodes": demand_nodes,
        "demand_regimes": {
            "base_demands": base_demands,
            "regimes": regimes,
            "risk_spread": round(rng.uniform(0.22, 0.34), 3),
        },
        "availability": availability,
        "risk_base_shift_per_edge": risk_base_shift_per_edge,
        "edge_capacity": capacities,
        "edge_cost": costs,
        "loss_penalty": _loss_penalty(rng, (1.3, 1.9)),
        "initial_inventory": initial_inventory,
        "supply_injection": supply_injection,
        "weights": _weights(rng, (0.04, 0.1), (0.55, 1.0)),
        "cvar": {"lambda": round(rng.uniform(0.35, 0.6), 2), "tau": 0.25},
    }


LEVEL_BUILDERS = {
    1: _build_level1_case,
    2: _build_level2_case,
    3: _build_level3_case,
    4: _build_level4_case,
    5: _build_level5_case,
}


def _render_preview(config_path: Path, seed: int, preview_dir: Path) -> Path | None:
    if plt is None or nx is None:
        return None
    preview_dir.mkdir(parents=True, exist_ok=True)
    cfg = generate_episode_config_from_file(config_path, seed)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(range(cfg.num_nodes))
    for idx, (src, dst) in enumerate(cfg.edges):
        graph.add_edge(src, dst, key=idx, idx=idx)

    simple = nx.DiGraph()
    simple.add_nodes_from(graph.nodes())
    simple.add_edges_from({(src, dst) for src, dst in cfg.edges})
    pos = _layout_positions(simple, seed) or {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_graph, ax_demand = axes
    ax_graph.axis("off")

    node_colors = []
    for node in graph.nodes():
        if node in cfg.supply_nodes:
            node_colors.append("#2ecc71")
        elif node in cfg.demand_nodes:
            node_colors.append("#e74c3c")
        else:
            node_colors.append("#3498db")

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        ax=ax_graph,
        node_color=node_colors,
        edgecolors="black",
        node_size=520,
    )
    nx.draw_networkx_labels(graph, pos=pos, ax=ax_graph, font_size=9)

    pair_to_indices: Dict[Tuple[int, int], List[int]] = {}
    for idx, (src, dst) in enumerate(cfg.edges):
        pair_to_indices.setdefault((src, dst), []).append(idx)

    label_annotations: List[Tuple[float, float, str]] = []
    for (src, dst), indices in pair_to_indices.items():
        count = len(indices)
        if count == 1:
            rads = [0.08]
        else:
            step_rad = 0.14 / max(1, count - 1)
            start = -step_rad * (count - 1) / 2
            rads = [start + i * step_rad for i in range(count)]

        # Label offsets are spaced independently of rad to reduce overlap.
        offset_step = 0.18
        offset_start = -offset_step * (count - 1) / 2

        for i, (rad, e_idx) in enumerate(zip(rads, indices)):
            nx.draw_networkx_edges(
                graph,
                pos=pos,
                ax=ax_graph,
                edgelist=[(src, dst, e_idx)],
                edge_color="#7f8c8d",
                width=1.6,
                arrows=True,
                arrowsize=12,
                connectionstyle=f"arc3,rad={rad}",
            )
            label_text = f"C:{cfg.edge_capacity[e_idx]:.1f}\nK:{cfg.edge_cost[e_idx]:.2f}"
            p1, p2 = pos[src], pos[dst]
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            dist = math.hypot(dx, dy) or 1e-6
            ox, oy = -dy / dist, dx / dist
            dist_for_offset = max(dist, 0.3)
            offset = (offset_start + i * offset_step) * dist_for_offset
            lx, ly = mx + ox * offset, my + oy * offset
            label_annotations.append((lx, ly, label_text))

    for lx, ly, text in label_annotations:
        ax_graph.text(
            lx,
            ly,
            text,
            fontsize=7,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.95, edgecolor="#555", linewidth=0.3, pad=0.6),
        )
    ax_graph.set_title(f"Level {cfg.level} | horizon {cfg.horizon} | nodes {cfg.num_nodes}")

    for node in cfg.demand_nodes:
        series = [cfg.demand_series[t][node] for t in range(cfg.horizon)]
        ax_demand.plot(range(cfg.horizon), series, label=f"node {node}")
    ax_demand.set_xlabel("t")
    ax_demand.set_ylabel("demand")
    ax_demand.set_title("Demand series")
    ax_demand.grid(True, alpha=0.3)
    if cfg.demand_nodes:
        ax_demand.legend(fontsize=8)

    fig.tight_layout()
    out_path = preview_dir / f"{config_path.stem}_preview.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def generate_cases(
    level: int,
    num_cases: int,
    seed: int,
    out_dir: Path,
    preview: bool,
    preview_dir: Path | None,
    size_factor: float,
    stem_base: str | None,
    build_opts: BuildOptions,
) -> List[CaseInfo]:
    builder = LEVEL_BUILDERS.get(level)
    if builder is None:
        raise ValueError(f"Unsupported level: {level}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: List[CaseInfo] = []

    for idx in range(num_cases):
        case_seed = seed + idx
        rng = random.Random(case_seed)
        config = builder(rng, size_factor, build_opts)
        base = stem_base or f"level{level}_gen"
        stem = f"{base}_{idx + 1:03d}"
        path = out_dir / f"{stem}.yaml"
        _write_yaml(config, path)

        preview_path = _render_preview(path, case_seed, preview_dir or out_dir) if preview else None
        summary = f"horizon={config['horizon']}, nodes={config['num_nodes']}, edges={len(config['edges'])}"
        cases.append(CaseInfo(path=path, preview_path=preview_path, summary=summary))

    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate randomized supply chain test cases with optional previews.")
    parser.add_argument("--level", type=int, choices=list(LEVEL_BUILDERS.keys()), required=True, help="Curriculum level (1-5).")
    parser.add_argument("--num_cases", type=int, default=3, help="Number of cases to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed; increments per case.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory to write YAML configs.")
    parser.add_argument("--preview", action="store_true", help="Also render a static preview PNG for each case.")
    parser.add_argument("--preview_dir", type=Path, default=DEFAULT_PREVIEW_DIR, help="Directory to write preview images (if --preview).")
    parser.add_argument("--size_factor", type=float, default=1.0, help="Scale node/edge/horizon ranges upward (>1.0) or downward (<1.0).")
    parser.add_argument("--supply_count", type=int, help="Override supply node count for layered levels (>=2).")
    parser.add_argument("--depth_range", type=str, help="Depth range 'min,max' for layered graph (levels 2-5).")
    parser.add_argument("--width_range", type=str, help="Width range 'min,max' per layer (levels 2-5).")
    parser.add_argument("--allow_cross_edges", action=BooleanOptionalAction, default=None, help="Enable/disable cross edges (None keeps level defaults).")
    parser.add_argument("--direct_supply_shortcuts", action=BooleanOptionalAction, default=None, help="Enable/disable supply→demand shortcuts (None keeps level defaults).")
    parser.add_argument("--shortcut_fraction", type=float, help="Fraction of demand nodes to connect with supply shortcuts (0.0-1.0).")
    parser.add_argument("--cross_edge_factor", type=float, help="Multiplier for cross-edge count (1.0 keeps default random count).")
    parser.add_argument("--cost_base_range", type=str, help="Base per-hop cost range 'lo,hi' (overrides level defaults).")
    parser.add_argument("--cost_shortcut_mode", choices=["premium", "discount", "mixed"], default="premium", help="How to price shortcuts: premium=高コスト, discount=低コスト, mixed=ランダム。")
    parser.add_argument("--stem_base", type=str, help="Base name for output files (default: level{level}_gen). stem_base_XYZ -> stem_base_XYZ_001.yaml ...")
    parser.add_argument("--stem_prefix", type=str, help="(Deprecated) Prefix for filenames; use --stem_base instead.")
    parser.add_argument("--allow_demand_edges", action=BooleanOptionalAction, default=None, help="Allow edges between demand nodes (same layer).")
    parser.add_argument("--demand_edge_fraction", type=float, help="Fraction of demand nodes used to create demand-to-demand edges (default 0.2).")
    parser.add_argument("--allow_multi_edges", action=BooleanOptionalAction, default=None, help="Allow duplicate edges (parallel arcs).")
    parser.add_argument("--multi_edge_prob", type=float, help="Probability per edge to add a duplicate when --allow_multi_edges (default 0.15).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_opts = BuildOptions(
        supply_count=args.supply_count,
        depth_range=_parse_range(args.depth_range, int) if args.depth_range else None,
        width_range=_parse_range(args.width_range, int) if args.width_range else None,
        allow_cross_edges=args.allow_cross_edges,
        direct_supply_shortcuts=args.direct_supply_shortcuts,
        shortcut_fraction=args.shortcut_fraction,
        cross_edge_factor=args.cross_edge_factor,
        cost_base_range=_parse_range(args.cost_base_range, float) if args.cost_base_range else None,
        cost_shortcut_mode=args.cost_shortcut_mode,
        allow_demand_edges=args.allow_demand_edges,
        demand_edge_fraction=args.demand_edge_fraction,
        allow_multi_edges=args.allow_multi_edges,
        multi_edge_prob=args.multi_edge_prob,
    )
    cases = generate_cases(
        level=args.level,
        num_cases=args.num_cases,
        seed=args.seed,
        out_dir=args.out_dir,
        preview=args.preview,
        preview_dir=args.preview_dir,
        size_factor=max(args.size_factor, 0.5),
        stem_base=args.stem_base or (f"{args.stem_prefix}_level{args.level}_gen" if args.stem_prefix else None),
        build_opts=build_opts,
    )

    for info in cases:
        print(f"[level {args.level}] wrote {info.path} ({info.summary})")
        if info.preview_path:
            print(f"  preview -> {info.preview_path}")
        elif args.preview and plt is None:
            print("  preview skipped (matplotlib/networkx not available)")


if __name__ == "__main__":
    main()
