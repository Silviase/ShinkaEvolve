from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

from examples.supply_chain.episode_configs import EpisodeConfig
from examples.supply_chain.utils import bool_matrix, non_negative, sinusoidal_series, zero_matrix

CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def _load_yaml(level: int) -> Dict[str, Any]:
    path = CONFIG_DIR / f"level{level}.yaml"
    return _load_yaml_file(path, level)


def _load_yaml_file(path: Path, level_hint: int | None = None) -> Dict[str, Any]:
    if not path.exists():
        msg = (
            f"Config file not found: {path}"
            if level_hint is None
            else f"Config file not found for level {level_hint}: {path}"
        )
        raise ValueError(msg)
    text = path.read_text()
    data: Any
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        try:
            stripped = "\n".join(
                ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
            )
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:  # pragma: no cover - clarity for users without yaml
            raise ValueError(
                "Failed to parse config. Install PyYAML or supply JSON-compatible YAML."
            ) from exc
    if not isinstance(data, dict):
        raise ValueError("Config file must be a mapping.")
    return data


def _float_list(values: Sequence[Any], length: int) -> List[float]:
    data = list(values)
    if len(data) != length:
        raise ValueError(f"Expected list of length {length}, got {len(data)}")
    return [float(x) for x in data]


def _edge_tuples(edges_raw: Iterable[Sequence[Any]]) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for pair in edges_raw:
        if len(pair) != 2:
            raise ValueError(f"Invalid edge entry (expected length 2): {pair}")
        edges.append((int(pair[0]), int(pair[1])))
    return edges


def _loss_penalty_list(raw_value: Any, edge_count: int) -> List[float]:
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        values = list(raw_value)
        if len(values) != edge_count:
            raise ValueError("loss_penalty length mismatch")
        return [float(x) for x in values]
    return [float(raw_value) for _ in range(edge_count)]


def _weights(cfg: Mapping[str, Any]) -> Tuple[float, float, float]:
    return (
        float(cfg.get("supply_reward_weight", 1.0)),
        float(cfg.get("cost_weight", 0.1)),
        float(cfg.get("risk_weight", 0.0)),
    )


def _cvar_params(cfg: Mapping[str, Any]) -> Tuple[float, float]:
    return float(cfg.get("lambda", 0.0)), float(cfg.get("tau", 0.2))


def _apply_supply_injection(
    horizon: int, num_nodes: int, injection_cfg: Mapping[str, Any] | None
) -> List[List[float]] | None:
    if not injection_cfg:
        return None
    per_period = injection_cfg.get("per_period", [])
    matrix = zero_matrix(horizon, num_nodes)
    for entry in per_period:
        node = int(entry.get("node", 0))
        amount = float(entry.get("amount", 0.0))
        if node < 0 or node >= num_nodes:
            raise ValueError(f"Invalid node index in supply_injection: {node}")
        for t in range(horizon):
            matrix[t][node] += amount
    return matrix


def _build_level1(cfg: Dict[str, Any], rng: random.Random) -> EpisodeConfig:
    horizon = int(cfg["horizon"])
    num_nodes = int(cfg["num_nodes"])
    edges = _edge_tuples(cfg["edges"])
    supply_nodes = [int(x) for x in cfg.get("supply_nodes", [])]
    demand_nodes = [int(x) for x in cfg.get("demand_nodes", [])]
    edge_count = len(edges)

    demand_series = zero_matrix(horizon, num_nodes)
    for entry in cfg.get("demand_uniform", []):
        node = int(entry["node"])
        low = float(entry["low"])
        high = float(entry["high"])
        value = rng.uniform(low, high)
        for t in range(horizon):
            demand_series[t][node] = value

    inv_cfg = cfg.get("initial_inventory", {})
    if isinstance(inv_cfg, Sequence) and not isinstance(inv_cfg, (str, bytes, dict)):
        initial_inventory = [float(x) for x in inv_cfg]
    else:
        min_val = float(inv_cfg.get("min", 0.0))
        scale = float(inv_cfg.get("scale", 1.0))
        total_demand = sum(demand_series[0][idx] for idx in demand_nodes)
        first_node = max(min_val, total_demand * scale)
        initial_inventory = [first_node] + [0.0] * (num_nodes - 1)

    availability_default = bool(cfg.get("availability_default", True))
    risk_default = float(cfg.get("risk_default", 0.0))
    availability_series = bool_matrix(horizon, edge_count, availability_default)
    risk_series = zero_matrix(horizon, edge_count, risk_default)

    edge_capacity = _float_list(cfg["edge_capacity"], edge_count)
    edge_cost = _float_list(cfg["edge_cost"], edge_count)
    loss_penalty = _loss_penalty_list(cfg.get("loss_penalty", 1.0), edge_count)

    weight_cfg = cfg.get("weights", {})
    supply_reward_weight, cost_weight, risk_weight = _weights(weight_cfg)
    cvar_lambda, cvar_tau = _cvar_params(cfg.get("cvar", {}))

    return EpisodeConfig(
        level=int(cfg["level"]),
        horizon=horizon,
        num_nodes=num_nodes,
        edges=edges,
        supply_nodes=supply_nodes,
        demand_nodes=demand_nodes,
        initial_inventory=initial_inventory,
        demand_series=demand_series,
        edge_capacity=edge_capacity,
        edge_cost=edge_cost,
        availability_series=availability_series,
        risk_series=risk_series,
        loss_penalty=loss_penalty,
        supply_reward_weight=supply_reward_weight,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
        cvar_lambda=cvar_lambda,
        cvar_tau=cvar_tau,
    )


def _apply_wave_series(
    horizon: int, num_nodes: int, waves: Sequence[Mapping[str, Any]]
) -> List[List[float]]:
    series = zero_matrix(horizon, num_nodes)
    for entry in waves:
        node = int(entry["node"])
        base = float(entry.get("base", 0.0))
        amplitude = float(entry.get("amplitude", 0.0))
        freq = float(entry.get("freq", 1.0))
        phase = float(entry.get("phase", 0.0))
        func_name = str(entry.get("func", "sin")).lower()
        trig = math.sin if func_name == "sin" else math.cos
        for t in range(horizon):
            series[t][node] = non_negative(base + amplitude * trig(t * freq + phase))
    return series


def _build_level2(cfg: Dict[str, Any], rng: random.Random) -> EpisodeConfig:
    horizon = int(cfg["horizon"])
    num_nodes = int(cfg["num_nodes"])
    edges = _edge_tuples(cfg["edges"])
    supply_nodes = [int(x) for x in cfg.get("supply_nodes", [])]
    demand_nodes = [int(x) for x in cfg.get("demand_nodes", [])]
    edge_count = len(edges)

    demand_series = _apply_wave_series(horizon, num_nodes, cfg.get("demand_waves", []))

    initial_inventory = [float(x) for x in cfg.get("initial_inventory", [])]

    availability_series = bool_matrix(horizon, edge_count, bool(cfg.get("availability_default", True)))
    risk_series = zero_matrix(horizon, edge_count, float(cfg.get("risk_default", 0.0)))

    edge_capacity = _float_list(cfg["edge_capacity"], edge_count)
    edge_cost = _float_list(cfg["edge_cost"], edge_count)
    loss_penalty = _loss_penalty_list(cfg.get("loss_penalty", 1.0), edge_count)

    supply_injection = _apply_supply_injection(
        horizon, num_nodes, cfg.get("supply_injection")
    )

    weight_cfg = cfg.get("weights", {})
    supply_reward_weight, cost_weight, risk_weight = _weights(weight_cfg)
    cvar_lambda, cvar_tau = _cvar_params(cfg.get("cvar", {}))

    return EpisodeConfig(
        level=int(cfg["level"]),
        horizon=horizon,
        num_nodes=num_nodes,
        edges=edges,
        supply_nodes=supply_nodes,
        demand_nodes=demand_nodes,
        initial_inventory=initial_inventory,
        demand_series=demand_series,
        edge_capacity=edge_capacity,
        edge_cost=edge_cost,
        availability_series=availability_series,
        risk_series=risk_series,
        loss_penalty=loss_penalty,
        supply_reward_weight=supply_reward_weight,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
        cvar_lambda=cvar_lambda,
        cvar_tau=cvar_tau,
        supply_injection=supply_injection,
    )


def _build_level3(cfg: Dict[str, Any], rng: random.Random) -> EpisodeConfig:
    horizon = int(cfg["horizon"])
    num_nodes = int(cfg["num_nodes"])
    edges = _edge_tuples(cfg["edges"])
    supply_nodes = [int(x) for x in cfg.get("supply_nodes", [])]
    demand_nodes = [int(x) for x in cfg.get("demand_nodes", [])]
    edge_count = len(edges)

    noise_std = float(cfg.get("noise_std", 0.0))
    base_series = _apply_wave_series(horizon, num_nodes, cfg.get("demand_waves", []))
    demand_series = zero_matrix(horizon, num_nodes)
    for t in range(horizon):
        for node in demand_nodes:
            base = base_series[t][node]
            demand_series[t][node] = non_negative(base + rng.gauss(0.0, noise_std))

    initial_inventory = [float(x) for x in cfg.get("initial_inventory", [])]

    availability_series = bool_matrix(horizon, edge_count, bool(cfg.get("availability_default", True)))
    risk_series = zero_matrix(horizon, edge_count, float(cfg.get("risk_default", 0.0)))

    edge_capacity = _float_list(cfg["edge_capacity"], edge_count)
    edge_cost = _float_list(cfg["edge_cost"], edge_count)
    loss_penalty = _loss_penalty_list(cfg.get("loss_penalty", 1.0), edge_count)

    supply_injection = _apply_supply_injection(
        horizon, num_nodes, cfg.get("supply_injection")
    )

    weight_cfg = cfg.get("weights", {})
    supply_reward_weight, cost_weight, risk_weight = _weights(weight_cfg)
    cvar_lambda, cvar_tau = _cvar_params(cfg.get("cvar", {}))

    return EpisodeConfig(
        level=int(cfg["level"]),
        horizon=horizon,
        num_nodes=num_nodes,
        edges=edges,
        supply_nodes=supply_nodes,
        demand_nodes=demand_nodes,
        initial_inventory=initial_inventory,
        demand_series=demand_series,
        edge_capacity=edge_capacity,
        edge_cost=edge_cost,
        availability_series=availability_series,
        risk_series=risk_series,
        loss_penalty=loss_penalty,
        supply_reward_weight=supply_reward_weight,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
        cvar_lambda=cvar_lambda,
        cvar_tau=cvar_tau,
        supply_injection=supply_injection,
    )


def _build_level4(cfg: Dict[str, Any], rng: random.Random) -> EpisodeConfig:
    horizon = int(cfg["horizon"])
    num_nodes = int(cfg["num_nodes"])
    edges = _edge_tuples(cfg["edges"])
    supply_nodes = [int(x) for x in cfg.get("supply_nodes", [])]
    demand_nodes = [int(x) for x in cfg.get("demand_nodes", [])]
    edge_count = len(edges)

    demand_series = zero_matrix(horizon, num_nodes)
    for entry in cfg.get("demand_sinusoidal", []):
        node = int(entry["node"])
        base = float(entry.get("base", 0.0))
        amplitude = float(entry.get("amplitude", 0.0))
        phase = float(entry.get("phase", 0.0))
        values = sinusoidal_series(horizon, base, amplitude, phase=phase)
        for t in range(horizon):
            demand_series[t][node] = values[t]

    avail_cfg = cfg.get("availability", {})
    open_prob_default = float(avail_cfg.get("open_probability", 0.85))
    open_prob_per_edge_raw = avail_cfg.get("per_edge")
    if open_prob_per_edge_raw is not None:
        if len(open_prob_per_edge_raw) != edge_count:
            raise ValueError("availability.per_edge length mismatch")
        open_prob_per_edge = [float(x) for x in open_prob_per_edge_raw]
    else:
        open_prob_per_edge = [open_prob_default for _ in range(edge_count)]

    availability_series: List[List[bool]] = []
    risk_cfg = cfg.get("risk", {})
    risk_range = risk_cfg.get("range", [0.0, 0.0])
    risk_min = float(risk_range[0] if isinstance(risk_range, (list, tuple)) else 0.0)
    risk_max = float(risk_range[1] if isinstance(risk_range, (list, tuple)) else 0.0)
    risk_per_edge_raw = risk_cfg.get("per_edge_range")
    if risk_per_edge_raw is not None:
        if len(risk_per_edge_raw) != edge_count:
            raise ValueError("risk.per_edge_range length mismatch")
        risk_ranges: List[Tuple[float, float]] = []
        for pair in risk_per_edge_raw:
            if len(pair) != 2:
                raise ValueError("risk.per_edge_range entries must have length 2")
            lo, hi = float(pair[0]), float(pair[1])
            risk_ranges.append((lo, hi))
    else:
        risk_ranges = [(risk_min, risk_max) for _ in range(edge_count)]

    risk_series: List[List[float]] = []
    for _ in range(horizon):
        avail_row: List[bool] = []
        risk_row: List[float] = []
        for e_idx, _ in enumerate(edges):
            is_open = rng.random() < open_prob_per_edge[e_idx]
            avail_row.append(is_open)
            lo, hi = risk_ranges[e_idx]
            risk_row.append(rng.uniform(lo, hi) if is_open else 0.0)
        availability_series.append(avail_row)
        risk_series.append(risk_row)

    edge_capacity = _float_list(cfg["edge_capacity"], edge_count)
    edge_cost = _float_list(cfg["edge_cost"], edge_count)
    loss_penalty = _loss_penalty_list(cfg.get("loss_penalty", 1.0), edge_count)

    supply_injection = _apply_supply_injection(
        horizon, num_nodes, cfg.get("supply_injection")
    )

    initial_inventory = [float(x) for x in cfg.get("initial_inventory", [])]

    weight_cfg = cfg.get("weights", {})
    supply_reward_weight, cost_weight, risk_weight = _weights(weight_cfg)
    cvar_lambda, cvar_tau = _cvar_params(cfg.get("cvar", {}))

    return EpisodeConfig(
        level=int(cfg["level"]),
        horizon=horizon,
        num_nodes=num_nodes,
        edges=edges,
        supply_nodes=supply_nodes,
        demand_nodes=demand_nodes,
        initial_inventory=initial_inventory,
        demand_series=demand_series,
        edge_capacity=edge_capacity,
        edge_cost=edge_cost,
        availability_series=availability_series,
        risk_series=risk_series,
        loss_penalty=loss_penalty,
        supply_reward_weight=supply_reward_weight,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
        cvar_lambda=cvar_lambda,
        cvar_tau=cvar_tau,
        supply_injection=supply_injection,
    )


def _build_level5(cfg: Dict[str, Any], rng: random.Random) -> EpisodeConfig:
    horizon = int(cfg["horizon"])
    num_nodes = int(cfg["num_nodes"])
    edges = _edge_tuples(cfg["edges"])
    supply_nodes = [int(x) for x in cfg.get("supply_nodes", [])]
    demand_nodes = [int(x) for x in cfg.get("demand_nodes", [])]
    edge_count = len(edges)

    base_demands = {int(k): float(v) for k, v in cfg.get("demand_regimes", {}).get("base_demands", {}).items()}
    regimes = cfg.get("demand_regimes", {}).get("regimes", [])
    risk_spread = float(cfg.get("demand_regimes", {}).get("risk_spread", 0.28))

    def regime_for_t(t: int) -> Mapping[str, Any]:
        for entry in regimes:
            until = entry.get("until")
            if until is None or t < int(until):
                return entry
        return regimes[-1] if regimes else {}

    demand_series = zero_matrix(horizon, num_nodes)
    risk_series: List[List[float]] = []
    availability_series: List[List[bool]] = []

    avail_cfg = cfg.get("availability", {})
    primary_prob = float(avail_cfg.get("primary", 0.85))
    secondary_prob = float(avail_cfg.get("secondary", 0.78))
    primary_edges = int(avail_cfg.get("primary_edges", edge_count))
    per_edge_open_raw = avail_cfg.get("per_edge")
    if per_edge_open_raw is not None:
        if len(per_edge_open_raw) != edge_count:
            raise ValueError("availability.per_edge length mismatch")
        open_prob_per_edge = [float(x) for x in per_edge_open_raw]
    else:
        open_prob_per_edge = [
            primary_prob if e_idx < primary_edges else secondary_prob
            for e_idx in range(edge_count)
        ]

    risk_base_shift_raw = cfg.get("risk_base_shift_per_edge")
    if risk_base_shift_raw is not None:
        if len(risk_base_shift_raw) != edge_count:
            raise ValueError("risk_base_shift_per_edge length mismatch")
        risk_base_shift = [float(x) for x in risk_base_shift_raw]
    else:
        risk_base_shift = [0.0 for _ in range(edge_count)]

    for t in range(horizon):
        reg = regime_for_t(t)
        scale = float(reg.get("scale", 1.0))
        volatility = float(reg.get("volatility", 0.0))
        risk_base = float(reg.get("risk_base", 0.0))
        for node, base_val in base_demands.items():
            demand_series[t][node] = non_negative(
                base_val * scale + rng.gauss(0.0, volatility)
            )

        avail_row: List[bool] = []
        risk_row: List[float] = []
        for e_idx, _ in enumerate(edges):
            is_open = rng.random() < open_prob_per_edge[e_idx]
            avail_row.append(is_open)
            edge_risk_base = max(0.0, risk_base + risk_base_shift[e_idx])
            risk_row.append(
                rng.uniform(edge_risk_base, edge_risk_base + risk_spread) if is_open else 0.0
            )
        availability_series.append(avail_row)
        risk_series.append(risk_row)

    edge_capacity = _float_list(cfg["edge_capacity"], edge_count)
    edge_cost = _float_list(cfg["edge_cost"], edge_count)
    loss_penalty = _loss_penalty_list(cfg.get("loss_penalty", 1.0), edge_count)

    supply_injection = _apply_supply_injection(
        horizon, num_nodes, cfg.get("supply_injection")
    )

    initial_inventory = [float(x) for x in cfg.get("initial_inventory", [])]

    weight_cfg = cfg.get("weights", {})
    supply_reward_weight, cost_weight, risk_weight = _weights(weight_cfg)
    cvar_lambda, cvar_tau = _cvar_params(cfg.get("cvar", {}))

    return EpisodeConfig(
        level=int(cfg["level"]),
        horizon=horizon,
        num_nodes=num_nodes,
        edges=edges,
        supply_nodes=supply_nodes,
        demand_nodes=demand_nodes,
        initial_inventory=initial_inventory,
        demand_series=demand_series,
        edge_capacity=edge_capacity,
        edge_cost=edge_cost,
        availability_series=availability_series,
        risk_series=risk_series,
        loss_penalty=loss_penalty,
        supply_reward_weight=supply_reward_weight,
        cost_weight=cost_weight,
        risk_weight=risk_weight,
        cvar_lambda=cvar_lambda,
        cvar_tau=cvar_tau,
        supply_injection=supply_injection,
    )


_LEVEL_BUILDERS = {
    1: _build_level1,
    2: _build_level2,
    3: _build_level3,
    4: _build_level4,
    5: _build_level5,
}


def generate_episode_config(level: int, seed: int) -> EpisodeConfig:
    rng = random.Random(seed)
    builder = _LEVEL_BUILDERS.get(level)
    if builder is None:
        raise ValueError(f"Unsupported level: {level}")
    cfg = _load_yaml(level)
    return builder(cfg, rng)


def generate_episode_config_from_file(config_path: Path | str, seed: int) -> EpisodeConfig:
    path = Path(config_path)
    cfg = _load_yaml_file(path)
    level = int(cfg.get("level", -1))
    builder = _LEVEL_BUILDERS.get(level)
    if builder is None:
        raise ValueError(f"Unsupported level in config ({level}) for file: {path}")
    rng = random.Random(seed)
    return builder(cfg, rng)


__all__ = ["generate_episode_config", "generate_episode_config_from_file"]
