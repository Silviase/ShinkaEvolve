from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# imageio is optional; fall back to Pillow-only GIF if unavailable.
try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    imageio = None

from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FILE_DIR = Path(__file__).resolve().parent
from examples.supply_chain.config_loader import (
    generate_episode_config,
    generate_episode_config_from_file,
)
from examples.supply_chain.initial import logistics_policy
from examples.supply_chain.simulation import simulate_episode


def load_policy(policy_path: Path):
    """Load logistics_policy(state) from an arbitrary Python file."""
    spec = importlib.util.spec_from_file_location("custom_logistics_policy", policy_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load module from {policy_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = getattr(module, "logistics_policy", None)
    if policy is None:
        raise AttributeError(f"{policy_path} does not define `logistics_policy`")
    return policy


def _build_graph(edges: List[Tuple[int, int]], seed: int) -> Tuple[nx.DiGraph, Dict[int, Tuple[float, float]]]:
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted({n for e in edges for n in e}))
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph, seed=seed)
    return graph, pos


def _scale(value: float, max_value: float, base: float, span: float) -> float:
    if max_value <= 1e-8:
        return base
    return base + span * (value / max_value)


def _draw_step(
    ax,
    graph: nx.DiGraph,
    pos: Dict[int, Tuple[float, float]],
    config,
    step: Dict[str, List[float]],
    max_inv: float,
    max_ship: float,
    max_demand: float,
) -> None:
    ax.clear()
    ax.axis("off")
    t = step["t"]
    inventory = step["inventory_start"]
    demand = step["demand"]
    shipments = step["shipments"]
    arrivals = step["arrivals"]
    available = step["edge_available"]

    node_sizes = [_scale(inventory[i], max_inv, base=400, span=800) for i in graph.nodes()]

    node_colors = []
    for idx in graph.nodes():
        if idx in config.supply_nodes:
            node_colors.append("#2ecc71")  # green
        elif idx in config.demand_nodes:
            node_colors.append("#e74c3c")  # red
        else:
            node_colors.append("#3498db")  # blue

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=node_sizes, node_color=node_colors, edgecolors="black")
    nx.draw_networkx_labels(
        graph,
        pos,
        labels={i: f"{i}\nInv:{inventory[i]:.1f}\nDem:{demand[i]:.1f}" for i in graph.nodes()},
        font_size=8,
        font_color="black",
    )

    edge_colors = []
    widths = []
    for e_idx, (src, dst) in enumerate(config.edges):
        ship_amt = shipments[e_idx]
        is_open = available[e_idx]
        widths.append(_scale(ship_amt, max_ship, base=1.0, span=4.0 if is_open else 0.0))
        if not is_open:
            edge_colors.append("#bdc3c7")  # gray for closed
        elif ship_amt > 1e-6:
            edge_colors.append("#f39c12")  # orange when used
        else:
            edge_colors.append("#7f8c8d")  # muted gray when idle

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=edge_colors,
        width=widths,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.07",
    )

    edge_labels = {}
    for e_idx, (src, dst) in enumerate(config.edges):
        if shipments[e_idx] > 1e-3:
            edge_labels[(src, dst)] = f"{shipments[e_idx]:.1f}"
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, rotate=False, ax=ax)

    ax.set_title(f"Level {config.level} | Step {t + 1}/{config.horizon}", fontsize=12)
    if max_demand > 1e-6:
        fill_rate = sum(step.get("fulfilled", [])) / max(1e-6, sum(demand))
        ax.text(0.01, 0.97, f"Step fill rate: {fill_rate:.2f}", transform=ax.transAxes, fontsize=9)


def render_frames(config, history: List[Dict[str, List[float]]], seed: int, out_dir: Path, fps: float) -> Path:
    graph, pos = _build_graph(config.edges, seed)
    max_inv = max(max(step["inventory_start"]) for step in history) if history else 1.0
    max_ship = max(max(step["shipments"]) for step in history) if history else 1.0
    max_demand = max(max(step["demand"]) for step in history) if history else 1.0

    frame_paths: List[Path] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_dir_path = Path(tmpdir)
        for step in history:
            fig, ax = plt.subplots(figsize=(7, 5))
            _draw_step(ax, graph, pos, config, step, max_inv, max_ship, max_demand)
            frame_file = tmp_dir_path / f"frame_{step['t']:03d}.png"
            fig.tight_layout()
            fig.savefig(frame_file, dpi=120)
            plt.close(fig)
            frame_paths.append(frame_file)

        gif_path = out_dir / "episode.gif"
        if imageio is not None:
            images = [imageio.imread(fp) for fp in frame_paths]
            imageio.mimsave(gif_path, images, fps=fps)
        else:
            pil_images = [Image.open(fp).convert("RGBA") for fp in frame_paths]
            duration_ms = int(1000 / max(fps, 1e-3))
            pil_images[0].save(
                gif_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=duration_ms,
                loop=0,
                disposal=2,
            )
    return gif_path


def _pick_default_config(level: int) -> Path | None:
    """Try repo defaults first; fall back to tests/level{level}_base.yaml."""
    primary = FILE_DIR / "configs" / f"level{level}.yaml"
    if primary.exists():
        return primary
    fallback = FILE_DIR / "configs" / "tests" / f"level{level}_base.yaml"
    if fallback.exists():
        return fallback
    return None


def main(
    level: int,
    seed: int,
    run_name: str,
    results_root: Path,
    fps: float,
    config_path: Path | None = None,
    policy_path: Path | None = None,
):
    results_dir = results_root / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    if config_path:
        config = generate_episode_config_from_file(config_path, seed)
    else:
        default_cfg = _pick_default_config(level)
        if default_cfg:
            config = generate_episode_config_from_file(default_cfg, seed)
        else:
            config = generate_episode_config(level, seed)

    policy_fn = logistics_policy if policy_path is None else load_policy(policy_path)
    result = simulate_episode(config, policy_fn, seed=seed, return_history=True)
    history = result.get("history", [])

    gif_path = render_frames(config, history, seed, results_dir, fps=fps)

    metrics_path = results_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump({k: v for k, v in result.items() if k != "history"}, f, indent=2)

    print(f"Saved GIF to {gif_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize logistics episodes as GIF.")
    parser.add_argument("--level", type=int, default=1, help="Curriculum level (1-5).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for scenario and layout.")
    parser.add_argument(
        "--run_name",
        type=str,
        default="demo",
        help="Subdirectory name under results/supply_chain.",
    )
    parser.add_argument(
        "--config_path",
        type=Path,
        help="Optional path to a config YAML/JSON; if set, overrides --level.",
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results") / "supply_chain",
        help="Root directory to store outputs.",
    )
    parser.add_argument("--fps", type=float, default=1.5, help="Frames per second for GIF.")
    parser.add_argument(
        "--policy_path",
        type=Path,
        help="Optional path to a Python file containing logistics_policy(state).",
    )
    args = parser.parse_args()
    main(
        level=args.level,
        seed=args.seed,
        run_name=args.run_name,
        results_root=args.results_root,
        fps=args.fps,
        config_path=args.config_path,
        policy_path=args.policy_path,
    )
