from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_metrics(metrics_path: Path, metric_key: str) -> float | None:
    with metrics_path.open() as f:
        data = json.load(f)
    value = data.get(metric_key)
    return float(value) if value is not None else None


def collect_generation_scores(run_dir: Path, metric_key: str) -> Tuple[List[int], List[float], float | None]:
    gen_pattern = re.compile(r"gen_(\d+)$")
    gens: List[int] = []
    scores: List[float] = []
    best_score: float | None = None

    for child in run_dir.iterdir():
        if not child.is_dir():
            continue
        metrics_path = child / "results" / "metrics.json"
        if not metrics_path.exists():
            continue

        if child.name == "best":
            best_score = load_metrics(metrics_path, metric_key)
        else:
            m = gen_pattern.match(child.name)
            if m:
                gen_num = int(m.group(1))
                score = load_metrics(metrics_path, metric_key)
                if score is not None:
                    gens.append(gen_num)
                    scores.append(score)

    # Sort by generation number
    paired = sorted(zip(gens, scores), key=lambda x: x[0])
    gens = [g for g, _ in paired]
    scores = [s for _, s in paired]
    return gens, scores, best_score


def load_policy_doc(main_py: Path) -> str:
    spec = importlib.util.spec_from_file_location("policy_module", main_py)
    if spec is None or spec.loader is None:
        return "Unable to load module."
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = getattr(module, "logistics_policy", None)
    if policy is None:
        return "`logistics_policy` not found."
    doc = getattr(policy, "__doc__", None)
    if not doc:
        return "No docstring."
    # Collapse whitespace for readability
    return " ".join(doc.split())


def main(run_dir: Path, metric_key: str, output_path: Path, show_policy_doc: bool) -> None:
    run_dir = run_dir.resolve()
    gens, scores, best_score = collect_generation_scores(run_dir, metric_key)

    if not gens:
        raise SystemExit(f"No generation metrics found under {run_dir}")

    print(f"Loaded {len(gens)} generations from {run_dir}")
    print(f"Metric: {metric_key}")
    for g, s in zip(gens, scores):
        print(f"gen_{g}: {s:.4f}")
    if best_score is not None:
        print(f"best: {best_score:.4f}")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(gens, scores, marker="o", label="gen score")
    if best_score is not None:
        plt.axhline(best_score, color="red", linestyle="--", label="best")
    plt.xlabel("generation")
    plt.ylabel(metric_key)
    plt.title(f"Score trajectory for {run_dir.name}")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")

    if show_policy_doc:
        print("\nPolicy docstrings per generation:")
        gen_pattern = re.compile(r"gen_(\d+)$")
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "best":
                tag = "best"
            else:
                m = gen_pattern.match(child.name)
                if not m:
                    continue
                tag = f"gen_{m.group(1)}"

            main_py = child / "main.py"
            if not main_py.exists():
                continue
            doc = load_policy_doc(main_py)
            print(f"{tag}: {doc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot gen-to-gen score trajectory for a Shinka run.")
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to run directory (e.g., results/shinka_supply_chain/.../2025.xx_xx...).",
    )
    parser.add_argument(
        "--metric_key",
        type=str,
        default="combined_score",
        help="Metric key to plot from metrics.json (default: combined_score).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path for the plot (default: <run_dir>/gen_scores.png).",
    )
    parser.add_argument(
        "--show_policy_doc",
        action="store_true",
        help="Print logistic_policy docstrings for each generation.",
    )
    args = parser.parse_args()
    out_path = args.output or (args.run_dir / "gen_scores.png")
    main(args.run_dir, args.metric_key, out_path, args.show_policy_doc)
