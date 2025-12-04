from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

FILE_DIR = Path(__file__).resolve().parent
REPO_ROOT = FILE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # Prefer full implementation if deps available
    from shinka.core import run_shinka_eval  # type: ignore
except Exception as e:  # pragma: no cover - optional deps (numpy/pandas/hydra)
    import importlib.util
    import json
    import os
    import pickle
    import time
    import statistics
    from typing import Optional, Tuple

    DEFAULT_METRICS_ON_ERROR = {
        "combined_score": 0.0,
        "execution_time_mean": 0.0,
        "execution_time_std": 0.0,
        "num_successful_runs": 0,
        "num_valid_runs": 0,
        "num_invalid_runs": 0,
        "all_validation_errors": [],
    }

    def _load_program(program_path: str) -> Any:
        spec = importlib.util.spec_from_file_location("program", program_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec for module at {program_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _save_json_results(
        results_dir: str,
        metrics: Dict[str, Any],
        correct: bool,
        error: Optional[str] = None,
    ) -> None:
        os.makedirs(results_dir, exist_ok=True)
        correct_payload = {"correct": correct, "error": error}
        with open(os.path.join(results_dir, "correct.json"), "w") as f:
            json.dump(correct_payload, f, indent=4)
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

    def run_shinka_eval(  # type: ignore
        program_path: str,
        results_dir: str,
        experiment_fn_name: str,
        num_runs: int,
        get_experiment_kwargs: Optional[Callable[[int], Dict[str, Any]]] = None,
        aggregate_metrics_fn: Optional[Callable[[List[Any]], Dict[str, Any]]] = None,
        validate_fn: Optional[Callable[[Any], Tuple[bool, Optional[str]]]] = None,
        default_metrics_on_error: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        effective_default = default_metrics_on_error.copy() if default_metrics_on_error else DEFAULT_METRICS_ON_ERROR.copy()
        overall_correct = True
        first_error: Optional[str] = None
        all_errors: List[str] = []
        num_valid = num_invalid = 0
        all_results: List[Any] = []
        execution_times: List[float] = []
        try:
            module = _load_program(program_path)
            if not hasattr(module, experiment_fn_name):
                raise AttributeError(f"Experiment function '{experiment_fn_name}' not found in {program_path}")
            experiment_fn = getattr(module, experiment_fn_name)
            for i in range(num_runs):
                kwargs = get_experiment_kwargs(i) if get_experiment_kwargs else {"seed": i + 1}
                start = time.perf_counter()
                run_result = experiment_fn(**kwargs)
                end = time.perf_counter()
                execution_times.append(end - start)
                all_results.append(run_result)
                if validate_fn:
                    valid, err = validate_fn(run_result)
                    if not valid:
                        num_invalid += 1
                        overall_correct = False
                        if err:
                            if not first_error:
                                first_error = f"Validation failed: {err}"
                            if err not in all_errors:
                                all_errors.append(err)
                    else:
                        num_valid += 1
                print(f"Run {i + 1}/{num_runs} completed in {end - start:.2f} seconds")
            metrics = aggregate_metrics_fn(all_results) if aggregate_metrics_fn else {"num_successful_runs": len(all_results)}
            metrics["execution_time_mean"] = statistics.mean(execution_times) if execution_times else 0.0
            metrics["execution_time_std"] = statistics.pstdev(execution_times) if len(execution_times) > 1 else 0.0
            if validate_fn:
                metrics["num_valid_runs"] = num_valid
                metrics["num_invalid_runs"] = num_invalid
                metrics["all_validation_errors"] = all_errors
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"Evaluation error: {exc}")
            metrics = {k: effective_default.get(k, v) for k, v in DEFAULT_METRICS_ON_ERROR.items()}
            if validate_fn:
                metrics.setdefault("num_valid_runs", num_valid)
                metrics.setdefault("num_invalid_runs", num_invalid or num_runs)
                metrics.setdefault("all_validation_errors", [str(exc)])
            first_error = str(exc)
            overall_correct = False
        if "extra_data" in metrics:
            os.makedirs(results_dir, exist_ok=True)
            extra = metrics.pop("extra_data")
            with open(os.path.join(results_dir, "extra.pkl"), "wb") as f:
                pickle.dump(extra, f)
        _save_json_results(results_dir, metrics, overall_correct, first_error)
        return metrics, overall_correct, first_error

from examples.supply_chain.config_loader import generate_episode_config_from_file
DEFAULT_CONFIG_DIR = FILE_DIR / "configs" / "tests"


def compute_cvar(values: Sequence[float], tau: float) -> float:
    """Computes CVaR on the worst tau fraction (larger values are worse)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values, reverse=True)
    cutoff = max(1, int(math.ceil(tau * len(sorted_vals))))
    return float(sum(sorted_vals[:cutoff]) / cutoff)


def validate_episode_result(result: Any) -> tuple[bool, str | None]:
    if not isinstance(result, dict):
        return False, "Result is not a dictionary."
    required_keys = ["level", "score", "fulfilled", "cost", "loss"]
    for key in required_keys:
        if key not in result:
            return False, f"Missing key '{key}' in result."
        try:
            value = float(result[key])
        except Exception:
            return False, f"Value for '{key}' is not numeric."
        if not math.isfinite(value):
            return False, f"Value for '{key}' is not finite."
    return True, None


def parse_config_dir(path_str: str | None) -> List[Path]:
    raw = Path(path_str) if path_str is not None else DEFAULT_CONFIG_DIR
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append((FILE_DIR / raw).resolve())
        candidates.append((REPO_ROOT / raw).resolve())
    config_dir: Path | None = None
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            config_dir = cand
            break
    if config_dir is None:
        raise argparse.ArgumentTypeError(
            f"Config dir not found. Tried: {', '.join(str(c) for c in candidates)}"
        )
    paths = sorted(p for p in config_dir.glob("*.yaml") if p.is_file())
    if not paths:
        raise argparse.ArgumentTypeError(f"No YAML configs found in {config_dir}")
    return paths


def extract_levels(config_paths: List[Path]) -> List[int]:
    levels: set[int] = set()
    for path in config_paths:
        cfg = generate_episode_config_from_file(path, seed=0)
        levels.add(int(cfg.level))
    return sorted(levels)


def make_kwargs_fn_from_configs(
    config_paths: List[Path], base_seed: int
) -> Callable[[int], Dict[str, Any]]:
    total = len(config_paths)

    def _kwargs(run_idx: int) -> Dict[str, Any]:
        cfg_path = config_paths[run_idx % total]
        seed = base_seed + run_idx
        return {"config_path": str(cfg_path), "seed": seed}

    return _kwargs


def make_aggregator(levels: List[int]) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        grouped: Dict[int, List[Dict[str, Any]]] = {lvl: [] for lvl in levels}
        for res in results:
            lvl = res.get("level")
            if lvl in grouped:
                grouped[lvl].append(res)

        public_metrics: Dict[str, Any] = {}
        private_metrics: Dict[str, Any] = {}
        level_scores: List[float] = []

        for lvl in levels:
            runs = grouped.get(lvl, [])
            if not runs:
                continue
            mean_score = (
                sum(float(r.get("normalized_score", r.get("score", 0.0))) for r in runs)
                / len(runs)
            )
            avg_fill_rate = (
                sum(float(r.get("fulfillment_rate", 0.0)) for r in runs) / len(runs)
            )
            mean_cost = sum(float(r.get("cost", 0.0)) for r in runs) / len(runs)

            public_metrics[f"level{lvl}_score"] = float(mean_score)
            public_metrics[f"level{lvl}_fill_rate"] = float(avg_fill_rate)
            private_metrics[f"level{lvl}_mean_cost"] = float(mean_cost)

            if lvl == 5:
                z_values = [
                    float(r.get("normalized_z_value", r.get("z_value", 0.0)))
                    for r in runs
                ]
                lam = float(runs[0].get("cvar_lambda", 0.0))
                tau = float(runs[0].get("cvar_tau", 0.25))
                cvar_value = compute_cvar(z_values, tau)
                risk_adjusted = avg_fill_rate - lam * cvar_value
                public_metrics["level5_cvar"] = float(cvar_value)
                public_metrics["level5_risk_adjusted"] = float(risk_adjusted)
                private_metrics["level5_z_values"] = z_values
                level_scores.append(risk_adjusted)
            else:
                level_scores.append(mean_score)

        combined_score = float(sum(level_scores) / len(level_scores)) if level_scores else 0.0
        public_metrics["levels_evaluated"] = levels
        metrics = {
            "combined_score": combined_score,
            "public": public_metrics,
            "private": private_metrics,
        }
        return metrics

    return _aggregate


def main(
    program_path: str,
    results_dir: str,
    base_seed: int,
    config_dir: str | None = None,
):
    # Resolve program_path relative to repo/examples dir if a bare filename is given.
    raw_prog = Path(program_path)
    if not raw_prog.is_absolute():
        candidates = [
            raw_prog,
            FILE_DIR / raw_prog,
            REPO_ROOT / raw_prog,
        ]
        for cand in candidates:
            if cand.exists():
                raw_prog = cand
                break
    program_path = str(raw_prog)

    # New default: run all YAMLs under configs/tests (including base levels and test cases).
    config_paths = parse_config_dir(config_dir)
    levels = extract_levels(config_paths)
    kwargs_fn = make_kwargs_fn_from_configs(config_paths, base_seed)
    aggregator = make_aggregator(levels)
    total_runs = len(config_paths)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=total_runs,
        get_experiment_kwargs=kwargs_fn,
        aggregate_metrics_fn=aggregator,
        validate_fn=validate_episode_result,
    )

    status = "success" if correct else "failed"
    print(f"Evaluation {status}. Combined score: {metrics.get('combined_score', 0.0):.4f}")
    if error_msg:
        print(f"First error: {error_msg}")
    print("Public metrics:")
    for key, value in metrics.get("public", {}).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate logistics policies across curriculum levels."
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default=str(FILE_DIR / "initial.py"),
        help="Path to the program containing 'run_experiment'.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/supply_chain",
        help="Directory to save metrics and correctness files.",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=13,
        help="Base seed; individual runs increment from this value.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=str(DEFAULT_CONFIG_DIR),
        help="Directory containing YAML configs to evaluate (defaults to configs/tests).",
    )
    args = parser.parse_args()
    main(
        program_path=args.program_path,
        results_dir=args.results_dir,
        base_seed=args.base_seed,
        config_dir=args.config_dir,
    )
