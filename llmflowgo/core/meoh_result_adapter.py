from __future__ import annotations
from datetime import datetime
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional

class EvaluationError(Exception):
    pass


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise EvaluationError(f"Failed to create spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore
    except Exception as e:
        raise EvaluationError(f"Error loading module {module_name} from {file_path}: {e}")
    return module


def compute_counts_metrics_from_best_solution(run_dir: Path, best_solution_code: str) -> Dict[str, Any]:
    """Evaluate best_solution_code with run artifacts to produce counts & metrics.
    Returns dict with keys: cloud, edge, device, makespan, energy.
    """
    framework_path = run_dir / "framework.py"
    evaluation_path = run_dir / "evaluation.py"

    # support multiple get_instance filenames
    gi_candidates = [
        run_dir / "get_instance.py",
        run_dir / "get_instance_adapted_corrected.py",
    ]
    get_instance_path = None
    for cand in gi_candidates:
        if cand.exists():
            get_instance_path = cand
            break
    if get_instance_path is None:
        raise EvaluationError("Required run artifact missing: get_instance.py")

    converted_instance_path = run_dir / "converted_instance.json"
    manifest_path = run_dir / "manifest.json"
    if not converted_instance_path.exists():
        try:
            import json
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    m = json.load(mf)
                ci_rel = m.get("converted_instance")
                if isinstance(ci_rel, str):
                    ci_candidate = run_dir / ci_rel if not Path(ci_rel).is_absolute() else Path(ci_rel)
                    if ci_candidate.exists():
                        converted_instance_path = ci_candidate
        except Exception:
            pass
        if not converted_instance_path.exists():
            data_ci = run_dir / "data" / "converted_instance.json"
            if data_ci.exists():
                converted_instance_path = data_ci

    for p in [framework_path, evaluation_path, get_instance_path, converted_instance_path]:
        if not Path(p).exists():
            raise EvaluationError(f"Required run artifact missing: {p}")

    framework_mod = _load_module("framework", framework_path)
    evaluation_mod = _load_module("evaluation", evaluation_path)
    get_instance_mod = _load_module("get_instance", get_instance_path)

    # Build problem instance
    if not hasattr(get_instance_mod, "get_problem_instance"):
        raise EvaluationError("get_problem_instance() not found in get_instance module")
    try:
        import inspect
        fn = getattr(get_instance_mod, "get_problem_instance")
        sig = inspect.signature(fn)
        if len(sig.parameters) == 0:
            problem_instance = fn()
        else:
            problem_instance = fn(converted_instance_path)
    except Exception as e:
        raise EvaluationError(f"Error building problem instance: {e}")

    # Execute best solution code within framework namespace
    try:
        compiled = compile(best_solution_code, filename="best_solution.py", mode="exec")
        exec(compiled, framework_mod.__dict__)
    except Exception as e:
        raise EvaluationError(f"Error executing best solution code: {e}")

    if not hasattr(framework_mod, "solve"):
        raise EvaluationError("solve() function not available in framework module after patch")

    try:
        solution = framework_mod.solve(problem_instance)
    except Exception as e:
        raise EvaluationError(f"Error calling solve(): {e}")

    if not hasattr(evaluation_mod, "evaluate_solution"):
        raise EvaluationError("evaluate_solution() not found in evaluation module")

    try:
        result = evaluation_mod.evaluate_solution(solution, problem_instance)
    except Exception as e:
        raise EvaluationError(f"Error during evaluation: {e}")

    required = {"cloud", "edge", "device", "makespan", "energy", "cost"}
    if not isinstance(result, dict) or not required.issubset(set(result.keys())):
        raise EvaluationError(f"Evaluation returned invalid result shape: {result}")

    # Load objective names if available
    objective_names = None
    try:
        import json as _json
        names_path = run_dir / "objective_names.json"
        if names_path.exists():
            with open(names_path, "r", encoding="utf-8") as nf:
                names_loaded = _json.load(nf)
            if isinstance(names_loaded, list):
                objective_names = names_loaded
    except Exception:
        pass

    return {
        "cloud": int(result["cloud"]),
        "edge": int(result["edge"]),
        "device": int(result["device"]),
        "makespan": float(result["makespan"]),
        "energy": float(result["energy"]),
        "cost": float(result["cost"]),
        "objective_names": objective_names,
    }


def build_final_result_payload(run_obj, counts_metrics: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Construct a forward-compatible results payload from run data and counts/metrics."""
    # Ensure datetimes are JSON-serializable
    def _to_iso(val):
        return val.isoformat() if isinstance(val, datetime) else val

    started = getattr(run_obj, "start_time", None)
    finished = getattr(run_obj, "end_time", None)

    payload = {
        "run": {
            "id": str(getattr(run_obj, "id", None)),
            "status": getattr(run_obj, "status", None),
            "startedAt": _to_iso(started),
            "finishedAt": _to_iso(finished),
        },
        "counts": {
            "cloud": counts_metrics.get("cloud"),
            "edge": counts_metrics.get("edge"),
            "device": counts_metrics.get("device"),
        },
        "metrics": {
            "makespan": counts_metrics.get("makespan"),
            "energy": counts_metrics.get("energy"),
            "cost": counts_metrics.get("cost"),
        },
        "meta": {
            "schema_version": "v1",
            "source": "meoh",
            "objective_names": counts_metrics.get("objective_names"),
        },
    }
    if extra:
        payload["artifacts"] = extra
    return payload