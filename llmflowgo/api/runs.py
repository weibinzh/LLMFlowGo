import uuid
from typing import List, Dict, Any, Optional
import multiprocessing
from pydantic import BaseModel
import shutil
from pathlib import Path
import importlib.util
import os
import ast

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from my_meoh_app.database import get_session
from my_meoh_app.models.run import Run, RunCreate, RunRead
from my_meoh_app.core.meoh_runner import run_meoh_optimization

router = APIRouter()

# Define project root and problem runs directory to locate run artifacts
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROBLEM_RUNS_DIR = PROJECT_ROOT / "problem_runs"


# --- DIAGNOSTIC: A simple sentinel task ---
def simple_background_task(message: str):
    """A task with no external dependencies to test the background runner."""
    import time
    print("\n" + "*"*25 + " SIMPLE BACKGROUND TASK LOG " + "*"*25)
    print(f"[*] Simple task started successfully.")
    print(f"[*] Message received: '{message}'")
    print(f"[*] This proves the background task runner IS WORKING.")
    time.sleep(2) # Keep it short
    print("*"*26 + " SIMPLE TASK FINISHED " + "*"*27 + "\n")
# --- END DIAGNOSTIC ---


@router.post("/runs", response_model=RunRead)
def create_run(
    *,
    session: Session = Depends(get_session),
    run_create: RunCreate, # Use the new clean model
):
    """
    Create a new run for a problem package.
    The standalone executor service will pick up pending runs.
    """
    # Create a Run instance from the RunCreate model
    db_run = Run.model_validate(run_create)
    db_run.status = "pending" # Explicitly set status

    session.add(db_run)
    session.commit()
    session.refresh(db_run)

    return db_run


@router.get("/runs")
def get_runs(
    problem_id: Optional[uuid.UUID] = None,
    session: Session = Depends(get_session)
):
    """
    Get all runs, optionally filtered by problem_id.
    """
    if problem_id:
        # If problem_id is provided, filter run records
        runs = session.exec(select(Run).where(Run.problem_id == problem_id).order_by(Run.created_at.desc())).all()
    else:
        # Otherwise return all run records
        runs = session.exec(select(Run).order_by(Run.created_at.desc())).all()
    return runs


@router.get("/runs/{run_id}")
def get_run_details(run_id: uuid.UUID, session: Session = Depends(get_session)):
    """
    Get details of a specific run.
    """
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.delete("/runs/{run_id}")
def delete_run(run_id: uuid.UUID, session: Session = Depends(get_session)):
    """
    Delete a specific run and its associated files.
    This will remove the run from the database and delete its directory from `problem_runs`.
    """
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Delete the run directory from the filesystem
    run_dir = PROBLEM_RUNS_DIR / str(run_id)
    if run_dir.is_dir():
        shutil.rmtree(run_dir)

    session.delete(run)
    session.commit()

    return {"message": "Run deleted successfully"}


# ---------------- NEW ENDPOINT: Final Counts ----------------
@router.get("/runs/{run_id}/final-counts")
def get_final_counts(run_id: uuid.UUID, session: Session = Depends(get_session)):
    """
    Evaluate the stored best solution code inside the run's isolated directory
    and return a dict with keys: cloud, edge, device, makespan, energy.
    """
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Ensure run completed and has best_solution_code
    if getattr(run, "status", None) != "completed":
        raise HTTPException(status_code=409, detail=f"Run status is '{run.status}', not completed")
    code_str = getattr(run, "best_solution_code", None)
    if not code_str:
        raise HTTPException(status_code=404, detail="Best solution code not available for this run")

    run_dir = PROBLEM_RUNS_DIR / str(run_id)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run directory not found on disk")

    framework_path = run_dir / "framework.py"
    evaluation_path = run_dir / "evaluation.py"

    # Support get_instance.py and get_instance_adapted_corrected.py
    gi_candidates = [
        run_dir / "get_instance.py",
        run_dir / "get_instance_adapted_corrected.py",
    ]
    get_instance_path = None
    for cand in gi_candidates:
        if Path(cand).exists():
            get_instance_path = cand
            break
    if get_instance_path is None:
        raise HTTPException(status_code=500, detail="Required run artifact missing: get_instance.py")

    # Resolve converted_instance.json path: manifest -> root -> data subdirectory
    converted_instance_path = run_dir / "converted_instance.json"
    manifest_path = run_dir / "manifest.json"
    if not converted_instance_path.exists():
        try:
            if manifest_path.exists():
                import json
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
            raise HTTPException(status_code=500, detail=f"Required run artifact missing: {p}")

    # Helper to load a module from file path
    def _load_module(module_name: str, file_path: Path):
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise HTTPException(status_code=500, detail=f"Failed to create spec for {file_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading module {module_name} from {file_path}: {e}")
        return module

    framework_mod = _load_module("framework", framework_path)
    evaluation_mod = _load_module("evaluation", evaluation_path)
    get_instance_mod = _load_module("get_instance", get_instance_path)

    # Build problem instance using the get_instance module
    if not hasattr(get_instance_mod, "get_problem_instance"):
        raise HTTPException(status_code=500, detail="get_problem_instance() not found in get_instance module")
    try:
        import inspect
        fn = getattr(get_instance_mod, "get_problem_instance")
        sig = inspect.signature(fn)
        if len(sig.parameters) == 0:
            problem_instance = fn()
        else:
            problem_instance = fn(converted_instance_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building problem instance: {e}")

    # Execute the best solution function code inside framework module namespace
    try:
        compiled = compile(code_str, filename="best_solution.py", mode="exec")
        exec(compiled, framework_mod.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing best solution code: {e}")

    # Ensure solve exists
    if not hasattr(framework_mod, "solve"):
        raise HTTPException(status_code=500, detail="solve() function not available in framework module after patch")

    # Call solve to obtain solution, then evaluate
    try:
        solution = framework_mod.solve(problem_instance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling solve(): {e}")

    if not hasattr(evaluation_mod, "evaluate_solution"):
        raise HTTPException(status_code=500, detail="evaluate_solution() not found in evaluation module")

    try:
        result = evaluation_mod.evaluate_solution(solution, problem_instance)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during evaluation: {e}")

    # Validate result keys
    expected_keys = {"cloud", "edge", "device", "makespan", "energy", "cost"}
    if not isinstance(result, dict) or not expected_keys.issubset(set(result.keys())):
        raise HTTPException(status_code=500, detail=f"Evaluation returned invalid result shape: {result}")

    return {
        "cloud": int(result["cloud"]),
        "edge": int(result["edge"]),
        "device": int(result["device"]),
        "makespan": float(result["makespan"]),
        "energy": float(result["energy"]),
        "cost": float(result["cost"]),
    }


# ---------------- NEW ENDPOINT: Standardized Results Payload ----------------
@router.get("/runs/{run_id}/results")
def get_run_results(run_id: uuid.UUID, session: Session = Depends(get_session)):
    """
    Return standardized final results payload for a run.
    If payload is missing, evaluate counts/metrics from best_solution_code as fallback.
    """
    run = session.get(Run, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # If payload already persisted, return it directly
    if getattr(run, "final_result_json", None):
        return run.final_result_json

    # Fallback: compute from best_solution_code if run completed
    if getattr(run, "status", None) != "completed":
        raise HTTPException(status_code=409, detail=f"Run status is '{run.status}', not completed")
    code_str = getattr(run, "best_solution_code", None)
    if not code_str:
        raise HTTPException(status_code=404, detail="Best solution code not available for this run")

    run_dir = PROBLEM_RUNS_DIR / str(run_id)
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail="Run directory not found on disk")

    # Use adapter for robust evaluation and payload construction
    from my_meoh_app.core.meoh_result_adapter import compute_counts_metrics_from_best_solution, build_final_result_payload

    try:
        cm = compute_counts_metrics_from_best_solution(run_dir, code_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute final results: {e}")

    payload = build_final_result_payload(run, cm)

    # Persist for future requests
    run.final_result_json = payload
    run.cloud_count = cm.get("cloud")
    run.edge_count = cm.get("edge")
    run.device_count = cm.get("device")
    run.makespan = cm.get("makespan")
    run.energy = cm.get("energy")
    session.add(run)
    session.commit()

    return payload
