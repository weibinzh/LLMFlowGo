import time
import uuid
import shutil
import os
import json
import sys
import psutil
import gc
from datetime import datetime, timezone
from sqlmodel import Session, select
import traceback
from contextlib import contextmanager
import numpy as np
from pathlib import Path
import importlib.util

from my_meoh_app.database import engine
from my_meoh_app.models.problem import ProblemPackage
from my_meoh_app.models.run import Run
from my_meoh_app.core.evaluation_runner import MEOHEvaluationWrapper
from my_meoh_app.tools.llm import get_llm_from_config
from my_meoh_app.core.meoh_result_adapter import compute_counts_metrics_from_best_solution, build_final_result_payload
from MEoh import MEoH, MEoHProfiler
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROBLEM_RUNS_DIR = PROJECT_ROOT / "problem_runs"
# Process Resource Monitoring and Cleanup Functions



def monitor_process_resources():
    """Monitor the resource usage of the current process"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        return {
            'memory_mb': memory_info.rss / 1024 / 1024,
            'cpu_percent': cpu_percent,
            'num_threads': process.num_threads(),
            'num_handles': process.num_handles() if hasattr(process, 'num_handles') else 'N/A'
        }
    except Exception:
        return None

def cleanup_process_resources():
    """Clean up resources used by the current process"""
    try:
        # Force garbage collection
        collected = gc.collect()
        print(f"[{os.getpid()}] Garbage collection: collected {collected} objects")
        
        # Clean up file handles (only on Unix systems)
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"[{os.getpid()}] File descriptor limits: soft={soft}, hard={hard}")
        except ImportError:
            pass
            
    except Exception as e:
        print(f"[{os.getpid()}] Error during resource cleanup: {e}")

def log_resource_usage(message=""):
    """Log resource usage of the current process"""
    resources = monitor_process_resources()
    if resources:
        print(f"[{os.getpid()}] Resource usage {message}: "
              f"Memory: {resources['memory_mb']:.1f}MB, "
              f"CPU: {resources['cpu_percent']:.1f}%, "
              f"Threads: {resources['num_threads']}")

# Database session management

@contextmanager
def get_session():
    """Provide a transactional scope around a series of operations."""
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def _append_run_log(session: Session, run: Run, message: str) -> None:
    """Add a line to the run log and commit to the database"""
    ts = datetime.now(timezone.utc).isoformat()
    prefix = f"[{ts}] "
    try:
        current = run.logs or ""
        # Keep log size reasonable
        if len(current) > 50000:
            current = current[-40000:]
        run.logs = (current + ("\n" if current else "") + prefix + message)
        session.add(run)
        session.commit()
    except Exception:
        pass


# Objective count detection function


def _get_num_objs(run_path: Path, template_program: str, task_description: str) -> int:
    """
    Run a single evaluation to determine the number of objectives.
    """
    print(f"[{os.getpid()}] Performing a dry run to determine the number of objectives...")
    
    eval_wrapper = MEOHEvaluationWrapper(run_path, template_program, task_description, PROJECT_ROOT, num_objs=0)
    scores = eval_wrapper.evaluate_program(template_program)
    
    if scores is None or not isinstance(scores, list) or not scores:
        log_path = run_path / 'subprocess_error.log'
        raise ValueError(
            f"Dry run failed. The evaluation subprocess returned an invalid result (likely None due to a crash). "
            f"Check the subprocess_error.log in the run directory '{run_path}' for the root cause."
        )
    
    first = scores[0]
    if isinstance(first, (int, float)):
        num_objs = len(scores)
    elif isinstance(first, (list, tuple)) and len(first) > 0:
        num_objs = len(first)
    else:
        raise TypeError("Dry run returned an unexpected shape; expected list of floats or list of list of floats.")

    print(f"[{os.getpid()}] Dry run successful. Number of objectives found: {num_objs}")
    return num_objs


# One-click optimization runner function


def run_meoh_optimization_oneclick(problem_id_str: str, run_id_str: str):
    """
    One-click optimization entry point: applies the one_click profile and delegates to the standard runner.
    Keeps the same call stack and resource management as run_meoh_optimization.
    """
    try:
        with get_session() as session:
            run = session.get(Run, uuid.UUID(run_id_str))
            if not run:
                print(f"[{run_id_str}] [one-click] Run not found; aborting.")
                return
            cfg = run.meoh_config or {}
            cfg["one_click"] = True
            run.meoh_config = cfg
            run.logs = (run.logs or "") + "\n[one-click] profile applied; executing via standard runner"
            session.add(run)
            session.commit()
    except Exception as e:
        print(f"[{run_id_str}] [one-click] Failed to apply profile before run: {e}")
    # Delegate execution to a standard runner (unified process and resource management)
    return run_meoh_optimization(problem_id_str, run_id_str)

# Main MEOH optimization runner function


def run_meoh_optimization(problem_id_str: str, run_id_str: str):
    """
    Run MEOH optimization in a background process.
    Requires its own database session.
    """
    # Validate UUID format
    try:
        problem_id = uuid.UUID(problem_id_str)
        run_id = uuid.UUID(run_id_str)
    except ValueError:
        print(f"[FATAL-ERROR] Invalid UUID string received in background process. Problem ID: '{problem_id_str}', Run ID: '{run_id_str}'")
        return

    # Main optimization logic
    original_cwd = os.getcwd()
    try:
        with get_session() as session:
            # --- Diagnostic information ---
            print("\n" + "="*80)
            print(f"[{run_id}] [DIAGNOSTICS] Starting MEOH optimization for problem: {problem_id}")
            print(f"[{run_id}] [DIAGNOSTICS] PID: {os.getpid()}")
            print(f"[{run_id}] [DIAGNOSTICS] Project Root: {PROJECT_ROOT}")
            print(f"[{run_id}] [DIAGNOSTICS] Problem Runs Dir: {PROBLEM_RUNS_DIR}")
            print(f"[{run_id}] [DIAGNOSTICS] Python Executable: {sys.executable}")
            print("="*80 + "\n")

            # Get run and problem package information
            run = session.get(Run, run_id)
            if not run:
                print(f"[{run_id}] [ERROR] Run with ID not found in the database.")
                return

            problem_package = session.get(ProblemPackage, problem_id)
            if not problem_package:
                run.status = "failed"
                run.logs = "Associated ProblemPackage not found in the database."
                session.add(run)
                session.commit()
                return

            # Record start time
            _append_run_log(session, run, "background task started")

            # Prepare run environment (using absolute paths)
            run_path = PROBLEM_RUNS_DIR / str(run.id)
            run_path.mkdir(parents=True, exist_ok=True)
            print(f"[{run_id}] Preparing run environment at {run_path}")

            run.status = "running"
            run.start_time = datetime.now(timezone.utc)
            session.add(run)
            session.commit()
            _append_run_log(session, run, "status set to running; preparing environment")
            
            # Copy data directory, smartly handle absolute/relative paths
            dest_data_dir = run_path / "data"
            if dest_data_dir.exists():
                shutil.rmtree(dest_data_dir)

            source_data_dir_str = problem_package.data_file_path
            if source_data_dir_str:
                source_data_dir = Path(source_data_dir_str)
                if not source_data_dir.is_absolute():
                    source_data_dir = PROJECT_ROOT / source_data_dir
                
                if source_data_dir.exists():
                    shutil.copytree(source_data_dir, dest_data_dir)
                    _append_run_log(session, run, "data directory prepared")
                else:
                    dest_data_dir.mkdir()
                    _append_run_log(session, run, f"WARNING: source data dir {source_data_dir} not found, created empty data dir.")
                    print(f"[{run_id}] WARNING: Source data directory {source_data_dir} not found. Created an empty data directory.")
            else:
                dest_data_dir.mkdir()
                _append_run_log(session, run, "no data_file_path provided, created empty data dir.")
                print(f"[{run_id}] No data_file_path provided. Created an empty data directory.")

            # Copy core files, smartly handle absolute/relative paths
            core_files_map = {
                'framework': problem_package.framework_file_path,
                'get_instance': problem_package.get_instance_file_path,
                'evaluation': problem_package.evaluation_file_path,
            }
            for file_key, source_path_str in core_files_map.items():
                if not source_path_str:
                    raise ValueError(
                        f"Problem package is misconfigured. The path for '{file_key}' is missing. "
                        f"Please re-create the problem package."
                    )

                source_path = Path(source_path_str)
                if not source_path.is_absolute():
                    source_path = PROJECT_ROOT / source_path
                
                if source_path.exists():
                    shutil.copy(source_path, run_path / source_path.name)
                else:
                    raise FileNotFoundError(
                        f"Core file for '{file_key}' not found at the configured path: {source_path}. "
                        f"Please check the problem package configuration."
                    )
            print(f"--- [BG_PROCESS_TRACE] [{datetime.now()}] Core files copied. ---")
            _append_run_log(session, run, "core files copied")

            # Ensure manifest.json exists in the run root directory, smartly handle absolute/relative paths
            if problem_package.framework_file_path:
                framework_path_str = problem_package.framework_file_path
                framework_path = Path(framework_path_str)
                if not framework_path.is_absolute():
                    framework_path = PROJECT_ROOT / framework_path

                source_manifest_path = framework_path.parent / "manifest.json"
                dest_manifest_path = run_path / "manifest.json"
                try:
                    if source_manifest_path.exists():
                        shutil.copy(source_manifest_path, dest_manifest_path)
                        print(f"[{run_id}] Copied manifest.json to run directory.")
                    else:
                        print(f"[{run_id}] WARNING: Could not find manifest at {source_manifest_path}. Will try to generate one.")
                    _append_run_log(session, run, "manifest ready")
                except Exception as gen_err:
                    print(f"[{run_id}] WARNING: Failed to prepare manifest.json: {gen_err}")
                    _append_run_log(session, run, f"manifest prepare warning: {gen_err}")

            # Create evaluation runner (pre-create for benchmarking)
            evaluation_runner = MEOHEvaluationWrapper(
                run_path, 
                problem_package.template_program_str,
                problem_package.task_description,
                PROJECT_ROOT,
                0 # Temporarily set to 0, will be updated after dry run
            )
            
            # Step 1: Evaluate the four baseline algorithms (run each five times and take the average)
            try:
                _append_run_log(session, run, "evaluating baseline algorithms (each 5 runs)")
                print(f"[{run_id}] Evaluating baseline algorithms (each 5 runs)...")
                run_dir = PROBLEM_RUNS_DIR / str(run.id)
                framework_path = run_dir / "framework.py"
                evaluation_path = run_dir / "evaluation.py"
                gi_candidates = [run_dir / "get_instance.py", run_dir / "get_instance_adapted_corrected.py"]
                get_instance_path = None
                for cand in gi_candidates:
                    if Path(cand).exists():
                        get_instance_path = cand
                        break
                if get_instance_path is None:
                    raise RuntimeError("Required run artifact missing: get_instance.py")
                converted_instance_path = run_dir / "converted_instance.json"
                manifest_path = run_dir / "manifest.json"
                if not converted_instance_path.exists():
                    try:
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
                        raise RuntimeError(f"Required run artifact missing: {p}")
                spec_fw = importlib.util.spec_from_file_location("framework", str(framework_path))
                fw_mod = importlib.util.module_from_spec(spec_fw); spec_fw.loader.exec_module(fw_mod)
                spec_ev = importlib.util.spec_from_file_location("evaluation", str(evaluation_path))
                ev_mod = importlib.util.module_from_spec(spec_ev); spec_ev.loader.exec_module(ev_mod)
                spec_gi = importlib.util.spec_from_file_location("get_instance", str(get_instance_path))
                gi_mod = importlib.util.module_from_spec(spec_gi); spec_gi.loader.exec_module(gi_mod)
                import inspect as _inspect
                fn = getattr(gi_mod, "get_problem_instance")
                sig = _inspect.signature(fn)
                if len(sig.parameters) == 0:
                    problem_instance = fn()
                else:
                    problem_instance = fn(converted_instance_path)
                baseline_defs = [
                    {"name": "FCFS", "path": PROJECT_ROOT / "baseline algorithm" / "FCFS" / "framework.py"},
                    {"name": "Maxmin", "path": PROJECT_ROOT / "baseline algorithm" / "Maxmin" / "framework.py"},
                    {"name": "Minmin", "path": PROJECT_ROOT / "baseline algorithm" / "Minmin" / "framework.py"},
                    {"name": "Random", "path": PROJECT_ROOT / "baseline algorithm" / "Random" / "framework.py"},
                ]
                baseline_out = []
                try:
                    tpl_scores = evaluation_runner.evaluate_program(problem_package.template_program_str)
                    if isinstance(tpl_scores, list) and len(tpl_scores) >= 4:
                        counts_tpl = [float(tpl_scores[0]), float(tpl_scores[1]), float(tpl_scores[2])]
                        metrics_neg = [float(x) for x in tpl_scores[3:]]
                        metrics_pos = [-x for x in metrics_neg]
                        tpl_avg = counts_tpl + metrics_pos
                        baseline_out.append({
                            "name": "Template",
                            "scores": tpl_avg,
                            "values": {
                                "cloud": int(round(tpl_avg[0])) if len(tpl_avg) > 0 else None,
                                "edge": int(round(tpl_avg[1])) if len(tpl_avg) > 1 else None,
                                "device": int(round(tpl_avg[2])) if len(tpl_avg) > 2 else None,
                                "makespan": float(tpl_avg[3]) if len(tpl_avg) > 3 else None,
                                "energy": float(tpl_avg[4]) if len(tpl_avg) > 4 else None,
                                "cost": float(tpl_avg[5]) if len(tpl_avg) > 5 else None,
                            }
                        })
                        print(f"[{run_id}] Baseline Template avg scores: {tpl_avg}")
                    else:
                        print(f"[{run_id}] [WARNING] Template baseline returned invalid scores: {tpl_scores}")
                except Exception as e:
                    print(f"[{run_id}] [WARNING] Template baseline failed: {e}")
                for bd in baseline_defs:
                    try:
                        spec_b = importlib.util.spec_from_file_location("baseline_framework", str(bd["path"]))
                        bmod = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(bmod)
                        if not hasattr(bmod, "solve"):
                            continue
                        runs = []
                        for _ in range(5):
                            solution = bmod.solve(problem_instance)
                            result = ev_mod.evaluate_solution(solution, problem_instance)
                            counts = [float(result.get("cloud", 0)), float(result.get("edge", 0)), float(result.get("device", 0))]
                            metrics = [float(result.get("makespan", 0.0)), float(result.get("energy", 0.0)), float(result.get("cost", 0.0))]
                            runs.append(counts + metrics)
                        avg = np.mean(np.array(runs, dtype=float), axis=0).tolist()
                        item = {
                            "name": bd["name"],
                            "scores": avg,
                            "values": {
                                "cloud": int(round(avg[0])) if len(avg) > 0 else None,
                                "edge": int(round(avg[1])) if len(avg) > 1 else None,
                                "device": int(round(avg[2])) if len(avg) > 2 else None,
                                "makespan": float(avg[3]) if len(avg) > 3 else None,
                                "energy": float(avg[4]) if len(avg) > 4 else None,
                                "cost": float(avg[5]) if len(avg) > 5 else None,
                            }
                        }
                        baseline_out.append(item)
                        print(f"[{run_id}] Baseline {bd['name']} avg scores: {avg}")
                    except Exception as e:
                        print(f"[{run_id}] [WARNING] Baseline {bd['name']} failed: {e}")
                if baseline_out:
                    run.baseline_scores = baseline_out
                    session.add(run)
                    session.commit()
                    print(f"[{run_id}] Baseline algorithms evaluated: {[x['name'] for x in baseline_out]}")
                    _append_run_log(session, run, "baseline algorithms evaluated")
                    try:
                        names = ["cloud","edge","device","makespan","energy","cost"]
                        names_path = run_dir / "objective_names.json"
                        with open(names_path, "w", encoding="utf-8") as nf:
                            json.dump(names, nf, ensure_ascii=False)
                    except Exception:
                        pass
                else:
                    print(f"[{run_id}] [WARNING] Baseline evaluation produced no results.")
                    _append_run_log(session, run, "[WARNING] baseline evaluation produced no results")
            except Exception as baseline_err:
                print(f"[{run_id}] [WARNING] Baseline evaluation error: {baseline_err}")
                _append_run_log(session, run, f"[WARNING] baseline evaluation process error: {baseline_err}")


            # Step 2: Perform a dry run to detect the number of objectives
            _append_run_log(session, run, "starting dry run to detect objectives")
            if run.baseline_scores:
                first_b = run.baseline_scores[0] if isinstance(run.baseline_scores, list) and run.baseline_scores else None
                num_objs = len(first_b.get("scores", [])) if isinstance(first_b, dict) else 0
                print(f"[{run_id}] Number of objectives detected from baseline run: {num_objs}")
            else:
                num_objs = _get_num_objs(
                    run_path=run_path,
                    template_program=problem_package.template_program_str,
                    task_description=problem_package.task_description
                )
            
            evaluation_runner.num_objs = num_objs
            _append_run_log(session, run, f"dry run ok; num_objs={num_objs}")

            # Step 3: Get LLM configuration from problem package
            llm_config = problem_package.llm_config or {}
            if not llm_config or not all(k in llm_config for k in ['apiKey', 'baseUrl', 'modelName']):
                raise ValueError(
                    f"Problem package {problem_package.id} does not have valid LLM configuration. "
                    f"Please configure the problem with LLM settings before running optimization."
                )
            
            # Step 4: Initialize LLM from configuration
            try:
                llm = get_llm_from_config(llm_config)
                print(f"[{run_id}] LLM initialized successfully")
            except Exception as llm_err:
                raise RuntimeError(f"Failed to initialize LLM: {llm_err}")

            # Step 5: Create performance profiler
            log_dir = run_path / 'logs'
            profiler = MEoHProfiler(
                log_dir=str(log_dir), 
                num_objs=num_objs,
                minimize=False
            )

            # Step 6: Configure MEOH parameters
            meoh_params = run.meoh_config or {}
            meoh_params['num_objs'] = num_objs
            meoh_params['debug_mode'] = False
            meoh_params['max_generations'] = meoh_params.get('max_generations', 10)
            meoh_params['max_sample_nums'] = meoh_params.get('max_sample_nums', 20)
            meoh_params['pop_size'] = meoh_params.get('pop_size', 20)

            _append_run_log(session, run, "initializing LLMFlowGo instance")
            log_resource_usage("before LLMFlowGo initialization")
            
            # Step 7: Create MEOH instance
            meoh_instance = MEoH(
                llm=llm,
                evaluation=evaluation_runner,
                profiler=profiler,
                **meoh_params
            )
            _append_run_log(session, run, "LLMFlowGo instance created; starting evolution")
            log_resource_usage("after LLMFlowGo initialization")

            # Step 8: Switch working directory to run directory
            os.chdir(run_path)
            print(f"[{run_id}] Changed current working directory to: {os.getcwd()}")
            
            # Step 9: Execute optimization
            final_population = meoh_instance.run()
            _append_run_log(session, run, "evolution finished")
            log_resource_usage("after LLMFlowGo evolution")

            try:
                if isinstance(profiler, MEoHProfiler):
                    profiler.register_population(meoh_instance._population)
                    _append_run_log(session, run, "snapshot final population and elitist to logs")
            except Exception as snap_err:
                _append_run_log(session, run, f"[WARNING] failed to snapshot final population: {snap_err}")
            
            _append_run_log(session, run, "saving optimization results to database")

            final_pop_data = []
            pareto_data = []
            best_solution = None
            best_solution_code = None
            result_summary = "Optimization completed, but summary could not be generated."
            result_analysis = "Optimization completed, but analysis could not be generated."
            generation_count = 0

            try:
                if hasattr(meoh_instance, '_population') and meoh_instance._population:
                    population_obj = meoh_instance._population
                    generation_count = getattr(population_obj, 'generation', 0)

                    # Step 10: Save final population
                    if hasattr(population_obj, 'population'):
                        for i, individual in enumerate(population_obj.population):
                            try:
                                final_pop_data.append({
                                    'id': f"ind_{i}",
                                    'generation': generation_count,
                                    'objectives': individual.score if hasattr(individual, 'score') else None,
                                    'fitness': getattr(individual, 'fitness', None),
                                    'code': str(individual)
                                })
                            except Exception as ind_err:
                                print(f"[{run_id}] [WARNING] Skipping an individual in final population due to error: {ind_err}")

                    # Step 11: Save and sort Pareto front solutions
                    if hasattr(population_obj, 'elitist'):
                        for i, elite in enumerate(population_obj.elitist):
                            try:
                                pareto_data.append({
                                    'id': f"pareto_{i}",
                                    'rank': 0,  
                                    'objectives': elite.score if hasattr(elite, 'score') else None,
                                    'crowding_distance': getattr(elite, 'crowding_distance', None),
                                    'code': str(elite)
                                })
                            except Exception as elite_err:
                                print(f"[{run_id}] [WARNING] Skipping an elite solution due to error: {elite_err}")

                        # Step 12: Sort Pareto front solutions by objective sum (descending)
                        if pareto_data:
                            pareto_data.sort(
                                key=lambda x: sum(x['objectives']) if isinstance(x['objectives'], (list, tuple)) else (x['objectives'] or float('-inf')),
                                reverse=True
                            )
                            for i, elite_solution in enumerate(pareto_data):
                                elite_solution['rank'] = i + 1

                    # Step 13: Determine best solution based on Pareto front
                    candidates = pareto_data if pareto_data else final_pop_data
                    if candidates:
                        pos_obj_list = []
                        valid_indices = []
                        for idx, cand in enumerate(candidates):
                            objs = cand.get('objectives')
                            if isinstance(objs, (list, tuple)) and all(o is not None for o in objs):
                                try:
                                    pos_obj = [-float(o) for o in objs]
                                except Exception:
                                    continue
                                pos_obj_list.append(pos_obj)
                                valid_indices.append(idx)
                        if pos_obj_list:
                            pos_np = np.array(pos_obj_list, dtype=float)
                            # Ideal point: take the minimum value in each dimension among all candidates (originally 'the smaller the better')
                            ideal = pos_np.min(axis=0)
                            # Distance to ideal point
                            diffs = pos_np - ideal
                            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
                            best_idx = valid_indices[int(np.argmin(dists))]
                            best_solution = candidates[best_idx]
                            best_solution_code = best_solution.get('code')
                        else:
                            sorted_candidates = sorted(
                                candidates,
                                key=lambda x: sum(x['objectives']) if isinstance(x['objectives'], (list, tuple)) else (x['objectives'] or float('-inf')),
                                reverse=True
                            )
                            if sorted_candidates:
                                best_solution = sorted_candidates[0]
                                best_solution_code = best_solution.get('code')

                    # 4. Generate summary and analysis
                    result_summary = f"Optimization completed after {generation_count} generations. Final population size: {len(final_pop_data)}. Pareto front size: {len(pareto_data)}."
                    
                    if best_solution and best_solution_code:
                        best_objectives = best_solution.get('objectives')
                        obj_names = evaluation_runner.objective_names if hasattr(evaluation_runner, 'objective_names') and evaluation_runner.objective_names else []

                        if not obj_names and isinstance(best_objectives, (list, tuple)):
                            obj_names = [f"Objective_{i+1}" for i in range(len(best_objectives))]
                        
                        pos_vals = []
                        if isinstance(best_objectives, (list, tuple)):
                            try:
                                pos_vals = [-float(v) for v in best_objectives]
                            except Exception:
                                pos_vals = []
                        
                        if obj_names and pos_vals and len(obj_names) == len(pos_vals):
                            objectives_str = ", ".join([f"{name}: {val:.4f}" for name, val in zip(obj_names, pos_vals)])
                        else:
                            objectives_str = str(best_objectives)

                        improvements_str = None
                        baseline_list = run.baseline_scores if hasattr(run, 'baseline_scores') else None
                        if baseline_list and isinstance(baseline_list, list) and pos_vals:
                            try:
                                b0 = baseline_list[0] if baseline_list else None
                                b_scores = b0.get('scores', []) if isinstance(b0, dict) else []
                                counts_names = [n for n in obj_names if str(n).lower() in ('cloud','edge','device')]
                                m_names = [n for n in obj_names if str(n).lower() not in ('cloud','edge','device')]
                                b_metrics = b_scores[3:] if len(b_scores) > 3 else []
                                v_metrics = pos_vals[3:] if len(pos_vals) > 3 else []
                                Lm = min(len(b_metrics), len(v_metrics))
                                if Lm > 0:
                                    improvements = []
                                    for i in range(Lm):
                                        name = m_names[i] if i < len(m_names) else f"Objective_{i+4}"
                                        b = float(b_metrics[i])
                                        v = float(v_metrics[i])
                                        if b != 0.0:
                                            imp = (b - v) / b * 100.0
                                            improvements.append(f"{name}: {imp:+.2f}%")
                                        else:
                                            improvements.append(f"{name}: N/A (baseline=0)")
                                    improvements_str = ", ".join(improvements)
                            except Exception:
                                improvements_str = None
                        
                        # Summary analysis text: Indicate that the criterion for selecting the best solution is the ideal point distance, and include a comparison with the baseline.
                        if improvements_str:
                            result_analysis = (
                                f"Found {len(pareto_data)} non-dominated solutions. "
                                f"The best solution (selected by ideal-point distance) has objectives: [{objectives_str}]. "
                                f"Compared to baseline: [{improvements_str}]."
                            )
                        else:
                            result_analysis = (
                                f"Found {len(pareto_data)} non-dominated solutions. "
                                f"The best solution (selected by ideal-point distance) has objectives: [{objectives_str}]."
                            )
                    else:
                        result_analysis = "Optimization ran but no optimal solution could be determined."

                else:
                    _append_run_log(session, run, "[WARNING] meoh_instance._population not found. No results to save.")

            except Exception as result_err:
                print(f"[{run_id}] [FATAL-ERROR] Failed to process and save results: {result_err}\n{traceback.format_exc()}")
                _append_run_log(session, run, f"[FATAL-ERROR] Failed during result processing: {result_err}")

            # 5. Assign all results to the run object
            run.final_population = final_pop_data
            run.pareto_front = pareto_data
            run.best_solution_code = best_solution_code
            run.result_summary = result_summary
            run.result_analysis = result_analysis

            # New: Evaluate the best solution to get final counts and metrics, then persist the snapshot
            try:
                run_dir = PROBLEM_RUNS_DIR / str(run.id)
                if best_solution_code:
                    cm = compute_counts_metrics_from_best_solution(run_dir, best_solution_code)
                    run.cloud_count = cm.get("cloud")
                    run.edge_count = cm.get("edge")
                    run.device_count = cm.get("device")
                    run.makespan = cm.get("makespan")
                    run.energy = cm.get("energy")
                    payload = build_final_result_payload(run, cm, extra={"raw": {
                        "final_population": final_pop_data,
                        "pareto_front": pareto_data,
                    }})
                    run.final_result_json = payload
                    _append_run_log(session, run, f"final counts computed: {cm}")
                else:
                    _append_run_log(session, run, "[WARNING] best_solution_code not available; skip final counts.")
            except Exception as e:
                _append_run_log(session, run, f"[WARNING] failed to compute final counts: {e}")

            _append_run_log(session, run, f"processed {len(final_pop_data)} individuals and {len(pareto_data)} elite solutions")

            # Completion banner
            print(f'''\n╔══════════════════════════════════════════════════════════════╗\n║            🎉 LLMFlowGo Optimization Completed! 🎉            ║\n╠══════════════════════════════════════════════════════════════╣\n║  Run ID: {run_id}                                            ║\n║  Completed At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Local Time)      ║\n╚══════════════════════════════════════════════════════════════╝\n''')
            _append_run_log(session, run, "🎉 LLMFlowGo optimization completed successfully!")
            
            # 更新运行状态
            run.status = "completed"
            run.end_time = datetime.now(timezone.utc)
            session.add(run)
            session.commit()
            _append_run_log(session, run, "run status updated to completed")
    except Exception as outer_err:
        print(f"[{run_id}] [FATAL-ERROR] Optimization failed: {outer_err}\n{traceback.format_exc()}")
        try:
            with get_session() as session2:
                run2 = session2.get(Run, run_id)
                if run2:
                    run2.status = "failed"
                    run2.end_time = datetime.now(timezone.utc)
                    run2.logs = (run2.logs or "") + f"\n[FATAL-ERROR] Optimization failed: {outer_err}"
                    session2.add(run2)
                    session2.commit()
        except Exception as log_err:
            print(f"[{run_id}] [ERROR] Failed to record failure status: {log_err}")
    finally:
        try:
            os.chdir(original_cwd)
        except Exception as chdir_err:
            print(f"[{run_id}] [WARNING] Failed to restore working directory: {chdir_err}")
        cleanup_process_resources()
        log_resource_usage("after cleanup")
