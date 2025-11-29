import multiprocessing
import sys
import time
import importlib.util
import os
import traceback
import ast
import numpy as np
from typing import List
import faulthandler # --- FINAL DEBUG WEAPON ---

# It's assumed that the 'MEoh' directory is in the python path.
# This will be the case when running uvicorn from the project root.
from MEoh.base.evaluate import Evaluation
from my_meoh_app.models.problem import ProblemPackage

def _load_module_from_path(module_name: str, file_path: str):
    """
    Dynamically loads a module from a given file path.
    This is a robust way to load modules in spawned subprocesses.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module '{module_name}' from {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Add the module to sys.modules to make it available for other imports
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Define a simple queue for communication
queue = multiprocessing.Queue()

def _isolated_execution(run_path: str, new_code_str: str, task_description: str, project_root: str, queue: multiprocessing.Queue, num_runs: int):
    """
    This function runs in a separate process to isolate the execution environment.
    """
    # --- CRITICAL FIX: Use the explicitly provided project_root ---
    # The parent process has determined the correct project root and passed it
    # as an argument. We add it to the path to ensure all modules can be imported.
    try:
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
    except Exception as e:
        # If this path logic fails, it's a fatal setup error.
        queue.put(f"FATAL: Subprocess failed to configure sys.path with root '{project_root}'. Error: {e}")
        return
    # --- END CRITICAL FIX ---
    
    # --- BOOTSTRAP DEBUG LOGGING ---
    # This log is created IMMEDIATELY upon entry to confirm the subprocess has started
    # and to check the received arguments before any other complex logic runs.
    run_path_abs = os.path.abspath(run_path)
    bootstrap_log_path = os.path.join(run_path_abs, "subprocess_bootstrap.log")
    try:
        with open(bootstrap_log_path, 'w', encoding='utf-8') as f:
            f.write(f"--- Subprocess Bootstrap Log ---\n")
            f.write(f"Timestamp: {time.time()}\n")
            f.write(f"Checkpoint 1: Bootstrap log opened.\n")
            f.write(f"Status: Subprocess initiated successfully.\n")
            f.write(f"Received run_path: {run_path_abs}\n")
            f.write(f"Is task_description None? {task_description is None}\n")
            if task_description is not None:
                f.write(f"Length of task_description: {len(task_description)}\n")
            f.write(f"Length of new_code_str: {len(new_code_str)}\n")
            f.write(f"Checkpoint 2: Bootstrap log content written.\n")
    except Exception as e:
        queue.put(f"CRITICAL BOOTSTRAP ERROR: {e}")
        return
    # --- END BOOTSTRAP LOGGING ---

    # --- MODIFICATION FOR HARD CRASH DEBUGGING ---
    faulthandler_log_path = os.path.join(run_path_abs, "faulthandler_crash.log")
    f_crash = None # Define beforehand
    try:
        with open(bootstrap_log_path, 'a', encoding='utf-8') as f: f.write("Checkpoint 3: About to open faulthandler log.\n")
        f_crash = open(faulthandler_log_path, 'w')
        with open(bootstrap_log_path, 'a', encoding='utf-8') as f: f.write("Checkpoint 4: Faulthandler log opened, about to enable.\n")
        faulthandler.enable(file=f_crash)
        with open(bootstrap_log_path, 'a', encoding='utf-8') as f: f.write("Checkpoint 5: Faulthandler enabled.\n")
    except Exception:
        faulthandler.enable(file=sys.stderr)
    # --- END MODIFICATION ---

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file_path = os.path.join(run_path_abs, "subprocess_error.log")

    try:
        with open(bootstrap_log_path, 'a', encoding='utf-8') as f: f.write("Checkpoint 6: About to change directory.\n")
        os.chdir(run_path_abs)
        with open(bootstrap_log_path, 'a', encoding='utf-8') as f: f.write(f"Checkpoint 7: Directory changed successfully to {os.getcwd()}.\n")

        # Redirect stdout and stderr to a log file for debugging
        with open(bootstrap_log_path, 'a', encoding='utf-8') as f: f.write("Checkpoint 8: About to open main error log for redirection.\n")
        with open(log_file_path, 'w', encoding='utf-8') as log_file:
            sys.stdout = log_file
            sys.stderr = log_file
            with open(bootstrap_log_path, 'a', encoding='utf-8') as f: f.write("Checkpoint 9: Stdout/Stderr redirected successfully.\n")

            # Add the run path to sys.path to allow imports
            # This is technically already done, but we'll re-verify
            if run_path_abs not in sys.path:
                 sys.path.insert(0, run_path_abs)

            # --- FIX: Load modules directly and robustly from file paths ---
            framework_module = _load_module_from_path("framework", os.path.join(run_path_abs, "framework.py"))
            get_instance_module = _load_module_from_path("get_instance", os.path.join(run_path_abs, "get_instance.py"))
            evaluation_module = _load_module_from_path("evaluation", os.path.join(run_path_abs, "evaluation.py"))

            # 3. Monkey Patch: Inject the MEOH-generated function into the framework module
            #    First, find the target function's name from the template code.
            tree = ast.parse(new_code_str)
            target_func_name = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    target_func_name = node.name
                    break
            if not target_func_name:
                raise ValueError("Could not find a function definition in the new code string.")

            #    Then, execute the new code to override the function in the loaded module's namespace
            exec(new_code_str, framework_module.__dict__)
            print(f"[Subprocess] Patched function '{target_func_name}' into framework module.")

            # --- FINAL FIX (This must be it): Initialize ALL local variables before the loop ---
            objective_keys = None
            all_scores_list = [] 
            # --- END FIX ---

            # 4. Run the evaluation cycle - use provided num_runs
            print(f"[Subprocess] Starting {num_runs} evaluation runs...")
            for i in range(num_runs):
                problem_instance = get_instance_module.get_problem_instance()
                if isinstance(problem_instance, dict):
                    problem_instance["objective"] = "sum"
                    if "time_budget_seconds" not in problem_instance:
                        problem_instance["time_budget_seconds"] = 10.0
                solution = framework_module.solve(problem_instance)
                scores_dict = evaluation_module.evaluate_solution(solution, problem_instance)

                if not isinstance(scores_dict, dict):
                    raise TypeError("evaluate_solution must return a dictionary.")
                
                # --- Final Robustness Fix ---
                # Check if the returned dictionary is empty, which is an invalid state.
                if not scores_dict:
                    raise ValueError(
                        "evaluation.py's evaluate_solution function returned an empty dictionary. "
                        "MEOH requires at least one objective score to optimize."
                    )
                # --- End of Fix ---

                if objective_keys is None:
                    counts_order = [k for k in ("cloud", "edge", "device") if k in scores_dict]
                    metrics_order = [k for k in scores_dict.keys() if k not in counts_order]
                    objective_keys = counts_order + metrics_order
                    print(f"[Subprocess] Determined objective output order: {objective_keys}")
                    try:
                        import json as _json
                        names_path = os.path.join(run_path_abs, "objective_names.json")
                        with open(names_path, "w", encoding="utf-8") as nf:
                            _json.dump(objective_keys, nf, ensure_ascii=False)
                    except Exception:
                        pass
                
                counts_order = [k for k in ("cloud", "edge", "device") if k in scores_dict]
                metrics_order = [k for k in objective_keys if k not in counts_order]
                counts_vals = [float(scores_dict[k]) for k in counts_order]
                metrics_vals = [float(-scores_dict[k]) for k in metrics_order]
                current_scores = counts_vals + metrics_vals
                all_scores_list.append(current_scores)
            
            # 5. Calculate the average scores and put them in the queue
            avg_scores = np.mean(all_scores_list, axis=0).tolist()
            print(f"[Subprocess] Finished {num_runs} runs. Average scores: {avg_scores}")
            queue.put(avg_scores)

    except BaseException as e:  # --- FINAL DEBUG: Catch EVERYTHING ---
        # Catching BaseException is generally discouraged, but here it's a powerful
        # tool to debug "hard crashes" (e.g., from C extensions like NumPy)
        # that don't get caught by a regular `except Exception`.
        tb_str = traceback.format_exc()
        # Also include the exception object itself for more context,
        # as some crashes might not produce a full traceback.
        error_msg = f"--- A CRITICAL ERROR OCCURRED IN SUBPROCESS ---\n"
        error_msg += f"Exception Type: {type(e).__name__}\n"
        error_msg += f"Exception Details: {e}\n\n"
        error_msg += f"Full Traceback:\n{tb_str}"
        
        print(error_msg, file=original_stderr)
        
        if 'log_file' in locals() and not log_file.closed:
            log_file.write(error_msg)
        
        queue.put(None) # Signal failure
    
    finally:
        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Close the faulthandler file handle if it's open
        if f_crash:
            f_crash.close()

        # Clean up path
        if run_path_abs in sys.path:
            sys.path.remove(run_path_abs)


class MEOHEvaluationWrapper(Evaluation):
    def __init__(self, run_path, template_program, task_description, project_root, num_objs=2):
        # Call parent constructor to set required attributes like safe_evaluate
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            safe_evaluate=True,  # Use safe evaluation by default
            timeout_seconds=300  # 增加超时时间到5分钟，给复杂算法更多执行时间
        )
        
        self.run_path = run_path
        self.project_root = project_root
        self.num_objs = num_objs
        print(f"[Main Process] MEOHEvaluationWrapper initialized for run path: {run_path}")

    def evaluate_program(self, program_str: str, callable_func: callable = None, **kwargs):
        """
        This method is called by the MEOH framework for each new candidate function.
        It uses a subprocess for safe, isolated execution.
        
        Args:
            program_str: The program code string to evaluate
            callable_func: The callable function object (ignored in our implementation)
            **kwargs: Additional keyword arguments
        """
        print(f"[Main Process] Evaluating new program...")
        
        queue = multiprocessing.Queue()
        try:
            template_str = getattr(self, 'template_program', None)
        except Exception:
            template_str = None
        is_template = False
        if isinstance(program_str, str) and isinstance(template_str, str):
            is_template = program_str.strip() == template_str.strip()
        
        num_runs = 5 if is_template else 1
        
        proc = multiprocessing.Process(
            target=_isolated_execution, 
            args=(self.run_path, program_str, self.task_description, self.project_root, queue, num_runs)
        )
        
        proc.start()
        # Set a timeout for the evaluation - increased to allow for complex algorithms
        timeout_seconds = 600 if is_template else 300
        proc.join(timeout=timeout_seconds)

        error_log_path = os.path.join(self.run_path, "subprocess_error.log")

        if proc.is_alive():
            print("[Main Process] Evaluation process timed out. Terminating.")
            try:
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"TIMEOUT: Evaluation subprocess exceeded {timeout_seconds}s and was terminated.\n")
            except Exception:
                pass
            proc.terminate()
            proc.join()
            # Signal failure to caller
            return None
        
        if queue.empty():
            print("[Main Process] Evaluation process finished but queue is empty.")
            try:
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write("ERROR: Subprocess finished without returning results (empty queue).\n")
            except Exception:
                pass
            return None

        result = queue.get()
        print(f"[Main Process] Got result from subprocess: {result}")
        
        # STRICT VALIDATION: Return None for all invalid results to prevent population injection
        if result is None:
            print("[Main Process] CRITICAL: Subprocess returned None, this should not happen!")
            try:
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write("CRITICAL ERROR: Subprocess returned None result.\n")
            except Exception:
                pass
            # CRITICAL: Return None to prevent injection into population
            return None
        
        # Validate result shape: expect a list of floats (vector) or list of list of floats
        if not isinstance(result, list) or not result:
            print(f"[Main Process] CRITICAL: Invalid result type/shape: {type(result)}")
            try:
                with open(error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"ERROR: Subprocess returned invalid result type/shape: {type(result)}.\n")
            except Exception:
                pass
            # CRITICAL: Return None to prevent injection into population
            return None
        
        # Additional validation: ensure all scores are valid numbers
        for i, score in enumerate(result):
            if not isinstance(score, (int, float)) or np.isnan(score) or np.isinf(score):
                print(f"[Main Process] CRITICAL: Invalid score at index {i}: {score}")
                try:
                    with open(error_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"ERROR: Invalid score at index {i}: {score}.\n")
                except Exception:
                    pass
                # CRITICAL: Return None to prevent injection into population
                return None
        
        print(f"[Main Process] Result validation passed: {result}")
        try:
            import json as _json
            names_path = os.path.join(self.run_path, "objective_names.json")
            if os.path.exists(names_path):
                with open(names_path, "r", encoding="utf-8") as nf:
                    names = _json.load(nf)
                if isinstance(names, list) and all(isinstance(x, str) for x in names):
                    self.objective_names = names
                    print(f"[Main Process] Loaded objective names: {self.objective_names}")
        except Exception:
            pass
        return result
