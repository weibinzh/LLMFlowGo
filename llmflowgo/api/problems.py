from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from sqlmodel import Session, select
from typing import List, Dict, Any, Optional
import uuid
import os
import shutil
import json
import ast
from pydantic import BaseModel
import aiofiles
import time
from datetime import datetime, timezone
import multiprocessing

from my_meoh_app.database import engine, get_session
from my_meoh_app.models.problem import ProblemPackage
from my_meoh_app.models.run import Run
from my_meoh_app.core.code_analyzer import analyze_python_file
from my_meoh_app.core.llm_service import get_llm_analysis, get_llm_summary_for_function
from my_meoh_app.core.llm_service import get_llm_potential_reason
from my_meoh_app.core.llm_service import recommend_algorithm_preset as llm_recommend_preset
from my_meoh_app.core.llm_service import suggest_optimization_description as llm_suggest_desc
from my_meoh_app.core.code_parser import get_function_calls, get_functions_details
from my_meoh_app.core.config_converter import ConfigConverter
from my_meoh_app.core.dag_analyzer import analyze_problem_instance, build_llm_input_payload
from my_meoh_app.core.llm_service import get_server_count_recommendation, get_server_counts_and_preset
from my_meoh_app.core.meoh_runner import run_meoh_optimization

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class LLMConfig(BaseModel):
    apiKey: str
    baseUrl: str
    modelName: str

class StartRunRequest(BaseModel):
    meoh_config: Dict[str, Any]

class PreciseBuildRunRequest(BaseModel):
    environmentConfig: Dict[str, Any]
    dagConfig: Dict[str, Any]
    llm_config: LLMConfig
    name: Optional[str] = None
    description: Optional[str] = None
    bounds: Optional[Dict[str, List[int]]] = None
    meoh_config: Optional[Dict[str, Any]] = None
    algorithmPreset: Optional[str] = None


router = APIRouter()

# The root of the application, which is the parent directory of 'my_meoh_app'
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROBLEMS_DIR = os.path.join(APP_ROOT, "user_problems")
os.makedirs(PROBLEMS_DIR, exist_ok=True)


@router.post("/problems/", response_model=ProblemPackage)
async def create_problem_package(
    name: str = Form(...),
    description: Optional[str] = Form(None),
    framework_file: UploadFile = File(...),
    get_instance_file: UploadFile = File(...),
    evaluation_file: UploadFile = File(...),
    environment_config: str = Form(...),
    dag_config: str = Form(...),
    session: Session = Depends(get_session)
):
    """
    Create a new problem package by uploading framework files and configuration data.
    """
    problem_id = uuid.uuid4()
    problem_dir = os.path.join(PROBLEMS_DIR, str(problem_id))
    os.makedirs(problem_dir, exist_ok=True)

    data_dir = os.path.join(problem_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    file_paths = {}

    # --- FINAL FIX: Use aiofiles for fully asynchronous file writing ---
    async def write_file(file_path: str, upload_file: UploadFile):
        contents = await upload_file.read()
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(contents)
        return contents.decode('utf-8')

    # Save uploaded code files
    framework_path = os.path.join(problem_dir, "framework.py")
    framework_content = await write_file(framework_path, framework_file)
    file_paths['framework_file_path'] = framework_path

    get_instance_path = os.path.join(problem_dir, "get_instance.py")
    await write_file(get_instance_path, get_instance_file)
    file_paths['get_instance_file_path'] = get_instance_path

    evaluation_path = os.path.join(problem_dir, "evaluation.py")
    await write_file(evaluation_path, evaluation_file)
    file_paths['evaluation_file_path'] = evaluation_path

    # Parse configuration data
    try:
        env_config = json.loads(environment_config)
        dag_config_data = json.loads(dag_config)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON configuration: {e}")

    # Use ConfigConverter to generate data files
    try:
        converter = ConfigConverter()
        
        full_config = {
            'environmentConfig': env_config,
            'dagConfig': dag_config_data
        }
        converted_data = converter.convert_full_config(full_config)
        
        # Save generated data files
        data_file_manifest = {}
        
        # Save DAG data (workload and edges)
        dag_data_path = os.path.join(data_dir, "dag.json")
        async with aiofiles.open(dag_data_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(converted_data['dag'], indent=4))
        data_file_manifest['dag_data'] = "data/dag.json"
        
        # Build converted_instance and save to data
        dag_obj = str(dag_config_data.get('objective', 'sum')) if isinstance(dag_config_data, dict) else 'sum'
        dag_ci = dict(converted_data['dag']); dag_ci['objective'] = dag_obj
        rr = {
            "cloud": [max(1, int(env_config.get('cloudCount', 1))), max(1, int(env_config.get('cloudCount', 1)) + 2)],
            "edge": [max(1, int(env_config.get('edgeCount', 1))), max(1, int(env_config.get('edgeCount', 1)) + 2)],
            "device": [max(1, int(env_config.get('deviceCount', 1))), max(1, int(env_config.get('deviceCount', 1)) + 2)],
        }
        cloud_spec = str(env_config.get('cloudSpec', 'cloud-large'))
        edge_spec = str(env_config.get('edgeSpec', 'edge-medium'))
        device_spec = str(env_config.get('deviceSpec', 'device-small'))
        c_cfg = converter.server_types['cloud'].get(cloud_spec, converter.server_types['cloud']['cloud-large'])
        e_cfg = converter.server_types['edge'].get(edge_spec, converter.server_types['edge']['edge-medium'])
        d_cfg = converter.server_types['device'].get(device_spec, converter.server_types['device']['device-small'])
        
        type_prototypes = {
            "cloud": {"capacity": float(c_cfg['mips'])/100.0, "power": float(c_cfg['power']), "price": float(c_cfg['price'])},
            "edge": {"capacity": float(e_cfg['mips'])/100.0, "power": float(e_cfg['power']), "price": float(e_cfg['price'])},
            "device": {"capacity": float(d_cfg['mips'])/100.0, "power": float(d_cfg['power']), "price": float(d_cfg['price'])},
        }

        conn_map = {
            ('cloud','cloud'): 'cloud_to_cloud',
            ('cloud','edge'): 'cloud_to_edge',
            ('cloud','device'): 'cloud_to_device',
            ('edge','cloud'): 'edge_to_cloud',
            ('edge','edge'): 'edge_to_edge',
            ('edge','device'): 'edge_to_device',
            ('device','cloud'): 'device_to_cloud',
            ('device','edge'): 'device_to_edge',
            ('device','device'): 'device_to_device',
        }
        network_defaults = {}
        for (a,b), key in conn_map.items():
            cfg = converter.network_config.get(key, {"bandwidth": 100.0, "power": 0.05})
            network_defaults[f"{a},{b}"] = {"bandwidth": float(cfg['bandwidth']), "link_power": float(cfg['power'])}

        converted_instance = {
            "dag": dag_ci,
            "recommended_ranges": rr,
            "type_prototypes": type_prototypes,
            "network_defaults": network_defaults,
        }
        converted_instance_path = os.path.join(data_dir, "converted_instance.json")
        async with aiofiles.open(converted_instance_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(converted_instance, indent=4))
        manifest_path = os.path.join(problem_dir, "manifest.json")
        manifest = {
            "converted_instance": "data/converted_instance.json",
            "dag_data": "data/dag.json"
        }
        async with aiofiles.open(manifest_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(manifest, indent=4))
        
        file_paths['data_file_path'] = data_dir
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration conversion failed: {e}")

    problem_package = ProblemPackage(
        id=problem_id,
        name=name,
        description=description,
        template_program_str=framework_content,
        **file_paths
    )
    
    session.add(problem_package)
    session.commit()
    session.refresh(problem_package)
    
    return problem_package


@router.get("/problems/", response_model=List[ProblemPackage])
def get_problem_packages(session: Session = Depends(get_session)):
    """
    Get a list of all created problem packages.
    """
    problems = session.exec(select(ProblemPackage)).all()
    return problems


@router.get("/problems/algorithm-presets")
def list_algorithm_presets():
    items = []
    root = os.path.join(APP_ROOT, "algo_presets")
    if os.path.isdir(root):
        for name in os.listdir(root):
            d = os.path.join(root, name)
            if os.path.isdir(d):
                label = name
                description = ""
                manifest_path = os.path.join(d, "manifest.json")
                try:
                    if os.path.exists(manifest_path):
                        with open(manifest_path, "r", encoding="utf-8") as mf:
                            m = json.load(mf)
                            label = str(m.get("label", label))
                            description = str(m.get("description", ""))
                except Exception:
                    pass
                items.append({"name": name, "label": label, "description": description})
    return {"presets": items}

class AnalyzePresetRequest(BaseModel):
    preset: str
    llm_config: LLMConfig
    dagConfig: Dict[str, Any] | None = None
    environmentConfig: Dict[str, Any] | None = None
    serverTypes: Dict[str, Any] | None = None

@router.post("/problems/analyze-preset-framework")
def analyze_preset_framework(request: AnalyzePresetRequest):
    preset_dir = os.path.join(APP_ROOT, "algo_presets", request.preset)
    framework_path = os.path.join(preset_dir, "framework.py")
    if not os.path.exists(framework_path):
        raise HTTPException(status_code=404, detail=f"Framework file not found: {framework_path}")
    try:
        functions = analyze_python_file(framework_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        with open(framework_path, "r", encoding="utf-8") as f:
            full_source_code = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        analysis = get_llm_analysis(
            api_key=request.llm_config.apiKey,
            base_url=request.llm_config.baseUrl,
            model_name=request.llm_config.modelName,
            full_source_code=full_source_code,
            functions=functions,
        )
        valid_names = {fn.get('name') for fn in functions}
        filtered = [item for item in analysis if str(item.get('name')) in valid_names]
        if not filtered:
            raise HTTPException(status_code=422, detail="LLM did not return any valid target functions from the analyzed list")
        best_item = sorted(filtered, key=lambda x: int(x.get('potential', 0)), reverse=True)[0]
        best_name = str(best_item.get('name'))
        src = next((fn.get('source_code') for fn in functions if str(fn.get('name')) == best_name), None)
        if not src:
            raise HTTPException(status_code=500, detail="Failed to locate source code for selected function")
        # Use structured metrics (same as the recommended server process)
        raw_dag = request.dagConfig or {}
        env_cfg = request.environmentConfig or {}
        try:
            analysis = analyze_problem_instance(raw_dag)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"DAG analysis failed: {e}")

        servers_info: Dict[str, Any] = {}
        if isinstance(request.serverTypes, dict):
            servers_info["server_types"] = {k: v for k, v in request.serverTypes.items() if k in ["cloud", "edge", "device"]}
        servers_info["counts"] = {
            "cloud": int(env_cfg.get("cloudCount", 0) or 0),
            "edge": int(env_cfg.get("edgeCount", 0) or 0),
            "device": int(env_cfg.get("deviceCount", 0) or 0),
        }
        servers_info["specs"] = {
            "cloudSpec": env_cfg.get("cloudSpec"),
            "edgeSpec": env_cfg.get("edgeSpec"),
            "deviceSpec": env_cfg.get("deviceSpec"),
        }

        try:
            structured_payload = build_llm_input_payload(analysis, servers_info)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to build LLM payload: {e}")

        # Pass the original DAG along with the structured summary to the LLM so that optimization directions can be generated based on the same metrics.
        dag_for_prompt = {"raw": raw_dag, "structured": structured_payload}

        desc_obj = llm_suggest_desc(
            api_key=request.llm_config.apiKey,
            base_url=request.llm_config.baseUrl,
            model_name=request.llm_config.modelName,
            dag_config=dag_for_prompt,
            environment_config=env_cfg,
        )
        reason = str(desc_obj.get("reason", ""))
        description = str(desc_obj.get("description", ""))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    print(f"[Algorithm Selection] preset={request.preset}, selected_function={best_name}")
    try:
        potential = best_item.get('potential')
        if potential is not None:
            print(f"[Algorithm Selection] score(potential)={potential}")
        fn_reason = best_item.get('reason')
        if fn_reason:
            print("[Algorithm Selection] function_reason:\n" + str(fn_reason))
    except Exception:
        pass
    if reason:
        print(f"[Algorithm Selection] reason: {reason}")
    if description:
        print("[Algorithm Selection] description:\n" + description)
    return {"preset": request.preset, "reason": reason, "description": description, "frameworkPath": framework_path}


@router.get("/problems/{problem_id}", response_model=ProblemPackage)
def get_problem_package_details(problem_id: uuid.UUID, session: Session = Depends(get_session)):
    """
    Get detailed information of a single problem package by ID.
    """
    problem = session.get(ProblemPackage, problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem package not found")
    return problem


class AnalysisRequest(BaseModel):
    llm_config: LLMConfig

class AnalysisResult(BaseModel):
    analysis: List[Dict]

class ConfigConversionRequest(BaseModel):
    environmentConfig: Dict[str, Any]
    dagConfig: Dict[str, Any]

@router.post("/convert-config")
async def convert_config(request: ConfigConversionRequest):
    """
    Convert environment and DAG configuration to standard JSON format
    """
    try:
        converter = ConfigConverter()

        full_config = {
            'environmentConfig': request.environmentConfig,
            'dagConfig': request.dagConfig
        }
        
        converted_data = converter.convert_full_config(full_config)
        
        return {
            "success": True,
            "data": converted_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration conversion failed: {str(e)}")

@router.post("/problems/{problem_id}/analyze", response_model=AnalysisResult)
def analyze_problem_framework(problem_id: uuid.UUID, request: AnalysisRequest, session: Session = Depends(get_session)):
    problem = session.get(ProblemPackage, problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem package not found")

    framework_path = problem.framework_file_path
    if not os.path.exists(framework_path):
        raise HTTPException(status_code=404, detail=f"Framework file not found at path: {framework_path}")

    # 1. Use AST to analyze code
    try:
        functions = analyze_python_file(framework_path)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {e}")

    if not functions:
        raise HTTPException(status_code=422, detail="No top-level functions found in the framework file or file is empty.")

    try:
        with open(framework_path, 'r', encoding='utf-8') as f:
            full_source_code = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read framework file: {e}")

    # 2. Call LLM service for analysis
    try:
        llm_analysis_result = get_llm_analysis(
            api_key=request.llm_config.apiKey,
            base_url=request.llm_config.baseUrl,
            model_name=request.llm_config.modelName,
            full_source_code=full_source_code,
            functions=functions
        )
        return AnalysisResult(analysis=llm_analysis_result)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during LLM analysis: {e}")


class ConfigureBody(BaseModel):
    target_function_name: str
    task_description: str
    llm_config: LLMConfig


def build_template_program(source_path: str, target_function_name: str) -> str:
    """
    Construct the template_program string. It contains all the imports and **the single** target function definition.
    """
    with open(source_path, 'r', encoding='utf-8') as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    
    imports = []
    target_function_source = None

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef) and node.name == target_function_name:
            target_function_source = ast.unparse(node)

    if target_function_source is None:
        raise ValueError(f"Target function '{target_function_name}' not found in the source file.")

    template_program = "\n".join(imports) + "\n\n" + target_function_source
    return template_program


@router.put("/problems/{problem_id}/configure", response_model=ProblemPackage)
def configure_problem_optimization(problem_id: uuid.UUID, config: ConfigureBody, session: Session = Depends(get_session)):
    """
    Save user configuration for optimization task and build the final task_description (prompt).
    """
    problem = session.get(ProblemPackage, problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem package not found")

    # 1. Build template_program_str
    try:
        template_str = build_template_program(problem.framework_file_path, config.target_function_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


    # 2. Static code analysis to find helper functions
    try:
        with open(problem.framework_file_path, 'r', encoding='utf-8') as f:
            full_source_code = f.read()
        tree = ast.parse(full_source_code)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse framework file content: {e}")

    target_function = config.target_function_name
    called_functions = get_function_calls(tree, target_function)
    if target_function in called_functions:
        called_functions.remove(target_function)
    
    function_details = get_functions_details(tree, called_functions)

    # 3. Generate LLM summaries for helper functions
    try:
        print(f"Starting to generate summaries for {len(function_details)} helper functions...")
        for detail in function_details:
            print(f"  -> Generating summary for '{detail['name']}'...")
            # Add a 1-second delay before each API call to avoid rate limiting or connection issues
            time.sleep(1)
            summary = get_llm_summary_for_function(
                api_key=config.llm_config.apiKey,
                base_url=config.llm_config.baseUrl,
                model_name=config.llm_config.modelName,
                function_source_code=detail['source_code']
            )
            detail['summary'] = summary # Replace with LLM-generated summary
        print("All summaries generated.")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate function summary using LLM: {e}")

    # 4. Build the final task_description (prompt)
    user_original_description = config.task_description

    helper_functions_context = []
    for func_info in function_details:
        context = f"""
Function: `{func_info['name']}`
- Parameters: {func_info['parameters']}
- Returns: {func_info['returns']}
- Summary: {func_info['summary']}
"""
        helper_functions_context.append(context)

    final_task_description = f"""Please help me design a heuristic strategy to {user_original_description}.

Requirement: Keep the generated code complexity as low as possible.

To do this, you can and should only use the following pre-defined helper functions. Do not reinvent them:
{''.join(helper_functions_context)}
"""

    # 5. Update database record
    problem.target_function_name = config.target_function_name
    problem.task_description = final_task_description  # Save the final constructed prompt
    problem.template_program_str = template_str
    
    # Save LLM config to the problem package for future optimization runs
    problem.llm_config = {
        "apiKey": config.llm_config.apiKey,
        "baseUrl": config.llm_config.baseUrl,
        "modelName": config.llm_config.modelName
    }
    
    session.add(problem)
    session.commit()
    session.refresh(problem)
    
    return problem



class RecommendServerCountsRequest(BaseModel):
    dagConfig: Dict[str, Any]
    serverTypes: Dict[str, Any]
    llm_config: Optional[LLMConfig] = None
    bounds: Optional[Dict[str, List[int]]] = None

@router.post("/dag/recommend-server-counts")
def recommend_server_counts(request: RecommendServerCountsRequest):
    """
    Accept DAG configuration and server specs from the frontend, analyze the DAG, and use LLM to return min/max ranges for cloud, edge, and device server counts.
    """
    # 1) Basic validation
    dag_cfg = request.dagConfig
    if not isinstance(dag_cfg, dict) or ("workload" not in dag_cfg or "edges" not in dag_cfg):
        raise HTTPException(status_code=422, detail="Invalid dagConfig: missing 'workload' or 'edges'")

    # 2) Perform DAG analysis
    try:
        analysis = analyze_problem_instance(dag_cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DAG analysis failed: {e}")

    # 3) Build LLM input payload (pass through fixed single specs, do not use default bounds)
    servers_info: Dict[str, Any] = {}
    servers_info.update({k: v for k, v in request.serverTypes.items() if k in ["cloud", "edge", "device"]})
    payload = build_llm_input_payload(analysis, servers_info)

    # 4) Parse and validate LLM configuration (required)
    if not request.llm_config or not all([request.llm_config.apiKey, request.llm_config.baseUrl, request.llm_config.modelName]):
        raise HTTPException(status_code=400, detail="LLM configuration is required and must include apiKey, baseUrl, and modelName.")
    api_key = request.llm_config.apiKey
    base_url = request.llm_config.baseUrl
    model_name = request.llm_config.modelName

    # 5) In one LLM round, return server ranges and algorithm preset recommendation (restricted to predefined candidates)
    algo_root = os.path.join(APP_ROOT, "algo_presets")
    candidate_names = [name for name in os.listdir(algo_root) if os.path.isdir(os.path.join(algo_root, name))] if os.path.isdir(algo_root) else []
    try:
        combined = get_server_counts_and_preset(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            payload=payload,
            candidate_names=candidate_names or ["ga", "pso"],
        )
        recommendation = {
            "cloudRange": combined["cloudRange"],
            "edgeRange": combined["edgeRange"],
            "deviceRange": combined["deviceRange"],
        }
        algo_preset = combined.get("preset")
        algo_reason = combined.get("reason")
        recommendation["algoPreset"] = algo_preset
        recommendation["algoReason"] = algo_reason
    except (ValueError, RuntimeError) as e:
        # Fallback to two calls: first ranges, then algorithm
        try:
            recommendation = get_server_count_recommendation(api_key=api_key, base_url=base_url, model_name=model_name, payload=payload)
        except (ValueError, RuntimeError) as e2:
            raise HTTPException(status_code=500, detail=str(e2))
        # Simplify to choose by candidate names only (no source loading)
        algo_preset = ("ga" if str(dag_cfg.get("objective", "")).lower() in ("makespan", "time", "latency") else "pso")
        algo_reason = "Heuristic fallback based on objective."

        recommendation["algoPreset"] = algo_preset
        recommendation["algoReason"] = algo_reason

    print(f"[LLM] Server ranges computed successfully: cloud {recommendation.get('cloudRange')}, edge {recommendation.get('edgeRange')}, device {recommendation.get('deviceRange')} | Algo: {algo_preset}\nReason: {algo_reason}")
    return {
        "success": True,
        "message": "LLM recommended server count ranges successfully",
        "recommendation": recommendation,
        "algoPreset": algo_preset,
        "algoReason": algo_reason,
        "analysis": {
            "graph_summary": analysis.get("graph_summary", {}),
            "metrics": analysis.get("metrics", {}),
        }
    }


class PreciseCreateRequest(BaseModel):
    user_problem_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    llm_config: LLMConfig


@router.post("/problems/precise-create", response_model=ProblemPackage)
def precise_create_problem(req: PreciseCreateRequest, session: Session = Depends(get_session)):
    """
    Register an existing user_problem directory as a new ProblemPackage, and automatically save
    the target function name and task description based on AST analysis (calling LLM to generate
    helper function summaries).
    """
    user_dir = os.path.join(PROBLEMS_DIR, req.user_problem_id)
    if not os.path.isdir(user_dir):
        raise HTTPException(status_code=404, detail=f"User problem directory not found: {user_dir}")

    framework_path = os.path.join(user_dir, "framework.py")
    get_instance_path = os.path.join(user_dir, "get_instance.py")
    evaluation_path = os.path.join(user_dir, "evaluation.py")
    data_dir = os.path.join(user_dir, "data")

    for p in [framework_path, get_instance_path, evaluation_path, data_dir]:
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"Required file or directory missing: {p}")

    # Use directory name as problem ID to ensure one-to-one mapping with the filesystem
    raw_id = req.user_problem_id.strip()
    candidate = raw_id
    # Support 32-digit hex directory names without hyphens: auto-convert to standard UUID format
    if len(raw_id) == 32 and all(c in "0123456789abcdefABCDEF" for c in raw_id):
        candidate = f"{raw_id[0:8]}-{raw_id[8:12]}-{raw_id[12:16]}-{raw_id[16:20]}-{raw_id[20:32]}"
    try:
        problem_id = uuid.UUID(candidate)
    except ValueError:
        raise HTTPException(status_code=400, detail="user_problem_id must be a valid UUID folder name")
    # Avoid duplicate registration
    existing = session.get(ProblemPackage, problem_id)
    if existing:
        raise HTTPException(status_code=409, detail="ProblemPackage with this ID already exists")

    name = req.name or f"Precise-{req.user_problem_id[:8]}"

    try:
        with open(framework_path, 'r', encoding='utf-8') as f:
            framework_content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read framework: {e}")

    problem = ProblemPackage(
        id=problem_id,
        name=name,
        description=req.description,
        framework_file_path=framework_path,
        get_instance_file_path=get_instance_path,
        evaluation_file_path=evaluation_path,
        data_file_path=data_dir,
        template_program_str=framework_content,
    )

    # AST function analysis
    try:
        functions = analyze_python_file(framework_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Code analysis failed: {e}")

    target_fn = None
    try:
        analysis = get_llm_analysis(
            api_key=req.llm_config.apiKey,
            base_url=req.llm_config.baseUrl,
            model_name=req.llm_config.modelName,
            full_source_code=framework_content,
            functions=functions,
        )
        valid_names = {fn.get('name') for fn in functions}
        filtered = [item for item in analysis if str(item.get('name')) in valid_names]
        if not filtered:
            raise HTTPException(status_code=422, detail="LLM did not return any valid target functions from the analyzed list")
        best_item = sorted(filtered, key=lambda x: int(x.get('potential', 0)), reverse=True)[0]
        target_fn = str(best_item.get('name'))
        print(f"[LLM] Selected target function: {target_fn} (score={best_item.get('potential')})")
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=500, detail=f"LLM analysis failed: {e}")
    except HTTPException:
        raise

    if not target_fn:
        raise HTTPException(status_code=422, detail="No target function selected by LLM")

    # Build template_program_str (imports + target function definition)
    try:
        template_str = build_template_program(framework_path, target_fn)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # Build final task_description: generate summaries for called helper functions
    try:
        with open(framework_path, 'r', encoding='utf-8') as f:
            full_source_code = f.read()
        tree = ast.parse(full_source_code)
        called_functions = get_function_calls(tree, target_fn)
        if target_fn in called_functions:
            called_functions.remove(target_fn)
        function_details = get_functions_details(tree, called_functions)

        for detail in function_details:
            time.sleep(1)
            summary = get_llm_summary_for_function(
                api_key=req.llm_config.apiKey,
                base_url=req.llm_config.baseUrl,
                model_name=req.llm_config.modelName,
                function_source_code=detail['source_code']
            )
            detail['summary'] = summary

        helper_context = []
        for func_info in function_details:
            helper_context.append(f"""
Function: `{func_info['name']}`
- Parameters: {func_info['parameters']}
- Returns: {func_info['returns']}
- Summary: {func_info['summary']}
""")

        base_desc = req.description or "search optimal server counts (cloud/edge/device) with GA and scheduling evaluation"
        final_task_desc = f"""Please help me design a heuristic strategy to {base_desc}.

Requirement: Keep the generated code complexity as low as possible.

Use only the following helper functions and do not reinvent them:
{''.join(helper_context)}
"""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to construct task description: {e}")

    # Write to database
    problem.target_function_name = target_fn
    problem.task_description = final_task_desc
    problem.template_program_str = template_str
    problem.llm_config = {
        "apiKey": req.llm_config.apiKey,
        "baseUrl": req.llm_config.baseUrl,
        "modelName": req.llm_config.modelName,
    }

    session.add(problem)
    session.commit()
    session.refresh(problem)

    return problem


@router.post("/problems/create-problem-package")
def create_problem_package(req: PreciseBuildRunRequest, session: Session = Depends(get_session)):
    """
    Create a problem package:
    - Copy the three specified source files into a new user_problem directory
    - Generate data JSON files from the provided DAG and environment configuration
    - Automatically run AST+LLM analysis to register as a ProblemPackage
    Return { problem }
    """
    # 1) Create a new user problem directory
    problem_id = uuid.uuid4()
    user_problem_id_str = str(problem_id)
    user_dir = os.path.join(PROBLEMS_DIR, user_problem_id_str)
    os.makedirs(user_dir, exist_ok=True)
    data_dir = os.path.join(user_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    # 2) Copy source files to the user directory
    if req.algorithmPreset:
        preset_dir = os.path.join(APP_ROOT, "algo_presets", req.algorithmPreset)
        src_framework = os.path.join(preset_dir, "framework.py")
        src_get_instance = os.path.join(preset_dir, "get_instance.py")
        src_evaluation = os.path.join(preset_dir, "evaluation.py")
    else:
        DEFAULT_FILES_ROOT = r"e:\\final_software"
        src_framework = os.path.join(DEFAULT_FILES_ROOT, "framework.py")
        src_get_instance = os.path.join(DEFAULT_FILES_ROOT, "get_instance_adapted_corrected.py")
        src_evaluation = os.path.join(DEFAULT_FILES_ROOT, "evaluation.py")
    for p in [src_framework, src_get_instance, src_evaluation]:
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail=f"Source file not found: {p}")

    framework_path = os.path.join(user_dir, "framework.py")
    get_instance_path = os.path.join(user_dir, "get_instance.py")
    evaluation_path = os.path.join(user_dir, "evaluation.py")
    shutil.copyfile(src_framework, framework_path)
    shutil.copyfile(src_get_instance, get_instance_path)
    shutil.copyfile(src_evaluation, evaluation_path)

    # 3) Generate data JSON files based on the configuration
    try:
        converter = ConfigConverter()
        full_config = {
            'environmentConfig': req.environmentConfig,
            'dagConfig': req.dagConfig
        }
        converted_data = converter.convert_full_config(full_config)

        with open(os.path.join(data_dir, "dag.json"), 'w', encoding='utf-8') as f:
            json.dump(converted_data['dag'], f, indent=2, ensure_ascii=False)

        env_cfg = req.environmentConfig or {}
        dag_obj = str(req.dagConfig.get('objective', 'sum')) if isinstance(req.dagConfig, dict) else 'sum'
        dag_ci = dict(converted_data['dag'])
        dag_ci['objective'] = dag_obj

        rr = req.bounds or {
            "cloud": [max(1, int(env_cfg.get('cloudCount', 1))), max(1, int(env_cfg.get('cloudCount', 1)) + 2)],
            "edge": [max(1, int(env_cfg.get('edgeCount', 1))), max(1, int(env_cfg.get('edgeCount', 1)) + 2)],
            "device": [max(1, int(env_cfg.get('deviceCount', 1))), max(1, int(env_cfg.get('deviceCount', 1)) + 2)],
        }

        cloud_spec = str(env_cfg.get('cloudSpec', 'cloud-large'))
        edge_spec = str(env_cfg.get('edgeSpec', 'edge-large'))
        device_spec = str(env_cfg.get('deviceSpec', 'device-large'))
        c_cfg = converter.server_types['cloud'].get(cloud_spec, converter.server_types['cloud']['cloud-large'])
        e_cfg = converter.server_types['edge'].get(edge_spec, converter.server_types['edge']['edge-large'])
        d_cfg = converter.server_types['device'].get(device_spec, converter.server_types['device']['device-large'])
        
        # Adjust to evaluator-required fields and lowercase type names
        type_prototypes = {
            "cloud": {"capacity": float(c_cfg['mips'])/100.0, "power": float(c_cfg['power']), "price": float(c_cfg['price'])},
            "edge": {"capacity": float(e_cfg['mips'])/100.0, "power": float(e_cfg['power']), "price": float(e_cfg['price'])},
            "device": {"capacity": float(d_cfg['mips'])/100.0, "power": float(d_cfg['power']), "price": float(d_cfg['price'])},
        }

        # Add network_defaults to satisfy get_instance strict validation
        conn_map = {
            ('cloud','cloud'): 'cloud_to_cloud',
            ('cloud','edge'): 'cloud_to_edge',
            ('cloud','device'): 'cloud_to_device',
            ('edge','cloud'): 'edge_to_cloud',
            ('edge','edge'): 'edge_to_edge',
            ('edge','device'): 'edge_to_device',
            ('device','cloud'): 'device_to_cloud',
            ('device','edge'): 'device_to_edge',
            ('device','device'): 'device_to_device',
        }
        network_defaults = {}
        for (a,b), key in conn_map.items():
            cfg = converter.network_config.get(key, {"bandwidth": 100.0, "power": 0.05})
            network_defaults[f"{a},{b}"] = {"bandwidth": float(cfg['bandwidth']), "link_power": float(cfg['power'])}

        # Complete converted_instance including network_defaults
        converted_instance = {
            "dag": dag_ci,
            "recommended_ranges": rr,
            "type_prototypes": type_prototypes,
            "network_defaults": network_defaults,
        }
        with open(os.path.join(data_dir, "converted_instance.json"), 'w', encoding='utf-8') as f:
            json.dump(converted_instance, f, indent=2, ensure_ascii=False)

        # Always save server count ranges (bounds)
        bounds_obj = req.bounds or rr
        with open(os.path.join(data_dir, "bounds.json"), 'w', encoding='utf-8') as f:
            json.dump(bounds_obj, f, indent=2, ensure_ascii=False)

        # Write manifest (includes bounds)
        manifest = {
            "converted_instance": "data/converted_instance.json",
            "dag_data": "data/dag.json",
            "bounds": "data/bounds.json",
        }
        with open(os.path.join(user_dir, "manifest.json"), 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate data files: {e}")

    # 4) Create ProblemPackage and automatically analyze and fill description
    try:
        # Read framework source
        with open(framework_path, 'r', encoding='utf-8') as f:
            framework_content = f.read()

        # AST function list
        try:
            functions = analyze_python_file(framework_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Code analysis failed: {e}")

        # Select target function (LLM)
        target_fn = None
        try:
            analysis = get_llm_analysis(
                api_key=req.llm_config.apiKey,
                base_url=req.llm_config.baseUrl,
                model_name=req.llm_config.modelName,
                full_source_code=framework_content,
                functions=functions,
            )
            valid_names = {fn.get('name') for fn in functions}
            filtered = [item for item in analysis if str(item.get('name')) in valid_names]
            if not filtered:
                raise HTTPException(status_code=422, detail="LLM did not return any valid target functions from the analyzed list")
            best_item = sorted(filtered, key=lambda x: int(x.get('potential', 0)), reverse=True)[0]
            target_fn = str(best_item.get('name'))
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=500, detail=f"LLM analysis failed: {e}")

        if not target_fn:
            raise HTTPException(status_code=422, detail="No target function selected by LLM")

        try:
            print(f"[Algorithm Selection] preset={req.algorithmPreset}")
            print(f"[Algorithm Selection] functions={[fn.get('name') for fn in functions]}")
            print(f"[Algorithm Selection] selected_function={target_fn}")
            try:
                potential = best_item.get('potential')
                if potential is not None:
                    print(f"[Algorithm Selection] score(potential)={potential}")
                fn_reason = best_item.get('reason')
                if fn_reason:
                    print("[Algorithm Selection] function_reason:\n" + str(fn_reason))
            except Exception:
                pass
        except Exception:
            pass

        # Build template program
        try:
            template_str = build_template_program(framework_path, target_fn)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        # Generate task description (includes summaries of called helper functions)
        try:
            tree = ast.parse(framework_content)
            called_functions = get_function_calls(tree, target_fn)
            if target_fn in called_functions:
                called_functions.remove(target_fn)
            function_details = get_functions_details(tree, called_functions)
            for detail in function_details:
                summary = get_llm_summary_for_function(
                    api_key=req.llm_config.apiKey,
                    base_url=req.llm_config.baseUrl,
                    model_name=req.llm_config.modelName,
                    function_source_code=detail['source_code']
                )
                detail['summary'] = summary

            helper_context = []
            for func_info in function_details:
                helper_context.append(f"""
Function: `{func_info['name']}`
- Parameters: {func_info['parameters']}
- Returns: {func_info['returns']}
- Summary: {func_info['summary']}
""")
            base_desc = req.description or "search optimal server counts (cloud/edge/device) with GA and scheduling evaluation"
            final_task_desc = f"""Please help me design a heuristic strategy to {base_desc}.

Requirement: Keep the generated code complexity as low as possible.

Use only the following helper functions and do not reinvent them:
{''.join(helper_context)}
"""
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to construct task description: {e}")

        try:
            print("[Algorithm Selection] task_description:\n" + final_task_desc)
        except Exception:
            pass

        problem_name = req.name or f"Precise-built Problem {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        new_problem = ProblemPackage(
            id=problem_id,
            name=problem_name,
            description=req.description or "Precisely built via UI",
            framework_file_path=framework_path,
            get_instance_file_path=get_instance_path,
            evaluation_file_path=evaluation_path,
            data_file_path=data_dir,
            target_function_name=target_fn,
            task_description=final_task_desc,
            template_program_str=template_str,
            llm_config={
                "apiKey": req.llm_config.apiKey,
                "baseUrl": req.llm_config.baseUrl,
                "modelName": req.llm_config.modelName,
            },
        )
        session.add(new_problem)
        session.commit()
        session.refresh(new_problem)
        return {"problem": new_problem}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create problem package: {e}")

@router.post("/problems/{problem_id}/start-run")
def start_run(problem_id: str, req: StartRunRequest, session: Session = Depends(get_session)):
    problem = session.get(ProblemPackage, problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="Problem not found")

    try:
        run_id = uuid.uuid4()
        # Set initial status to 'pending'; background executor (run_executor) will pick it up and create the directory
        new_run = Run(
            id=run_id,
            problem_id=problem.id,
            meoh_config=req.meoh_config,
            status="pending",
        )
        session.add(new_run)
        session.commit()
        session.refresh(new_run)

        return {"run": new_run}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start run: {e}")

@router.post("/problems/precise-build-run")
def precise_build_and_run(req: PreciseBuildRunRequest, session: Session = Depends(get_session)):
    """
    One-click workflow:
    - Call create_problem_package to create the problem package
    - Call start_run to start the run
    Return { problem, run }
    """
    # 1. Create problem package
    try:
        print(f"[Precise Build] preset={req.algorithmPreset}")
        if req.description:
            print("[Precise Build] optimization_description:\n" + str(req.description))
    except Exception:
        pass
    problem_response = create_problem_package(req, session)
    problem = problem_response["problem"]

    # 2. Start run
    start_run_req = StartRunRequest(meoh_config=req.meoh_config or {})
    run_response = start_run(problem.id, start_run_req, session)
    run = run_response["run"]

    return {
        "problem": problem,
        "run": run,
    }


class RecommendPresetRequest(BaseModel):
    dagConfig: Dict[str, Any]
    environmentConfig: Dict[str, Any]
    llm_config: LLMConfig

@router.post("/problems/recommend-algorithm-preset")
def recommend_algorithm_preset(req: RecommendPresetRequest):
    root = os.path.join(APP_ROOT, "algo_presets")
    items = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            d = os.path.join(root, name)
            if os.path.isdir(d):
                fx = os.path.join(d, "framework.py")
                gi = os.path.join(d, "get_instance.py")
                ev = os.path.join(d, "evaluation.py")
                try:
                    with open(fx, "r", encoding="utf-8") as f: f_str = f.read()[:4000]
                except Exception:
                    f_str = ""
                try:
                    with open(gi, "r", encoding="utf-8") as f: gi_str = f.read()[:4000]
                except Exception:
                    gi_str = ""
                try:
                    with open(ev, "r", encoding="utf-8") as f: ev_str = f.read()[:4000]
                except Exception:
                    ev_str = ""
                items.append({"name": name, "framework": f_str, "get_instance": gi_str, "evaluation": ev_str})
    result = llm_recommend_preset(
        api_key=req.llm_config.apiKey,
        base_url=req.llm_config.baseUrl,
        model_name=req.llm_config.modelName,
        dag_config=req.dagConfig,
        environment_config=req.environmentConfig,
        candidates=items,
    )
    return {"preset": result.get("preset", "ga"), "reason": result.get("reason", "")}

class SuggestDescriptionRequest(BaseModel):
    dagConfig: Dict[str, Any]
    environmentConfig: Dict[str, Any]
    llm_config: LLMConfig

@router.post("/problems/suggest-optimization-description")
def suggest_optimization_description(req: SuggestDescriptionRequest):
    result = llm_suggest_desc(
        api_key=req.llm_config.apiKey,
        base_url=req.llm_config.baseUrl,
        model_name=req.llm_config.modelName,
        dag_config=req.dagConfig,
        environment_config=req.environmentConfig,
    )
    return {"description": result.get("description", ""), "reason": result.get("reason", "")}

@router.delete("/problems/{problem_id}")
def delete_problem_package(problem_id: uuid.UUID, session: Session = Depends(get_session)):
    # Locate the problem package
    problem = session.get(ProblemPackage, problem_id)
    if not problem:
        raise HTTPException(status_code=404, detail="ProblemPackage not found")

    # Delete associated runs first to satisfy FK constraints
    from my_meoh_app.models.run import Run
    runs = session.exec(select(Run).where(Run.problem_id == problem_id)).all()
    for r in runs:
        session.delete(r)
    session.commit()

    # Remove the problem directory and files
    try:
        problem_dir = os.path.join(PROBLEMS_DIR, str(problem_id))
        if os.path.exists(problem_dir):
            shutil.rmtree(problem_dir, ignore_errors=True)
    except Exception:
        # Ignore file deletion errors
        pass

    # Delete the problem record
    session.delete(problem)
    session.commit()
    return {"status": "deleted", "id": str(problem_id)}
