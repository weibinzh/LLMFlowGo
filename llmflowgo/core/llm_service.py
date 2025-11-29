import time
from typing import List, Dict, Any, Optional
import json
from openai import OpenAI
import traceback
import re


def get_llm_analysis(
    api_key: str,
    base_url: str,
    model_name: str,
    full_source_code: str,
    functions: List[Dict[str, str]],
    server_tiers: Optional[Dict[str, str]] = None,
    dag_structure: Optional[Dict[str, Any]] = None,
) -> List[Dict]:
    """
    Using a real LLM API to analyze the optimization potential of a list of functions。

    Args:
        api_key: OpenAI API Key.
        base_url: Base URL of the LLM API.
        model_name: Name of the model to use.
        full_source_code: Complete Python source code as a string.
        functions: List of function information extracted from AST analysis.

    Returns:
        A list of dictionaries, each containing 'name', 'potential', and 'reason' for each function.
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"Error creating OpenAI client: {e}")
        raise ValueError(f"Failed to create OpenAI client. Check your API Key or Base URL. Error: {e}")

    function_names = [func['name'] for func in functions]

    context = {
        "serverTiers": server_tiers or {"cloud": "large", "edge": "medium", "device": "small"},
    }
    if dag_structure is not None:
        context["dag"] = dag_structure

    prompt = f"""
As an expert in heuristic design and meta-evolution, analyze the given Python source and evaluate each top-level function's exploration space capability.

A function has strong exploration-space capability if it offers tunable heuristic decision logic—such as scoring rules, fitness evaluation, neighborhood construction, or selection/mutation strategies—allowing optimization algorithms to explore alternatives effectively.
Functions limited to I/O, data loading, fixed control flow, or simple utilities lack such tunable structure and therefore have low exploration space.

Additional Constraint (Max 2 callees):
- Prefer candidates whose internal call footprint is small (calls to other functions ≤ 2).
- If a function calls more than two other functions, cap its potential at 5 and mention this explicitly in the reason.
- Heavy orchestrators with many calls should receive low scores regardless of other factors.

Environment context (JSON):
{json.dumps(context, ensure_ascii=False)}

Analyze code:
```python
{full_source_code}
```

Top-level functions:
{', '.join(function_names)}

Task:
For each listed function, assess how suitable it is as the primary target for heuristic evolution, focusing strictly on exploration space capability (richness, tunability, and impact on search behavior). Apply the Max-2-callees constraint above when scoring. Ignore non-heuristic or orchestration code.

Output Format:
Return a JSON array. Each item MUST have keys:
- \"name\": function name (string)
- \"potential\": exploration space score from 1 (lowest) to 10 (highest) (integer)
- \"reason\": concise justification (string)

Example:
```json
[
  {{
    "name": "evaluate_fitness",
    "potential": 10,
    "reason": "Defines core scoring signal; highly tunable, few callees (≤2)."
  }},
  {{
    "name": "load_data",
    "potential": 1,
    "reason": "I/O utility; no heuristic logic to evolve."
  }}
]
```

Now, provide the analysis for: {', '.join(function_names)}.
"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            print(f"--- Attempting LLM Analysis (Attempt {attempt + 1}/{max_retries}) ---")
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant designed to output JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                response_format={"type": "json_object"},
            )
            
            response_content = chat_completion.choices[0].message.content
            print("--- LLM Response Received ---")
            
     
            # Try to extract JSON from a markdown code block
            json_match = re.search(r"```(json)?\s*([\s\S]*?)\s*```", response_content)
            if json_match:
                print("--- Found JSON in markdown block, extracting... ---")
                response_content = json_match.group(2)

            # The response is expected to be a JSON string, which needs to be parsed.
            # The prompt now asks for a JSON array directly.
            # Let's assume the top-level key might be 'analysis' or it's the array directly.
            parsed_response = json.loads(response_content)

            # Check if the response is a list. If it's a dict with one key, assume the list is inside.
            if isinstance(parsed_response, dict) and len(parsed_response.keys()) == 1:
                analysis_result = list(parsed_response.values())[0]
            else:
                analysis_result = parsed_response

            if not isinstance(analysis_result, list):
                 raise ValueError("LLM response is not a list as expected.")

            # Validate structure of each item
            for item in analysis_result:
                if not all(key in item for key in ["name", "potential", "reason"]):
                    raise ValueError(f"Invalid item in LLM response: {item}")

            print("--- LLM Analysis Complete and Parsed Successfully ---")
            return analysis_result

        except Exception as e:
            print(f"--- LLM Analysis Failed (Attempt {attempt + 1}/{max_retries}) ---")
            print(traceback.format_exc())
            if attempt + 1 == max_retries:
                # Re-raise a more user-friendly exception to be caught by the API endpoint
                raise RuntimeError(f"Failed to get a valid response from the LLM after {max_retries} attempts. Error: {e}")
            time.sleep(2) # Wait before retrying
    
    # This part should not be reachable
    return []


def get_llm_summary_for_function(
    api_key: str,
    base_url: str,
    model_name: str,
    function_source_code: str
) -> str:
    """
    Using a LLM to generate a concise summary of a single Python function's role in the exploration space.
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI client: {e}")

    prompt = f"""
As a seasoned Python engineer, read the function source below and, in one sentence, explain this function's role in the exploration space.

Rules:
- Answer in English, concise and accurate.
- Sentence length no more than 20 words.
- Focus on its role in the exploration space; do not describe implementation details.

Function source:
```python
{function_source_code}
```

One-sentence explanation:
"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that provides concise function summaries.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=model_name,
                temperature=0.2, # Lower temperature for more deterministic summaries
                max_tokens=50,
            )
            summary = chat_completion.choices[0].message.content.strip()
            # Basic validation
            if not summary or len(summary.split()) > 25: # a bit of leeway
                 raise ValueError("Generated summary is invalid or too long.")
            return summary
        except Exception as e:
            print(f"--- LLM Summary Generation Failed (Attempt {attempt + 1}/{max_retries}) ---")
            print(traceback.format_exc())
            if attempt + 1 == max_retries:
                raise RuntimeError(f"Failed to generate a valid summary for the function. Error: {e}")
            time.sleep(1)
    
    return "Summary generation failed." # Should not be reached


def get_server_count_recommendation(
    api_key: str,
    base_url: str,
    model_name: str,
    payload: Dict
) -> Dict[str, List[int]]:
    """
    Using a LLM to recommend integer count ranges for three server types (cloud, edge, device) based on DAG structure metrics and available server specifications.
    """
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI client: {e}")

    bounds = payload.get("servers", {}).get("bounds")

    if "serverTiers" not in payload:
        payload["serverTiers"] = {"cloud": "large", "edge": "medium", "device": "small"}

    prompt = f"""
You are an expert cloud-and-edge scheduler. Given a DAG's structure metrics and available server specifications, recommend integer count ranges for three server types.

Constraints and rules:
- ONLY return a JSON object with keys: cloudRange, edgeRange, deviceRange.
- Each value must be an array of two integers: [min, max].
- Ensure min <= max. Return integers only.
- Balance parallelism and critical-path length: higher parallelism widens edge/device ranges; long critical path can raise cloud upper bound.
- Consider CCR: higher communication-to-computation ratios favor larger edge/device ranges to reduce latency.
- Use server tiers mapping: cloud=large, edge=medium, device=small.
- Do NOT include any commentary or extra keys.

Inputs:
DAG metrics and summaries (JSON):
{json.dumps(payload, ensure_ascii=False)}

Output JSON example:
{{"cloudRange": [2, 5], "edgeRange": [6, 12], "deviceRange": [40, 50]}}
"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            response_content = chat_completion.choices[0].message.content.strip()

            json_match = re.search(r"```(json)?\s*([\s\S]*?)\s*```", response_content)
            if json_match:
                response_content = json_match.group(2)

            result = json.loads(response_content)
            if isinstance(result, dict) and set(result.keys()) == {"recommendation"}:
                result = result["recommendation"]

            if all(k in result for k in ["cloudCount", "edgeCount", "deviceCount"]):
                result = {
                    "cloudRange": [int(result["cloudCount"]), int(result["cloudCount"])],
                    "edgeRange": [int(result["edgeCount"]), int(result["edgeCount"])],
                    "deviceRange": [int(result["deviceCount"]), int(result["deviceCount"])],
                }

            # Validation structure is range
            for key in ["cloudRange", "edgeRange", "deviceRange"]:
                if key not in result:
                    raise ValueError(f"LLM response missing key: {key}")
                arr = result[key]
                if not (isinstance(arr, list) and len(arr) == 2):
                    raise ValueError(f"LLM response key '{key}' is not a [min,max] array: {arr}")
                # Try to convert elements to integers
                try:
                    arr = [int(float(arr[0])), int(float(arr[1]))]
                except Exception:
                    raise ValueError(f"LLM response key '{key}' contains non-integer values: {arr}")
                # Order validation, no default bounds
                lo = min(arr[0], arr[1])
                hi = max(arr[0], arr[1])
                result[key] = [lo, hi]

            # Success: ranges parsed and normalized. Suppressing duplicate log here; API route logs once.
            return {
                "cloudRange": result["cloudRange"],
                "edgeRange": result["edgeRange"],
                "deviceRange": result["deviceRange"],
            }
        except Exception as e:
            print(f"--- LLM Server Recommendation Failed (Attempt {attempt + 1}/{max_retries}) ---")
            print(traceback.format_exc())
            if attempt + 1 == max_retries:
                raise RuntimeError("Failed to get a valid server count recommendation from LLM.")
            time.sleep(1)

    raise RuntimeError("Failed to get a valid response from the LLM.")


def get_server_counts_and_preset(
    api_key: str,
    base_url: str,
    model_name: str,
    payload: Dict,
    candidate_names: List[str],
) -> Dict[str, Any]:
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI client: {e}")

    if "serverTiers" not in payload:
        payload["serverTiers"] = {"cloud": "large", "edge": "medium", "device": "small"}

    prompt = (
        "You are an expert cloud-and-edge scheduler. Given DAG metrics and available server specs, "
        "recommend integer count ranges for cloud/edge/device and select ONE algorithm preset.\n\n"
        "Rules:\n"
        "- ONLY return a JSON object with keys: cloudRange, edgeRange, deviceRange, preset, reason.\n"
        "- Each *Range must be [min, max] integers with min <= max.\n"
        f"- preset MUST be one of: {json.dumps(candidate_names, ensure_ascii=False)}.\n"
        "- Use server tiers mapping: cloud=large, edge=medium, device=small.\n"
        "- Do NOT include any commentary or extra keys.\n\n"
        "Inputs (JSON):\n"
        f"{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Output example:\n"
        "{\"cloudRange\": [2,5], \"edgeRange\": [6,12], \"deviceRange\": [40,50], \"preset\": \"ga\", \"reason\": \"Prefer GA for makespan on heterogeneous DAG\"}"
    )

    max_retries = 2
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            response_content = chat_completion.choices[0].message.content.strip()
            m = re.search(r"```(json)?\s*([\s\S]*?)\s*```", response_content)
            if m:
                response_content = m.group(2)
            result = json.loads(response_content)

            # normalize ranges
            for key in ["cloudRange", "edgeRange", "deviceRange"]:
                if key not in result:
                    raise ValueError(f"LLM response missing key: {key}")
                arr = result[key]
                if not (isinstance(arr, list) and len(arr) == 2):
                    raise ValueError(f"Key '{key}' must be [min,max]: {arr}")
                arr = [int(float(arr[0])), int(float(arr[1]))]
                lo, hi = min(arr[0], arr[1]), max(arr[0], arr[1])
                result[key] = [lo, hi]

            preset = str(result.get("preset", "")).strip()
            reason = str(result.get("reason", "")).strip()
            if not preset:
                raise ValueError("LLM did not return 'preset'.")
            if candidate_names and preset not in candidate_names:
                raise ValueError(f"Preset '{preset}' not in candidates: {candidate_names}")

            return {
                "cloudRange": result["cloudRange"],
                "edgeRange": result["edgeRange"],
                "deviceRange": result["deviceRange"],
                "preset": preset,
                "reason": reason,
            }
        except Exception:
            if attempt + 1 == max_retries:
                # final fallback
                raise RuntimeError("Failed to get combined server+algorithm recommendation from LLM.")
            time.sleep(1)

def recommend_algorithm_preset(
    api_key: str,
    base_url: str,
    model_name: str,
    dag_config: Dict,
    environment_config: Dict,
    candidates: List[Dict]
) -> Dict[str, str]:
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI client: {e}")

    payload = {
        "dag": dag_config,
        "environment": environment_config,
        "candidates": [
            {
                "name": c.get("name"),
                "framework": c.get("framework", ""),
                "get_instance": c.get("get_instance", ""),
                "evaluation": c.get("evaluation", ""),
            }
            for c in candidates
        ],
    }

    prompt = (
        "You are an algorithm framework selector. Given a DAG and environment, "
        "choose the best algorithm preset from the candidates and explain briefly. "
        "Return a JSON object with keys 'preset' and 'reason'. Do not include extra keys.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

    max_retries = 2
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = chat_completion.choices[0].message.content.strip()
            m = re.search(r"```(json)?\s*([\s\S]*?)\s*```", content)
            if m:
                content = m.group(2)
            result = json.loads(content)
            preset = str(result.get("preset", "")).strip()
            reason = str(result.get("reason", "")).strip()
            if not preset:
                raise ValueError("LLM did not return 'preset'.")
            return {"preset": preset, "reason": reason}
        except Exception:
            if attempt + 1 == max_retries:
                obj = str(dag_config.get("objective", "")).lower() if isinstance(dag_config, dict) else ""
                fallback = "ga" if obj in ("makespan", "time", "latency") else "pso"
                return {"preset": fallback, "reason": "Heuristic fallback based on objective."}
            time.sleep(1)


def suggest_optimization_description(
    api_key: str,
    base_url: str,
    model_name: str,
    dag_config: Dict,
    environment_config: Dict,
) -> Dict[str, str]:
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI client: {e}")

    payload = {
        "dag": dag_config,
        "environment": environment_config,
    }
    prompt = f"""You are designing the next-step heuristic strategy for algorithm improvement.

Goal:
- Produce a concise and actionable plan specifying WHICH heuristic strategies to use (e.g., scheduling priority rules, fitness/scoring design, neighborhood/move operators, selection/crossover/mutation tuning, CCR-aware placement, critical-path handling, load balancing, parallelism exploitation).
- Base guidance STRICTLY on the provided DAG structure metrics and environment/servers info.
- Do NOT diagnose or critique data quality; do NOT suggest changing data. Focus ONLY on algorithmic heuristics.

Output format:
Return a JSON object with exactly two keys:
- description: a short, actionable heuristic strategy plan (specific tactics to apply)
- reason: a brief justification tied directly to DAG metrics (critical path/parallelism/CCR/entropy) and environment

Inputs (JSON):
{json.dumps(payload, ensure_ascii=False)}"""

    max_retries = 2
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            content = chat_completion.choices[0].message.content.strip()
            m = re.search(r"```(json)?\s*([\s\S]*?)\s*```", content)
            if m:
                content = m.group(2)
            result = json.loads(content)
            desc = str(result.get("description", "")).strip()
            reason = str(result.get("reason", "")).strip()
            if not desc:
                raise ValueError("LLM did not return 'description'.")
            return {"description": desc, "reason": reason}
        except Exception:
            if attempt + 1 == max_retries:
                return {
                    "description": "Balance makespan and energy via scheduling and counts tuning.",
                    "reason": "Generic direction for DAG scheduling under heterogeneous resources.",
                }
            time.sleep(1)

def get_llm_potential_reason(
    api_key: str,
    base_url: str,
    model_name: str,
    function_source_code: str,
) -> str:
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        raise ValueError(f"Failed to create OpenAI client: {e}")

    prompt = (
        "In no more than 20 words, propose how to optimize this function for heuristic tuning. Focus on actionable, tunable strategies; do not list shortcomings.\n\n"
        + "```python\n" + function_source_code + "\n```"
    )

    max_retries = 2
    for attempt in range(max_retries):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers concisely."},
                    {"role": "user", "content": prompt},
                ],
                model=model_name,
                temperature=0.2,
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception:
            if attempt + 1 == max_retries:
                return "Tunable heuristic logic offers room for optimization."
            time.sleep(1)
