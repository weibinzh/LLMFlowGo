from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque
import math

"""
DAG Hierarchy and Task Load Analysis Module

This module provides the parsing of DAG graph data (successors or edge lists) in task scheduling problems into a hierarchical structure, and summarizes each layer's number of tasks, workload, data volume, etc., for subsequent LLM analysis and preliminary server quantity configuration.

It supports two common data formats:
- problem_instance style (output of get_instance_adapted):
  {
    "workload": {task_id: float, ...},
    "successors": {u: [v1, v2, ...], ...},
    "data_to_transfer": {(u, v): float, ...}  # Optional
  }
- dag.json style:
  {
    "workload": {"0": {"taskAmount": 1.0, "dataAmount": 1.0}, ...},
    "edges": [{"from": u, "to": v}, ...]
  }
"""

# ---------------------------------------------------------------------------
# Low-level Utility Functions
# ---------------------------------------------------------------------------

def _normalize_int_keys(d: Dict[Any, Any]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except Exception:
            out[k] = v
    return out


def _build_successors_from_edges(edges: List[Dict[str, int]]) -> Dict[int, List[int]]:
    succ: Dict[int, List[int]] = defaultdict(list)
    for e in edges:
        u = int(e.get("from"))
        v = int(e.get("to"))
        succ[u].append(v)
    nodes = set(succ.keys()) | {v for vs in succ.values() for v in vs}
    for n in nodes:
        succ.setdefault(n, [])
    return {int(k): list(map(int, v)) for k, v in succ.items()}


def _build_predecessors(successors: Dict[int, List[int]]) -> Dict[int, List[int]]:
    preds: Dict[int, List[int]] = defaultdict(list)
    for u, vs in successors.items():
        for v in vs:
            preds[v].append(u)
    nodes = set(successors.keys()) | set(preds.keys())
    for n in nodes:
        preds.setdefault(n, [])
    return {int(k): list(map(int, v)) for k, v in preds.items()}


def _topological_levels(successors: Dict[int, List[int]],
                        predecessors: Optional[Dict[int, List[int]]] = None) -> Tuple[Dict[int, int], List[int]]:
    """
    Use Kahn's algorithm to perform topological sorting and calculate the level of each node:
    level(u) = 0 if indegree(u) == 0, else max(level(p) for p in preds(u)) + 1
    Returns (levels, topo_order)
    """
    if predecessors is None:
        predecessors = _build_predecessors(successors)

    indeg = {u: len(predecessors.get(u, [])) for u in set(successors.keys()) | set(predecessors.keys())}
    q = deque([u for u, d in indeg.items() if d == 0])
    levels: Dict[int, int] = {}
    topo: List[int] = []

    while q:
        u = q.popleft()
        topo.append(u)
        preds_u = predecessors.get(u, [])
        if not preds_u:
            levels[u] = 0
        else:
            levels[u] = max(levels[p] for p in preds_u) + 1
        for v in successors.get(u, []):
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return levels, topo


def _extract_workload_maps(workload: Dict[Any, Any]) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Convert different styles of workload to two tables:
    - task_amount_map: {task_id: float}
    - data_amount_map: {task_id: float} (0 if missing)
    Supports:
    - {task_id: float}
    - {task_id: {"taskAmount": float, "dataAmount": float}}
    """
    task_amount_map: Dict[int, float] = {}
    data_amount_map: Dict[int, float] = {}

    workload_norm = _normalize_int_keys(workload)

    sample_val = next(iter(workload_norm.values())) if workload_norm else 0
    if isinstance(sample_val, dict):
        for k, v in workload_norm.items():
            task_amount_map[int(k)] = float(v.get("taskAmount", 0))
            data_amount_map[int(k)] = float(v.get("dataAmount", 0))
    else:
        for k, v in workload_norm.items():
            task_amount_map[int(k)] = float(v)
            data_amount_map[int(k)] = 0.0

    all_nodes = set(workload_norm.keys())
    for n in all_nodes:
        task_amount_map.setdefault(int(n), 0.0)
        data_amount_map.setdefault(int(n), 0.0)

    return task_amount_map, data_amount_map


# ---------------------------------------------------------------------------
# High-level analysis function
# ---------------------------------------------------------------------------

def analyze_problem_instance(problem_instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Input a unified problem instance dictionary, output the DAG hierarchy and task amount summary.

    Supports two input styles:
    1) Instance with successors:
       {
         "workload": {task_id: float 或 {taskAmount, dataAmount}},
         "successors": {u: [v, ...]},
         "data_to_transfer": {(u, v): float} 可选
       }
    2) Instance with edges:
       {
         "workload": {"0": {"taskAmount": 1.0, "dataAmount": 1.0}, ...},
         "edges": [{"from": u, "to": v}, ...]
       }

    Returns a dictionary with the following fields:
    - levels: {task_id: level}
    - level_nodes: {level: [task_ids...]}
    - level_summary: [
        {"level": L, "task_count": c, "total_task_amount": x, "total_data_amount": y, "avg_task_amount": a}
      ]
    - graph_summary: {"num_tasks": N, "depth": D, "total_task_amount": X, "total_data_amount": Y}
    - entry_nodes: [task_ids]
    - exit_nodes: [task_ids]
    - topo_order: [task_ids]  # 若图有环则可能不含全部节点
    """
    # Analyze workload
    if "workload" not in problem_instance:
        raise ValueError("problem_instance 缺少 'workload' 字段")
    task_amount_map, data_amount_map = _extract_workload_maps(problem_instance["workload"])

    # Build successors
    successors: Dict[int, List[int]]
    if "successors" in problem_instance:
        raw_succ = problem_instance["successors"]
        successors = {int(u): [int(v) for v in vs] for u, vs in _normalize_int_keys(raw_succ).items()}
    elif "edges" in problem_instance:
        successors = _build_successors_from_edges(problem_instance["edges"])  # 支持 dag.json 风格
    else:
        raise ValueError("problem_instance 需要包含 'successors' 或 'edges'")

    # Preprocess predecessors
    predecessors = _build_predecessors(successors)

    # entry / exit
    all_nodes = sorted(set(successors.keys()) | set(predecessors.keys()))
    entry_nodes = [u for u in all_nodes if len(predecessors.get(u, [])) == 0]
    exit_nodes = [u for u in all_nodes if len(successors.get(u, [])) == 0]

    # Topological levels
    levels, topo = _topological_levels(successors, predecessors)
    max_level = max(levels.values()) if levels else 0

    # Level to nodes mapping
    level_nodes: Dict[int, List[int]] = defaultdict(list)
    for u, L in levels.items():
        level_nodes[L].append(u)
    for L in level_nodes:
        level_nodes[L].sort()

    # Level summary statistics
    level_summary: List[Dict[str, Any]] = []
    total_task_amount = 0.0
    total_data_amount = 0.0
    for L in range(max_level + 1):
        nodes = level_nodes.get(L, [])
        t_amount = float(sum(task_amount_map.get(u, 0.0) for u in nodes))
        d_amount = float(sum(data_amount_map.get(u, 0.0) for u in nodes))
        count = len(nodes)
        avg_t = (t_amount / count) if count > 0 else 0.0
        level_summary.append({
            "level": L,
            "task_count": count,
            "total_task_amount": t_amount,
            "total_data_amount": d_amount,
            "avg_task_amount": avg_t,
        })
        total_task_amount += t_amount
        total_data_amount += d_amount

    graph_summary = {
        "num_tasks": len(all_nodes),
        "depth": max_level + 1 if levels else 0,
        "total_task_amount": total_task_amount,
        "total_data_amount": total_data_amount,
    }

    # Five angle metrics
    critical_path = _critical_path(successors, predecessors, task_amount_map)
    ccr = _ccr_metrics(task_amount_map, data_amount_map, problem_instance.get("data_to_transfer"))
    parallelism = _parallelism_metrics(level_summary, critical_path)
    entropy = _entropy_metrics(level_summary, successors, predecessors, task_amount_map)
    metrics = {
        "critical_path": critical_path,
        "ccr": ccr,
        "parallelism": parallelism,
        "entropy": entropy,
    }

    return {
        "levels": levels,
        "level_nodes": dict(level_nodes),
        "level_summary": level_summary,
        "graph_summary": graph_summary,
        "entry_nodes": entry_nodes,
        "exit_nodes": exit_nodes,
        "topo_order": topo,
        "metrics": metrics,
    }


def build_llm_input_payload(analysis: Dict[str, Any],
                            servers_fixed_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convert the analysis results into an input payload for the LLM for step two: have the LLM provide the quantity configuration for the three types of servers.

    servers_fixed_info：
    {
      "server_types": ["edge", "fog", "cloud"],
      "bounds": {"edge": [0, 50], "fog": [0, 20], "cloud": [1, 10]},
      "constraints": {"max_power_w": 5000, "max_cost": 1000}
    }
    """
    payload = {
        "graph_summary": analysis.get("graph_summary", {}),
        "level_summary": analysis.get("level_summary", []),
        "entry_nodes": analysis.get("entry_nodes", []),
        "exit_nodes": analysis.get("exit_nodes", []),
        # Concurrency profile（每层任务数）
        "concurrency_profile": [lvl.get("task_count", 0) for lvl in analysis.get("level_summary", [])],
    }
    # Five angle metrics summary（供 LLM 使用）
    payload["metrics"] = analysis.get("metrics", {})
    if servers_fixed_info:
        payload["servers"] = servers_fixed_info
    return payload


def _critical_path(successors: Dict[int, List[int]],
                   predecessors: Dict[int, List[int]],
                   task_weight_map: Dict[int, float]) -> Dict[str, float]:
    """
    Critical Path (Longest Path) Calculation：
    - length_unweighted: Number of nodes in the longest path (unweighted)
    - length_weighted: Sum of weights along the longest path (weighted by task amount)
    Using topological order for DP.
    """
    # Using topological order with DP calculation for critical path
    _, topo = _topological_levels(successors, predecessors)
    dp_w: Dict[int, float] = {}
    dp_len: Dict[int, int] = {}
    for u in topo:
        preds = predecessors.get(u, [])
        if not preds:
            dp_w[u] = float(task_weight_map.get(u, 0.0))
            dp_len[u] = 1
        else:
            dp_w[u] = max(dp_w[p] for p in preds) + float(task_weight_map.get(u, 0.0))
            dp_len[u] = max(dp_len[p] for p in preds) + 1

    cp_weight = max(dp_w.values()) if dp_w else 0.0
    cp_len = max(dp_len.values()) if dp_len else 0
    return {"length_unweighted": int(cp_len), "length_weighted": float(cp_weight)}


def _ccr_metrics(task_amount_map: Dict[int, float],
                 data_amount_map: Dict[int, float],
                 data_to_transfer: Optional[Dict[Any, Any]] = None) -> Dict[str, float]:
    """
    CCR Calculation：Communication/Computation Ratio
    - Node Data CCR (ccr_node): total_data_node / total_task_amount
    - Edge Data CCR (ccr_edge): sum(data_to_transfer) / total_task_amount (if provided)
    """
    total_task = float(sum(task_amount_map.values()))
    total_data_node = float(sum(data_amount_map.values()))
    ccr_node = (total_data_node / total_task) if total_task > 0 else 0.0

    total_edge_data = 0.0
    if isinstance(data_to_transfer, dict):
        for _, v in data_to_transfer.items():
            try:
                total_edge_data += float(v)
            except Exception:
                pass
    ccr_edge = (total_edge_data / total_task) if total_task > 0 else 0.0

    return {
        "total_task_amount": total_task,
        "total_data_node": total_data_node,
        "total_data_edge": total_edge_data,
        "ccr_node": ccr_node,
        "ccr_edge": ccr_edge,
    }


def _parallelism_metrics(level_summary: List[Dict[str, Any]],
                         critical_path: Dict[str, float]) -> Dict[str, Any]:
    """
    Parallelism Estimation：
    - max_width: Maximum layer width (peak number of tasks that can run simultaneously)
    - avg_width/std_width: Mean and standard deviation of layer widths
    - average_parallelism_weighted: Total task amount / Critical path weight
    - average_parallelism_unweighted: Total tasks / Critical path length
    - peak_level: Level number with maximum width
    - concurrency_profile: List of widths per level
    """
    widths = [int(ls.get("task_count", 0)) for ls in level_summary]
    max_width = max(widths) if widths else 0
    avg_width = (sum(widths) / len(widths)) if widths else 0.0
    var = (sum((w - avg_width) ** 2 for w in widths) / len(widths)) if widths else 0.0
    std_width = math.sqrt(var)

    total_task_amount = float(sum(ls.get("total_task_amount", 0.0) for ls in level_summary))
    total_tasks = int(sum(widths))
    cp_weight = float(critical_path.get("length_weighted", 0.0))
    cp_len = int(critical_path.get("length_unweighted", 0))

    avg_par_w = (total_task_amount / cp_weight) if cp_weight > 0 else 0.0
    avg_par_u = (total_tasks / cp_len) if cp_len > 0 else 0.0

    peak_level = max(range(len(widths)), key=lambda i: widths[i]) if widths else 0

    return {
        "max_width": int(max_width),
        "avg_width": float(avg_width),
        "std_width": float(std_width),
        "average_parallelism_weighted": float(avg_par_w),
        "average_parallelism_unweighted": float(avg_par_u),
        "peak_level": int(peak_level),
        "concurrency_profile": widths,
    }


def _shannon_entropy_from_counts(counts: List[int]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            ent -= p * math.log(p + 1e-12)
    return float(ent)


def _shannon_entropy_from_weights(weights: List[float]) -> float:
    total = float(sum(weights))
    if total <= 0:
        return 0.0
    ent = 0.0
    for w in weights:
        if w > 0:
            p = w / total
            ent -= p * math.log(p + 1e-12)
    return float(ent)


def _entropy_metrics(level_summary: List[Dict[str, Any]],
                     successors: Dict[int, List[int]],
                     predecessors: Dict[int, List[int]],
                     task_amount_map: Dict[int, float]) -> Dict[str, float]:
    """
    Structure Entropy：
    - level_entropy_nodes: Shannon entropy based on node count distribution per level
    - level_entropy_workload: Shannon entropy based on task amount distribution per level
    - degree_entropy_out/in: Shannon entropy based on out/in-degree distribution
    """
    node_counts = [int(ls.get("task_count", 0)) for ls in level_summary]
    workload_weights = [float(ls.get("total_task_amount", 0.0)) for ls in level_summary]
    level_entropy_nodes = _shannon_entropy_from_counts(node_counts)
    level_entropy_workload = _shannon_entropy_from_weights(workload_weights)

    nodes = sorted(set(successors.keys()) | set(predecessors.keys()))
    out_degrees = [len(successors.get(u, [])) for u in nodes]
    in_degrees = [len(predecessors.get(u, [])) for u in nodes]

    def counts_distribution(values: List[int]) -> List[int]:
        freq: Dict[int, int] = defaultdict(int)
        for v in values:
            freq[v] += 1
        return list(freq.values())

    out_degree_counts = counts_distribution(out_degrees)
    in_degree_counts = counts_distribution(in_degrees)

    degree_entropy_out = _shannon_entropy_from_counts(out_degree_counts)
    degree_entropy_in = _shannon_entropy_from_counts(in_degree_counts)

    return {
        "level_entropy_nodes": float(level_entropy_nodes),
        "level_entropy_workload": float(level_entropy_workload),
        "degree_entropy_out": float(degree_entropy_out),
        "degree_entropy_in": float(degree_entropy_in),
    }


__all__ = [
    "analyze_problem_instance",
    "build_llm_input_payload",
]