"""
Dynamic parameter discovery and calculated heuristic guessing.
Supports both internal Python agents and external UCI binaries without hardcoding.
"""

import subprocess
from typing import Any, Dict, List


def discover_agent_parameters(engine_command_or_agent: Any) -> List[Dict[str, Any]]:
    """
    Introspects an agent to discover its controllable knobs dynamically.
    For UCI binaries, parses 'option name ... type ... min ... max'.
    """
    discovered_knobs = []

    if isinstance(engine_command_or_agent, str):
        try:
            proc = subprocess.Popen(
                [engine_command_or_agent],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            proc.stdin.write("uci\n")
            proc.stdin.flush()

            while True:
                line = proc.stdout.readline()
                if not line or line.strip() == "uciok":
                    break
                line = line.strip()
                if line.startswith("option name"):
                    parts = line.split()
                    name_idx = parts.index("name") + 1
                    type_idx = parts.index("type")
                    opt_name = " ".join(parts[name_idx:type_idx])
                    opt_type = parts[type_idx + 1]

                    knob_info = {"name": opt_name, "type": opt_type, "source": "uci_option"}
                    if "min" in parts and "max" in parts:
                        knob_info["min"] = float(parts[parts.index("min") + 1])
                        knob_info["max"] = float(parts[parts.index("max") + 1])
                    discovered_knobs.append(knob_info)
            proc.terminate()
        except Exception:
            pass # Fallback if UCI binary fails to load
    elif hasattr(engine_command_or_agent, "get_configurable_knobs"):
        discovered_knobs = engine_command_or_agent.get_configurable_knobs()

    # FIX: these two need a "source" key too -- form_prioritized_hypotheses()
    # filters on k["source"] == "uci_option", and a plain k["source"] lookup
    # (no .get()) on an entry missing the key raised KeyError the moment any
    # blunder reached the "perturb_options" branch. Tagged "search_control"
    # here so that filter now excludes them cleanly instead of crashing.
    universal_search_knobs = [
        {"name": "depth_escalation", "type": "search_control", "dimension": "max_depth", "source": "search_control"},
        {"name": "time_budget_escalation", "type": "search_control", "dimension": "time_budget_ms", "source": "search_control"},
    ]
    return universal_search_knobs + discovered_knobs


def form_prioritized_hypotheses(blunder: Dict[str, Any], available_knobs: List[Dict]) -> List[Dict]:
    """
    Analyzes blunder telemetry to make calculated guesses, assigning a confidence
    score to prioritize which empirical tests to run first.
    """
    depth = blunder.get("agent_depth", 0)
    neural_disagree_val = blunder.get("neural_disagreement")
    neural_disagree = abs(neural_disagree_val) if neural_disagree_val is not None else 0.0
    trace = blunder.get("agent_pv_trace", [])
    
    max_eval_in_trace = max([t.get("eval", 0.0) for t in trace]) if trace else 0.0
    scored_hypotheses = []
    
    # 1. Horizon Search (Depth)
    depth_score = 50
    if depth < 5:
        depth_score += 30
    if max_eval_in_trace < 2.0:
        depth_score += 15
        
    scored_hypotheses.append({
        "action": "scale_depth",
        "test_sequence": [depth + 2, depth + 4],
        "description": "Test search horizon expansion",
        "confidence": depth_score
    })

    # 2. Compute Choke (Time Budget)
    time_score = 40
    if depth <= 3: 
        time_score += 45
        
    scored_hypotheses.append({
        "action": "scale_time",
        "test_sequence": [600, 2000], 
        "description": "Test compute starvation",
        "confidence": time_score
    })

    # 3. Static Distortions (Evaluator/Material Weights)
    eval_score = 30
    if neural_disagree > 2.0:
        eval_score += 50
    if max_eval_in_trace > 5.0:
        eval_score += 40
        
    # .get() rather than k["source"]: defensive against any future knob
    # source that forgets to tag itself (see discover_agent_parameters fix).
    eval_knobs = [k for k in available_knobs if k.get("source") == "uci_option"]
    if eval_knobs:
        scored_hypotheses.append({
            "action": "perturb_options",
            "knobs": eval_knobs,
            "description": "Test evaluation weight adjustments",
            "confidence": eval_score
        })

    scored_hypotheses.sort(key=lambda x: x["confidence"], reverse=True)
    return scored_hypotheses