"""
Dynamic parameter discovery and calculated heuristic guessing.
Supports both internal Python agents and external UCI binaries without hardcoding.
"""

import subprocess
from typing import Any, Dict, List

# The Diagnostic Taxonomy for Dashboard / LLM mapping
TAXONOMY = {
    "ERR_MEASUREMENT_ARTIFACT": "Agent succeeded on in-process re-run; original failure was due to multiprocessing startup overhead starving the wall-clock time.",
    "ERR_HORIZON_BLIND": "Search depth too shallow to see the tactical refutation.",
    "ERR_COMPUTE_CHOKE": "Engine exhausted clock budget before reaching an informative depth.",
    "ERR_MATERIAL_OVERCOMPENSATION": "Agent overvalued material gain, blinding it to a forced mate or severe tactical loss.",
    "ERR_EVAL_UNCALIBRATED": "Static evaluation fundamentally misunderstands the positional reality.",
    "ERR_PASSIVITY_BIAS": "Agent avoids active tactical lines for passive shuffling."
}


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

    # Tagged "search_control" here so that filter now excludes them cleanly instead of crashing.
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
    depth = blunder.get("agent_depth", 3)
    time_used = blunder.get("time_used", blunder.get("agent_time_ms", 300))
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
        "code": "ERR_HORIZON_BLIND",
        "action": "scale_depth",
        "test_sequence": [depth + 2, depth + 4],
        "description": TAXONOMY["ERR_HORIZON_BLIND"],
        "confidence": depth_score
    })

    # 2. Compute Choke (Time Budget)
    time_score = 40
    if depth <= 3: 
        time_score += 45
        
    scored_hypotheses.append({
        "code": "ERR_COMPUTE_CHOKE",
        "action": "scale_time",
        "test_sequence": [600, 2000], 
        "description": TAXONOMY["ERR_COMPUTE_CHOKE"],
        "confidence": time_score
    })

    # 3. Static Distortions (Evaluator/Material Weights)
    eval_score = 30
    if neural_disagree > 2.0:
        eval_score += 50
    if max_eval_in_trace > 5.0:
        eval_score += 40
        
    eval_knobs = [k for k in available_knobs if k.get("source") == "uci_option"]
    if eval_knobs:
        scored_hypotheses.append({
            "code": "ERR_MATERIAL_OVERCOMPENSATION",
            "action": "perturb_options",
            "knobs": eval_knobs,
            "description": TAXONOMY["ERR_MATERIAL_OVERCOMPENSATION"],
            "confidence": eval_score
        })

    # Sort standard hypotheses by confidence
    scored_hypotheses.sort(key=lambda x: x["confidence"], reverse=True)

    # 0. ALWAYS insert the Control Re-run (Measurement Artifact) at the absolute top
    # If this passes, the blunder wasn't real—it was OS pool starvation.
    control_hypothesis = {
        "code": "ERR_MEASUREMENT_ARTIFACT",
        "action": "control_rerun",
        "params": {"max_depth": depth, "time_budget_ms": time_used},
        "description": TAXONOMY["ERR_MEASUREMENT_ARTIFACT"],
        "confidence": 1000  # Force to index 0
    }
    
    return [control_hypothesis] + scored_hypotheses