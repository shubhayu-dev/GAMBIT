"""
Diagnostic experiments (docs/10 Week 6, docs/09_experiment_protocol.md).

Runs the candidate hypotheses as controlled experiments on the weak-bucket 
positions dynamically, reading from the experiment_schema.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from engine.agents import SearchAgent  # noqa: E402
from engine.evaluation import evaluate, evaluate_material_only  # noqa: E402
from experiments.runner import run_experiment  # noqa: E402

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def _paired_comparison(records_a: List[Dict], records_b: List[Dict], label_a: str, label_b: str) -> Dict:
    """Compares |regret| between two conditions run on the same positions."""
    abs_a = [abs(r["regret"]) for r in records_a]
    abs_b = [abs(r["regret"]) for r in records_b]
    mean_a = sum(abs_a) / len(abs_a) if abs_a else None
    mean_b = sum(abs_b) / len(abs_b) if abs_b else None

    p_value = None
    if HAVE_SCIPY and len(abs_a) >= 6 and any(a != b for a, b in zip(abs_a, abs_b)):
        try:
            _, p_value = wilcoxon(abs_a, abs_b)
        except ValueError:
            p_value = None

    improvement_pct = None
    if mean_a and mean_a > 0 and mean_b is not None:
        improvement_pct = round(100.0 * (1 - mean_b / mean_a), 1)

    return {
        "label_a": label_a, "mean_abs_regret_a": round(mean_a, 3) if mean_a is not None else None,
        "label_b": label_b, "mean_abs_regret_b": round(mean_b, 3) if mean_b is not None else None,
        "b_improves_on_a_by_pct": improvement_pct,
        "wilcoxon_p_value": round(p_value, 4) if p_value is not None else None,
        "n": len(abs_a),
    }


def run_diagnostic_suite(weak_positions: List[Dict], hypotheses: List[Dict], time_budget_ms: int = 300, seed: int = 100) -> Dict:
    """Dynamically executes A/B tests based on the experiment_schema of each hypothesis."""
    if len(weak_positions) < 2:
        raise ValueError("Need at least 2 weak-bucket positions to run diagnostic experiments")

    evidence = {}

    for hyp in hypotheses:
        hyp_id = hyp["id"]
        schema = hyp.get("experiment_schema")
        
        if not schema:
            continue
            
        variable = schema["variable"]
        control_val = schema["control"]
        treatment_val = schema["treatment"]
        
        # Helper to construct agent and runner parameters dynamically
        def build_kwargs(val):
            agent_kwargs = {"max_depth": 3, "eval_fn": evaluate} # Defaults
            run_kwargs = {"time_budget_ms": time_budget_ms, "seed": seed, "n_workers": 4}
            
            if variable == "max_depth":
                agent_kwargs["max_depth"] = val
            elif variable == "evaluator":
                agent_kwargs["eval_fn"] = evaluate_material_only if val == "material" else evaluate
            elif variable == "time_budget_ms":
                run_kwargs["time_budget_ms"] = val
                agent_kwargs["max_depth"] = 10 # Unlock depth to test pure time limits
                
            return agent_kwargs, run_kwargs

        control_agent_kw, control_run_kw = build_kwargs(control_val)
        treatment_agent_kw, treatment_run_kw = build_kwargs(treatment_val)

        # Run Control
        control_res = run_experiment(
            SearchAgent, control_agent_kw, weak_positions, 
            condition=f"{hyp_id}_control", 
            benchmark_description=f"Diagnostic {hyp_id} Control", **control_run_kw
        )
        
        # Run Treatment
        treatment_res = run_experiment(
            SearchAgent, treatment_agent_kw, weak_positions, 
            condition=f"{hyp_id}_treatment", 
            benchmark_description=f"Diagnostic {hyp_id} Treatment", **treatment_run_kw
        )

        label_a = f"{variable}={control_val}"
        label_b = f"{variable}={treatment_val}"
        
        # Store evidence using the hypothesis ID directly as the key
        evidence[hyp_id] = _paired_comparison(control_res["records"], treatment_res["records"], label_a, label_b)

    return evidence


def rank_hypotheses(evidence: Dict) -> List[Dict]:
    """Ranks hypotheses by improvement magnitude (the 'better' condition vs the 'worse' one)."""
    ranked = []
    for hyp_id, result in evidence.items():
        improvement = result["b_improves_on_a_by_pct"] or 0
        ranked.append({"hypothesis": hyp_id, "improvement_pct": improvement,
                        "p_value": result["wilcoxon_p_value"], **result})
    ranked.sort(key=lambda r: r["improvement_pct"], reverse=True)
    return ranked