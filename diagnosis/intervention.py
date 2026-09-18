"""
Validation layer for GAMBIT.
Extracts the dominant empirical fix from the proven ledger and tests it 
on a held-out dataset to verify generalization and prevent overfitting.
"""

from typing import Any, Dict, List, Tuple
from collections import Counter

# Reuse the universal agent query function from the active diagnostic loop
from diagnosis.diagnostic_experiment import _query_agent


def extract_dominant_fix(proven_ledger: List[Dict]) -> Tuple[str, Dict[str, Any]]:
    """
    Scans the proven ledger to find the most frequently successful parameter tweak.
    Returns a description of the fix and the exact configuration kwargs to run it.
    """
    successful_tests = []
    
    for record in proven_ledger:
        for log in record.get("investigation_log", []):
            if log.get("success"):
                successful_tests.append(log.get("test"))
                break  # Only count the first successful fix per position

    if not successful_tests:
        return "No generalized fix found", {}

    # Find the most common successful parameter injection
    dominant_test_string = Counter(successful_tests).most_common(1)[0][0]
    
    # Parse the string back into kwargs (e.g., "depth=6" -> {"max_depth": 6})
    intervention_kwargs = {}
    if "=" in dominant_test_string:
        param, val_str = dominant_test_string.split("=")
        
        # Type casting inference
        try:
            val = int(val_str.replace("ms", ""))
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str

        # Map to internal variables
        if param == "depth":
            intervention_kwargs["max_depth"] = val
        elif param == "time":
            intervention_kwargs["time_budget_ms"] = val
        else:
            intervention_kwargs[param] = val

    return dominant_test_string, intervention_kwargs


def evaluate_intervention(
    agent_target: Any, 
    held_out_positions: List[Dict[str, Any]], 
    baseline_kwargs: Dict[str, Any],
    intervention_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Runs the dominant fix on a blind, held-out dataset.
    Compares baseline performance against the intervened performance.
    """
    baseline_regret_sum = 0.0
    intervention_regret_sum = 0.0
    baseline_matches = 0
    intervention_matches = 0
    total = len(held_out_positions)

    if total == 0:
        return {}

    for pos in held_out_positions:
        fen = pos["position"]
        ref_move = pos["reference_move"]
        
        # Calculate baseline if it wasn't pre-computed in the dataset
        if "regret" in pos and pos.get("chosen_move") is not None:
            baseline_regret = pos["regret"]
            baseline_move = pos["chosen_move"]
        else:
            baseline_move = _query_agent(agent_target, fen, baseline_kwargs)
            # In a full system, you would query your oracle/evaluator here for exact regret.
            # For this validation, we use a binary match proxy if the oracle isn't live.
            baseline_regret = 0.0 if baseline_move == ref_move else 1.0 

        # Run the intervened agent
        treated_move = _query_agent(agent_target, fen, intervention_kwargs)
        treated_regret = 0.0 if treated_move == ref_move else 1.0

        baseline_regret_sum += baseline_regret
        intervention_regret_sum += treated_regret
        
        if baseline_move == ref_move:
            baseline_matches += 1
        if treated_move == ref_move:
            intervention_matches += 1

    mean_baseline_regret = baseline_regret_sum / total
    mean_intervention_regret = intervention_regret_sum / total
    
    regret_reduction_pct = 0.0
    if mean_baseline_regret > 0:
        regret_reduction_pct = ((mean_baseline_regret - mean_intervention_regret) / mean_baseline_regret) * 100

    return {
        "n_positions": total,
        "baseline_accuracy": (baseline_matches / total) * 100,
        "intervention_accuracy": (intervention_matches / total) * 100,
        "mean_baseline_regret": round(mean_baseline_regret, 3),
        "mean_intervention_regret": round(mean_intervention_regret, 3),
        "regret_reduction_pct": round(regret_reduction_pct, 2),
        "generalized": regret_reduction_pct > 15.0  # Threshold to prove it wasn't just noise
    }