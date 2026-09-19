"""
Validation layer for GAMBIT.
Extracts the dominant empirical fix from the proven ledger and tests it
on a held-out dataset to verify generalization and prevent overfitting.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter

# Reuse the universal agent query function from the active diagnostic loop
from diagnosis.diagnostic_experiment import _query_agent


def extract_dominant_fix(proven_ledger: List[Dict]) -> Tuple[str, Dict[str, Any]]:
    """
    Scans the proven ledger to find the most frequently successful parameter tweak.
    Returns a description of the fix and the exact configuration kwargs to run it.

    Note: this reports frequency of single-position matches across the
    ledger, not statistical significance -- it tells you which knob is
    worth testing on held-out data next (evaluate_intervention below), not
    that it's already confirmed to work.
    """
    successful_tests = []

    for record in proven_ledger:
        for log in record.get("investigation_log", []):
            if log.get("success"):
                successful_tests.append(log.get("test"))
                break  # Only count the first successful fix per position

    if not successful_tests:
        return "No candidate fix found in the diagnostic set", {}

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
    intervention_kwargs: Dict[str, Any],
    oracle_regret_fn: Optional[Callable[[str, str], float]] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Runs the dominant fix on a blind, held-out dataset.
    Compares baseline performance against the intervened performance.

    FIX (see chat writeup): the previous version compared a precomputed,
    continuous `pos["regret"]` for the baseline against a binary 0/1
    match-proxy for the intervention whenever no oracle was supplied --
    two different scales fed into the same "regret_reduction_pct"
    subtraction, which is not a meaningful number. This version always
    scores BOTH sides with the same metric: real regret via
    `oracle_regret_fn(fen, move) -> float` if you pass one, otherwise the
    binary match-proxy for both. Which metric was actually used is now
    reported explicitly in the result instead of left implicit.
    """
    if not held_out_positions:
        return {}

    def score(fen: str, move: Optional[str], ref_move: str) -> float:
        if oracle_regret_fn is not None:
            return oracle_regret_fn(fen, move) if move is not None else float("inf")
        return 0.0 if move == ref_move else 1.0

    metric_label = "oracle_regret" if oracle_regret_fn is not None else "binary_match_proxy"

    baseline_regret_sum = 0.0
    intervention_regret_sum = 0.0
    baseline_matches = 0
    intervention_matches = 0
    total = len(held_out_positions)

    for pos in held_out_positions:
        fen = pos["position"]
        ref_move = pos["reference_move"]

        baseline_move = _query_agent(agent_target, fen, baseline_kwargs, seed=seed)
        treated_move = _query_agent(agent_target, fen, intervention_kwargs, seed=seed)

        baseline_regret = score(fen, baseline_move, ref_move)
        treated_regret = score(fen, treated_move, ref_move)

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
        "regret_metric": metric_label,
        "baseline_accuracy": (baseline_matches / total) * 100,
        "intervention_accuracy": (intervention_matches / total) * 100,
        "mean_baseline_regret": round(mean_baseline_regret, 3),
        "mean_intervention_regret": round(mean_intervention_regret, 3),
        "regret_reduction_pct": round(regret_reduction_pct, 2),
        # Threshold is a heuristic sanity check, not a significance test --
        # there's still no p-value here, on purpose: don't report one we
        # didn't compute (see llm_reasoner.py).
        "generalized": regret_reduction_pct > 15.0,
    }