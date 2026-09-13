"""
Diagnostic experiments (docs/10 Week 6, docs/09_experiment_protocol.md).

Runs the three hypotheses from hypotheses.py as controlled experiments on the
weak-bucket positions only (never the held-out set -- that's reserved for
Week 7's final validation). Each hypothesis becomes a paired A/B comparison:
same positions, one factor changed, everything else held constant.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from agents import SearchAgent  # noqa: E402
from evaluation import evaluate, evaluate_material_only  # noqa: E402
from runner import run_experiment  # noqa: E402

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


def run_diagnostic_suite(weak_positions: List[Dict], time_budget_ms: int = 300, seed: int = 100) -> Dict:
    """Runs H1 (depth), H2 (evaluation quality), H3 (time budget) on the
    given weak-bucket positions and returns evidence for each."""
    if len(weak_positions) < 2:
        raise ValueError("Need at least 2 weak-bucket positions to run diagnostic experiments")

    evidence = {}

    # H1 - search depth
    shallow = run_experiment(SearchAgent, {"max_depth": 2}, weak_positions, condition="H1_depth_2",
                              time_budget_ms=time_budget_ms, seed=seed, n_workers=4,
                              benchmark_description="Week 6 diagnostic: weak bucket")
    deep = run_experiment(SearchAgent, {"max_depth": 4}, weak_positions, condition="H1_depth_4",
                           time_budget_ms=time_budget_ms, seed=seed, n_workers=4,
                           benchmark_description="Week 6 diagnostic: weak bucket")
    evidence["H1_search_depth"] = _paired_comparison(shallow["records"], deep["records"], "depth=2", "depth=4")

    # H2 - evaluation quality (ablation)
    material_only = run_experiment(SearchAgent, {"max_depth": 3, "eval_fn": evaluate_material_only},
                                    weak_positions, condition="H2_eval_material_only",
                                    time_budget_ms=time_budget_ms, seed=seed, n_workers=4,
                                    benchmark_description="Week 6 diagnostic: weak bucket")
    full_eval = run_experiment(SearchAgent, {"max_depth": 3, "eval_fn": evaluate},
                                weak_positions, condition="H2_eval_full",
                                time_budget_ms=time_budget_ms, seed=seed, n_workers=4,
                                benchmark_description="Week 6 diagnostic: weak bucket")
    evidence["H2_evaluation_quality"] = _paired_comparison(
        material_only["records"], full_eval["records"], "material-only eval", "material+PST eval"
    )

    # H3 - time budget
    short_budget = run_experiment(SearchAgent, {"max_depth": 4}, weak_positions, condition="H3_time_100ms",
                                   time_budget_ms=100, seed=seed, n_workers=4,
                                   benchmark_description="Week 6 diagnostic: weak bucket")
    long_budget = run_experiment(SearchAgent, {"max_depth": 4}, weak_positions, condition="H3_time_600ms",
                                  time_budget_ms=600, seed=seed, n_workers=4,
                                  benchmark_description="Week 6 diagnostic: weak bucket")
    evidence["H3_time_budget"] = _paired_comparison(short_budget["records"], long_budget["records"],
                                                     "100ms budget", "600ms budget")

    return evidence


def rank_hypotheses(evidence: Dict) -> List[Dict]:
    """Ranks hypotheses by improvement magnitude (the 'better' condition vs
    the 'worse' one). Larger, more significant improvement = stronger support
    for that hypothesis being the dominant cause."""
    ranked = []
    for hyp_id, result in evidence.items():
        improvement = result["b_improves_on_a_by_pct"] or 0
        ranked.append({"hypothesis": hyp_id, "improvement_pct": improvement,
                        "p_value": result["wilcoxon_p_value"], **result})
    ranked.sort(key=lambda r: r["improvement_pct"], reverse=True)
    return ranked
