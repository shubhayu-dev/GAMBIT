"""
Intervention + held-out validation (docs/10 Week 7,
docs/09_experiment_protocol.md step 7: "validate on held-out set, once").

Implements the doc's own worked example: adaptive search depth based on
position complexity, via AdaptiveSearchAgent (engine/agents.py). The
diagnostic-set experiments in Week 6 inform *which* hypothesis to act on;
this module applies the resulting intervention and checks it on positions
that were never touched during diagnosis.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from agents import SearchAgent, AdaptiveSearchAgent  # noqa: E402
from runner import run_experiment  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from metrics import mean_abs_regret  # noqa: E402


def apply_intervention_and_validate(holdout_positions: List[Dict], base_depth: int = 3,
                                     boosted_depth: int = 5, complexity_threshold: int = 30,
                                     time_budget_ms: int = 300, seed: int = 500) -> Dict:
    """
    Runs the baseline agent (fixed depth) and the adaptive-depth intervention
    on the SAME held-out positions, once, and reports before/after. No
    re-tuning against these results -- if it doesn't generalize, that's a
    valid, reportable finding (docs/09_experiment_protocol.md step 7).
    """
    if len(holdout_positions) < 1:
        raise ValueError("Need at least 1 held-out position to validate")

    before = run_experiment(
        SearchAgent, {"max_depth": base_depth}, holdout_positions,
        condition="held_out_before", time_budget_ms=time_budget_ms, seed=seed, n_workers=4,
        benchmark_description="Week 7 held-out validation",
    )
    after = run_experiment(
        AdaptiveSearchAgent,
        {"base_depth": base_depth, "boosted_depth": boosted_depth, "complexity_threshold": complexity_threshold},
        holdout_positions, condition="held_out_after", time_budget_ms=time_budget_ms, seed=seed, n_workers=4,
        benchmark_description="Week 7 held-out validation",
    )

    mean_before = mean_abs_regret(before["records"])
    mean_after = mean_abs_regret(after["records"])
    improvement_pct = (round(100.0 * (1 - mean_after / mean_before), 1)
                        if mean_before and mean_before > 0 else None)

    return {
        "n_positions": len(holdout_positions),
        "before": {"agent": "SearchAgent (fixed depth)", "mean_abs_regret": round(mean_before, 3) if mean_before is not None else None},
        "after": {"agent": "AdaptiveSearchAgent (complexity-based depth)", "mean_abs_regret": round(mean_after, 3) if mean_after is not None else None},
        "improvement_pct": improvement_pct,
        "before_records": before["records"],
        "after_records": after["records"],
    }
