"""
Classical vs neural disagreement (docs/11_integration_addendum.md, section
"10. Classical vs Neural Disagreement" in the pasted integration spec).

For each position, compares the classical evaluator's static score against
the neural value net's raw prediction (no search on either side -- this
isolates evaluation disagreement from search-depth differences) and looks
for correlation with decision regret: does the agent make worse decisions
specifically where the two evaluators disagree most?
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from environment import Board  # noqa: E402
from evaluation import evaluate as classical_evaluate  # noqa: E402
from neural_agent import NeuralAgent  # noqa: E402


def compute_disagreement(records: List[Dict], neural_agent: NeuralAgent) -> List[Dict]:
    """Adds classical_value, neural_value, and disagreement (absolute
    difference, in pawns) to each record, keyed off its `position` FEN."""
    enriched = []
    for r in records:
        board = Board(r["position"])
        classical_value = classical_evaluate(board)
        neural_value = neural_agent.raw_value(board)
        enriched.append({
            **r,
            "classical_value": round(classical_value, 3),
            "neural_value": round(neural_value, 3),
            "disagreement": round(abs(classical_value - neural_value), 3),
        })
    return enriched


def disagreement_regret_correlation(enriched_records: List[Dict]) -> Dict:
    """Pearson correlation between disagreement and |regret| -- a positive,
    non-trivial correlation would support using disagreement as an
    independent diagnostic signal, per the integration spec."""
    disagreements = [r["disagreement"] for r in enriched_records]
    abs_regrets = [abs(r["regret"]) for r in enriched_records]
    n = len(disagreements)
    if n < 3:
        return {"n": n, "correlation": None, "note": "too few positions for a meaningful correlation"}

    mean_d = sum(disagreements) / n
    mean_r = sum(abs_regrets) / n
    cov = sum((d - mean_d) * (r - mean_r) for d, r in zip(disagreements, abs_regrets)) / n
    std_d = (sum((d - mean_d) ** 2 for d in disagreements) / n) ** 0.5
    std_r = (sum((r - mean_r) ** 2 for r in abs_regrets) / n) ** 0.5
    correlation = cov / (std_d * std_r) if std_d > 0 and std_r > 0 else None

    return {
        "n": n,
        "mean_disagreement": round(mean_d, 3),
        "mean_abs_regret": round(mean_r, 3),
        "correlation": round(correlation, 3) if correlation is not None else None,
    }
