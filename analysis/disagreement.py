"""
Classical vs neural disagreement (docs/11_integration_addendum.md).

For each position, compares the classical evaluator's static score against
the neural value net's raw prediction, isolating evaluation disagreement.
It calculates the Pearson correlation and p-value to determine if the agent
makes worse decisions specifically where the two evaluators disagree most.
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from engine.environment import Board  # noqa: E402
from engine.evaluation import evaluate as classical_evaluate  # noqa: E402
from engine.neural_agent import NeuralAgent  # noqa: E402

try:
    from scipy.stats import pearsonr
    import numpy as np
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def compute_disagreement(records: List[Dict], neural_agent: NeuralAgent) -> List[Dict]:
    """Adds classical_value, neural_value, and disagreement (absolute
    difference, in pawns) to each record."""
    enriched = []
    for r in records:
        board = Board(r["position"])
        classical_value = classical_evaluate(board)
        try:
            neural_value = neural_agent.raw_value(board)
        except Exception:
            neural_value = 0.0  # Fallback if model prediction fails
            
        enriched.append({
            **r,
            "classical_value": round(classical_value, 3),
            "neural_value": round(neural_value, 3),
            "disagreement": round(abs(classical_value - neural_value), 3),
        })
    return enriched


def disagreement_regret_correlation(enriched_records: List[Dict]) -> Dict:
    """Pearson correlation between disagreement and |regret|.
    Returns correlation and p-value (if scipy is available) to verify significance."""
    # Use .get() to prevent crashes if a record is missing the regret key
    disagreements = [r.get("disagreement", 0.0) for r in enriched_records]
    abs_regrets = [abs(r.get("regret", 0.0)) for r in enriched_records]
    n = len(disagreements)
    
    if n < 3:
        return {"n": n, "correlation": None, "p_value": None, "note": "Too few positions"}

    if HAVE_SCIPY:
        # Check for zero variance to prevent scipy from throwing warnings/errors
        if np.std(disagreements) == 0 or np.std(abs_regrets) == 0:
            corr, p_val = 0.0, 1.0
        else:
            corr, p_val = pearsonr(disagreements, abs_regrets)
    else:
        # Manual fallback (No p-value)
        mean_d = sum(disagreements) / n
        mean_r = sum(abs_regrets) / n
        cov = sum((d - mean_d) * (r - mean_r) for d, r in zip(disagreements, abs_regrets)) / n
        std_d = (sum((d - mean_d) ** 2 for d in disagreements) / n) ** 0.5
        std_r = (sum((r - mean_r) ** 2 for r in abs_regrets) / n) ** 0.5
        corr = cov / (std_d * std_r) if std_d > 0 and std_r > 0 else 0.0
        p_val = None

    return {
        "n": n,
        "mean_disagreement": round(sum(disagreements) / n, 3),
        "mean_abs_regret": round(sum(abs_regrets) / n, 3),
        "correlation": round(corr, 3) if corr is not None else None,
        "p_value": round(p_val, 4) if p_val is not None else None
    }