"""
Decision-quality metrics (docs/08_metrics_spec.md), computed from trajectory
records produced by experiments/runner.py.
"""

import statistics
from typing import Dict, List, Optional

DEFAULT_REGRET_THRESHOLD = 0.5  # pawns; see docs/08 - tunable per phase, kept
                                  # uniform here for the MVP


def decision_accuracy(records: List[Dict], threshold: float = DEFAULT_REGRET_THRESHOLD) -> Optional[float]:
    """% of decisions with |regret| <= threshold."""
    if not records:
        return None
    hits = sum(1 for r in records if abs(r["regret"]) <= threshold)
    return 100.0 * hits / len(records)


def mean_regret(records: List[Dict]) -> Optional[float]:
    if not records:
        return None
    return sum(r["regret"] for r in records) / len(records)


def mean_abs_regret(records: List[Dict]) -> Optional[float]:
    if not records:
        return None
    return sum(abs(r["regret"]) for r in records) / len(records)


def regret_stdev(records: List[Dict]) -> float:
    vals = [r["regret"] for r in records]
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def mean_time_used(records: List[Dict]) -> Optional[float]:
    if not records:
        return None
    return sum(r["time_used"] for r in records) / len(records)
