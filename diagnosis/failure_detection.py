"""
Failure detection (docs/10 Week 5 goal: "Failure patterns -> hypotheses").
Thin wrapper over analysis/profile.py's weakest_dimension -- kept as its own
module because diagnosis/ owns "what's the primary weakness", while
analysis/ owns "here are the numbers".
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))
from profile import weakest_dimension  # noqa: E402

DIMENSION_LABELS = {
    "opening": "Opening play",
    "middlegame": "Middlegame play",
    "endgame": "Endgame play",
    "tactical_proxy": "Tactical (high-complexity) positions",
    "positional_proxy": "Positional (low-complexity) positions",
    "defensive_proxy": "Defensive (inferior) positions",
}


def detect_primary_weakness(profile: Dict) -> Tuple[str, Dict]:
    """Returns (dimension_key, dimension_stats) for the weakest measured
    dimension with enough samples to be meaningful."""
    dim = weakest_dimension(profile)
    return dim, profile[dim]


def describe_weakness(dim: str, stats: Dict) -> str:
    label = DIMENSION_LABELS.get(dim, dim)
    return (f"Primary weakness: {label}. Decision accuracy "
            f"{stats['decision_accuracy_pct']}% over {stats['n']} positions "
            f"(mean regret {stats['mean_regret']}), versus other dimensions.")
