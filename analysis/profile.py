"""
Behavioral capability profile (docs/04 Layer "Behavioral Analysis",
docs/08_metrics_spec.md "Behavioral profile dimensions").

Deviation, flagged per Constitution Rule 7: the spec's tactical/positional/
defensive/endgame categories are meant to come from a curated, human-labeled
benchmark (see docs/07). Without network access to real Lichess puzzle data,
this module derives proxy categories directly from position features instead:

- opening / middlegame / endgame: from Board.game_phase() (unchanged, exact)
- tactical / positional: proxied by branching factor (legal move count) --
  high-branching positions tend to have more tactical candidate moves,
  low-branching positions tend to be quieter/positional. This is a coarse
  proxy, not a real tag.
- defensive: proxied by whether the position is worse for the side to move
  (their own reference-engine evaluation is negative from their perspective)
  -- i.e. they're defending, matching docs/07's definition of "inferior
  positions, avoiding tactical collapse" at the position level rather than a
  curated puzzle label.

Swap these proxies for real curated tags once a categorized benchmark is
available -- nothing downstream (diagnosis, experiments, intervention) reads
the tags directly; they only consume the profile dict this module produces.
"""

import statistics
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from environment import Board  # noqa: E402

from metrics import decision_accuracy, mean_regret, mean_abs_regret, mean_time_used  # noqa: E402


def _position_complexity(fen: str) -> int:
    return len(Board(fen).legal_moves())


def _side_to_move(fen: str) -> str:
    return fen.split()[1]


def tag_records(records: List[Dict]) -> List[Dict]:
    """Adds complexity, tactical/positional bucket, and defensive-bucket tags
    to each record (does not mutate the input)."""
    complexities = [_position_complexity(r["position"]) for r in records]
    median_complexity = statistics.median(complexities) if complexities else 0

    tagged = []
    for r, complexity in zip(records, complexities):
        turn = _side_to_move(r["position"])
        perspective_eval = r["evaluation_before"] if turn == "w" else -r["evaluation_before"]
        tagged.append({
            **r,
            "complexity": complexity,
            "tactical_bucket": "high_complexity" if complexity >= median_complexity else "low_complexity",
            "is_defensive": perspective_eval < 0,
        })
    return tagged


def _bucket_stats(records: List[Dict], threshold: float) -> Dict:
    return {
        "n": len(records),
        "decision_accuracy_pct": round(decision_accuracy(records, threshold), 1) if records else None,
        "mean_regret": round(mean_regret(records), 3) if records else None,
        "mean_abs_regret": round(mean_abs_regret(records), 3) if records else None,
    }


def build_profile(records: List[Dict], threshold: float = 0.5) -> Dict:
    """Builds the multidimensional capability profile from a trajectory dataset."""
    tagged = tag_records(records)

    profile = {
        "overall": _bucket_stats(tagged, threshold),
        "opening": _bucket_stats([r for r in tagged if r["game_phase"] == "opening"], threshold),
        "middlegame": _bucket_stats([r for r in tagged if r["game_phase"] == "middlegame"], threshold),
        "endgame": _bucket_stats([r for r in tagged if r["game_phase"] == "endgame"], threshold),
        "tactical_proxy": _bucket_stats([r for r in tagged if r["tactical_bucket"] == "high_complexity"], threshold),
        "positional_proxy": _bucket_stats([r for r in tagged if r["tactical_bucket"] == "low_complexity"], threshold),
        "defensive_proxy": _bucket_stats([r for r in tagged if r["is_defensive"]], threshold),
        "consistency": {
            "regret_stdev": round(statistics.pstdev([r["regret"] for r in tagged]), 3) if len(tagged) > 1 else 0.0,
        },
        "efficiency": {
            "mean_time_used_s": round(mean_time_used(tagged), 4) if tagged else None,
        },
    }
    return profile


def weakest_dimension(profile: Dict, dimensions=("opening", "middlegame", "endgame",
                                                   "tactical_proxy", "positional_proxy", "defensive_proxy")) -> str:
    """Returns the profile dimension with the lowest decision accuracy among
    those with enough samples to be meaningful (n >= 2)."""
    candidates = {
        name: profile[name]["decision_accuracy_pct"]
        for name in dimensions
        if profile[name]["n"] >= 2 and profile[name]["decision_accuracy_pct"] is not None
    }
    if not candidates:
        return "overall"
    return min(candidates, key=candidates.get)
