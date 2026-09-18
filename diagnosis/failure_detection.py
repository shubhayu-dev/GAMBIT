"""
Failure detection and blunder isolation for GAMBIT.
Identifies weak capability dimensions and isolates actionable blunder records.
"""

from typing import Dict, List, Optional, Tuple


def detect_primary_weakness(profile: Dict) -> Tuple[str, Dict]:
    """
    Scans the capability profile to identify the dimension with the lowest
    decision accuracy and highest regret.
    """
    candidate_dims = [
        "tactical_proxy",
        "positional_proxy",
        "defensive_proxy",
        "middlegame",
        "endgame",
        "opening",
    ]
    worst_dim = "tactical_proxy"
    min_accuracy = 101.0
    max_regret = -1.0

    for dim in candidate_dims:
        stats = profile.get(dim)
        if not stats or stats.get("n", 0) < 5:
            continue
        acc = stats.get("decision_accuracy_pct", 100.0)
        regret = stats.get("mean_regret", 0.0)

        if (acc < min_accuracy) or (acc == min_accuracy and regret > max_regret):
            min_accuracy = acc
            max_regret = regret
            worst_dim = dim

    return worst_dim, profile.get(worst_dim, {})


def describe_weakness(weak_dim: str, weak_stats: Dict) -> str:
    """Generates a human-readable summary of the detected failure dimension."""
    acc = weak_stats.get("decision_accuracy_pct", 0.0)
    n = weak_stats.get("n", 0)
    regret = weak_stats.get("mean_regret", 0.0)
    dim_label = weak_dim.replace("_proxy", "").replace("_", " ").title()
    return (
        f"Primary weakness: {dim_label} positions. Decision accuracy {acc:.1f}% "
        f"over {n} positions (mean regret {regret:.3f}), versus other dimensions."
    )


def extract_blunders(
    records: List[Dict],
    top_k: int = 5,
    min_regret: float = 1.0,
    disagreement_records: Optional[List[Dict]] = None
) -> List[Dict]:
    """
    Filters and extracts the most severe decision errors.
    Merges neural probe evaluation disagreement if available.
    """
    disagree_map = {}
    if disagreement_records:
        for r in disagreement_records:
            disagree_map[r.get("position")] = r.get("disagreement_pawns")

    blunder_list = []
    for r in records:
        regret = r.get("regret", 0.0)
        if regret >= min_regret and not r.get("decision_match", False):
            fen = r.get("position")
            blunder_list.append({
                "puzzle_id": r.get("puzzle_id"),
                "game_id": r.get("game_id"),
                "position": fen,
                "chosen_move": r.get("chosen_move"),
                "reference_move": r.get("reference_move"),
                "regret": round(regret, 3),
                "agent_depth": r.get("agent_depth_reached") or r.get("agent_depth", 0),
                "agent_pv_line": r.get("agent_pv_line", ""),
                "agent_pv_trace": r.get("agent_pv_trace", []),
                "reference_pv_line": r.get("reference_pv_line", ""),
                "neural_disagreement": disagree_map.get(fen, 0.0),
            })

    blunder_list.sort(key=lambda x: x["regret"], reverse=True)
    return blunder_list[:top_k]