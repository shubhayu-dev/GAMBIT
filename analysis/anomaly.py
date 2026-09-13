"""
Anomaly/pattern detection (docs/11_integration_addendum.md, section "12.
Machine Learning for Behavioral Analysis" in the pasted integration spec).

Uses IsolationForest over a small feature set derived from trajectory
records (regret, time used, position complexity, classical/neural
disagreement) to flag unusual decisions -- not necessarily bad ones, just
statistically unusual relative to the rest of the dataset. This is a
genuine unsupervised ML technique, distinct from both the hand-coded
evaluation and the supervised neural value net.
"""

from typing import Dict, List

try:
    from sklearn.ensemble import IsolationForest
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

GAME_PHASE_CODE = {"opening": 0, "middlegame": 1, "endgame": 2}


def _feature_row(r: Dict) -> List[float]:
    return [
        r.get("regret", 0.0),
        r.get("time_used", 0.0),
        r.get("complexity", 0.0),
        r.get("disagreement", 0.0),
        GAME_PHASE_CODE.get(r.get("game_phase"), -1),
    ]


def detect_anomalies(enriched_records: List[Dict], contamination: float = 0.15, seed: int = 0) -> List[Dict]:
    """
    Returns the input records with an added `is_anomaly` bool and
    `anomaly_score` (lower = more anomalous, per IsolationForest's
    convention) for each. Records need `regret`, `time_used`, `complexity`,
    `disagreement`, and `game_phase` -- run analysis/profile.py's
    tag_records() and analysis/disagreement.py's compute_disagreement()
    first to populate `complexity` and `disagreement`.
    """
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required for anomaly detection")
    if len(enriched_records) < 10:
        # IsolationForest needs a reasonable sample size to mean anything;
        # below that, don't pretend to detect anomalies.
        return [{**r, "is_anomaly": False, "anomaly_score": None} for r in enriched_records]

    X = [_feature_row(r) for r in enriched_records]
    model = IsolationForest(contamination=contamination, random_state=seed)
    model.fit(X)
    predictions = model.predict(X)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)

    return [
        {**r, "is_anomaly": bool(pred == -1), "anomaly_score": round(float(score), 4)}
        for r, pred, score in zip(enriched_records, predictions, scores)
    ]


def summarize_anomalies(anomaly_records: List[Dict]) -> Dict:
    anomalies = [r for r in anomaly_records if r.get("is_anomaly")]
    if not anomalies:
        return {"n_anomalies": 0, "n_total": len(anomaly_records), "common_phase": None, "mean_regret_anomalous": None}

    phase_counts: Dict[str, int] = {}
    for r in anomalies:
        phase_counts[r.get("game_phase", "unknown")] = phase_counts.get(r.get("game_phase", "unknown"), 0) + 1
    common_phase = max(phase_counts, key=phase_counts.get) if phase_counts else None

    return {
        "n_anomalies": len(anomalies),
        "n_total": len(anomaly_records),
        "anomaly_rate_pct": round(100.0 * len(anomalies) / len(anomaly_records), 1),
        "common_phase": common_phase,
        "phase_breakdown": phase_counts,
        "mean_regret_anomalous": round(sum(abs(r["regret"]) for r in anomalies) / len(anomalies), 3),
        "mean_regret_normal": round(
            sum(abs(r["regret"]) for r in anomaly_records if not r.get("is_anomaly")) /
            max(1, len(anomaly_records) - len(anomalies)), 3
        ) if len(anomaly_records) > len(anomalies) else None,
    }
