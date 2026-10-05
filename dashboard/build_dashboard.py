"""
Rebuilds data/gambit_report.html from already-saved pipeline outputs
(data/pipeline_summary.json + the most recent data/trajectories_*.jsonl),
without re-running the full pipeline (which needs the raw Lichess file).

Run from the repo root:
    python3 dashboard/build_dashboard.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from report_html import build_report_html  # noqa: E402

DATA_DIR = ROOT / "data"
N_BLUNDER_EXAMPLES = 5


def _latest_trajectory_file():
    candidates = sorted(DATA_DIR.glob("trajectories_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        return None
    return candidates[0]


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    summary_path = DATA_DIR / "pipeline_summary.json"
    if not summary_path.exists():
        print(f"No {summary_path} found -- run run_full_pipeline.py first.")
        return

    summary = json.loads(summary_path.read_text())

    traj_path = _latest_trajectory_file()
    blunder_examples = []
    if traj_path:
        records = _load_jsonl(traj_path)
        # Worst mistakes first, by regret.
        worst = sorted(records, key=lambda r: -r.get("regret", 0.0))[:N_BLUNDER_EXAMPLES]
        # Keep one clean "correct move" example too, so the report isn't
        # all failures -- the first decent-complexity position the agent
        # got right.
        correct = next((r for r in records if r.get("decision_match")), None)
        blunder_examples = worst + ([correct] if correct and correct not in worst else [])
        print(f"  Loaded {len(records)} trajectory records from {traj_path.name}")
        print(f"  Selected {len(worst)} worst blunders + "
              f"{'1 correct example' if correct else '0 correct examples'} for the board visuals.")
    else:
        print("  No trajectories_*.jsonl found in data/ -- report will have no board visuals.")

    deviations = [
        "Evaluations use curated Lichess puzzle positions with 20-pawn clipped regret against "
        "ground-truth reference moves.",
        "Trajectory data is stored as JSONL.",
        "Dashboard is static HTML with inline SVG board diagrams (no external assets, no JS).",
        "Rebuilt from saved pipeline_summary.json + trajectories JSONL rather than a fresh "
        "pipeline run (raw Lichess source file not present in this environment).",
    ]

    html = build_report_html(
        agent_name=summary["agent"],
        profile=summary["profile"],
        weakness_dim=summary["weak_dimension"],
        weakness_desc=_weakness_desc(summary),
        ranked_hypotheses=summary.get("hypotheses_ranked", []),
        intervention_result=summary.get("intervention", {}),
        deviations=deviations,
        llm_diagnosis=summary.get("llm_diagnosis") or None,
        blunder_examples=blunder_examples,
    )

    out_path = DATA_DIR / "gambit_report.html"
    out_path.write_text(html)
    print(f"  Report written to {out_path}")


def _weakness_desc(summary):
    """pipeline_summary.json doesn't store the human-readable weakness
    sentence (run_full_pipeline.py builds it inline and only stores the
    dimension name) -- reconstruct an equivalent one from the profile."""
    dim = summary["weak_dimension"]
    stats = summary["profile"].get(dim, {})
    acc = stats.get("decision_accuracy_pct")
    n = stats.get("n")
    regret = stats.get("mean_abs_regret")
    label = dim.replace("_proxy", "").replace("_", " ")
    if acc is None or n is None:
        return f"Weakest measured dimension: {label} (insufficient data)."
    return (
        f"The weakest measured dimension is <strong>{label}</strong>: only {acc:.0f}% decision "
        f"accuracy across {n} positions, with an average mistake size of {regret:.2f} pawns when it "
        f"does go wrong — the largest gap of any capability bucket tested."
    )


if __name__ == "__main__":
    main()
