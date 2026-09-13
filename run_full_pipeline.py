import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Ensure all internal project modules resolve cleanly
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "diagnosis"))
sys.path.insert(0, str(ROOT / "dashboard"))

from experiments.lichess_data import load_lichess_positions, RAW_ZST_PATH
from engine.agents import SearchAgent
from experiments.benchmark import diagnostic_holdout_split
from experiments.runner import run_experiment
from analysis.profile import build_profile, tag_records
from analysis.report import format_profile_report
from diagnosis.failure_detection import detect_primary_weakness, describe_weakness
from diagnosis.hypotheses import generate_hypotheses
from diagnosis.diagnostic_experiment import run_diagnostic_suite, rank_hypotheses
from diagnosis.intervention import apply_intervention_and_validate
from dashboard.report_html import build_report_html

DEVIATIONS = [
    "No network access in this sandbox: Stockfish and python-chess could not be installed "
    "(Week 1) — the chess environment and reference evaluator were built from scratch, "
    "verified via perft.",
    "Trajectory data is stored as JSONL, not Parquet.",
    "The dashboard is static HTML, not a live Streamlit app.",
    "Evaluations use curated Lichess puzzle positions with 20-pawn clipped regret against "
    "ground-truth reference moves.",
]


def main():
    print("=" * 60)
    print("GAMBIT — full pipeline run (Weeks 4-8)")
    print("=" * 60)

    # --- Week 4 setup: benchmark + baseline trajectory dataset ---
    print("\n[1/6] Loading Lichess benchmark + diagnostic/held-out split...")
    positions, _ = load_lichess_positions(path=RAW_ZST_PATH, max_positions=150, validate=True)

    # Map 'phase' explicitly to resolve KeyError in benchmark.diagnostic_holdout_split
    for p in positions:
        p["phase"] = p.get("lichess_phase") or p.get("internal_phase") or "unknown"

    diagnostic, holdout = diagnostic_holdout_split(positions, holdout_fraction=0.3, seed=2)
    print(f"  {len(positions)} positions -> {len(diagnostic)} diagnostic / {len(holdout)} held-out")

    print("\n[2/6] Running baseline agent over the diagnostic set (Week 4)...")
    agent_name = "SearchAgent(max_depth=3)"
    baseline = run_experiment(
        SearchAgent,
        {"max_depth": 3},
        diagnostic,
        condition="baseline_profile",
        time_budget_ms=300,
        reference_depth=2,
        seed=10,
        n_workers=4,
        benchmark_description="Week 4-8 pipeline: diagnostic set",
    )
    profile = build_profile(baseline["records"])
    print(format_profile_report(agent_name, profile))

    # --- Week 5: failure detection + hypotheses ---
    print("\n[3/6] Detecting primary weakness + generating hypotheses (Week 5)...")
    weak_dim, weak_stats = detect_primary_weakness(profile)
    weakness_desc = describe_weakness(weak_dim, weak_stats)
    print(f"  {weakness_desc}")
    hyps = generate_hypotheses(weak_dim)
    for h in hyps:
        print(f"  {h['id']}: {h['name']} — {h['statement']}")

    # --- Week 6: diagnostic experiments ---
    print("\n[4/6] Running diagnostic experiments for H1/H2/H3 (Week 6)...")
    tagged = tag_records(baseline["records"])
    weak_records = {
        "opening": [r for r in tagged if r["game_phase"] == "opening"],
        "middlegame": [r for r in tagged if r["game_phase"] == "middlegame"],
        "endgame": [r for r in tagged if r["game_phase"] == "endgame"],
        "tactical_proxy": [r for r in tagged if r["tactical_bucket"] == "high_complexity"],
        "positional_proxy": [r for r in tagged if r["tactical_bucket"] == "low_complexity"],
        "defensive_proxy": [r for r in tagged if r["is_defensive"]],
    }.get(weak_dim, tagged)

    # Preserve reference moves and metadata so diagnostic experiments evaluate against Lichess
    weak_positions = [
        {
            "fen": r["position"],
            "phase": r["game_phase"],
            "reference_move_lichess": r.get("reference_move"),
            "puzzle_id": r.get("puzzle_id"),
            "game_id": r.get("game_id"),
        }
        for r in weak_records
    ]
    if len(weak_positions) < 2:
        print(f"  Weak bucket too small ({len(weak_positions)} positions) — falling back to full diagnostic set")
        weak_positions = diagnostic

    evidence = run_diagnostic_suite(weak_positions, time_budget_ms=300, seed=100)
    ranked = rank_hypotheses(evidence)
    for r in ranked:
        print(
            f"  {r['hypothesis']}: {r['label_a']} -> {r['label_b']}, "
            f"regret {r['mean_abs_regret_a']} -> {r['mean_abs_regret_b']} "
            f"({r['improvement_pct']}% improvement, p={r['p_value']})"
        )

    # --- Week 7: intervention + held-out validation ---
    print("\n[5/6] Applying intervention + validating on held-out set (Week 7)...")
    intervention_result = apply_intervention_and_validate(
        holdout,
        base_depth=3,
        boosted_depth=5,
        complexity_threshold=30,
        time_budget_ms=300,
        seed=500,
    )
    print(f"  Before (fixed depth):    mean |regret| = {intervention_result['before']['mean_abs_regret']}")
    print(f"  After (adaptive depth):  mean |regret| = {intervention_result['after']['mean_abs_regret']}")
    print(
        f"  Improvement: {intervention_result['improvement_pct']}% on "
        f"{intervention_result['n_positions']} held-out positions"
    )

    # --- Week 8: report ---
    print("\n[6/6] Building final report (Week 8)...")
    html = build_report_html(
        agent_name, profile, weak_dim, weakness_desc, ranked, intervention_result, DEVIATIONS
    )
    out_path = ROOT / "data" / "gambit_report.html"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(html)
    print(f"  Report written to {out_path}")

    summary_path = ROOT / "data" / "pipeline_summary.json"
    summary = {
        "agent": agent_name,
        "n_diagnostic": len(diagnostic),
        "n_holdout": len(holdout),
        "profile": profile,
        "weak_dimension": weak_dim,
        "hypotheses_ranked": ranked,
        "intervention": {k: v for k, v in intervention_result.items() if not k.endswith("_records")},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary JSON written to {summary_path}")

    print("\n" + "=" * 60)
    print("Full OBSERVE -> EVALUATE -> DIAGNOSE -> EXPERIMENT -> INTERVENE -> VALIDATE loop complete.")
    print("=" * 60)
    return out_path


if __name__ == "__main__":
    main()