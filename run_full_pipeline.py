import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "diagnosis"))
sys.path.insert(0, str(ROOT / "dashboard"))

from engine.neural_agent import NeuralAgent
from analysis.disagreement import compute_disagreement, disagreement_regret_correlation
from experiments.lichess_data import load_lichess_positions, RAW_ZST_PATH
from engine.agents import SearchAgent
from experiments.benchmark import diagnostic_holdout_split
from experiments.runner import run_experiment
from analysis.profile import build_profile, tag_records
from analysis.anomaly import detect_anomalies
from analysis.report import format_profile_report
from diagnosis.hypotheses import generate_hypotheses
from diagnosis.failure_detection import detect_primary_weakness, describe_weakness
from diagnosis.llm_reasoner import build_evidence_payload, call_llm_reasoner
from diagnosis.diagnostic_experiment import run_diagnostic_suite, rank_hypotheses
from diagnosis.intervention import apply_intervention_and_validate
from dashboard.report_html import build_report_html

DEVIATIONS = [
    "Evaluations use curated Lichess puzzle positions with 20-pawn clipped regret against "
    "ground-truth reference moves.",
    "Trajectory data is stored as JSONL.",
    "Dashboard is static HTML.",
]


def main():
    print("=" * 60)
    print("GAMBIT — full pipeline run (Weeks 4-8)")
    print("=" * 60)

    # --- 1. Load Ground Truth Data ---
    print("\n[1/6] Loading Lichess benchmark + diagnostic/held-out split...")
    positions, _ = load_lichess_positions(path=RAW_ZST_PATH, max_positions=150, validate=True)
    for p in positions:
        p["phase"] = p.get("lichess_phase") or p.get("internal_phase") or "unknown"

    diagnostic, holdout = diagnostic_holdout_split(positions, holdout_fraction=0.3, seed=2)
    print(f"  {len(positions)} positions -> {len(diagnostic)} diagnostic / {len(holdout)} held-out")

    # --- 2. Baseline Evaluation ---
    print("\n[2/6] Running baseline agent over the diagnostic set...")
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

    # --- 2b. Anomaly Detection (Step 9) ---
    print("\n[2b/6] Running anomaly detection...")
    anomaly_summary = None
    try:
        anomalies = detect_anomalies(baseline["records"])
        n_anomalies = len(anomalies) if isinstance(anomalies, list) else anomalies.get("n_anomalies", 0)
        anomaly_summary = {"n_anomalies": n_anomalies}
        print(f"  Identified {n_anomalies} anomalous decision states.")
    except Exception as e:
        print(f"  Anomaly detection skipped: {e}")

    # --- Step 10: Classical vs. Neural Disagreement ---
    print("\n[2c/6] Computing Classical vs Neural Evaluation Disagreement...")
    try:
        n_agent = NeuralAgent()
        enriched = compute_disagreement(baseline["records"], n_agent)
        disagreement_stats = disagreement_regret_correlation(enriched)
        
        corr = disagreement_stats.get('correlation')
        pval = disagreement_stats.get('p_value')
        sig = "Significant" if pval is not None and pval < 0.05 else "Not significant"
        print(f"  Mean Disagreement: {disagreement_stats['mean_disagreement']} pawns")
        print(f"  Correlation with Regret: {corr} (p={pval}) -> {sig}")
    except Exception as e:
        print(f"  Disagreement analysis skipped: {e}")
        disagreement_stats = None

    # --- 3. Failure Detection & Hypotheses ---
    print("\n[3/6] Detecting primary weakness + generating candidate hypotheses...")
    weak_dim, weak_stats = detect_primary_weakness(profile)
    weakness_desc = describe_weakness(weak_dim, weak_stats)
    print(f"  {weakness_desc}")
    
    # 3a. Candidate hypotheses
    hyps = generate_hypotheses(weak_dim)
    for h in hyps:
        print(f"  {h['id']}: {h['name']} — {h['statement']}")
        schema = h["experiment_schema"]
        print(f"      -> Test Plan: {schema['variable']} (Control: {schema['control']} vs Treatment: {schema['treatment']})")

    # 3b. Extract glass-box blunder case studies for the LLM
    blunders = sorted(
        [r for r in baseline["records"] if r.get("regret", 0.0) > 0 and not r.get("decision_match", False)],
        key=lambda r: r.get("regret", 0.0),
        reverse=True
    )
    
    # MICRO-BATCHING: Only send the top 2 worst blunders to the LLM
    case_studies = [
        {
            "position": b.get("position"),
            "puzzle_id": b.get("puzzle_id"),
            "chosen_move": b.get("chosen_move"),
            "reference_move": b.get("reference_move"),
            "regret": b.get("regret"),
            "agent_depth": b.get("agent_depth_reached"),
            "agent_pv_line": b.get("agent_pv_line"),
            "agent_pv_trace": b.get("agent_pv_trace", []),
            "reference_pv_line": b.get("reference_pv_line", ""),
        }
        for b in blunders[:2]
    ]

    # 3c. Gemini LLM Reasoning
    print("\n  [LLM] Asking Gemini to reason with glass-box telemetry...")
    llm_diagnosis = {}
    try:
        evidence = build_evidence_payload(
            weak_dimension=weak_dim, 
            weak_stats=weak_stats, 
            hypotheses=hyps,
            disagreement_stats=disagreement_stats,
            anomaly_summary=anomaly_summary,
            case_studies=case_studies
        )
        print("\n" + "-" * 40)
        print("  PAYLOAD BEING SENT TO GEMINI:")
        print(json.dumps(evidence, indent=2))
        print("-" * 40 + "\n")
        llm_diagnosis = call_llm_reasoner(evidence)
        
        print("\n" + "─" * 50)
        print(" GEMINI GLASS-BOX DIAGNOSIS")
        print("─" * 50)
        print(f" Interpretation: {llm_diagnosis.get('interpretation')}")
        print(f" Top Hypothesis: {llm_diagnosis.get('best_supported_hypothesis')}")
        print(f" Next Experiment: {llm_diagnosis.get('suggested_next_experiment')}")
        
        case_diagnoses = llm_diagnosis.get("case_study_analysis", [])
        if case_diagnoses:
            print("\n Blunder Case Studies (Move-by-Move Refutation):")
            for cs in case_diagnoses:
                print(f"   • Position: {cs.get('position')}")
                if "divergence_analysis" in cs:
                    print(f"     Divergence: {cs.get('divergence_analysis')}")
                print(f"     Diagnosis:  {cs.get('diagnosis')}")
        print("─" * 50 + "\n")
    except Exception as e:
        print(f"  [LLM Skipped]: {e}")

    # --- 4. Diagnostic Experiments ---
    print("\n[4/6] Running diagnostic experiments for H1/H2/H3...")
    tagged = tag_records(baseline["records"])
    weak_records = {
        "opening": [r for r in tagged if r["game_phase"] == "opening"],
        "middlegame": [r for r in tagged if r["game_phase"] == "middlegame"],
        "endgame": [r for r in tagged if r["game_phase"] == "endgame"],
        "tactical_proxy": [r for r in tagged if r["tactical_bucket"] == "high_complexity"],
        "positional_proxy": [r for r in tagged if r["tactical_bucket"] == "low_complexity"],
        "defensive_proxy": [r for r in tagged if r["is_defensive"]],
    }.get(weak_dim, tagged)

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
        weak_positions = diagnostic

    evidence = run_diagnostic_suite(weak_positions, hypotheses=hyps, time_budget_ms=300, seed=100)
    ranked = rank_hypotheses(evidence)
    for r in ranked:
        print(
            f"  {r['hypothesis']}: {r['label_a']} -> {r['label_b']}, "
            f"regret {r['mean_abs_regret_a']} -> {r['mean_abs_regret_b']} "
            f"({r['improvement_pct']}% improvement, p={r['p_value']})"
        )

    # --- 5. Evidence-Gated Intervention ---
    print("\n[5/6] Evaluating diagnostic evidence for intervention gating...")
    top_hyp = ranked[0] if ranked else None
    has_valid_evidence = (
        top_hyp is not None 
        and top_hyp.get("improvement_pct", 0) > 0 
        and top_hyp.get("p_value", 1.0) < 0.05
    )

    if has_valid_evidence and top_hyp["hypothesis"] == "H1":
        intervention_result = apply_intervention_and_validate(
            holdout, base_depth=3, boosted_depth=5, complexity_threshold=30, time_budget_ms=300, seed=500
        )
    elif has_valid_evidence and top_hyp["hypothesis"] == "H3":
        print(f"  Intervention confirmed: Time budget expansion based on {top_hyp['hypothesis']}.")
        before_eval = run_experiment(SearchAgent, {"max_depth": 3}, holdout, time_budget_ms=100, reference_depth=2, seed=500)
        after_eval = run_experiment(SearchAgent, {"max_depth": 3}, holdout, time_budget_ms=600, reference_depth=2, seed=500)
        imp = ((before_eval["mean_regret"] - after_eval["mean_regret"]) / max(1e-6, before_eval["mean_regret"])) * 100
        intervention_result = {
            "status": "applied",
            "type": "time_budget_expansion",
            "before": {"mean_abs_regret": round(before_eval["mean_regret"], 3)},
            "after": {"mean_abs_regret": round(after_eval["mean_regret"], 3)},
            "improvement_pct": round(imp, 2),
            "n_positions": len(holdout),
        }
        print(f"  Before (100ms): mean regret = {intervention_result['before']['mean_abs_regret']}")
        print(f"  After (600ms):  mean regret = {intervention_result['after']['mean_abs_regret']}")
        print(f"  Held-out Improvement: {intervention_result['improvement_pct']}%")
    else:
        print("  Intervention skipped: no hypothesis demonstrated statistically significant improvement.")
        intervention_result = {
            "status": "rejected",
            "improvement_pct": 0.0,
            "n_positions": len(holdout),
            "before": {"mean_abs_regret": 0.0},
            "after": {"mean_abs_regret": 0.0},
        }

    # --- 6. Build Final Report ---
    print("\n[6/6] Building final report...")
    html = build_report_html(
        agent_name, profile, weak_dim, weakness_desc, ranked, intervention_result, DEVIATIONS, llm_diagnosis
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
        "llm_diagnosis": llm_diagnosis,
        "intervention": {k: v for k, v in intervention_result.items() if not k.endswith("_records")},
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary JSON written to {summary_path}")

    print("\n" + "=" * 60)
    print("Full loop complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()