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
from analysis.profile import build_profile
from analysis.anomaly import detect_anomalies
from analysis.report import format_profile_report

from diagnosis.failure_detection import detect_primary_weakness, describe_weakness, extract_blunders
from diagnosis.diagnostic_experiment import run_empirical_investigation
from diagnosis.intervention import extract_dominant_fix, evaluate_intervention
from dashboard.report_html import build_report_html
# FIXED: Swapped call_llm_reasoner for call_local_reasoner
from diagnosis.llm_reasoner import build_evidence_payload, call_local_reasoner 

DEVIATIONS = [
    "Evaluations use curated Lichess puzzle positions with 20-pawn clipped regret against "
    "ground-truth reference moves.",
    "Trajectory data is stored as JSONL.",
    "Dashboard is static HTML.",
]


def main():
    print("=" * 60)
    print("GAMBIT — full pipeline run")
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

    # --- 3. Failure Detection + Empirical Investigation ---
    print("\n[3/6] Detecting primary weakness + isolating blunders...")
    weak_dim, weak_stats = detect_primary_weakness(profile)
    weakness_desc = describe_weakness(weak_dim, weak_stats)
    print(f"  {weakness_desc}")

    blunders = extract_blunders(
        baseline["records"], top_k=5, min_regret=1.0,
        disagreement_records=enriched if disagreement_stats else None,
    )
    print(f"  Isolated {len(blunders)} blunders (top by regret) to investigate.")

    proven_ledger = run_empirical_investigation(SearchAgent, blunders, seed=100)
    for rec in proven_ledger:
        print(f"  [{rec['validation_status']}] {rec['proven_cause']}")

    # FIXED: Replaced Gemini call with local Ollama
    print("\n  [LLM] Asking local Ollama to reason over the investigation ledger...")
    llm_diagnosis = {}
    try:
        evidence = build_evidence_payload(
            weak_dimension=weak_dim,
            weak_stats=weak_stats,
            disagreement_stats=disagreement_stats,
            anomaly_summary=anomaly_summary,
            proven_ledger=proven_ledger[:2],
        )
        print("\n" + "-" * 40)
        print("  PAYLOAD BEING SENT TO LOCAL OLLAMA:")
        print(json.dumps(evidence, indent=2))
        print("-" * 40 + "\n")
        
        # Pointed directly to Qwen2.5 as instructed
        llm_diagnosis = call_local_reasoner(evidence, model_name="qwen2.5:14b")
        print("\n" + "─" * 50)
        print(" LOCAL LLM DIAGNOSIS")
        print("─" * 50)
        print(f" Interpretation:      {llm_diagnosis.get('interpretation')}")
        print(f" Proven fixes summary: {llm_diagnosis.get('proven_fixes_summary')}")
        print(f" Architectural rec:   {llm_diagnosis.get('architectural_recommendation')}")
        if llm_diagnosis.get("caveats"):
            print(f" Caveats:             {llm_diagnosis.get('caveats')}")

        case_diagnoses = llm_diagnosis.get("case_study_analysis", [])
        if case_diagnoses:
            print("\n Blunder Case Studies:")
            for cs in case_diagnoses:
                print(f"   • Position: {cs.get('position')}")
                if "divergence_analysis" in cs:
                    print(f"     Divergence: {cs.get('divergence_analysis')}")
                print(f"     Fix explanation: {cs.get('empirical_fix_explanation')}")
        print("─" * 50 + "\n")
    except Exception as e:
        print(f"  [LLM Skipped]: {e}")

    llm_diagnosis_for_report = dict(llm_diagnosis) if llm_diagnosis else {}
    if llm_diagnosis:
        llm_diagnosis_for_report["best_supported_hypothesis"] = llm_diagnosis.get("proven_fixes_summary")
        llm_diagnosis_for_report["case_study_analysis"] = [
            {**cs, "diagnosis": cs.get("empirical_fix_explanation")}
            for cs in llm_diagnosis.get("case_study_analysis", [])
        ]

    # --- 4. In-sample check of the dominant candidate fix ---
    print("\n[4/6] Extracting the dominant candidate fix from the ledger...")
    dominant_fix_desc, intervention_kwargs = extract_dominant_fix(proven_ledger)
    print(f"  Dominant candidate: {dominant_fix_desc} -> {intervention_kwargs}")

    if intervention_kwargs:
        diagnostic_check = evaluate_intervention(
            SearchAgent, blunders, baseline_kwargs={"max_depth": 3}, intervention_kwargs=intervention_kwargs, seed=100,
        )
        print(
            f"  In-sample (same blunders used to find the fix -- expected to look good, "
            f"NOT evidence of generalization): "
            f"regret {diagnostic_check.get('mean_baseline_regret')} -> {diagnostic_check.get('mean_intervention_regret')} "
            f"({diagnostic_check.get('regret_reduction_pct')}% reduction, metric={diagnostic_check.get('regret_metric')})"
        )
    else:
        diagnostic_check = {}
        print("  No candidate fix to check -- no investigation_log entry recovered a reference move.")

    ranked = [{
        "hypothesis": "Empirical",
        "label_a": "baseline",
        "label_b": dominant_fix_desc,
        "mean_abs_regret_a": diagnostic_check.get("mean_baseline_regret"),
        "mean_abs_regret_b": diagnostic_check.get("mean_intervention_regret"),
        "improvement_pct": diagnostic_check.get("regret_reduction_pct"),
        "p_value": None,
    }] if intervention_kwargs else []

    # --- 5. Held-out validation of the dominant fix ---
    print("\n[5/6] Validating the dominant fix on held-out positions never used above...")
    if intervention_kwargs:
        holdout_for_eval = [
            {"position": p["fen"], "reference_move": p.get("reference_move_lichess")}
            for p in holdout
        ]
        held_out_eval = evaluate_intervention(
            SearchAgent, holdout_for_eval, baseline_kwargs={"max_depth": 3}, intervention_kwargs=intervention_kwargs, seed=500,
        )
        print(
            f"  Held-out: regret {held_out_eval.get('mean_baseline_regret')} -> "
            f"{held_out_eval.get('mean_intervention_regret')} "
            f"({held_out_eval.get('regret_reduction_pct')}% reduction, metric={held_out_eval.get('regret_metric')})"
        )
        print(f"  Generalized: {held_out_eval.get('generalized')}")

        intervention_result = {
            "status": "applied" if held_out_eval.get("generalized") else "rejected",
            "type": dominant_fix_desc,
            "before": {"mean_abs_regret": held_out_eval.get("mean_baseline_regret", 0.0)},
            "after": {"mean_abs_regret": held_out_eval.get("mean_intervention_regret", 0.0)},
            "improvement_pct": held_out_eval.get("regret_reduction_pct", 0.0),
            "n_positions": held_out_eval.get("n_positions", len(holdout)),
            "regret_metric": held_out_eval.get("regret_metric"),
        }
    else:
        print("  Skipped: no candidate fix from step 4 to validate.")
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
        agent_name, profile, weak_dim, weakness_desc, ranked, intervention_result, DEVIATIONS, llm_diagnosis_for_report
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