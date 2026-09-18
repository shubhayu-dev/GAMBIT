"""
LLM reasoning layer powered by Google Gemini.
Translates empirical diagnostic evidence and the Proven Ledger into post-mortems.
"""

import json
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are the lead diagnostic reasoning engine of GAMBIT, an automated chess AI debugging framework.
You receive:
1. Aggregate capability metrics.
2. A Proven Ledger of empirical experiments where parameter interventions (depth expansion, evaluation weights, time budgets) were tested directly against blunders.

Your job:
1. Analyze why the agent blundered initially using the principal variations.
2. State the exact tactical or strategic reality the agent missed.
3. Explain WHY the empirically proven parameter fix successfully eliminated the blunder based on chess search mechanics.

Output strict JSON only.

Expected JSON schema:
{
  "interpretation": "High-level summary of agent failure mode across the cohort.",
  "proven_fixes_summary": "Summary of which empirical interventions reliably restored optimal play.",
  "case_study_analysis": [
     {
       "position": "FEN string",
       "divergence_analysis": "Step 1: Note deviation ply. Step 2: Detail tactical line missed.",
       "empirical_fix_explanation": "Explain why the tested parameter fix resolved the mistake."
     }
  ],
  "architectural_recommendation": "Next code-level fix for the engine developer."
}"""


def build_evidence_payload(
    weak_dimension: str,
    weak_stats: Dict,
    disagreement_stats: Optional[Dict] = None,
    anomaly_summary: Optional[Dict] = None,
    proven_ledger: Optional[List[Dict]] = None,
) -> Dict:
    """Builds the comprehensive diagnostic payload using proven empirical results."""
    return {
        "failure_pattern": f"low_decision_accuracy_in_{weak_dimension}",
        "conditions": {
            "dimension": weak_dimension,
            "n_positions": weak_stats.get("n"),
            "decision_accuracy_pct": weak_stats.get("decision_accuracy_pct"),
            "mean_regret": weak_stats.get("mean_regret"),
        },
        "classical_neural_disagreement": disagreement_stats,
        "anomaly_summary": anomaly_summary,
        "proven_ledger": proven_ledger or [],
    }


def call_llm_reasoner(
    evidence: Dict,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash"
) -> Dict:
    """Sends diagnostic telemetry to Gemini with exponential backoff for rate limits."""
    os.makedirs("data", exist_ok=True)
    with open("data/llm_evidence_payload.json", "w") as f:
        json.dump(evidence, f, indent=2)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError("Install the google-genai library: pip install google-genai")

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not configured.")

    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.2,
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=json.dumps(evidence, indent=2),
                config=config,
            )
            return json.loads(response.text)
        except Exception as e:
            error_str = str(e)
            if "503" in error_str and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                print(f"\n  [LLM Warning] 503 Capacity Spike. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\n  [LLM Failed]: {e}")
                raise e

    return {}