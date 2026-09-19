"""
LLM reasoning layer powered by Google Gemini.
Translates empirical diagnostic evidence and the investigation ledger into post-mortems.

FIXES (see chat writeup):
1. Restored the anti-fabrication instruction ("do not invent numbers, moves,
   or claims not present in the payload") that this version had dropped.
   That instruction existed for a reason -- the LLM is the last line
   between raw telemetry and a human reading a report, and this project's
   own audit history (train/test leakage, the regret-formula bug) is
   exactly the kind of thing an unconstrained LLM writeup would have
   confidently narrated over instead of catching.
2. Each proven_ledger entry now carries a `validation_status` field (see
   diagnosis/diagnostic_experiment.py) that's either "unresolved" or
   "single_position_match (unconfirmed -- ...)" -- never "proven" outright,
   since a single knob bump matching one reference move once is not a
   validated fix. The prompt now explicitly tells the model to reflect
   that hedge in its language instead of asserting the fix "successfully
   eliminated" anything.
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
2. An investigation ledger of empirical experiments where parameter interventions (depth expansion, evaluation weights, time budgets) were tested against specific blunders.

Your job:
1. Analyze why the agent blundered initially using the principal variations.
2. State the exact tactical or strategic reality the agent missed.
3. Explain the mechanism by which the tested parameter change could plausibly resolve the mistake, based on chess search mechanics.

Rules:
- Do not fabricate numbers, moves, positions, or claims that are not present in the payload. If the payload doesn't contain enough information to explain something, say so explicitly rather than guessing.
- Every ledger entry has a `validation_status` field. If it is anything other than a fully validated/generalized result, your language MUST reflect that: use hedged phrasing ("recovered the reference move on this position", "a candidate fix") rather than assertive phrasing ("resolved", "eliminated", "fixed"). Only use confirmatory language ("resolved", "fixed") for entries explicitly marked as validated/generalized.
- If a case's investigation_log shows no successful test, say plainly that no tested intervention explained the blunder -- do not invent one.

Output strict JSON only.

Expected JSON schema:
{
  "interpretation": "High-level summary of agent failure mode across the cohort.",
  "proven_fixes_summary": "Summary of which interventions were tested and which (if any) are actually validated vs. still candidates.",
  "case_study_analysis": [
     {
       "position": "FEN string",
       "divergence_analysis": "Step 1: Note deviation ply. Step 2: Detail tactical line missed.",
       "empirical_fix_explanation": "Explain the candidate fix and its mechanism, hedged per validation_status. If unresolved, say so."
     }
  ],
  "caveats": ["Limitations in the evidence -- e.g. single-position matches not yet validated on held-out data"],
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
    # Reverted to gemini-3.6-flash: this version had silently downgraded to
    # gemini-2.5-flash with no comment explaining why. An unlogged model
    # swap changes what a report says without anyone deciding to change it
    # -- if that downgrade was intentional (e.g. cost or availability),
    # leave a comment saying so; if not, this is the original value,
    # consistent with the commentary module (analysis/commentary.py).
    model_name: str = "gemini-3.6-flash"
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