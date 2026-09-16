"""
LLM reasoning layer powered by Google Gemini.
"""

import json
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()

def build_evidence_payload(
    weak_dimension: str,
    weak_stats: Dict,
    hypotheses: List[Dict],
    disagreement_stats: Optional[Dict] = None,
    anomaly_summary: Optional[Dict] = None,
    case_studies: Optional[List[Dict]] = None
) -> Dict:
    """Builds the diagnostic payload, including glass-box PV traces for top blunders."""
    return {
        "failure_pattern": f"low_decision_accuracy_in_{weak_dimension}",
        "conditions": {
            "dimension": weak_dimension,
            "n_positions": weak_stats.get("n"),
            "decision_accuracy_pct": weak_stats.get("decision_accuracy_pct"),
            "mean_regret": weak_stats.get("mean_regret"),
        },
        "candidate_hypotheses": [
            {"id": h["id"], "name": h["name"], "statement": h["statement"]} 
            for h in hypotheses
        ],
        "classical_neural_disagreement": disagreement_stats,
        "anomaly_summary": anomaly_summary,
        "blunder_case_studies": case_studies or [],
    }

SYSTEM_PROMPT = """You are the diagnostic reasoning layer of GAMBIT, an AI agent evaluation \
framework. You act as a senior chess AI developer debugging a search-based agent. \
You receive quantitative metrics AND glass-box move telemetry (Principal Variations, \
evaluation traces) for specific blunders. 

Your job:
1. Interpret the aggregate evidence to evaluate the candidate hypotheses.
2. Analyze the 'blunder_case_studies'. Compare the agent's expected line (agent_pv_line) \
against the ground truth (reference_pv_line). 
3. Diagnose EXACTLY why the agent blundered.

You MUST use a Chain of Thought. For every case study, you must first fill out the 'divergence_analysis' \
field where you explicitly state the exact ply where the agent_pv deviates from the reference_pv, \
and state the tactical reality the agent missed. Only then may you write the 'diagnosis'.

Output strict JSON only. Do not fabricate numbers or moves not present in the payload.

Expected JSON schema:
{
  "interpretation": "High-level summary of the agent's failure mode.",
  "best_supported_hypothesis": "H1",
  "case_study_analysis": [
     {
       "position": "FEN string of the blunder",
       "divergence_analysis": "Step 1: State the deviation ply. Step 2: Explain the missed tactic.",
       "diagnosis": "The final conclusion of the algorithmic blind spot based on the divergence."
     }
  ],
  "caveats": ["List of limitations in the data"],
  "suggested_next_experiment": "What to test next"
}"""

def call_llm_reasoner(evidence: Dict, api_key: Optional[str] = None, model_name: str = "gemini-3.6-flash") -> Dict:
    """
    Calls the Google Gemini API with `evidence` and returns the parsed JSON reasoning output.
    Uses the modern google-genai SDK with exponential backoff for 503 capacity errors.
    """
    # Dump the evidence locally just in case the API fails completely
    os.makedirs("data", exist_ok=True)
    with open("data/llm_evidence_payload.json", "w") as f:
        json.dump(evidence, f, indent=2)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "The `google-genai` package isn't installed. "
            "Run: pip install google-genai"
        )

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY found (env var or api_key argument).")

    client = genai.Client(api_key=key)
    
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.2 
    )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=json.dumps(evidence, indent=2),
                config=config
            )
            return json.loads(response.text)
            
        except Exception as e:
            error_str = str(e)
            if "503" in error_str and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                print(f"\n  [LLM Warning] 503 High Demand. Retrying in {wait_time} seconds (Attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                print(f"\n  [LLM Error] API failed after {attempt+1} attempts. Payload saved to data/llm_evidence_payload.json")
                raise e
    
    return {}