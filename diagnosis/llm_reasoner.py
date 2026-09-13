"""
LLM reasoning layer powered by Google Gemini.
"""

import json
import os
from typing import Dict, List, Optional


def build_evidence_payload(weak_dimension: str, weak_stats: Dict, hypotheses: List[Dict],
                            disagreement_stats: Optional[Dict] = None,
                            anomaly_summary: Optional[Dict] = None) -> Dict:
    return {
        "failure_pattern": f"low_decision_accuracy_in_{weak_dimension}",
        "conditions": {
            "dimension": weak_dimension,
            "n_positions": weak_stats.get("n"),
            "decision_accuracy_pct": weak_stats.get("decision_accuracy_pct"),
            "mean_regret": weak_stats.get("mean_regret"),
        },
        "candidate_hypotheses": [{"id": h["id"], "name": h["name"], "statement": h["statement"]} for h in hypotheses],
        "classical_neural_disagreement": disagreement_stats,
        "anomaly_summary": anomaly_summary,
    }


SYSTEM_PROMPT = """You are the diagnostic reasoning layer of GAMBIT, an AI agent evaluation \
framework. You receive structured, already-computed quantitative evidence about a chess-playing \
agent's behavior. Your job: interpret the evidence, comment on which candidate hypothesis (if any) \
it actually supports, note anything the evidence does NOT establish, and suggest what additional \
experiment would sharpen the diagnosis. Do not declare a cause with more confidence than the \
evidence warrants. Do not fabricate numbers not present in the payload. Output JSON only, with keys: \
"interpretation", "best_supported_hypothesis" (or null), "caveats" (list), "suggested_next_experiment"."""


def call_llm_reasoner(evidence: Dict, api_key: Optional[str] = None, model_name: str = "gemini-1.5-flash") -> Dict:
    """
    Calls the Google Gemini API with `evidence` and returns the parsed JSON reasoning output.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            "The `google-generativeai` package isn't installed. "
            "pip install google-generativeai, set GEMINI_API_KEY, and this function will work."
        )

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY found (env var or api_key argument).")

    genai.configure(api_key=key)
    
    # Enforce strict JSON output at the API level
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
        generation_config={"response_mime_type": "application/json"}
    )

    response = model.generate_content(json.dumps(evidence, indent=2))
    return json.loads(response.text)