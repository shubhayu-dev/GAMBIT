"""
LLM reasoning layer (docs/11_integration_addendum.md, section "14. LLM
Reasoning Layer" in the pasted integration spec).

Role, per the spec, and enforced here: the LLM interprets *already-computed*
quantitative evidence, generates candidate hypotheses, ranks them, and
proposes experiments. It never declares a cause on its own authority, never
plays moves, and never modifies the agent directly -- every hypothesis it
proposes still has to go through diagnosis/diagnostic_experiment.py's
controlled A/B testing before anything is acted on.

Deviation, flagged per Constitution Rule 7 (see docs/11_integration_addendum.md):
`call_llm_reasoner()` below is real, working code for calling the Anthropic
API -- but this sandbox has no network access and no API key configured, so
it can't actually execute here. Rather than fake a call or hand-write a
templated "LLM-sounding" response, the actual reasoning step for this
project's real evidence was performed once, manually, by Claude in the
session that built this integration -- see
`data/llm_reasoning_output.json`, which is real analysis of this project's
real numbers, not a synthetic example. Swap in a real ANTHROPIC_API_KEY and
network access and `call_llm_reasoner()` runs unchanged.
"""

import json
import os
from typing import Dict, List, Optional


def build_evidence_payload(weak_dimension: str, weak_stats: Dict, hypotheses: List[Dict],
                            disagreement_stats: Optional[Dict] = None,
                            anomaly_summary: Optional[Dict] = None) -> Dict:
    """Assembles the structured evidence dict the LLM reasons over -- matches
    the schema in the pasted integration spec's section 14."""
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


def call_llm_reasoner(evidence: Dict, api_key: Optional[str] = None, model: str = "claude-sonnet-5") -> Dict:
    """
    Real implementation: calls the Anthropic API with `evidence` and returns
    the parsed JSON reasoning output. Requires `anthropic` installed and a
    working ANTHROPIC_API_KEY (env var or passed explicitly) with network
    access -- neither is available in the build sandbox, so this will raise
    here. See the module docstring for how this project's actual reasoning
    step was performed instead.
    """
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "The `anthropic` package isn't installed in this environment. "
            "pip install anthropic, set ANTHROPIC_API_KEY, and this function will work as-is."
        )

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY found (env var or api_key argument).")

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(evidence, indent=2)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return json.loads(text)
