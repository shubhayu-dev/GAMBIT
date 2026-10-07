"""
LLM reasoning layer powered by Google Gemini and Local LLMs (Ollama).
Translates empirical diagnostic evidence and the investigation ledger into post-mortems.

FIXES:
1. Restored the anti-fabrication instruction ("do not invent numbers, moves,
   or claims not present in the payload").
2. Each proven_ledger entry now carries a `validation_status` field.
3. Added semantic PV translation (`annotate_pv_line`) to cure LLM FEN-blindness.
4. Added `call_local_reasoner` for offline, rate-limit-free diagnostics via Ollama.
5. Added explicit tools=[] override in Gemini config to prevent AFC warnings.
"""

import json
import os
import time
import requests
import chess
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
- Do NOT mention general concepts like "controlling the center" or "opening diagonals" unless the specific piece that moved is directly responsible for that action. Base your tactical analysis on the semantic English PV lines provided.

CRITICAL RULE: Look exactly at the 'proven_cause' string. IF IT SAYS 'No tested knob recovered the reference move', YOU MUST STATE THAT NO FIX WORKED. DO NOT INVENT A FIX.

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


def annotate_pv_line(fen: str, uci_moves: List[str]) -> List[str]:
    """
    Translates raw UCI moves into semantic English descriptions.
    Provides the LLM with tactical realities (captures, checks) it cannot "see".
    """
    if not isinstance(uci_moves, list) or not uci_moves:
        return []

    board = chess.Board(fen)
    annotated_line = []
    
    piece_names = {
        chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
        chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"
    }

    for uci_move in uci_moves:
        try:
            move = chess.Move.from_uci(uci_move)
            if move not in board.legal_moves:
                annotated_line.append(f"{uci_move} (Illegal move hallucinated by agent)")
                break
            
            moving_piece = board.piece_at(move.from_square)
            piece_name = piece_names[moving_piece.piece_type] if moving_piece else "Piece"
            is_capture = board.is_capture(move)
            
            board.push(move)
            is_checkmate = board.is_checkmate()
            is_check = board.is_check()
            
            action = "captures on" if is_capture else "moves to"
            target = chess.square_name(move.to_square)
            tactical = ", Checkmate!" if is_checkmate else (", Check" if is_check else "")
                
            annotated_line.append(f"{uci_move} ({piece_name} {action} {target}{tactical})")
        except Exception:
            annotated_line.append(f"{uci_move} (Unparseable move)")
            break

    return annotated_line


def build_evidence_payload(
    weak_dimension: str,
    weak_stats: Dict,
    disagreement_stats: Optional[Dict] = None,
    anomaly_summary: Optional[Dict] = None,
    proven_ledger: Optional[List[Dict]] = None,
) -> Dict:
    """Builds the comprehensive diagnostic payload using proven empirical results and semantic annotations."""
    
    # Enrich the ledger to cure FEN-blindness
    enriched_ledger = []
    for entry in (proven_ledger or []):
        enriched = entry.copy()
        fen = entry.get("position")
        if fen:
            # FIXED: Added the explicit 1-move tactical annotations for immediate checkmate/capture detection
            ref = entry.get("reference_move")
            orig = entry.get("original_move")
            if ref:
                enriched["reference_move_tactical"] = annotate_pv_line(fen, [ref])[0]
            if orig:
                enriched["original_move_tactical"] = annotate_pv_line(fen, [orig])[0]

            # Maintained original PV line annotations as well for deeper context
            if "agent_pv_line" in entry:
                enriched["agent_pv_line_semantic"] = annotate_pv_line(fen, entry["agent_pv_line"])
            if "reference_pv_line" in entry:
                enriched["reference_pv_line_semantic"] = annotate_pv_line(fen, entry["reference_pv_line"])
        enriched_ledger.append(enriched)

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
        "proven_ledger": enriched_ledger,
    }


def call_local_reasoner(
    evidence: Dict, 
    model_name: str = "qwen2.5:7b-instruct", 
    host: str = "http://localhost:11434"
) -> Dict:
    """Sends diagnostic telemetry to a local Ollama instance."""
    os.makedirs("data", exist_ok=True)
    with open("data/llm_evidence_payload_local.json", "w") as f:
        json.dump(evidence, f, indent=2)

    prompt = f"{SYSTEM_PROMPT}\n\nEVIDENCE PAYLOAD:\n{json.dumps(evidence, indent=2)}"
    
    try:
        response = requests.post(f"{host}/api/generate", json={
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2, "num_ctx": 8192}
        }, timeout=180)
        response.raise_for_status()
        return json.loads(response.json()["response"])
    except Exception as e:
        print(f"\n  [Local LLM Failed]: {e}")
        return {
            "interpretation": "Local LLM failed to process.",
            "proven_fixes_summary": "Error reaching Ollama.",
            "case_study_analysis": [],
            "caveats": [str(e)],
            "architectural_recommendation": "N/A"
        }


def call_llm_reasoner(
    evidence: Dict,
    api_key: Optional[str] = None,
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
    
    # FIXED: Added tools=[] to explicitly disable automatic function calling 
    # to prevent parsing errors when Gemini tries to call functions instead of responding
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.2,
        tools=[] 
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