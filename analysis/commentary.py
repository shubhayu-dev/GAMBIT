import json
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


def build_commentary_payload(moves: List[Dict]) -> Dict:
    """Build the batched payload for one chunk of unexplained moves."""
    
    def format_eval(pawns: float) -> str:
        """
        Converts floating point pawns to formatted strings.
        Intercepts massive internal engine constants (e.g. 10000100) 
        and formats them as 'Forced Mate' so the LLM doesn't misinterpret them.
        """
        cp = round(pawns * 100)
        if abs(cp) > 900000:
            return "+Forced Mate" if cp > 0 else "-Forced Mate"
        return f"{cp} cp"

    return {
        "moves": [
            {
                "ply": m["ply"],
                "move": m["move"],
                "fen_before": m["fen_before"],
                "eval_before": format_eval(m["eval_before"]),
                "eval_after": format_eval(m["eval_after"]),
                "eval_delta": format_eval(m["eval_after"] - m["eval_before"]),
                "engine_pv_from_here": m.get("pv_from_here", []),
                "n_legal_moves_available": m.get("n_legal_at_root"),
                "n_moves_actually_compared": len(m.get("alternatives", [])),
                "moves_compared_by_search": [
                    {"move": a["move"], "score": format_eval(a["score"]), "fully_searched": a["fully_searched"]}
                    for a in m.get("alternatives", [])[:8]
                ],
                "move_description": m.get("move_description"),
            }
            for m in moves
        ]
    }


SYSTEM_PROMPT = """You are the move-commentary layer of GAMBIT, an AI agent evaluation \
framework. You explain, in plain language a club-level chess player can follow, why the \
AGENT chose this move over the other options ITS OWN SEARCH actually looked at -- not just \
what the move does on the board, and not what an idealized unlimited-time engine would have \
played.

You are given a list of moves. For each one you get: the position before the move (FEN), the \
move itself, the evaluation before and after, the engine's expected principal variation \
continuing from the chosen move, and: n_legal_moves_available (total legal moves), \
n_moves_actually_compared (how many the search reached before its time budget ran out), and \
moves_compared_by_search -- the actual moves it evaluated, in the order it tried them, each \
with a score and a fully_searched flag. You also get move_description -- a plain-language, \
PRE-COMPUTED, ground-truth statement of what the move actually is.

HARD RULE on piece identity: move_description is ALWAYS correct about piece identity, capture, \
and check -- it is not a hint, it is ground truth. Do NOT re-derive or second-guess which piece \
moved, what (if anything) was captured, or whether the move gives check.

CRITICAL CONTEXT: this engine runs on a wall-clock time budget. It frequently does NOT compare \
every legal move -- alpha-beta cutoffs and the time limit mean the search may stop after only a \
handful of moves. A move with fully_searched: false was cut short (its score may just be a \
shallow static evaluation, not a real search result) -- treat its score as unreliable and say so. \
n_moves_actually_compared being smaller than n_legal_moves_available is not an error, \
it's the time budget -- and it is often the real answer to "why did the agent make this choice".

Rules:
1. Base every explanation ONLY on the data given for that move. Do not invent threats, plans, \
or moves that are not present in the payload.
2. The explanation must be a COMPARISON grounded in moves_compared_by_search, not a \
description of the played move alone. Reference at least one specific alternative from that \
list and its score.
3. HARD RULE, no exceptions: an alternative with fully_searched: false may be MENTIONED only \
to note that the search didn't get a reliable read on it -- never as evidence the played move \
was correct OR incorrect. 
4. Reference concrete squares and pieces rather than vague praise ("a good move").
5. Keep each explanation to 1-2 sentences.
6. Classify each move into exactly one tag: "tactical", "positional", "blunder", or "book".
7. If the eval swing is large but you cannot see why from the PV or the compared moves, say \
so explicitly rather than guessing.
8. eval_before, eval_after, and score fields are formatted strings (e.g., "15 cp", "+Forced Mate").
9. HARD RULE AGAINST FILLER: Do NOT mention general concepts like "controlling the center", \
"development", or "opening the long diagonal" unless the exact piece that moved is directly \
responsible for that specific action in this specific turn.

Output strict JSON only. No preamble, no markdown fences.

Expected JSON schema:
{
  "commentary": [
    {
      "ply": 14,
      "move": "e5f3",
      "explanation": "...",
      "tag": "tactical"
    }
  ]
}
"""


def _call_gemini(
    payload: Dict,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash",
    max_retries: int = 5,
) -> List[Dict]:
    """Original Gemini path."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "The google-genai package isn't installed. Run: pip install google-genai"
        )

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No GEMINI_API_KEY found (env var or api_key argument).")

    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        temperature=0.2,
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=json.dumps(payload, indent=2),
                config=config,
            )
            parsed = json.loads(response.text)
            return parsed.get("commentary", [])
        except Exception as e:
            error_str = str(e)
            if "503" in error_str and attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                print(
                    f"\n  [Commentary Warning] 503 High Demand. "
                    f"Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(wait_time)
            else:
                print(
                    f"\n  [Commentary Error] Gemini API failed after {attempt + 1} attempts. "
                    f"Payload saved to data/commentary_payload_latest.json"
                )
                raise e

    return []


def _call_ollama(
    payload: Dict,
    model_name: str = "qwen2.5:7b-instruct",
    host: str = "http://localhost:11434",
    max_retries: int = 3,
    num_ctx: int = 8192,
    num_predict: int = 4096,
    timeout_s: int = 180,
) -> List[Dict]:
    """Calls a local Ollama server instead of Gemini."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("The requests package isn't installed. Run: pip install requests")

    url = f"{host}/api/chat"
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": num_ctx, "num_predict": num_predict},
    }

    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=body, timeout=timeout_s)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            parsed = json.loads(content)
            return parsed.get("commentary", [])
        except requests.exceptions.ConnectionError as e:
            last_error = e
            print(
                f"\n  [Commentary Warning] Can't reach Ollama at {host} -- is it running? "
                f"(ollama serve, or check curl {host}/api/tags). "
                f"Attempt {attempt + 1}/{max_retries}."
            )
            time.sleep(3)
        except (json.JSONDecodeError, KeyError) as e:
            last_error = e
            print(
                f"\n  [Commentary Warning] Ollama returned something that wasn't the expected "
                f"JSON shape (attempt {attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(1)

    print(
        f"\n  [Commentary Error] Ollama call failed after {max_retries} attempts: {last_error}. "
        f"Payload saved to data/commentary_payload_latest.json"
    )
    raise RuntimeError(f"Ollama commentary call failed: {last_error}")


def call_commentary_llm(
    payload: Dict,
    provider: str = "ollama",
    **kwargs,
) -> List[Dict]:
    os.makedirs("data", exist_ok=True)
    with open("data/commentary_payload_latest.json", "w") as f:
        json.dump(payload, f, indent=2)

    if provider == "ollama":
        return _call_ollama(payload, **kwargs)
    elif provider == "gemini":
        return _call_gemini(payload, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider!r} (expected 'ollama' or 'gemini')")


def generate_deterministic_comment(m: Dict) -> Optional[Dict]:
    """
    Rule-based check: Identifies standard development or obvious recaptures.
    If conditions are met, returns a deterministically generated comment 
    to bypass the LLM, saving compute and batch space.
    """
    regret = m.get("regret", 0.0)
    eval_before = m.get("eval_before", 0.0)
    eval_after = m.get("eval_after", 0.0)
    eval_delta = abs(eval_after - eval_before)
    
    # Only skip LLM if regret is essentially zero and evaluation shift is < 50cp (0.5 pawns)
    if regret <= 0.05 and eval_delta < 0.5:
        desc = m.get("move_description", "")
        desc_lower = desc.lower()
        
        if "captures" in desc_lower or "takes" in desc_lower:
            return {
                "ply": m["ply"],
                "move": m["move"],
                "explanation": f"A straightforward recapture or even trade ({desc}).",
                "tag": "positional"
            }
        elif m.get("ply", 0) <= 20: # Typical early opening phase
            return {
                "ply": m["ply"],
                "move": m["move"],
                "explanation": f"Standard opening development ({desc}).",
                "tag": "book"
            }
        else:
            return {
                "ply": m["ply"],
                "move": m["move"],
                "explanation": f"A solid, natural continuation that maintains the balance ({desc}).",
                "tag": "positional"
            }
    return None


def chunk_moves(moves: List[Dict], chunk_size: int = 40) -> List[List[Dict]]:
    return [moves[i:i + chunk_size] for i in range(0, len(moves), chunk_size)]


def annotate_game(
    unexplained_moves: List[Dict],
    chunk_size: Optional[int] = None,
    retry_missing_individually: bool = True,
    **llm_kwargs,
) -> List[Dict]:
    """
    End-to-end: chunk -> call -> flatten back into one list of commentary dicts in ply order.
    """
    provider = llm_kwargs.get("provider", "ollama")
    if chunk_size is None:
        # FIXED: Reduced Ollama batch size to max 3 plies to prevent JSON truncation
        chunk_size = 3 if provider == "ollama" else 40

    all_commentary: List[Dict] = []
    needs_llm: List[Dict] = []

    # FIXED: Pre-filter before prompting
    for m in unexplained_moves:
        det_comment = generate_deterministic_comment(m)
        if det_comment:
            all_commentary.append(det_comment)
        else:
            needs_llm.append(m)

    # Process remaining moves with the LLM
    for chunk in chunk_moves(needs_llm, chunk_size=chunk_size):
        payload = build_commentary_payload(chunk)
        result = call_commentary_llm(payload, **llm_kwargs)

        by_ply = {m["ply"]: m for m in chunk}
        requested_plies = set(by_ply.keys())
        returned_plies = {c["ply"] for c in result}
        missing = requested_plies - returned_plies
        
        if missing:
            print(
                f"\n  [Commentary Warning] Asked for commentary on {len(requested_plies)} "
                f"moves, got {len(returned_plies)} back. Missing plies: {sorted(missing)}."
            )
            if retry_missing_individually and chunk_size > 1:
                print(f"  Retrying {len(missing)} missing move(s) individually...")
                for ply in sorted(missing):
                    single_payload = build_commentary_payload([by_ply[ply]])
                    try:
                        single_result = call_commentary_llm(single_payload, **llm_kwargs)
                    except Exception as e:
                        single_result = []
                        print(f"    ply {ply}: retry itself failed ({e})")
                    
                    if single_result:
                        result.extend(single_result)
                    else:
                        print(f"    ply {ply}: still no commentary even alone.")

        all_commentary.extend(result)
        
    return sorted(all_commentary, key=lambda c: c.get("ply", 0))