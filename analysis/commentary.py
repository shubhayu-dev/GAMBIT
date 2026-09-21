"""
Human-readable per-move commentary — LLM layer.

Scope note (see PROJECT_CONSTITUTION.md, docs/11_integration_addendum.md style):
this is deliberately a SEPARATE component from diagnosis/llm_reasoner.py.
llm_reasoner.py's job is diagnostic reasoning that feeds the DIAGNOSE step of
the OBSERVE->EVALUATE->DIAGNOSE->EXPERIMENT->INTERVENE->VALIDATE loop -- it
proposes/evaluates hypotheses about *why the agent is systematically weak*.
This module answers a different question a human reviewing one game would
ask: "why did the engine play this move, right here." It reuses the same
LLM-calling conventions (strict JSON, retry/backoff, no fabricated moves) but
must not be merged into the diagnostic payload -- doing so would blur a
scope boundary the constitution is explicit about.

Design decisions this module encodes (see chat writeup):
  - Only moves the rule-based pass (rule_based_commentary.py) couldn't
    explain are sent here -- forced moves, only-legal-moves, and simple
    recaptures are filtered out before this module ever runs.
  - One LLM call per CHUNK of moves (default: one call per game), not one
    call per move, so the model has surrounding context and cost/latency
    stays proportional to games, not plies.
  - Strict JSON in, strict JSON out, matching llm_reasoner.py's approach.
    The model is instructed never to reference moves outside what it's given.
"""

import json
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()


def build_commentary_payload(moves: List[Dict]) -> Dict:
    """Build the batched payload for one chunk of unexplained moves.

    Each entry in `moves` is expected to have:
        ply: int
        move: str (UCI)
        fen_before: str
        eval_before: float   # centipawns, White's perspective
        eval_after: float
        pv_from_here: List[str]   # UCI moves, the engine's expected
                                   # continuation after this move (may be [])
        alternatives: List[Dict]  # engine.search.search_best_move()'s
                                   # info["root_trace"] for this move --
                                   # [{"move": uci, "score": cp, "fully_searched": bool}, ...],
                                   # in the order the live search actually
                                   # tried them (NOT sorted/reconstructed).
                                   # This is what the SAME time-bounded call
                                   # that chose the move actually compared it
                                   # against -- see engine/search.py's
                                   # alpha_beta root_trace docstring for why
                                   # a fresh unconstrained re-search is a
                                   # different, less honest answer here.
        n_legal_at_root: int       # total legal moves available, so the LLM
                                   # can say "compared 4 of 27" rather than
                                   # implying every option was weighed.
    """
    # FIX: engine.evaluation.evaluate() returns pawn-scale floats (a queen
    # is ~9.0, not ~900), but this payload's fields were named "_cp"
    # (centipawns) without actually converting -- so the LLM was told these
    # were centipawns while seeing values like 0.04. Converting for real
    # here (x100, rounded to int) so the field name and the value agree,
    # and so the numbers read like the standard chess-engine convention a
    # model has actually seen a lot of in training (whole-number centipawns)
    # rather than tiny unfamiliar fractions.
    def to_cp(pawns: float) -> int:
        return round(pawns * 100)

    return {
        "moves": [
            {
                "ply": m["ply"],
                "move": m["move"],
                "fen_before": m["fen_before"],
                "eval_before_cp": to_cp(m["eval_before"]),
                "eval_after_cp": to_cp(m["eval_after"]),
                "eval_delta_cp": to_cp(m["eval_after"] - m["eval_before"]),
                "engine_pv_from_here": m.get("pv_from_here", []),
                "n_legal_moves_available": m.get("n_legal_at_root"),
                "n_moves_actually_compared": len(m.get("alternatives", [])),
                # Live search comparison -- see docstring above. Capped so
                # the payload doesn't balloon; the played move is always
                # included since it's always in root_trace.
                "moves_compared_by_search": [
                    {"move": a["move"], "score_cp": to_cp(a["score"]), "fully_searched": a["fully_searched"]}
                    for a in m.get("alternatives", [])[:8]
                ],
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
move itself, the evaluation before and after (centipawns, White's perspective, positive favors \
White), the engine's expected principal variation continuing from the chosen move, and: \
`n_legal_moves_available` (total legal moves in the position), `n_moves_actually_compared` \
(how many the search reached before its time budget ran out), and `moves_compared_by_search` \
-- the actual moves it evaluated, in the order it tried them, each with a score and a \
`fully_searched` flag.

CRITICAL CONTEXT: this engine runs on a wall-clock time budget. It frequently does NOT compare \
every legal move -- alpha-beta cutoffs and the time limit mean the search may stop after only a \
handful of moves, especially later in the game when there are more legal moves to consider. A \
move with `fully_searched: false` was cut short (its score may just be a shallow static \
evaluation, not a real search result) -- treat its score as unreliable and say so if you use it. \
`n_moves_actually_compared` being much smaller than `n_legal_moves_available` is not an error, \
it's the time budget -- and it is often the real answer to "why did the agent make this choice": \
it may not have had time to even look at a better option.

Rules:
1. Base every explanation ONLY on the data given for that move. Do not invent threats, plans, \
or moves that are not present in the payload -- including moves NOT in `moves_compared_by_search`. \
Never claim the agent "considered" or "rejected" a move that isn't in that list, even if you \
recognize it as objectively good -- the agent's search never saw it.
2. The explanation must be a COMPARISON grounded in `moves_compared_by_search`, not a \
description of the played move alone. Reference at least one specific alternative from that \
list and its score, and say concretely why the search preferred the played move over it. If \
`n_moves_actually_compared` is small relative to `n_legal_moves_available`, say so plainly -- \
e.g. "the search only had time to compare 3 of 27 legal moves, and this scored best among \
those" -- rather than implying a thorough comparison that didn't happen.
3. HARD RULE, no exceptions: an alternative with `fully_searched: false` may be MENTIONED only \
to note that the search didn't get a reliable read on it -- never as evidence the played move \
was correct OR incorrect, and never with a comparative claim like "was better/worse" attached \
to its score. Do NOT write a sentence that both (a) uses a fully_searched: false score to claim \
it was "better" or "worse" than the played move, AND (b) says in the same breath that it wasn't \
fully evaluated -- that is a direct contradiction and is the single most important failure mode \
to avoid. If every entry besides the played move has `fully_searched: false`, or \
`moves_compared_by_search` has only the played move, say plainly that no reliable comparison \
exists for this move -- do not manufacture one from an unreliable score.
4. Reference concrete squares and pieces (e.g. "opens the long diagonal for the bishop on \
c1") rather than vague praise ("a good move").
5. Keep each explanation to 1-2 sentences.
6. Classify each move into exactly one tag: "tactical" (wins/loses material or forces a \
sequence), "positional" (improves piece placement, pawn structure, king safety without \
immediate material change), "blunder" (large negative eval swing for the mover), or \
"book" (standard/developing move with no significant eval swing).
7. If the eval swing is large but you cannot see why from the PV or the compared moves, say \
so explicitly rather than guessing.
8. eval_before_cp/eval_after_cp/score_cp are centipawns (100 = one pawn). A queen-level swing \
looks like ~900, not ~9 -- don't reinterpret these as raw pawn counts.

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
    """Original Gemini path -- unchanged, kept available in case you want to
    switch back or compare quality against the local model."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "The `google-genai` package isn't installed. Run: pip install google-genai"
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
    """Calls a local Ollama server instead of Gemini.

    Differences from the Gemini path worth knowing about:
    - No SDK -- plain HTTP POST to Ollama's /api/chat, since Ollama has no
      official Python client comparable to google-genai.
    - Ollama's JSON mode (`"format": "json"`) forces syntactically valid
      JSON but does NOT enforce a specific schema the way Gemini's
      response_mime_type + our schema description together tend to in
      practice -- qwen2.5 usually follows the schema described in
      SYSTEM_PROMPT, but validate a few real responses yourself before
      trusting this at scale (see the note in `annotate_game` about
      chunk_size for local models).
    - `num_ctx` defaults to 8192 here, not Ollama's own 2048 default. A
      40-move payload with moves_compared_by_search attached is easily a
      few thousand tokens; leaving Ollama's default in place means the
      start of your prompt (the system instructions) can silently fall out
      of the context window with no error raised. If you use a large
      chunk_size, raise this further.
    - `num_predict` (max OUTPUT tokens) defaults here to 4096, NOT left at
      Ollama/the model's own default. Many quantized model builds ship
      with a small default baked into the Modelfile (128-256 is common) --
      nowhere near enough to cover a JSON array of explanations for a full
      chunk of moves. When that limit is hit mid-generation, `format:
      "json"` closes out whatever partial JSON exists rather than erroring,
      so you silently get back 1-2 entries instead of the whole chunk with
      no error at all. This is the most likely explanation if you send N
      moves and get back far fewer than N with no warning printed.
    - No native "high demand" 503 -- retries here are for connection
      refused (server not running / crashed) and malformed JSON output
      (small local models occasionally wrap JSON in prose despite
      instructions), not rate limiting.
    """
    try:
        import requests
    except ImportError:
        raise RuntimeError("The `requests` package isn't installed. Run: pip install requests")

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
                f"(`ollama serve`, or check `curl {host}/api/tags`). "
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
    """Dispatches to the local Ollama model (default, per current setup) or
    Gemini (provider="gemini", kept available for comparison/fallback).
    Extra kwargs pass straight through to whichever backend is chosen --
    e.g. model_name, host, num_ctx for Ollama; model_name, api_key,
    max_retries for Gemini.
    """
    os.makedirs("data", exist_ok=True)
    with open("data/commentary_payload_latest.json", "w") as f:
        json.dump(payload, f, indent=2)

    if provider == "ollama":
        return _call_ollama(payload, **kwargs)
    elif provider == "gemini":
        return _call_gemini(payload, **kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider!r} (expected 'ollama' or 'gemini')")


def chunk_moves(moves: List[Dict], chunk_size: int = 40) -> List[List[Dict]]:
    """Split a game's unexplained moves into chunks for batching.

    Default of 40 was sized for Gemini. For a local 7B model over HTTP on a
    laptop, a smaller chunk (e.g. 15-20) is worth trying first -- smaller
    prompts mean faster generation and less risk of the model losing track
    of the schema partway through a long JSON array. Pass chunk_size
    explicitly to annotate_game() to override.
    """
    return [moves[i:i + chunk_size] for i in range(0, len(moves), chunk_size)]


def annotate_game(
    unexplained_moves: List[Dict],
    chunk_size: int = 40,
    retry_missing_individually: bool = True,
    **llm_kwargs,
) -> List[Dict]:
    """End-to-end: chunk -> call -> flatten back into one list of commentary
    dicts in ply order. This is what a caller (e.g. the dashboard, or a
    post-processing script over stored trajectories) should call after
    rule_based_commentary.classify_game() has filtered out the obvious moves.

    llm_kwargs pass through to call_commentary_llm -- e.g.
    annotate_game(moves, provider="ollama", model_name="qwen2.5:7b-instruct")
    or annotate_game(moves, provider="gemini").

    Prints a warning for any move that went into a chunk but has no
    matching entry in what came back. FIX: this used to just print the
    warning and move on, still silently missing those moves in the final
    result. Root cause (see chat writeup) turned out not to be a token
    budget -- even a batch of 2 moves came back with only 1 -- so a smaller
    7B model appears to just stop after one item under JSON-mode decoding
    regardless of chunk size. Rather than trust any batch size to be
    reliable, retry_missing_individually=True (default) automatically
    retries every missing move ALONE (chunk of 1) as a fallback, since a
    single-item array is by far the easiest case for a small model to get
    right. This costs one extra call per dropped move but only runs for
    the moves that actually got dropped.
    """
    all_commentary: List[Dict] = []
    for chunk in chunk_moves(unexplained_moves, chunk_size=chunk_size):
        payload = build_commentary_payload(chunk)
        result = call_commentary_llm(payload, **llm_kwargs)

        by_ply = {m["ply"]: m for m in chunk}
        requested_plies = set(by_ply.keys())
        returned_plies = {c["ply"] for c in result}
        missing = requested_plies - returned_plies
        if missing:
            print(
                f"\n  [Commentary Warning] Asked for commentary on {len(requested_plies)} "
                f"moves in this batch, got {len(returned_plies)} back. Missing plies: "
                f"{sorted(missing)}. Likely cause: the model stopped generating after "
                f"one JSON array entry rather than covering the whole batch (a known "
                f"issue with smaller models under JSON-mode decoding, not necessarily "
                f"a token-budget problem)."
            )
            if retry_missing_individually:
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
                        print(f"    ply {ply}: still no commentary even alone -- giving up on this move.")

        all_commentary.extend(result)
    return sorted(all_commentary, key=lambda c: c["ply"])