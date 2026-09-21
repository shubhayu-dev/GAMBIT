"""
Runnable demo: plays a short self-play game with the real SearchAgent,
builds a trajectory in the shape rule_based_commentary/commentary expect,
runs the rule-based filter, and (if GEMINI_API_KEY is set) sends whatever's
left to the LLM commentary layer.

Run from engine/ (needs environment.py, agents.py, evaluation.py, search.py
on the path) with analysis/ also on the path:

    cd GAMBIT-main
    PYTHONPATH=engine:analysis python3 analysis/run_commentary_demo.py

Rule-based part works with zero setup. The LLM part additionally needs:
    pip install google-genai python-dotenv
    export GEMINI_API_KEY=your_key_here
(or a .env file with GEMINI_API_KEY=... in the working directory)

If no key is set, the script still runs and just reports how many moves
*would* go to the LLM, without making the call.
"""

import os
import sys

# FIX: previously this script checked os.environ directly, but the .env
# file was only ever loaded inside commentary.py -- which wasn't imported
# until AFTER this check already needed the key to be present. A .env file
# was silently never picked up; only a real shell `export` worked. Loading
# it here, before the check, fixes that.
from dotenv import load_dotenv
load_dotenv()

from environment import Board, START_FEN
from agents import SearchAgent
from evaluation import evaluate

from rule_based_commentary import classify_game

N_PLIES = 20
SEARCH_DEPTH = 3
TIME_BUDGET_MS = 200


def play_game_and_build_trajectory():
    board = Board(START_FEN)
    white = SearchAgent(max_depth=SEARCH_DEPTH)
    black = SearchAgent(max_depth=SEARCH_DEPTH)

    trajectory = []
    for ply in range(1, N_PLIES + 1):
        if board.is_terminal():
            break
        agent = white if board.turn == "w" else black
        eval_before = evaluate(board)
        fen_before = board.fen()

        move = agent.get_move(board, time_budget_ms=TIME_BUDGET_MS, seed=ply)
        info = agent.last_search_info
        board_after = board.apply_move(move)

        trajectory.append({
            "ply": ply,
            "board_before": board,
            "move": move,
            "board_after": board_after,
            "eval_before": eval_before,
            "eval_after": info.get("score", evaluate(board_after)),
            "fen_before": fen_before,
            # pv is already a list of UCI strings (search_best_move's "pv" field);
            # pv[0] is the move itself, the rest is the expected continuation.
            "pv_from_here": info.get("pv", [])[1:],
            "depth_reached": info.get("depth_reached", SEARCH_DEPTH),
            # Live-captured comparison data -- see search.py's alpha_beta
            # root_trace docstring. This is what the SAME time-bounded call
            # that chose `move` actually compared it against, not a
            # reconstruction with different (unlimited-time) rules.
            "root_trace": info.get("root_trace", []),
            "n_legal_at_root": info.get("n_legal_at_root"),
        })
        board = board_after

    return trajectory


def main():
    print(f"Playing a {N_PLIES}-ply self-play game (SearchAgent depth={SEARCH_DEPTH})...")
    trajectory = play_game_and_build_trajectory()
    print(f"Game produced {len(trajectory)} plies.\n")

    results = classify_game(trajectory)

    handled = [r for r in results if r.handled]
    unhandled = [r for r in results if not r.handled]

    print(f"Rule-based pass handled {len(handled)}/{len(results)} moves without any LLM call:")
    for r in handled:
        print(f"  ply {r.ply:>2} {r.move_uci:>6}  [{r.classification}]  {r.explanation}")

    print(f"\n{len(unhandled)} moves need LLM commentary:")
    for r in unhandled:
        print(f"  ply {r.ply:>2} {r.move_uci:>6}  (no rule matched)")

    if not unhandled:
        print("\nNothing to send to the LLM this game -- done.")
        return

    # Using a local Ollama server (qwen2.5) instead of Gemini -- no API key
    # needed, but the server itself needs to be running.
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=2)
    except Exception:
        print(
            "\nCan't reach Ollama at http://localhost:11434 -- skipping the LLM call. "
            "Start it with `ollama serve` (and make sure you've pulled a model, e.g. "
            "`ollama pull qwen2.5:7b-instruct`), then rerun."
        )
        return

    from commentary import annotate_game

    unhandled_plies = {r.ply for r in unhandled}
    trajectory_by_ply = {step["ply"]: step for step in trajectory}

    llm_input = []
    for ply in sorted(unhandled_plies):
        step = trajectory_by_ply[ply]
        llm_input.append({
            "ply": step["ply"],
            "move": step["move"].uci(),
            "fen_before": step["fen_before"],
            "eval_before": step["eval_before"],
            "eval_after": step["eval_after"],
            "pv_from_here": step["pv_from_here"],
            # Live-captured root_trace, not a reconstruction -- see
            # search.py/commentary.py for why that distinction matters.
            "alternatives": step["root_trace"],
            "n_legal_at_root": step["n_legal_at_root"],
        })

    print("\nCalling the local Ollama commentary layer (qwen2.5)...")
    # Smaller chunk_size than Gemini's default -- see chunk_moves()'s
    # docstring: faster generation, less risk of a small local model losing
    # the JSON schema partway through a long batch. Tune based on how your
    # model actually does with 15 vs. the full game in one call.
    commentary = annotate_game(
        llm_input, provider="ollama", model_name="qwen2.5:7b-instruct", chunk_size=15,
    )
    print("\nLLM commentary:")
    for c in commentary:
        print(f"  ply {c['ply']:>2} {c['move']:>6}  [{c.get('tag')}]  {c['explanation']}")


if __name__ == "__main__":
    sys.exit(main())