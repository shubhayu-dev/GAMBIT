"""
Benchmark construction (docs/07_dataset_benchmark_spec.md).

Deviation, flagged per Constitution Rule 7: the spec calls for sampling from
public Lichess games/puzzles. This sandbox has no network access to fetch
that data, so positions here are generated via self-play between baseline
agents instead, then tagged by game phase. This is a placeholder benchmark
for exercising the experiment runner end to end -- swap in real curated
Lichess positions (tactical/positional/defensive/endgame categories per the
spec) once network access is available. Nothing about the runner or schema
below depends on where positions came from.
"""

import random
from typing import List, Dict

from environment import Board, START_FEN
from agents import RandomAgent


def generate_benchmark(n_games: int = 20, seed: int = 0) -> List[Dict]:
    """Self-play random games, sampling one mid-game position per game.
    Returns a list of {"fen": ..., "phase": ..., "source_game": ...} dicts."""
    rng = random.Random(seed)
    positions = []
    for game_idx in range(n_games):
        board = Board(START_FEN)
        agent_w, agent_b = RandomAgent(), RandomAgent()
        plies = rng.randint(6, 50)
        for p in range(plies):
            if board.is_terminal():
                break
            agent = agent_w if board.turn == "w" else agent_b
            move = agent.get_move(board, time_budget_ms=50, seed=seed * 10000 + game_idx * 1000 + p)
            board = board.apply_move(move)
        if not board.is_terminal():
            positions.append({
                "fen": board.fen(),
                "phase": board.game_phase(),
                "source_game": game_idx,
            })
    return positions


def diagnostic_holdout_split(positions: List[Dict], holdout_fraction: float = 0.3, seed: int = 0):
    """Stratified-by-phase split per docs/07: diagnostic set used for hypothesis
    testing, held-out set touched only once, for final validation."""
    rng = random.Random(seed)
    by_phase: Dict[str, List[Dict]] = {}
    for pos in positions:
        by_phase.setdefault(pos["phase"], []).append(pos)

    diagnostic, holdout = [], []
    for phase, items in by_phase.items():
        items = items[:]
        rng.shuffle(items)
        cut = max(1, int(len(items) * holdout_fraction)) if len(items) > 1 else 0
        holdout += items[:cut]
        diagnostic += items[cut:]
    return diagnostic, holdout
