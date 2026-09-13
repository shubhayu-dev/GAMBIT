"""
Generates (position, outcome) training pairs for the neural value model by
running fast self-play games with RandomAgent, per the integration spec's
"value target: game result" option (docs/11_integration_addendum.md, section
6 of the pasted design doc). This is a genuinely learned target -- not a
distillation of the hand-coded classical evaluator -- even though it's
self-play data rather than real Lichess games (same network-access
deviation as the rest of the project; see docs/11 for the swap-in plan once
network access exists).
"""

import random
from typing import Dict, List, Tuple

from environment import Board, START_FEN
from agents import RandomAgent


def generate_value_training_data(n_games: int = 250, max_plies: int = 60,
                                  sample_every: int = 2, seed: int = 42) -> Tuple[List[str], List[float]]:
    """
    Returns (fens, targets) where each target is the eventual game result
    from the perspective of the side to move AT that recorded position:
    +1.0 win, 0.0 draw, -1.0 loss (unfinished games at max_plies are
    scored as a draw, i.e. 0.0 -- a coarse but honest choice, not a win).
    """
    rng = random.Random(seed)
    all_fens, all_targets = [], []

    for game_idx in range(n_games):
        board = Board(START_FEN)
        agent_w, agent_b = RandomAgent(), RandomAgent()
        game_positions = []  # (fen, side_to_move)

        ply = 0
        while not board.is_terminal() and ply < max_plies:
            if ply % sample_every == 0:
                game_positions.append((board.fen(), board.turn))
            agent = agent_w if board.turn == "w" else agent_b
            move = agent.get_move(board, time_budget_ms=20, seed=seed * 100000 + game_idx * 1000 + ply)
            board = board.apply_move(move)
            ply += 1

        result = board.result()  # '1-0' / '0-1' / '1/2-1/2' / None (unfinished)
        for fen, stm in game_positions:
            if result == "1-0":
                target = 1.0 if stm == "w" else -1.0
            elif result == "0-1":
                target = -1.0 if stm == "w" else 1.0
            else:  # draw or unfinished
                target = 0.0
            all_fens.append(fen)
            all_targets.append(target)

    return all_fens, all_targets


def generate_value_training_data_with_game_ids(n_games: int = 250, max_plies: int = 60,
                                                sample_every: int = 2, seed: int = 42) -> List[Dict]:
    """
    Audit fix (docs/12_audit_report.md section 4 / docs/14_final_audit_summary.md
    Step 1): same generation as generate_value_training_data, but returns
    each position tagged with its source game_idx, so callers can split
    train/val/test at the GAME level rather than the position level.
    Positions from the same self-play game are highly correlated (adjacent
    plies of one game share almost the entire board state) -- splitting
    individual positions randomly, as the original training pipeline did,
    lets near-duplicate positions from the same game land in both train and
    test, inflating apparent generalization.
    """
    rng = random.Random(seed)
    records = []

    for game_idx in range(n_games):
        board = Board(START_FEN)
        agent_w, agent_b = RandomAgent(), RandomAgent()
        ply = 0
        while not board.is_terminal() and ply < max_plies:
            if ply % sample_every == 0:
                records.append({"fen": board.fen(), "side_to_move": board.turn, "game_idx": game_idx})
            agent = agent_w if board.turn == "w" else agent_b
            move = agent.get_move(board, time_budget_ms=20, seed=seed * 100000 + game_idx * 1000 + ply)
            board = board.apply_move(move)
            ply += 1

    return records


def game_level_split(records: List[Dict], train_frac: float = 0.7, val_frac: float = 0.15,
                      seed: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Splits records into train/val/test by game_idx, not by individual
    position, so no game's positions appear in more than one split."""
    rng = random.Random(seed)
    game_ids = sorted(set(r["game_idx"] for r in records))
    rng.shuffle(game_ids)

    n_train = int(len(game_ids) * train_frac)
    n_val = int(len(game_ids) * val_frac)
    train_ids = set(game_ids[:n_train])
    val_ids = set(game_ids[n_train:n_train + n_val])
    test_ids = set(game_ids[n_train + n_val:])

    train = [r for r in records if r["game_idx"] in train_ids]
    val = [r for r in records if r["game_idx"] in val_ids]
    test = [r for r in records if r["game_idx"] in test_ids]
    return train, val, test
