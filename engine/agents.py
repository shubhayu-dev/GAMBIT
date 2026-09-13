"""
Baseline agents implementing the Agent interface (docs/06_agent_interface.md).

No Stockfish binary was available in this sandbox (no network access to fetch
it), so `MaterialAgent` stands in as the "reference" agent for now: a fixed-depth
minimax + material/mobility evaluator. It implements the exact same interface a
real Stockfish-UCI wrapper would, so swapping one in later (Week 2+) requires no
changes anywhere else in the pipeline — see the `Agent` base class contract.
"""

import random
import time
from typing import Optional

from environment import Board, Move, WHITE, BLACK
from evaluation import evaluate, evaluate_material_only
from search import search_best_move

PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}


class Agent:
    """Interface every GAMBIT agent must implement (docs/06)."""

    def name(self) -> str:
        raise NotImplementedError

    def get_move(self, board: Board, time_budget_ms: int, seed: Optional[int] = None) -> Move:
        raise NotImplementedError

    def config(self) -> dict:
        return {}


class RandomAgent(Agent):
    """Baseline: picks uniformly among legal moves. Deterministic given a seed."""

    def name(self) -> str:
        return "RandomAgent"

    def get_move(self, board: Board, time_budget_ms: int, seed: Optional[int] = None) -> Move:
        rng = random.Random(seed)
        return rng.choice(board.legal_moves())

    def config(self) -> dict:
        return {}


def material_score(board: Board) -> float:
    """Static evaluation: material balance + light mobility term, from White's POV."""
    score = 0
    for piece in board.board:
        if piece == ".":
            continue
        value = PIECE_VALUES[piece.upper()]
        score += value if piece.isupper() else -value
    mobility = len(board.legal_moves())
    score += mobility if board.turn == WHITE else -mobility
    return score / 100.0  # roughly pawn units


class MaterialAgent(Agent):
    """
    Fixed-depth minimax with alpha-beta pruning over material_score.
    Placeholder reference agent standing in for Stockfish (see module docstring).
    """

    def __init__(self, depth: int = 3):
        self.depth = depth

    def name(self) -> str:
        return f"MaterialAgent(depth={self.depth})"

    def config(self) -> dict:
        return {"depth": self.depth, "eval": "material+mobility"}

    def get_move(self, board: Board, time_budget_ms: int, seed: Optional[int] = None) -> Move:
        best_move, _ = self._search(board, self.depth, float("-inf"), float("inf"))
        return best_move

    def _search(self, board: Board, depth: int, alpha: float, beta: float):
        if depth == 0 or board.is_terminal():
            if board.is_checkmate():
                return None, (-99999 if board.turn == WHITE else 99999)
            return None, material_score(board)

        legal = board.legal_moves()
        maximizing = board.turn == WHITE
        best_move = legal[0]
        best_score = float("-inf") if maximizing else float("inf")

        for move in legal:
            _, score = self._search(board.apply_move(move), depth - 1, alpha, beta)
            if maximizing:
                if score > best_score:
                    best_score, best_move = score, move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score, best_move = score, move
                beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_move, best_score


def evaluate_position(board: Board, depth: int = 3) -> float:
    """Reference evaluation used to compute decision regret (docs/08_metrics_spec.md)."""
    _, score = MaterialAgent(depth=depth)._search(board, depth, float("-inf"), float("inf"))
    return score


class SearchAgent(Agent):
    """
    The real 'agent under test' for GAMBIT (docs/22 MVP scope: "custom
    search-based agent"). Iterative-deepening alpha-beta over a pluggable
    evaluation function (defaults to material+PST+mobility from
    evaluation.py), with basic move ordering. `eval_fn` is swappable so Week 6
    can test the "evaluation weakness" hypothesis by ablating it, and later an
    adaptive-depth intervention can wrap this same class.
    """

    def __init__(self, max_depth: int = 4, eval_fn=None):
        self.max_depth = max_depth
        self.eval_fn = eval_fn if eval_fn is not None else evaluate
        self.last_search_info = {}

    def name(self) -> str:
        eval_name = getattr(self.eval_fn, "__name__", "custom_eval")
        return f"SearchAgent(max_depth={self.max_depth}, eval={eval_name})"

    def config(self) -> dict:
        return {
            "max_depth": self.max_depth,
            "eval": getattr(self.eval_fn, "__name__", "custom_eval"),
            "search": "iterative_deepening_alphabeta",
        }

    def get_move(self, board: Board, time_budget_ms: int, seed: Optional[int] = None) -> Move:
        move, info = search_best_move(board, self.max_depth, time_budget_ms, self.eval_fn)
        self.last_search_info = info
        return move


class AdaptiveSearchAgent(Agent):
    """
    Week 7 intervention: adaptive search depth based on position complexity
    (branching factor), instead of SearchAgent's fixed depth. Complexity
    thresholds and the depth bump are parameters so the diagnostic-set
    tuning (docs/09_experiment_protocol.md) and held-out validation are
    clearly separated -- this class only implements the mechanism.
    """

    def __init__(self, base_depth: int = 3, boosted_depth: int = 5,
                 complexity_threshold: int = 30, eval_fn=None):
        self.base_depth = base_depth
        self.boosted_depth = boosted_depth
        self.complexity_threshold = complexity_threshold
        self.eval_fn = eval_fn if eval_fn is not None else evaluate

    def name(self) -> str:
        return (f"AdaptiveSearchAgent(base={self.base_depth}, boosted={self.boosted_depth}, "
                f"threshold={self.complexity_threshold})")

    def config(self) -> dict:
        return {
            "base_depth": self.base_depth,
            "boosted_depth": self.boosted_depth,
            "complexity_threshold": self.complexity_threshold,
            "eval": getattr(self.eval_fn, "__name__", "custom_eval"),
        }

    def get_move(self, board: Board, time_budget_ms: int, seed: Optional[int] = None) -> Move:
        complexity = len(board.legal_moves())
        depth = self.boosted_depth if complexity >= self.complexity_threshold else self.base_depth
        move, _info = search_best_move(board, depth, time_budget_ms, self.eval_fn)
        return move
