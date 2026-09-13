"""
Neural value model (docs/11_integration_addendum.md, "5. Modern AI Agent").

Two honest notes up front, both about the training target -- see the addendum
for the full write-up:

1. Self-play *game outcome* (win/draw/loss) was tried first, per the
   integration spec's preferred target. At this sandbox's scale (RandomAgent
   self-play, single CPU, no time to run thousands of long games) only ~12%
   of games reach a decisive result within 150 plies -- the rest hit the ply
   cap undecided. That makes outcome-based labels overwhelmingly "draw",
   which isn't a useful training signal. `self_play_data.py` still implements
   this honestly (it's the right target once real games/compute are
   available) but isn't what trains the model below.
2. Instead, the model here is trained to approximate `MaterialAgent`'s
   fixed-depth search evaluation on a broad sample of self-play positions --
   i.e. function approximation / distillation of the classical evaluator,
   not an independently-discovered signal. This is a real, legitimate ML
   technique (it's literally how engines like Stockfish's NNUE were
   originally bootstrapped), but it means this model cannot know anything
   the classical evaluator doesn't already encode. Swap in real game-outcome
   or real Lichess-derived labels once there's enough compute/data for
   outcome labels to not be almost-all-draws.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Tuple

from environment import Board, WHITE
from features import board_to_features, FEATURE_DIM
from agents import Agent, evaluate_position
from search import search_best_move

try:
    from sklearn.neural_network import MLPRegressor
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "neural_value_model.pkl"


def generate_distillation_training_data(positions: List[dict], reference_depth: int = 2,
                                          clip_pawns: float = 20.0) -> Tuple[List[List[float]], List[float]]:
    """Labels each position's feature vector with MaterialAgent's fixed-depth
    evaluation (in pawns, from White's perspective) -- see module docstring
    for why this target was chosen over sparse self-play outcomes.

    Mate scores found within the reference search (~+-99999) are clipped to
    +-clip_pawns: a handful of such outliers in a ~2000-position sample were
    enough to dominate the MLP's squared-error loss and wreck the fit
    everywhere else (train/test R^2 went negative). Clipping is standard
    practice for training on engine evaluations for exactly this reason --
    it says "very good/bad", not a literal pawn count, past that point."""
    X, y = [], []
    for pos in positions:
        board = Board(pos["fen"])
        value = evaluate_position(board, depth=reference_depth)
        value = max(-clip_pawns, min(clip_pawns, value))
        X.append(board_to_features(board))
        y.append(value)
    return X, y


def train_value_network(X: List[List[float]], y: List[float], seed: int = 0) -> "MLPRegressor":
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required to train the neural value model")
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=500,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=15,
    )
    model.fit(X, y)
    return model


def save_model(model, path: Path = MODEL_PATH):
    path.parent.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: Path = MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def neural_evaluate(board: Board, model) -> float:
    """Same signature/scale convention (pawns, White's perspective) as
    engine/evaluation.py's evaluate(), so it's a drop-in eval_fn for the
    existing alpha-beta search in search.py -- no search code changes needed
    to use a learned evaluator instead of a hand-coded one."""
    x = [board_to_features(board)]
    return float(model.predict(x)[0])


class NeuralAgent(Agent):
    """
    The 'modern AI' agent (docs/11_integration_addendum.md). Same
    iterative-deepening alpha-beta search as SearchAgent, but the leaf
    evaluation comes from a trained MLPRegressor instead of hand-coded
    material+PST weights. This isolates exactly one variable (where the
    evaluation numbers come from) so classical-vs-neural comparisons
    (analysis/disagreement.py) are apples-to-apples.
    """

    def __init__(self, model=None, model_path: Optional[Path] = None, max_depth: int = 3):
        self.model = model if model is not None else load_model(model_path or MODEL_PATH)
        self.max_depth = max_depth

    def name(self) -> str:
        return f"NeuralAgent(max_depth={self.max_depth})"

    def config(self) -> dict:
        return {"max_depth": self.max_depth, "eval": "mlp_value_net", "search": "iterative_deepening_alphabeta"}

    def _eval_fn(self, board: Board) -> float:
        return neural_evaluate(board, self.model)

    def get_move(self, board: Board, time_budget_ms: int, seed: Optional[int] = None):
        move, _info = search_best_move(board, self.max_depth, time_budget_ms, self._eval_fn)
        return move

    def raw_value(self, board: Board) -> float:
        """Direct value-net prediction with no search -- used for
        classical/neural disagreement analysis, which compares evaluations,
        not just final moves."""
        return neural_evaluate(board, self.model)
