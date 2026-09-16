"""
PyTorch Neural Value Model (docs/11_integration_addendum.md).

This agent uses a deep neural network to evaluate chess positions statically.
Unlike the classical evaluator (which relies on search depth to see tactics),
this model is trained to recognize positional patterns, king safety, and 
long-term compensation instantly.

Updated to support batch GPU evaluation for high-throughput search trees.
"""

import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from environment import Board
from features import board_to_features, create_feature_batch, FEATURE_DIM
from agents import Agent, evaluate_position
from search import search_best_move

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "pytorch_value_net.pth"


class ChessValueNet(nn.Module):
    """
    A robust Feed-Forward Neural Network for static board evaluation.
    This architecture is heavily regularized to prevent memorization and 
    deep enough to learn complex positional heuristics natively.
    """
    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.fc4 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


def neural_batch_evaluate(boards: List[Board], model: Optional[nn.Module], device: str = "cpu") -> List[float]:
    """
    Evaluates a batch of positions simultaneously on the GPU.
    Returns a list of float evaluation scores (in pawn units).
    """
    if not HAVE_TORCH or model is None or len(boards) == 0:
        return [0.0] * len(boards)

    model.eval()
    with torch.no_grad():
        x = create_feature_batch(boards, device=device)
        preds = model(x)
        return preds.view(-1).cpu().tolist()


def neural_evaluate(board: Board, model: Optional[nn.Module], device: str = "cpu") -> float:
    """
    Generates a static evaluation of a single board using the PyTorch model.
    """
    scores = neural_batch_evaluate([board], model, device=device)
    return scores[0] if scores else 0.0


class NeuralAgent(Agent):
    """
    The neural agent. Uses iterative-deepening alpha-beta search accelerated 
    by batched GPU evaluations over leaf nodes and sibling branches.
    """

    def __init__(self, model_path: Optional[Path] = None, max_depth: int = 3):
        super().__init__()
        self.max_depth = max_depth
        self.device = "cuda" if HAVE_TORCH and torch.cuda.is_available() else "cpu"
        self.last_search_info = {}
        
        if not HAVE_TORCH:
            print("WARNING: PyTorch not installed. NeuralAgent will return 0.0 eval.")
            self.model = None
        else:
            self.model = ChessValueNet().to(self.device)
            path = model_path or MODEL_PATH
            if path.exists():
                self.model.load_state_dict(torch.load(path, map_location=self.device))
                self.model.eval()
            else:
                print(f"WARNING: No trained weights found at {path}. Model is untrained.")

    def name(self) -> str:
        return f"NeuralAgent(PyTorch, max_depth={self.max_depth})"

    def config(self) -> dict:
        return {
            "max_depth": self.max_depth, 
            "eval": "pytorch_value_net", 
            "search": "iterative_deepening_alphabeta_batched",
            "device": self.device,
        }

    def _eval_fn(self, board: Board) -> float:
        return neural_evaluate(board, self.model, self.device)

    def _batch_eval_fn(self, boards: List[Board]) -> List[float]:
        return neural_batch_evaluate(boards, self.model, self.device)

    def get_move(self, board: Board, time_budget_ms: int, seed: Optional[int] = None):
        move, info = search_best_move(
            board,
            self.max_depth,
            time_budget_ms,
            eval_fn=self._eval_fn,
            batch_eval_fn=self._batch_eval_fn,
        )
        self.last_search_info = info
        return move

    def raw_value(self, board: Board) -> float:
        """Direct single prediction with no tree search."""
        return neural_evaluate(board, self.model, self.device)

    def raw_batch_values(self, boards: List[Board]) -> List[float]:
        """Direct batch predictions with no tree search."""
        return neural_batch_evaluate(boards, self.model, self.device)