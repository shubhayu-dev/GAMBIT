"""
Board -> feature vector for the neural value model (engine/neural_agent.py).

Updated to support batch tensor construction for sibling batching and GPU acceleration.
"""

from typing import List

try:
    import numpy as np
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

from environment import Board, WHITE

PIECE_ORDER = "PNBRQKpnbrqk"  # 12 planes: white pieces, then black pieces
FEATURE_DIM = 12 * 64 + 1


def board_to_features(board: Board) -> List[float]:
    """769-dim feature vector: 12 one-hot piece planes (64 squares each) +
    1 side-to-move flag. Flattened for the MLP; reshape to (12, 8, 8) for a
    future CNN."""
    features = [0.0] * (12 * 64)
    for square, piece in enumerate(board.board):
        if piece == ".":
            continue
        plane = PIECE_ORDER.index(piece)
        features[plane * 64 + square] = 1.0
    features.append(1.0 if board.turn == WHITE else 0.0)
    return features


def create_feature_batch(boards: List[Board], device: str = "cpu"):
    """
    Converts a list of Board objects into a single stacked PyTorch tensor
    of shape (N, 769) to allow batched GPU evaluation.
    """
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch is required for batch feature extraction.")

    n = len(boards)
    if n == 0:
        return torch.empty((0, FEATURE_DIM), dtype=torch.float32, device=device)

    batch = np.zeros((n, FEATURE_DIM), dtype=np.float32)

    for i, board in enumerate(boards):
        for square, piece in enumerate(board.board):
            if piece != ".":
                plane = PIECE_ORDER.index(piece)
                batch[i, plane * 64 + square] = 1.0

        if board.turn == WHITE:
            batch[i, -1] = 1.0

    return torch.from_numpy(batch).to(device)