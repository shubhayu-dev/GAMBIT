"""
Board -> feature vector for the neural value model (engine/neural_agent.py).

Deviation, flagged per Constitution Rule 7 (see docs/11_integration_addendum.md):
the integration spec calls for a CNN over 8x8 feature planes. No `torch`/
`tensorflow` is installable in this sandbox (no network), so the model here
is an sklearn MLPRegressor instead -- a real trained neural network, just not
convolutional. To keep the *input representation* faithful to the spec
anyway, boards are still encoded as 12 one-hot piece planes (a real CNN could
consume this same encoding unchanged, reshaped to 12x8x8, whenever torch
becomes available) rather than hand-picked scalar features.
"""

from typing import List

from environment import Board, WHITE

PIECE_ORDER = "PNBRQKpnbrqk"  # 12 planes: white pieces, then black pieces


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


FEATURE_DIM = 12 * 64 + 1
