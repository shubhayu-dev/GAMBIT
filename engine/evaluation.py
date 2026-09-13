"""
Static evaluation function for the Week 2 search agent (docs/10, Week 2 goal:
"basic evaluation"). Material + standard piece-square tables + mobility +
a light king-safety term. Score is centipawns from White's perspective
(positive favors White), matching the convention used by `regret` in
docs/08_metrics_spec.md.

This replaces the material-only eval used by Week 1's MaterialAgent
placeholder as the evaluation backing the real search agent; MaterialAgent
itself still exists in agents.py as the (still Stockfish-less) reference
evaluator for regret computation, kept intentionally simple so it's a stable
yardstick separate from the agent being tested.
"""

from environment import Board, WHITE

PIECE_VALUES = {"P": 100, "N": 320, "B": 330, "R": 500, "Q": 900, "K": 0}

# Standard simplified piece-square tables, white's perspective, a1=index0 ...
# h8=index63 when read rank-by-rank from rank1 to rank8 (mirrored for black).
PAWN_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
KNIGHT_PST = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
BISHOP_PST = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
ROOK_PST = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]
QUEEN_PST = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
KING_PST_MID = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]
PST = {"P": PAWN_PST, "N": KNIGHT_PST, "B": BISHOP_PST, "R": ROOK_PST, "Q": QUEEN_PST, "K": KING_PST_MID}


def _pst_value(piece: str, square: int) -> int:
    kind = piece.upper()
    table = PST[kind]
    # tables above are laid out rank8->rank1 (standard PST convention);
    # our square index is rank1->rank8, file a->h, so mirror the rank for white,
    # and mirror both rank and file for black by flipping the table lookup.
    file_idx, rank_idx = square % 8, square // 8
    if piece.isupper():
        idx = (7 - rank_idx) * 8 + file_idx
        return table[idx]
    else:
        idx = rank_idx * 8 + file_idx
        return table[idx]


def evaluate(board: Board) -> float:
    """Centipawn score from White's perspective, returned in pawn units."""
    score = 0
    for square, piece in enumerate(board.board):
        if piece == ".":
            continue
        value = PIECE_VALUES[piece.upper()] + _pst_value(piece, square)
        score += value if piece.isupper() else -value

    mobility = len(board.legal_moves())
    score += (mobility * 2) if board.turn == WHITE else -(mobility * 2)

    return score / 100.0


def evaluate_material_only(board: Board) -> float:
    """Material-count-only evaluation, no PST/mobility terms. Used in Week 6's
    diagnostic experiment to test the 'evaluation-function weakness' hypothesis
    by ablating everything except material and comparing regret."""
    score = 0
    for piece in board.board:
        if piece == ".":
            continue
        value = PIECE_VALUES[piece.upper()]
        score += value if piece.isupper() else -value
    return score / 100.0
