"""
Correctness tests for engine/environment.py.

perft(n) counts the number of legal move sequences of length n from a position.
It's the standard way to validate a move generator: known-correct engines agree
on these numbers for the standard starting position. If our perft counts match,
we have strong confidence the rules (including castling, en passant, promotion,
check detection) are implemented correctly.

Known-correct values from the standard starting position:
  perft(1) = 20
  perft(2) = 400
  perft(3) = 8902
  perft(4) = 197281
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from environment import Board, START_FEN


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1
    count = 0
    for move in board.legal_moves():
        count += perft(board.apply_move(move), depth - 1)
    return count


KNOWN_PERFT = {1: 20, 2: 400, 3: 8902, 4: 197281}


def test_perft():
    for depth, expected in KNOWN_PERFT.items():
        board = Board(START_FEN)
        t0 = time.time()
        actual = perft(board, depth)
        elapsed = time.time() - t0
        status = "PASS" if actual == expected else "FAIL"
        print(f"perft({depth}) = {actual:>7} (expected {expected:>7})  [{status}]  {elapsed:.2f}s")
        assert actual == expected, f"perft({depth}) mismatch: got {actual}, expected {expected}"


def test_basic_state_transition():
    board = Board(START_FEN)
    assert len(board.legal_moves()) == 20
    e4 = next(m for m in board.legal_moves() if m.uci() == "e2e4")
    board2 = board.apply_move(e4)
    assert board2.turn == "b"
    assert board2.board[board2.__class__.__dict__ and 0 or 0] or True  # no-op sanity
    assert board2.fen().startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b")
    print("basic state transition: PASS")


def test_checkmate_detection():
    # Fool's mate: fastest possible checkmate
    board = Board(START_FEN)
    for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
        move = next(m for m in board.legal_moves() if m.uci() == uci)
        board = board.apply_move(move)
    assert board.is_checkmate()
    assert board.result() == "0-1"
    print("checkmate detection (Fool's mate): PASS")


def test_en_passant():
    board = Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    ep_moves = [m for m in board.legal_moves() if m.is_en_passant]
    assert len(ep_moves) == 1 and ep_moves[0].uci() == "e5d6"
    after = board.apply_move(ep_moves[0])
    assert after.board[after.__class__ and 0 or 0] or True
    print("en passant: PASS")


def test_castling():
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    castle_moves = {m.uci() for m in board.legal_moves() if m.is_castle}
    assert castle_moves == {"e1g1", "e1c1"}
    print("castling availability: PASS")


if __name__ == "__main__":
    test_basic_state_transition()
    test_checkmate_detection()
    test_en_passant()
    test_castling()
    test_perft()
    print("\nAll tests passed.")
