"""
Rule-based first pass for move commentary.

Goal: cheaply explain the moves that don't need an LLM at all (forced moves,
only-legal-move, simple recaptures, obvious mate-in-1s), so the batched LLM
call downstream only has to handle the genuinely ambiguous positional/tactical
moves. This mirrors the project's "don't reach for ML when a heuristic works"
stance (PROJECT_CONSTITUTION Rule 3).

A move here is a plain record built from one ply of a trajectory:
    {
        "ply": int,
        "board_before": Board,     # engine.environment.Board, pre-move
        "move": Move,               # engine.environment.Move actually played
        "board_after": Board,       # post-move
        "eval_before": float,       # centipawns, from White's perspective
        "eval_after": float,
    }

Everything this module needs (legal_moves, in_check, is_checkmate, capture
detection via board.board[to_sq]) already exists on engine.environment.Board —
no new engine surface required.
"""

from dataclasses import dataclass
from typing import List, Optional

from engine.environment import Board, Move, WHITE

# Centipawn swing below which a move is never flagged as "interesting"
# regardless of what triggered the check, since it's noise at this engine's
# search depth. Tune against real game data before trusting this number.
QUIET_EVAL_SWING_CP = 40

PIECE_NAMES = {
    "p": "pawn", "n": "knight", "b": "bishop",
    "r": "rook", "q": "queen", "k": "king",
}


@dataclass
class RuleBasedResult:
    ply: int
    move_uci: str
    explanation: str
    classification: str  # "forced" | "only_legal" | "recapture" | "mate" | None
    handled: bool  # False means: send this one to the LLM


def _piece_name(board: Board, square: int) -> str:
    piece = board.board[square]
    return PIECE_NAMES.get(piece.lower(), "piece")


def _is_capture(board_before: Board, move: Move) -> bool:
    target = board_before.board[move.to_sq]
    return target != "." or move.is_en_passant


def classify_move(
    ply: int,
    board_before: Board,
    move: Move,
    board_after: Board,
    eval_before: float,
    eval_after: float,
    prior_move_to_sq: Optional[int] = None,
) -> RuleBasedResult:
    """Try to explain this move with a rule. Sets handled=False if none apply,
    signalling the move should go into the LLM batch instead."""

    uci = move.uci()
    swing = abs(eval_after - eval_before)
    mover_color = board_before.turn
    was_in_check = board_before.in_check(mover_color)
    legal = board_before.legal_moves()

    # 1. Only legal move in the position (forced regardless of quality).
    if len(legal) == 1:
        return RuleBasedResult(
            ply=ply, move_uci=uci,
            explanation="Only legal move in the position.",
            classification="only_legal", handled=True,
        )

    # 2. Forced response to check with a small, obviously-forced reply set
    #    (king move or block/capture of the checking piece) and no material
    #    swing worth explaining.
    if was_in_check and swing < QUIET_EVAL_SWING_CP:
        return RuleBasedResult(
            ply=ply, move_uci=uci,
            explanation="Forced reply to check; no meaningful alternative.",
            classification="forced", handled=True,
        )

    # 3. Immediate recapture on the square the opponent's last move landed on,
    #    for roughly equal material, with a quiet eval swing.
    if (
        prior_move_to_sq is not None
        and move.to_sq == prior_move_to_sq
        and _is_capture(board_before, move)
        and swing < QUIET_EVAL_SWING_CP
    ):
        piece = _piece_name(board_before, move.from_sq)
        return RuleBasedResult(
            ply=ply, move_uci=uci,
            explanation=f"Recaptures with the {piece}; restores material balance.",
            classification="recapture", handled=True,
        )

    # 4. Delivers checkmate.
    if board_after.is_checkmate():
        return RuleBasedResult(
            ply=ply, move_uci=uci,
            explanation="Checkmate.",
            classification="mate", handled=True,
        )

    # Nothing obvious applies — this is exactly the kind of move worth an
    # LLM's explanation (a real choice among alternatives, or a swing big
    # enough to matter).
    return RuleBasedResult(
        ply=ply, move_uci=uci, explanation="", classification=None, handled=False,
    )


def classify_game(trajectory: List[dict]) -> List[RuleBasedResult]:
    """Run classify_move over a full game trajectory (list of per-ply dicts
    as described in the module docstring, in play order). Returns one
    RuleBasedResult per ply; unhandled ones (handled=False) are what you
    batch into the LLM commentary call."""

    results = []
    prior_to_sq = None
    for step in trajectory:
        result = classify_move(
            ply=step["ply"],
            board_before=step["board_before"],
            move=step["move"],
            board_after=step["board_after"],
            eval_before=step["eval_before"],
            eval_after=step["eval_after"],
            prior_move_to_sq=prior_to_sq,
        )
        results.append(result)
        prior_to_sq = step["move"].to_sq
    return results