"""
Search: iterative deepening alpha-beta with simple move ordering.
Week 2 goal per docs/10_development_plan.md: "Position -> search -> decision.
Minimax, alpha-beta, basic evaluation."
"""

import time
from typing import Optional, Tuple

from environment import Board, Move, WHITE

MATE_SCORE = 100000


def _move_order_key(board: Board, move: Move) -> int:
    """Rough MVV-LVA-style ordering: try captures of valuable pieces first,
    with cheap attackers preferred, so alpha-beta prunes more branches sooner."""
    target = board.board[move.to_sq]
    if target == ".":
        return 0
    from environment import Board as _B
    victim_value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}.get(target.upper(), 0)
    attacker = board.board[move.from_sq]
    attacker_value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}.get(attacker.upper(), 0)
    return victim_value * 10 - attacker_value


def alpha_beta(board: Board, depth: int, alpha: float, beta: float,
               eval_fn, deadline: Optional[float] = None, node_counter: Optional[list] = None) -> Tuple[Optional[Move], float]:
    if node_counter is not None:
        node_counter[0] += 1

    if deadline is not None and time.time() > deadline:
        return None, eval_fn(board)

    if board.is_checkmate():
        return None, (-MATE_SCORE - depth if board.turn == WHITE else MATE_SCORE + depth)
    if board.is_stalemate() or board.halfmove >= 100:
        return None, 0.0
    if depth == 0:
        return None, eval_fn(board)

    legal = sorted(board.legal_moves(), key=lambda m: -_move_order_key(board, m))
    maximizing = board.turn == WHITE
    best_move = legal[0]
    best_score = float("-inf") if maximizing else float("inf")

    for move in legal:
        _, score = alpha_beta(board.apply_move(move), depth - 1, alpha, beta, eval_fn, deadline, node_counter)
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
        if deadline is not None and time.time() > deadline:
            break

    return best_move, best_score


def search_best_move(board: Board, max_depth: int, time_budget_ms: int, eval_fn) -> Tuple[Move, dict]:
    """Iterative deepening: search depth 1, 2, 3... until time budget runs out,
    always keeping the best move found by the last fully-completed depth.
    Returns (move, info) where info logs the depth actually reached and total
    nodes visited across all iterations (docs/12_audit_report.md section 9
    asks for node counts alongside regret/time for controlled experiments)."""
    deadline = time.time() + time_budget_ms / 1000.0
    best_move, best_score, depth_reached = None, 0.0, 0
    node_counter = [0]

    for depth in range(1, max_depth + 1):
        if time.time() > deadline:
            break
        move, score = alpha_beta(board, depth, float("-inf"), float("inf"), eval_fn, deadline, node_counter)
        if move is not None:
            best_move, best_score, depth_reached = move, score, depth
        if time.time() > deadline:
            break

    if best_move is None:  # depth 1 didn't even finish (extreme time pressure) - fall back
        legal = board.legal_moves()
        best_move = legal[0]

    return best_move, {"depth_reached": depth_reached, "score": best_score, "nodes": node_counter[0]}
