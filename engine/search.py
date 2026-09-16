"""
Search: iterative deepening alpha-beta with move ordering,
Principal Variation (PV) extraction, and Sibling Batching for GPU acceleration.
"""

import time
from typing import Callable, List, Optional, Tuple

from environment import Board, Move, WHITE, BLACK

MATE_SCORE = 100000


def _move_to_str(move: Move) -> str:
    """Safe string serializer for moves across chess library representations."""
    if hasattr(move, "uci"):
        return move.uci()
    return str(move)


def _move_order_key(board: Board, move: Move) -> int:
    """Rough MVV-LVA-style ordering: try captures of valuable pieces first,
    with cheap attackers preferred, so alpha-beta prunes branches sooner."""
    target = board.board[move.to_sq]
    if target == ".":
        return 0
    victim_value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}.get(target.upper(), 0)
    attacker = board.board[move.from_sq]
    attacker_value = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}.get(attacker.upper(), 0)
    return victim_value * 10 - attacker_value


def alpha_beta(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    eval_fn: Callable[[Board], float],
    deadline: Optional[float] = None,
    node_counter: Optional[list] = None,
    batch_eval_fn: Optional[Callable[[List[Board]], List[float]]] = None,
) -> Tuple[Optional[Move], float, List[Move]]:
    """Alpha-beta search with sibling GPU batching at the leaf frontier."""
    if node_counter is not None:
        node_counter[0] += 1

    if deadline is not None and time.time() > deadline:
        return None, eval_fn(board), []

    if board.is_checkmate():
        score = -MATE_SCORE - depth if board.turn == WHITE else MATE_SCORE + depth
        return None, score, []
    if board.is_stalemate() or board.halfmove >= 100:
        return None, 0.0, []
    if depth == 0:
        return None, eval_fn(board), []

    legal = sorted(board.legal_moves(), key=lambda m: -_move_order_key(board, m))
    if not legal:
        return None, eval_fn(board), []

    maximizing = board.turn == WHITE

    # --- Sibling GPU Batching at Frontier (depth == 1) ---
    if depth == 1 and batch_eval_fn is not None:
        child_boards = [board.apply_move(m) for m in legal]
        scores = [0.0] * len(legal)
        eval_indices = []
        eval_boards = []

        # Filter out terminal states before GPU dispatch
        for idx, cb in enumerate(child_boards):
            if cb.is_checkmate():
                scores[idx] = float(MATE_SCORE if cb.turn == BLACK else -MATE_SCORE)
            elif cb.is_stalemate() or cb.halfmove >= 100:
                scores[idx] = 0.0
            else:
                eval_indices.append(idx)
                eval_boards.append(cb)

        # Single batched forward pass on GPU
        if eval_boards:
            batch_scores = batch_eval_fn(eval_boards)
            for idx, score_val in zip(eval_indices, batch_scores):
                scores[idx] = float(score_val)

        if node_counter is not None:
            node_counter[0] += len(legal)

        best_score = float("-inf") if maximizing else float("inf")
        best_move = legal[0]

        for move, score in zip(legal, scores):
            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, best_score)
            if beta <= alpha:
                break

        return best_move, best_score, [best_move]

    # --- Standard Alpha-Beta Search (depth > 1 or classical fallback) ---
    best_move = legal[0]
    best_pv: List[Move] = [best_move]
    best_score = float("-inf") if maximizing else float("inf")

    for move in legal:
        next_board = board.apply_move(move)
        _, score, child_pv = alpha_beta(
            next_board,
            depth - 1,
            alpha,
            beta,
            eval_fn,
            deadline,
            node_counter,
            batch_eval_fn=batch_eval_fn,
        )

        if maximizing:
            if score > best_score:
                best_score = score
                best_move = move
                best_pv = [move] + child_pv
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score = score
                best_move = move
                best_pv = [move] + child_pv
            beta = min(beta, best_score)

        if beta <= alpha:
            break
        if deadline is not None and time.time() > deadline:
            break

    return best_move, best_score, best_pv


def search_best_move(
    board: Board,
    max_depth: int,
    time_budget_ms: int,
    eval_fn: Callable[[Board], float],
    batch_eval_fn: Optional[Callable[[List[Board]], List[float]]] = None,
) -> Tuple[Move, dict]:
    """Iterative deepening: search depth 1, 2, 3... until time budget runs out.

    Returns (best_move, info), where info contains:
      - depth_reached
      - score
      - nodes
      - pv: List of move strings in the principal variation
      - pv_line: Space-separated PV string
      - pv_trace: List of {"move": ..., "eval": ...} tracking evaluation per ply
    """
    deadline = time.time() + time_budget_ms / 1000.0
    best_move, best_score, depth_reached = None, 0.0, 0
    best_pv: List[Move] = []
    node_counter = [0]

    for depth in range(1, max_depth + 1):
        if time.time() > deadline:
            break
        move, score, pv = alpha_beta(
            board,
            depth,
            float("-inf"),
            float("inf"),
            eval_fn,
            deadline,
            node_counter,
            batch_eval_fn=batch_eval_fn,
        )
        if move is not None:
            best_move, best_score, depth_reached, best_pv = move, score, depth, pv
        if time.time() > deadline:
            break

    if best_move is None:
        legal = board.legal_moves()
        best_move = legal[0] if legal else None
        best_pv = [best_move] if best_move else []
        best_score = eval_fn(board)

    # Step-by-step evaluation trace along the expected PV line
    pv_trace = []
    curr_board = board
    for m in best_pv:
        if m is None:
            continue
        try:
            curr_board = curr_board.apply_move(m)
            pv_trace.append({
                "move": _move_to_str(m),
                "eval": round(float(eval_fn(curr_board)), 3),
            })
        except Exception:
            break

    pv_str_list = [_move_to_str(m) for m in best_pv if m is not None]

    info = {
        "depth_reached": depth_reached,
        "score": round(best_score, 3) if isinstance(best_score, float) else best_score,
        "nodes": node_counter[0],
        "pv": pv_str_list,
        "pv_line": " ".join(pv_str_list),
        "pv_trace": pv_trace,
    }

    return best_move, info