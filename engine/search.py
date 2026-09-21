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
    is_root: bool = False,
    root_trace: Optional[list] = None,
) -> Tuple[Optional[Move], float, List[Move]]:
    """Alpha-beta search with sibling GPU batching at the leaf frontier.

    is_root/root_trace: when is_root=True, appends one entry per root move
    actually processed to root_trace (a list the caller provides), as
    {"move": Move, "score": float, "fully_searched": bool}. This captures
    what THIS SAME live, time-bounded call actually compared -- as opposed
    to reconstructing a comparison afterward with a fresh, unconstrained
    search (see evaluate_root_candidates below, and the chat writeup: those
    two answer different questions -- "what did the agent's real decision
    weigh" vs. "what's actually best given unlimited time" -- and conflating
    them produced a misleading comparison where the played move looked like
    it lost to alternatives that, in the live game, the search never
    actually got time to reach or full-depth evaluate).
    `fully_searched=False` means either the deadline had already passed
    before this move's subtree was searched (its score is an immediate
    static eval, not a real depth-based one), or the deadline was crossed
    partway through it (its score may reflect a truncated subtree).
    Moves never reached at all (because an earlier move's alpha-beta cutoff
    or the deadline stopped the loop first) simply have no entry -- their
    absence from root_trace IS the information "the search never compared
    this move to the one it chose."
    """
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
            if is_root and root_trace is not None:
                root_trace.append({"move": move, "score": float(score), "fully_searched": True})
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
        # Recorded BEFORE the recursive call: if the deadline had already
        # passed, this move's search never really happened -- the recursive
        # call below will hit the deadline check at its own entry and
        # return an immediate static eval, not a real depth-based score.
        already_late = deadline is not None and time.time() > deadline

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

        if is_root and root_trace is not None:
            # Recorded AFTER too: if the deadline passed anywhere during
            # this move's own subtree, its score may reflect a subtree that
            # was truncated partway through, not a genuine depth-(depth-1)
            # value -- mark it un-fully-searched either way.
            now_late = deadline is not None and time.time() > deadline
            root_trace.append({
                "move": move,
                "score": float(score),
                "fully_searched": not (already_late or now_late),
            })

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


def evaluate_root_candidates(
    board: Board,
    depth: int,
    eval_fn: Callable[[Board], float],
    deadline: Optional[float] = None,
    batch_eval_fn: Optional[Callable[[List[Board]], List[float]]] = None,
) -> List[Tuple[Move, float]]:
    """Scores every legal root move independently with a FULL, unconstrained
    alpha-beta window per move (no shared deadline, no shared pruning across
    siblings), returning [(move, score), ...] sorted best-first.

    IMPORTANT -- this answers "what's actually the best move at this depth
    given unlimited time," which is a DIFFERENT question from "what did the
    agent's real, time-bounded search actually compare before choosing."
    For the latter -- i.e. for move commentary explaining the agent's real
    decision -- use search_best_move()'s `info["root_trace"]` instead, which
    is captured live, inside the same deadline-bound call that picked the
    move. Conflating the two produced a misleading comparison during
    development (see chat writeup): a move that looks like it "lost" to
    something in this function's output may simply never have been reached
    at all by the live, time-pressured search, which is a different and
    more relevant fact than "it scored lower."

    Still useful for: checking whether a time-budget increase would help
    (compare this function's top pick against what the agent actually
    played), or other retrospective what-if analysis -- just don't present
    its output as "what the agent considered."

    Costs roughly (branching factor) times a single alpha_beta call at this
    depth. Not intended for use during live play/search.
    """
    legal = board.legal_moves()
    maximizing = board.turn == WHITE
    results = []

    for move in legal:
        next_board = board.apply_move(move)
        _, score, _ = alpha_beta(
            next_board,
            depth - 1,
            float("-inf"),
            float("inf"),
            eval_fn,
            deadline,
            node_counter=None,
            batch_eval_fn=batch_eval_fn,
        )
        results.append((move, score))

    results.sort(key=lambda pair: pair[1], reverse=maximizing)
    return results


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
      - root_trace: List of {"move", "score", "fully_searched"} for every root
        move THIS call actually compared before picking best_move -- captured
        live, from the same deadline-bound search, not reconstructed
        afterward. A move played but not the top score in root_trace is not
        a contradiction: alpha-beta cutoffs and the time budget mean not
        every legal move gets compared. See evaluate_root_candidates() for
        the separate (and NOT equivalent) question of what an unconstrained
        search would have found.
      - n_legal_at_root: total legal moves available, for comparing against
        len(root_trace) to see how much of the option space was actually
        considered under the time budget.
    """
    deadline = time.time() + time_budget_ms / 1000.0
    best_move, best_score, depth_reached = None, 0.0, 0
    best_pv: List[Move] = []
    best_root_trace: List[dict] = []
    n_legal_at_root = len(board.legal_moves())
    node_counter = [0]

    for depth in range(1, max_depth + 1):
        if time.time() > deadline:
            break
        this_depth_trace: list = []
        move, score, pv = alpha_beta(
            board,
            depth,
            float("-inf"),
            float("inf"),
            eval_fn,
            deadline,
            node_counter,
            batch_eval_fn=batch_eval_fn,
            is_root=True,
            root_trace=this_depth_trace,
        )
        if move is not None:
            best_move, best_score, depth_reached, best_pv = move, score, depth, pv
            # This is the trace from the SAME call that produced best_move --
            # i.e. exactly what the live search compared before choosing it,
            # deadline truncation and all. See alpha_beta's root_trace
            # docstring for why this must come from this call, not a
            # separate reconstruction.
            best_root_trace = this_depth_trace
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

    # Serialize root_trace's Move objects to UCI strings for JSON-friendliness
    # downstream (analysis/commentary.py consumes this directly). n_legal
    # is included so a consumer can tell "we compared 4 of 27 legal moves"
    # apart from "we compared 4 of 4" -- the difference between a search
    # that ran out of time and one that genuinely had few options.
    root_trace_serialized = [
        {
            "move": _move_to_str(entry["move"]),
            "score": round(entry["score"], 3),
            "fully_searched": entry["fully_searched"],
        }
        for entry in best_root_trace
    ]

    info = {
        "depth_reached": depth_reached,
        "score": round(best_score, 3) if isinstance(best_score, float) else best_score,
        "nodes": node_counter[0],
        "root_trace": root_trace_serialized,
        "n_legal_at_root": n_legal_at_root,
        "pv": pv_str_list,
        "pv_line": " ".join(pv_str_list),
        "pv_trace": pv_trace,
    }

    return best_move, info