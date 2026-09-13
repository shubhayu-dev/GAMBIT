import sys
import time
import uuid
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Type

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))
from engine.environment import Board  # noqa: E402
from engine.agents import Agent, MaterialAgent, evaluate_position  # noqa: E402

from experiments.data_store import (  # noqa: E402
    DATA_DIR, init_db, register_agent, register_benchmark, register_experiment,
    register_run, write_trajectory_jsonl,
)


def _evaluate_one_position(args) -> Dict:
    agent_class, agent_kwargs, pos, time_budget_ms, reference_depth, seed, condition = args
    fen = pos["fen"]
    board = Board(fen)
    agent = agent_class(**agent_kwargs)

    eval_before = evaluate_position(board, depth=reference_depth)
    
    t0 = time.time()
    chosen_move = agent.get_move(board, time_budget_ms=time_budget_ms, seed=seed)
    time_used = time.time() - t0
    search_info = getattr(agent, "last_search_info", {}) or {}

    # GROUND TRUTH OVERRIDE: Use Lichess puzzle solution if available, 
    # otherwise fallback to internal MaterialAgent baseline.
    if "reference_move_lichess" in pos:
        uci_str = pos["reference_move_lichess"]
        reference_move = next((m for m in board.legal_moves() if m.uci() == uci_str), None)
    else:
        reference_move, _ = MaterialAgent(depth=reference_depth)._search(
            board, reference_depth, float("-inf"), float("inf")
        )

    next_board = board.apply_move(chosen_move)
    eval_after = evaluate_position(next_board, depth=reference_depth)

    if reference_move is not None and reference_move.uci() != chosen_move.uci():
        reference_board = board.apply_move(reference_move)
        eval_after_reference = evaluate_position(reference_board, depth=reference_depth)
    else:
        eval_after_reference = eval_after  

    CLIP_PAWNS = 20.0
    eval_after_clipped = max(-CLIP_PAWNS, min(CLIP_PAWNS, eval_after))
    eval_after_reference_clipped = max(-CLIP_PAWNS, min(CLIP_PAWNS, eval_after_reference))

    if board.turn == "w":
        regret = eval_after_reference_clipped - eval_after_clipped
        single_move_eval_delta = eval_before - eval_after
    else:
        regret = eval_after_clipped - eval_after_reference_clipped
        single_move_eval_delta = eval_after - eval_before

    return {
        "position": fen,
        "agent": agent.name(),
        "chosen_move": chosen_move.uci() if chosen_move else None,
        "reference_move": reference_move.uci() if reference_move else None,
        "evaluation_before": round(eval_before, 3),
        "evaluation_after": round(eval_after, 3),
        "regret": max(0.0, round(regret, 3)), # Enforce non-negative regret mathematically
        "single_move_eval_delta": round(single_move_eval_delta, 3),
        "search_depth": reference_depth,
        "agent_depth_reached": search_info.get("depth_reached"),
        "agent_nodes_searched": search_info.get("nodes"),
        "time_used": round(time_used, 4),
        "game_phase": board.game_phase(),                 # Internal heuristic (Option C)
        "lichess_phase": pos.get("lichess_phase"),        # Ground truth tag (Option C)
        "puzzle_id": pos.get("puzzle_id"),
        "game_id": pos.get("game_id"),                    # Audit Item 12.3 fix
        "rating": pos.get("rating"),
        "condition": condition,
    }


def run_experiment(agent_class: Type[Agent], agent_kwargs: Optional[dict], positions: List[Dict],
                    condition: str = "default", time_budget_ms: int = 300, reference_depth: int = 2,
                    seed: int = 0, n_workers: int = 4, benchmark_description: str = "self-play sample") -> Dict:
    
    agent_kwargs = agent_kwargs or {}
    # Pass the entire `pos` dictionary into args to retain metadata
    tasks = [
        (agent_class, agent_kwargs, pos, time_budget_ms, reference_depth, seed + i, condition)
        for i, pos in enumerate(positions)
    ]

    with Pool(processes=n_workers) as pool:
        records = pool.map(_evaluate_one_position, tasks)

    mean_regret = sum(r["regret"] for r in records) / len(records) if records else 0.0

    conn = init_db()
    agent_instance = agent_class(**agent_kwargs)
    agent_id = f"{agent_class.__name__}_{hash(frozenset(agent_kwargs.items())) & 0xffff}"
    register_agent(conn, agent_id, agent_instance.name(), agent_instance.config())

    benchmark_id = f"bench_{len(positions)}_{hash(tuple(p['fen'] for p in positions)) & 0xffff}"
    register_benchmark(conn, benchmark_id, benchmark_description, len(positions))

    experiment_id = f"exp_{uuid.uuid4().hex[:8]}"
    register_experiment(conn, experiment_id, agent_id, benchmark_id, condition,
                         time_budget_ms, reference_depth, seed)

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    trajectory_path = DATA_DIR / f"trajectories_{run_id}.jsonl"
    write_trajectory_jsonl(records, trajectory_path)
    register_run(conn, run_id, experiment_id, len(records), mean_regret, str(trajectory_path))
    conn.close()

    return {"records": records, "mean_regret": mean_regret, "trajectory_path": trajectory_path,
            "experiment_id": experiment_id, "run_id": run_id}