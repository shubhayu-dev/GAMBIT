"""
Runs iterative empirical search across dynamically formulated hypotheses.
Tests candidate fixes per blunder before passing results to the LLM.

FIX (see chat writeup): this module previously imported python-chess and
called `agent.select_move(chess.Board(fen), ...)`. Neither exists anywhere
else in this codebase -- engine/README.md is explicit that python-chess was
never a dependency, and every agent in engine/agents.py exposes
`get_move(board: environment.Board, time_budget_ms, seed) -> Move`, not
`select_move`. That made the internal-agent path crash on first use
(ModuleNotFoundError, then AttributeError once chess was installed). Fixed
below to use this project's own Board/get_move directly.

RIGOR NOTE: recovering the exact reference move on ONE position after a
knob bump is a much weaker claim than the project's earlier paired
Wilcoxon test across a whole diagnostic set (n=1, no significance test, and
a position can have more than one objectively fine move -- a miss doesn't
prove the position is unresolved, and a hit doesn't prove the knob is the
real cause). This version keeps the same search strategy but stops calling
the result "proven": every case record below carries a `validation_status`
field, and nothing is labeled resolved without that caveat attached. Actual
validation happens downstream in diagnosis/intervention.py, which checks
whether the dominant fix here generalizes to held-out positions.
"""

import subprocess
from typing import Any, Dict, List, Optional

from environment import Board  # this project's own board -- not python-chess

from diagnosis.hypotheses import discover_agent_parameters, form_prioritized_hypotheses

UNCONFIRMED = "single_position_match (unconfirmed -- see intervention.py for held-out validation)"


def _query_agent(
    agent_target: Any, fen: str, run_kwargs: Dict[str, Any], seed: int = 0
) -> Optional[str]:
    """Invokes either an external UCI engine binary (agent_target is a str
    path) or an internal Agent subclass (agent_target is a class), and
    returns the chosen move as a UCI string, or None if no move was made."""
    if isinstance(agent_target, str):
        proc = subprocess.Popen(
            [agent_target],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )

        def send_cmd(cmd: str):
            proc.stdin.write(f"{cmd}\n")
            proc.stdin.flush()

        send_cmd("uci")
        send_cmd("isready")

        for k, v in run_kwargs.items():
            if k not in ("max_depth", "time_budget_ms"):
                send_cmd(f"setoption name {k} value {v}")

        send_cmd(f"position fen {fen}")

        go_parts = ["go"]
        if "max_depth" in run_kwargs:
            go_parts.append(f"depth {run_kwargs['max_depth']}")
        if "time_budget_ms" in run_kwargs:
            go_parts.append(f"movetime {run_kwargs['time_budget_ms']}")
        send_cmd(" ".join(go_parts))

        chosen_move = None
        try:
            while True:
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("bestmove"):
                    chosen_move = line.split()[1]
                    break
        finally:
            proc.terminate()
        return chosen_move

    # Internal Python agent -- this project's own Board + Agent.get_move.
    board = Board(fen)
    constructor_kwargs = {k: v for k, v in run_kwargs.items() if k != "time_budget_ms"}
    agent = agent_target(**constructor_kwargs)
    time_ms = run_kwargs.get("time_budget_ms", 300)
    move = agent.get_move(board, time_budget_ms=time_ms, seed=seed)
    return move.uci() if move is not None else None


def run_empirical_investigation(
    agent_target: Any,
    blunders: List[Dict[str, Any]],
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """
    Searches, per blunder, for a knob setting that makes the agent play the
    reference move. Returns one record per blunder. Treat `proven_cause` as
    a candidate explanation, not a confirmed one -- check `validation_status`
    (and, for anything you rely on, run diagnosis.intervention.evaluate_intervention
    on held-out data before trusting it).
    """
    discovered_knobs = discover_agent_parameters(agent_target)
    ledger = []

    for b in blunders:
        fen = b["position"]
        ref_move = b["reference_move"]
        original_move = b["chosen_move"]

        case_record = {
            "position": fen,
            "puzzle_id": b.get("puzzle_id"),
            "original_move": original_move,
            "reference_move": ref_move,
            "original_regret": b.get("regret"),
            "investigation_log": [],
            "proven_cause": None,
            "validation_status": "unresolved",
        }

        ranked_hypotheses = form_prioritized_hypotheses(b, discovered_knobs)
        solved = False

        for hyp in ranked_hypotheses:
            if solved:
                break

            if hyp["action"] == "scale_depth":
                for target_depth in hyp["test_sequence"]:
                    result_move = _query_agent(agent_target, fen, {"max_depth": target_depth}, seed=seed)
                    case_record["investigation_log"].append({
                        "test": f"depth={target_depth}", "success": (result_move == ref_move)
                    })
                    if result_move == ref_move:
                        case_record["proven_cause"] = f"Horizon Limitation (matched reference move at depth {target_depth})"
                        case_record["validation_status"] = UNCONFIRMED
                        solved = True
                        break

            elif hyp["action"] == "scale_time":
                for t_budget in hyp["test_sequence"]:
                    result_move = _query_agent(agent_target, fen, {"time_budget_ms": t_budget}, seed=seed)
                    case_record["investigation_log"].append({
                        "test": f"time={t_budget}ms", "success": (result_move == ref_move)
                    })
                    if result_move == ref_move:
                        case_record["proven_cause"] = f"Compute Starvation (matched reference move at {t_budget}ms)"
                        case_record["validation_status"] = UNCONFIRMED
                        solved = True
                        break

            elif hyp["action"] == "perturb_options":
                for eval_knob in hyp["knobs"]:
                    if solved:
                        break
                    test_val = eval_knob.get("max", 100000) if "max" in eval_knob else "material_pst"
                    test_name = eval_knob["name"]
                    result_move = _query_agent(
                        agent_target, fen, {test_name: test_val, "max_depth": b.get("agent_depth", 3)}, seed=seed
                    )
                    case_record["investigation_log"].append({
                        "test": f"{test_name}={test_val}", "success": (result_move == ref_move)
                    })
                    if result_move == ref_move:
                        case_record["proven_cause"] = f"Static Heuristic Flaw (matched reference move via {test_name}={test_val})"
                        case_record["validation_status"] = UNCONFIRMED
                        solved = True
                        break

        if not solved:
            case_record["proven_cause"] = "No tested knob recovered the reference move."
            case_record["validation_status"] = "unresolved"

        ledger.append(case_record)

    return ledger


# ---------------------------------------------------------
# Legacy pipeline compatibility
# ---------------------------------------------------------
# These previously silently no-op'd (`pass` / `return []`), which is worse
# than a crash: a caller could believe run_diagnostic_suite() had actually
# run its A/B tests when nothing happened. They now fail loudly instead,
# pointing at the real replacement -- see run_full_pipeline.py for the
# current wiring.
def run_diagnostic_suite(weak_positions, hypotheses, time_budget_ms=300, seed=100):
    raise NotImplementedError(
        "run_diagnostic_suite() was replaced by run_empirical_investigation(). "
        "Update the caller -- see run_full_pipeline.py."
    )


def rank_hypotheses(evidence_records):
    raise NotImplementedError(
        "rank_hypotheses() was replaced by diagnosis.intervention.extract_dominant_fix(). "
        "Update the caller -- see run_full_pipeline.py."
    )