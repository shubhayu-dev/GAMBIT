"""
Runs iterative empirical search across dynamically formulated hypotheses.
Tests candidate fixes per blunder before passing results to the LLM.

FIX:
1. Safely checks agent constructor parameters via inspect.signature before passing
   probe arguments (like checkmate_value or use_quiescence) to avoid TypeError crashes.
2. Adds Control Re-Run (Artifact Filter), Checkmate Scoring, and Quiescence probes
   only when supported or handled gracefully.
"""

import inspect
import subprocess
from typing import Any, Dict, List, Optional

from engine.environment import Board
from diagnosis.hypotheses import discover_agent_parameters, form_prioritized_hypotheses

UNCONFIRMED = "single_position_match (unconfirmed -- see intervention.py for held-out validation)"


def _query_agent(
    agent_target: Any, fen: str, run_kwargs: Dict[str, Any], seed: int = 0
) -> Optional[str]:
    """Invokes either an external UCI engine binary or an internal Agent subclass."""
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

    # Internal Python agent -- check supported kwargs before initialization
    board = Board(fen)
    constructor_kwargs = {k: v for k, v in run_kwargs.items() if k != "time_budget_ms"}
    
    # Filter kwargs to only what agent_target.__init__ accepts
    sig = inspect.signature(agent_target.__init__)
    accepted_params = sig.parameters
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in accepted_params.values())

    valid_kwargs = {}
    attr_overrides = {}

    for k, v in constructor_kwargs.items():
        if has_var_kwargs or k in accepted_params:
            valid_kwargs[k] = v
        else:
            # Stash as attribute override in case agent sets it post-init
            attr_overrides[k] = v

    try:
        agent = agent_target(**valid_kwargs)
        for attr, val in attr_overrides.items():
            if hasattr(agent, attr):
                setattr(agent, attr, val)
    except TypeError:
        return None

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
    reference move. Returns one record per blunder.
    """
    discovered_knobs = discover_agent_parameters(agent_target)
    sig = inspect.signature(agent_target.__init__) if not isinstance(agent_target, str) else None
    has_var_kwargs = sig and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    supported_params = set(sig.parameters.keys()) if sig else set()

    ledger = []

    for b in blunders:
        fen = b["position"]
        ref_move = b["reference_move"]
        original_move = b["chosen_move"]

        orig_depth = b.get("agent_depth", 3)
        orig_time = b.get("agent_time_ms", 300)

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

        solved = False

        # --- PROBE 1: Control Re-Run (Artifact Filter) ---
        control_move = _query_agent(agent_target, fen, {"max_depth": orig_depth, "time_budget_ms": orig_time}, seed=seed)
        case_record["investigation_log"].append({
            "test": "control_rerun", "success": (control_move == ref_move)
        })
        if control_move == ref_move:
            case_record["proven_cause"] = "Measurement Artifact (multiprocessing startup overhead)"
            case_record["validation_status"] = "ERR_MEASUREMENT_ARTIFACT"
            solved = True

        # --- PROBE 2: Checkmate Scoring Probe (if supported) ---
        if not solved and (has_var_kwargs or "checkmate_value" in supported_params):
            mate_move = _query_agent(agent_target, fen, {"max_depth": orig_depth, "time_budget_ms": orig_time, "checkmate_value": 50000}, seed=seed)
            if mate_move is not None:
                case_record["investigation_log"].append({
                    "test": "checkmate_value=50000", "success": (mate_move == ref_move)
                })
                if mate_move == ref_move:
                    case_record["proven_cause"] = "Tactical Horizon Flaw (matched reference move by forcing immediate mate lines with checkmate_value=50000)"
                    case_record["validation_status"] = UNCONFIRMED
                    solved = True

        # --- PROBE 3: Quiescence Search Probe (if supported) ---
        if not solved and (has_var_kwargs or "use_quiescence" in supported_params):
            q_move = _query_agent(agent_target, fen, {"max_depth": orig_depth, "time_budget_ms": orig_time, "use_quiescence": True}, seed=seed)
            if q_move is not None:
                case_record["investigation_log"].append({
                    "test": "use_quiescence=True", "success": (q_move == ref_move)
                })
                if q_move == ref_move:
                    case_record["proven_cause"] = "Evaluation Flaw (matched reference move by forcing capture resolutions via Quiescence Search)"
                    case_record["validation_status"] = UNCONFIRMED
                    solved = True

        # --- Standard Depth / Time / Discovered Option Sweeps ---
        ranked_hypotheses = form_prioritized_hypotheses(b, discovered_knobs)

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
                        agent_target, fen, {test_name: test_val, "max_depth": orig_depth}, seed=seed
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