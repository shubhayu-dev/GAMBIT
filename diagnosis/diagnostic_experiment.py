"""
Runs iterative empirical search across dynamically formulated hypotheses.
Proves the fix before passing results to the LLM.
"""

import subprocess
import math
from typing import Any, Dict, List
import chess
from scipy.stats import wilcoxon

from diagnosis.hypotheses import discover_agent_parameters, form_prioritized_hypotheses


def _query_agent(agent_target: Any, fen: str, run_kwargs: Dict[str, Any]) -> str:
    """Invokes either internal class or external UCI subprocess and returns chosen UCI move string."""
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
            if k not in ["max_depth", "time_budget_ms"]:
                send_cmd(f"setoption name {k} value {v}")

        send_cmd(f"position fen {fen}")
        
        go_parts = ["go"]
        if "max_depth" in run_kwargs:
            go_parts.append(f"depth {run_kwargs['max_depth']}")
        if "time_budget_ms" in run_kwargs:
            go_parts.append(f"movetime {run_kwargs['time_budget_ms']}")
        send_cmd(" ".join(go_parts))

        chosen_move = None
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("bestmove"):
                chosen_move = line.split()[1]
                break
        proc.terminate()
        return chosen_move
    else:
        board = chess.Board(fen)
        agent_kw = {k: v for k, v in run_kwargs.items() if k != "time_budget_ms"}
        inst = agent_target(**agent_kw)
        time_ms = run_kwargs.get("time_budget_ms", 300)
        res = inst.select_move(board, time_budget_ms=time_ms)
        return res["move"].uci() if res.get("move") else None


def run_empirical_investigation(
    agent_target: Any,
    blunders: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Actively searches for fixes per blunder using ranked heuristic guessing.
    """
    discovered_knobs = discover_agent_parameters(agent_target)
    proven_ledger = []

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
        }

        ranked_hypotheses = form_prioritized_hypotheses(b, discovered_knobs)
        solved = False

        for hyp in ranked_hypotheses:
            if solved:
                break

            if hyp["action"] == "scale_depth":
                for target_depth in hyp["test_sequence"]:
                    result_move = _query_agent(agent_target, fen, {"max_depth": target_depth})
                    case_record["investigation_log"].append({
                        "test": f"depth={target_depth}", "success": (result_move == ref_move)
                    })
                    if result_move == ref_move:
                        case_record["proven_cause"] = f"Horizon Limitation (Resolved at depth {target_depth})"
                        solved = True
                        break

            elif hyp["action"] == "scale_time":
                for t_budget in hyp["test_sequence"]:
                    result_move = _query_agent(agent_target, fen, {"time_budget_ms": t_budget})
                    case_record["investigation_log"].append({
                        "test": f"time={t_budget}ms", "success": (result_move == ref_move)
                    })
                    if result_move == ref_move:
                        case_record["proven_cause"] = f"Compute Starvation (Resolved at {t_budget}ms)"
                        solved = True
                        break
            
            elif hyp["action"] == "perturb_options":
                for eval_knob in hyp["knobs"]:
                    if solved: break
                    test_val = eval_knob.get("max", 100000) if "max" in eval_knob else "material_pst"
                    test_name = eval_knob["name"]
                    result_move = _query_agent(agent_target, fen, {test_name: test_val, "max_depth": b.get("agent_depth", 3)})
                    case_record["investigation_log"].append({
                        "test": f"{test_name}={test_val}", "success": (result_move == ref_move)
                    })
                    if result_move == ref_move:
                        case_record["proven_cause"] = f"Static Heuristic Flaw (Resolved by setting {test_name}={test_val})"
                        solved = True
                        break

        if not solved:
            case_record["proven_cause"] = "Unresolved structural deficiency."

        proven_ledger.append(case_record)

    return proven_ledger

# ---------------------------------------------------------
# Legacy Pipeline Compatibility Functions (Leave Untouched)
# ---------------------------------------------------------
def run_diagnostic_suite(weak_positions, hypotheses, time_budget_ms=300, seed=100):
    pass # Retained for backward compatibility in run_full_pipeline.py if needed

def rank_hypotheses(evidence_records):
    return [] # Retained for backward compatibility in run_full_pipeline.py if needed