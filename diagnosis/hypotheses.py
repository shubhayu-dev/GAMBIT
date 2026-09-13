"""
Hypothesis generation (docs/10 Week 5, docs/09_experiment_protocol.md).

For the MVP, hypotheses are generated from a fixed candidate set tied to the
controllable knobs SearchAgent actually has (search depth, evaluation
function, time budget) -- per the Constitution, GAMBIT proposes testable
explanations, not vague diagnoses like "the agent is bad at X".
"""

from typing import Dict, List


def generate_hypotheses(dimension: str) -> List[Dict]:
    """Returns 3 candidate hypotheses for why the agent underperforms in
    `dimension`, each mapped to a concrete, testable experiment condition."""
    return [
        {
            "id": "H1",
            "name": "Search depth limitation",
            "statement": (f"The agent's search depth is insufficient to see far enough ahead "
                           f"in {dimension} positions, causing high regret."),
            "test": "Sweep search depth (shallow vs deep) at fixed time budget and evaluation "
                    "function; measure regret on the weak-bucket positions.",
        },
        {
            "id": "H2",
            "name": "Evaluation-function weakness",
            "statement": (f"The static evaluation function misjudges {dimension} positions "
                           f"even when search finds the right lines, causing high regret."),
            "test": "Ablate the evaluation function (material-only vs material+PST+mobility) "
                    "at fixed depth and time budget; measure regret on the weak-bucket positions.",
        },
        {
            "id": "H3",
            "name": "Time budget insufficiency",
            "statement": (f"The agent doesn't get enough thinking time to reach a useful depth "
                           f"in {dimension} positions within the current time budget."),
            "test": "Sweep time budget (short vs long) at fixed max depth; measure regret on "
                    "the weak-bucket positions.",
        },
    ]
