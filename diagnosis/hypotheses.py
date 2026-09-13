"""
Hypothesis Generation Registry (GAMBIT Step 11/12).

This module defines the architectural parameter registry for the current agent.
For the classical SearchAgent MVP, the hypotheses are deterministic and tied
directly to the agent's exposed knobs: search depth, evaluation heuristics,
and time budget. 

By explicitly defining the `experiment_schema`, this acts as the exact JSON 
template that the Gemini LLM Reasoner will use to dynamically generate and 
execute its own A/B tests in future pipeline iterations.
"""

from typing import Dict, List


def generate_hypotheses(dimension: str) -> List[Dict]:
    """
    Generates concrete, testable hypotheses for agent underperformance in a given dimension.
    Returns the hypothesis statement alongside the exact control/treatment parameters 
    required for diagnostic_experiment.py to execute the A/B test.
    """
    return [
        {
            "id": "H1",
            "name": "Search Depth Limitation",
            "statement": (f"The agent's search depth is insufficient to resolve tactical "
                          f"lines in {dimension} positions, resulting in high regret."),
            "test": "Sweep search depth (shallow vs deep) at fixed time budget.",
            "experiment_schema": {
                "variable": "max_depth",
                "control": 2,
                "treatment": 4
            }
        },
        {
            "id": "H2",
            "name": "Evaluation Function Weakness",
            "statement": (f"The static evaluation function misjudges {dimension} positions "
                          f"even when search explores the correct lines."),
            "test": "Ablate the evaluation function (material-only vs material+PST).",
            "experiment_schema": {
                "variable": "evaluator",
                "control": "material",
                "treatment": "material_pst"
            }
        },
        {
            "id": "H3",
            "name": "Time Budget Insufficiency",
            "statement": (f"The agent is compute-starved and cannot reach a useful depth "
                          f"in {dimension} positions within the allotted time."),
            "test": "Sweep time budget (100ms vs 600ms) with a high theoretical max depth.",
            "experiment_schema": {
                "variable": "time_budget_ms",
                "control": 100,
                "treatment": 600
            }
        },
    ]