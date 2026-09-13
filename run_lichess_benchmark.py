import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "engine"))
sys.path.insert(0, str(root_dir / "experiments"))

from engine.agents import SearchAgent
from experiments.lichess_data import load_lichess_positions, RAW_ZST_PATH
from experiments.runner import run_experiment

# Load 250 validated Lichess positions
positions, _ = load_lichess_positions(path=RAW_ZST_PATH, max_positions=250, validate=True)

# Test the agent against the dataset
results = run_experiment(
    agent_class=SearchAgent,
    agent_kwargs={"max_depth": 3},
    positions=positions,
    time_budget_ms=500,
    reference_depth=3,
    benchmark_description="Lichess Middlegame Patched Eval"
)

print(f"Run ID: {results['run_id']}")
print(f"Mean Regret: {results['mean_regret']}")
print(f"Saved to: {results['trajectory_path']}")