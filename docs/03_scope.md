# 03 — Project Scope

## MVP — must have
**Environment**: chess environment, state representation, legal move generation.
**Agent**: custom search-based agent, standardized agent interface, ≥1 baseline agent.
**Evaluation**: benchmark dataset, reference evaluation, decision regret,
performance metrics.
**Analysis**: trajectory logging, capability profiling, failure analysis.
**Diagnosis**: 2–3 candidate hypotheses, hypothesis ranking.
**Experiments**: ≥1 diagnostic experiment, ≥1 intervention.
**Validation**: held-out test set, before/after comparison.
**Presentation**: dashboard, visualizations, experiment report.

## Explicitly out of scope (unless MVP is already complete)
- Stockfish-level chess engine
- Deep reinforcement learning / large neural network training
- LLM-based autonomous agent
- Multi-agent RL
- Cloud-scale distributed infrastructure
- Multiple game environments simultaneously
- Automatic neural architecture modification
- Processing the entire Lichess dataset
- Complex production frontend

These become labeled FUTURE WORK if proposed mid-project.

## Success criteria
GAMBIT is successful if it can take an unknown agent through the full loop —
benchmark → trajectory → performance measurement → weakness identification →
hypothesis generation → diagnostic experiment → cause identification →
intervention → held-out evaluation → quantifiable improvement — end to end, even
on a small scale. Depth of the loop matters more than breadth of features.
