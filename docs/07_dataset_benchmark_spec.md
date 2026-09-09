# 07 — Dataset & Benchmark Specification

## Data principle
Public chess data is source material, not the deliverable. We are not downloading
or training on the entire Lichess database.

```
Public Chess Data (games/puzzles)
  ↓ Sampling
  ↓ Filtering
  ↓ Position Extraction
  ↓ Benchmark Construction
  ↓ Agent Experiments
  ↓ Generated Trajectory Dataset
```

Target size: thousands to tens of thousands of curated positions, sized to
available compute — not raw dataset volume.

## Benchmark categories (controlled, not random sampling)
- **Tactical** — forks, pins, skewers, discovered attacks, mating sequences
- **Positional** — pawn structures, mobility, king safety, piece activity
- **Defensive** — inferior positions, defensive resources, avoiding tactical collapse
- **Endgame** — pawn endings, rook endings, conversion, defense
- **Complexity** — tagged by branching factor, tactical depth, eval volatility,
  number of candidate moves
- **Time pressure** — same positions evaluated under varying compute budgets

## Diagnostic vs. held-out split
```
Dataset
  ├── Diagnostic Set   (used to detect failure patterns and select experiments)
  └── Held-out Set     (used only for final before/after validation)
```
Recommended starting split: ~70% diagnostic / ~30% held-out, stratified by
category so each split has proportional tactical/positional/defensive/endgame
coverage. **The held-out set must never be used to choose or tune the
intervention** — see `09_experiment_protocol.md`.

## Trajectory record schema
Every agent decision becomes one record:
```json
{
  "position": "<FEN>",
  "agent": "AgentA",
  "chosen_move": "Nf6",
  "reference_move": "Qh5",
  "evaluation_before": 1.2,
  "evaluation_after": -0.4,
  "regret": 1.6,
  "search_depth": 5,
  "time_used": 0.72,
  "game_phase": "middlegame",
  "category": "defensive",
  "complexity_score": 0.63,
  "experiment_id": "exp_0007",
  "condition": "depth_6"
}
```
Stored as Parquet for bulk trajectory data; SQLite holds the agents/experiments/
benchmarks/runs/configs tables that reference it.
