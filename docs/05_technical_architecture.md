# 05 — Technical Architecture

These are implementation preferences, not immutable requirements. If a better
technical choice comes up mid-project, evaluate it against scope, research value,
complexity, performance, and maintainability — don't swap tech just because it's
newer.

## Stack diagram
```
┌─────────────────────────────────────────────┐
│ GAMBIT UI — Python / Streamlit               │
└──────────────────────┬──────────────────────┘
                        │
┌──────────────────────▼──────────────────────┐
│ PYTHON RESEARCH LAYER                        │
│ Metrics | Statistics | Diagnosis | Analysis  │
└──────────────────────┬──────────────────────┘
                        │  Experiment API
┌──────────────────────▼──────────────────────┐
│ C++ CORE                                     │
│ Board → Moves → Search → Evaluation          │
│ Experiment Runner + Parallel Workers         │
└──────────────────────┬──────────────────────┘
                        │
┌──────────────────────▼──────────────────────┐
│ DATA — SQLite + Parquet + JSON metadata      │
└─────────────────────────────────────────────┘
```

## Recommended starting order (see `10_development_plan.md`)
Prototype the environment + a baseline agent in **Python** first (`python-chess` +
a UCI wrapper around Stockfish for reference evals) to get the full loop working
end to end quickly. Port performance-critical pieces (search, move generation) to
C++ only once the loop is proven — don't front-load C++ complexity before the
research question is de-risked.

## Technology stack
| Layer | Choice | Notes |
|---|---|---|
| Core engine | C++20, CMake, GoogleTest | board repr, move gen, search, eval, parallel experiment execution |
| Research layer | Python: NumPy, Pandas, SciPy, scikit-learn | metrics, stats, diagnosis; PyTorch only if actually needed later |
| Structured data | SQLite | agents, experiments, benchmarks, runs, configs |
| Bulk data | Parquet | trajectories, decisions, position features, eval results |
| Config/metadata | JSON | configuration, metadata, experiment definitions |
| Dashboard | Streamlit | fast to build, no reason to hand-roll a frontend for MVP |

## Parallelization strategy
Parallelize **independent experiments**, not internal search trees, for the MVP:
```
        EXPERIMENT
     ┌──────┼──────┐
  Worker1 Worker2 Worker3
  1000pos 1000pos 1000pos
     └──────┼──────┘
        Aggregator → Dataset
```
Parallel search-tree exploration is future work if throughput becomes a bottleneck.
