# GAMBIT
Game-theoretic Agent Monitoring and Behavior Intervention Testbed

An AI agent evaluation and diagnostic framework. Chess is the first environment —
not the point of the project. The point is the closed loop:

```
OBSERVE → EVALUATE → DIAGNOSE → EXPERIMENT → INTERVENE → VALIDATE
```

## Start here
1. Read `PROJECT_CONSTITUTION.md` first — paste it into any LLM session about
   this project. It's the source of truth on scope.
2. Read `/docs` in order (01 → 10) for the full Week 0 design.

## Repo layout
```
GAMBIT/
├── PROJECT_CONSTITUTION.md
├── docs/            # design specs (01-10)
├── engine/          # C++ core: board, moves, search, eval, agents
├── experiments/     # experiment runner, parallel execution
├── analysis/        # metrics, statistics, behavioral profiling
├── diagnosis/        # hypothesis generation, diagnostic experiments
├── dashboard/       # Streamlit UI
└── data/            # SQLite + Parquet + JSON (gitignored bulk data)
```

## Status
Week 0 (design) complete. Week 1 (chess environment) next — see
`docs/10_development_plan.md`.
