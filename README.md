# GAMBIT

**Game-theoretic Agent Monitoring and Behavior Intervention Testbed**

An evaluation and diagnostic framework for AI agents. Chess is the first environment, not the point of the project. The point is the closed loop:

```
OBSERVE → EVALUATE → DIAGNOSE → EXPERIMENT → INTERVENE → VALIDATE
   ▲                                                          │
   └──────────────────────────────────────────────────────────┘
```

Instead of reporting "the agent scored X", GAMBIT asks *where* an agent fails, *why*, whether a proposed fix actually works, and whether that fix generalizes to held-out data.

> **Status: research prototype, audited, not yet ready to scale.** The full loop runs end to end on self-play data plus a small real Lichess sample. Several early findings were later overturned by audit. See [Status and honest limitations](#status-and-honest-limitations) before trusting any number in this repo.

---

## Contents

- [Start here](#start-here)
- [Quick start](#quick-start)
- [How the loop works](#how-the-loop-works)
- [Repository layout](#repository-layout)
- [Key components](#key-components)
- [Status and honest limitations](#status-and-honest-limitations)
- [What the audit found](#what-the-audit-found)
- [Known deviations](#known-deviations)
- [Roadmap](#roadmap)
- [Documentation index](#documentation-index)

---

## Start here

1. Read **`PROJECT_CONSTITUTION.md`** first. It is the source of truth on scope. Paste it into any LLM session about this project.
2. Read **`docs/`** in order (01 → 10) for the Week 0 design.
3. Read **`docs/12_audit_report.md`** through **`docs/15_audit_round3.md`** before relying on any result. They supersede several earlier claims.
4. The final report from the main pipeline run is `data/gambit_report.html`.

## Quick start

See **`SETUP.md`** for environment setup, including running a local Ollama model for move commentary.

```bash
# End-to-end move commentary demo (local Ollama by default; Gemini also supported)
python analysis/run_commentary_demo.py
```

The regret audit tests are runnable directly:

```bash
python diagnosis/test_regret_audit.py
```

The full diagnostic pipeline entry point is `diagnosis/run_full_pipeline.py`.

## How the loop works

| Stage | What happens | Where |
|---|---|---|
| **Observe** | Agents play games; every decision is logged as a trajectory (SQLite + JSONL) | `engine/`, `experiments/` |
| **Evaluate** | Per-move regret against a reference move, decision accuracy by game phase, behavioral profiles | `analysis/`, `experiments/runner.py` |
| **Diagnose** | Per-blunder empirical search: does some knob setting recover the reference move on this exact position? | `diagnosis/` |
| **Experiment** | Controlled comparisons (e.g. search depth 2 vs 4) with real node counts | `diagnosis/`, `experiments/` |
| **Intervene** | Apply a candidate fix to the agent | `diagnosis/intervention.py` |
| **Validate** | Check the fix on a held-out set. Only this step earns the word "generalized" | `diagnosis/intervention.py` |

Every ledger entry from `diagnostic_experiment.run_empirical_investigation()` carries an explicit `validation_status` and is never silently "proven". A knob bump that fixes one position is a much weaker claim than a statistical test across a diagnostic set. Read the module docstring of `diagnosis/diagnostic_experiment.py` before trusting any case study's `proven_cause`.

## Repository layout

```
GAMBIT/
├── PROJECT_CONSTITUTION.md   # scope and source of truth
├── SETUP.md                  # environment + Ollama setup
├── CHANGE_LOG.md             # bug fixes with verification notes
├── docs/                     # design specs (01-10), addenda and audits (11-15)
├── engine/                   # chess rules, search, evaluation, agents
├── experiments/              # runner, parallel execution, Lichess data loader
├── analysis/                 # metrics, statistics, behavioral profiling,
│                             #   move commentary (see below)
├── diagnosis/                # hypothesis generation, diagnostic experiments,
│                             #   interventions, LLM reasoning layer
├── dashboard/                # Streamlit UI (see Known deviations)
└── data/                     # SQLite + JSONL + CSV (bulk data gitignored)
```

> Note: earlier versions of this README described `engine/` as a "C++ core". The implemented engine is pure Python. Update the Constitution if the C++ plan still stands.

## Key components

### Chess environment and agents (Weeks 1-2)
- Pure-Python rules engine, perft-verified through depth 4.
- Baseline agents, including random play.
- `SearchAgent`: iterative-deepening alpha-beta with piece-square-table evaluation (`engine/search.py`, `engine/evaluation.py`). Finds forced mates, takes free material, beats random play.

### Experiment framework (Week 3)
- Self-play benchmark generation with a diagnostic / held-out split.
- Parallel experiment runner (`experiments/runner.py`).
- SQLite + JSONL trajectory logging.
- 96 validated real Lichess puzzle positions (`data/lichess_puzzles_sample.csv`, `experiments/lichess_data.py`), sourced from the official Lichess dataset on HuggingFace.

### Diagnosis pipeline
Reworked from paired-hypothesis statistical testing to a per-blunder empirical search. The rewrite initially shipped with breaking bugs (dependence on python-chess and a nonexistent `select_move()`, a `KeyError` in `hypotheses.py`, mixed regret metrics in `intervention.py`, stale calls in `run_full_pipeline.py`). All were fixed and verified against the real engine. See `CHANGE_LOG.md`.

### Integration addendum (scope change, flagged)
A neural (MLP) value agent, classical-vs-neural disagreement analysis, IsolationForest anomaly detection, and a real-but-not-live LLM reasoning layer. Details and rationale: `docs/11_integration_addendum.md`. **The neural model is currently shelved as untrustworthy** (see audit findings).

### Move commentary system
A separate subsystem in `analysis/`, deliberately not part of `diagnosis/llm_reasoner.py` (that module feeds the DIAGNOSE step; commentary narrates moves for a human reader).

| File | Role |
|---|---|
| `rule_based_commentary.py` | Filters out moves that need no LLM: only legal move, forced check replies, simple recaptures, mate |
| `commentary.py` | Batches the rest into a strict-JSON LLM call. Providers: Ollama (default, `qwen2.5:7b-instruct`) or Gemini. Retries moves a batch dropped, one at a time |
| `commentary_report.py` | Renders a browser-openable HTML report: board snapshot, tag, explanation per move |
| `run_commentary_demo.py` | End-to-end runnable driver |

**Why the explanations are honest:** `engine/search.py` captures a `root_trace` during the live, time-bounded search that chose the move. It records every root move actually compared, plus whether the deadline cut that comparison short (`fully_searched: false`). The engine runs on a wall-clock budget, so it may compare only 3-4 of 25-30 legal moves in the middlegame. Reconstructing alternatives with a fresh unconstrained re-search would make the played move look like it lost to options the live search never reached. `evaluate_root_candidates()` remains only for the different question "what would unlimited time have found?" and is not used for commentary.

## Status and honest limitations

| Week | Scope | Status |
|---|---|---|
| 0 | Design | Complete |
| 1 | Chess environment | Complete |
| 2 | Search agent | Complete |
| 3 | Experiment framework | Complete |
| 4-8 | Metrics → diagnosis → intervention → validation → report | Complete; two null/inconclusive results reported as-is (Week 6 diagnostic experiments, Week 7 intervention validation) |
| Audit rounds 1-3 | Correctness audit | Complete |
| Scaling to 500-2000 positions | | **NOT READY** (two blockers below) |

**Blocking issues before scaling:**
1. The neural model must shrink, get far more data, or be shelved until a better target exists.
2. `game_phase()` must be recalibrated against real Lichess tags.

**Data provenance:** all positions outside the 96-position Lichess sample are self-play generated.

## What the audit found

Read these before quoting any earlier result.

| Finding | Impact |
|---|---|
| **Regret formula bug.** It never used the stored `reference_move`. Proven wrong with constructed positions (`diagnosis/test_regret_audit.py`) and fixed in `experiments/runner.py` | Reversed the Week 5 conclusion. Middlegame went from 0% to 100% decision accuracy; **Opening** is the actual weak dimension |
| **Unclipped mate scores.** Found by testing on real Lichess puzzles: mean regret reached 14,584 | Fixed by clipping, as for the neural target. Clipped mate regret is a bounded approximation, not exact regret. 10 constructed mate cases pass, with sign conventions verified |
| **Neural model leakage.** Position-level split leaked across games | With a game-level split, test R² is **-0.074**, worse than every baseline including a constant predictor. Root cause: 106,881 parameters on ~1,400 unique examples, plus a hard 2-ply-search target |
| **`game_phase()` miscalibration** | 76% agreement with real Lichess tags, always under-detecting endgames. Both classifiers are now stored separately rather than silently preferring one |
| **Depth hypothesis** (search depth 2 vs 4, corrected metric, real node counts) | No support found. No intervention was built on it |

Full write-ups: `docs/12_audit_report.md`, `docs/13_lichess_integration.md`, `docs/14_final_audit_summary.md`, `docs/15_audit_round3.md` (includes the claims-and-confidence table and exact next steps).

## Known deviations

Caused by no network access in the original build sandbox. Each has a documented drop-in fix in `docs/10_development_plan.md` and none requires changing the Agent interface or trajectory schema.

- Stockfish and python-chess (a pure-Python engine is used instead)
- Real Lichess benchmark data at scale (only 96 puzzles integrated so far)
- pyarrow / Parquet (SQLite + JSONL used instead)
- Streamlit dashboard (not run in the sandbox)
- A live LLM reasoning layer (implemented, not run live)

Note: `lichess.org` and `raw.githubusercontent.com` block automated fetching. The HuggingFace mirror of the official dataset does not.

## Roadmap

1. Recalibrate `game_phase()` against Lichess ground truth.
2. Resolve the neural model (shrink, add data, or shelve).
3. Scale from 96 to 500-2000 real positions.
4. Re-run diagnostic experiments on the corrected metric and real data.
5. Swap in Stockfish as the reference engine.
6. Validate any proposed intervention on held-out data before making claims.

Exact next steps: `docs/12_audit_report.md` §14 and `docs/15_audit_round3.md`.

## Documentation index

| Doc | Contents |
|---|---|
| `PROJECT_CONSTITUTION.md` | Scope, principles, source of truth |
| `docs/01`-`10` | Week 0 design and week-by-week development plan |
| `docs/11_integration_addendum.md` | Neural agent, anomaly detection, LLM layer |
| `docs/12_audit_report.md` | First correctness audit |
| `docs/13_lichess_integration.md` | Real data integration |
| `docs/14_final_audit_summary.md` | Files changed, tests, commands, stats |
| `docs/15_audit_round3.md` | Mate tests, neural root cause, READY/NOT READY decision |
| `CHANGE_LOG.md` | Diagnosis pipeline fixes |
| `SETUP.md` | Environment and Ollama setup |
