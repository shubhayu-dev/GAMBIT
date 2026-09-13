# 10 — 8-Week Development Plan

**Week 0 — Design (this document set).**
Problem statement, research questions, scope, abstract + technical architecture,
agent interface, dataset spec, metrics spec, benchmark design, git repo. ✅

**Week 1 — Chess environment.** Position → legal moves → state transition.
Recommend prototyping in Python (`python-chess`) before porting to C++.

**Week 2 — Agent.** Position → search → decision. Minimax, alpha-beta, basic
evaluation. Reference agent = Stockfish via UCI wrapper implementing the same
interface.

**Week 3 — Experiment framework.** Agent → benchmark → trajectory dataset.
Experiment runner, parallel execution (independent experiments, multiprocessing),
data logging to Parquet/SQLite.

**Week 4 — Evaluation.** Dataset → metrics → capability profile. GAMBIT should
function as a basic agent evaluation platform by end of this week.

**Week 5 — Behavioral diagnosis.** Failure patterns → hypotheses (2–4 candidates
per identified weakness).

**Week 6 — Diagnostic experiments.** Hypothesis → controlled experiment →
evidence, on the diagnostic set only.

**Week 7 — Intervention.** Diagnosis → targeted modification → held-out
evaluation, single pass, no re-tuning against held-out results.

**Week 8 — Integration.** Working system + dashboard (Streamlit) + report +
presentation.

## Team responsibility map
| Member | Owns | Learns |
|---|---|---|
| A — Core AI / Chess Engine | board repr, move gen, search, eval, agent, agent interface | C++, bitboards, minimax, alpha-beta |
| B — Experimentation & Systems | experiment runner, benchmark execution, parallelization, data storage | C++, threading/multiprocessing, reproducibility, data formats |
| C — AI Research & Diagnostics | metrics, statistical analysis, behavior analysis, failure detection, hypothesis engine | NumPy/Pandas/SciPy, statistics, experimental design |
| D — Integration & Visualization | dashboard, visualization, system integration, reports | Streamlit, viz, API integration |

**Team rule:** the four roles are not four independent mini-projects — every
member should understand the full pipeline (Environment → Agent → Experiment →
Data → Analysis → Diagnosis → Intervention → Validation), and interfaces between
components (agent interface, trajectory schema) must be frozen before parallel
implementation starts.

## Week 1 status — complete, with one noted deviation

**Deviation from the original plan (flagged per Constitution Rule 7):** the plan
in `05_technical_architecture.md` called for prototyping on `python-chess` with
Stockfish as the reference engine. This sandbox had no network access, so
neither could be installed. Instead:

- `engine/environment.py` implements chess rules directly (board, legal move
  generation including castling/en passant/promotion, state transition,
  check/checkmate/stalemate detection). Verified correct via perft against
  known-good values (20 / 400 / 8902 / 197281 at depths 1–4) in
  `engine/tests/test_environment.py`.
- `engine/agents.py` provides `RandomAgent` (baseline) and `MaterialAgent`
  (fixed-depth alpha-beta minimax over material + mobility), standing in for
  Stockfish as the **reference** evaluator until a real UCI engine is wired in.
  Both implement the exact `Agent` interface from `06_agent_interface.md`, so
  swapping in Stockfish later requires no changes elsewhere in the pipeline.
- `engine/demo_week1.py` runs the full `state → agent → action → transition`
  loop and writes trajectory records matching the `07_dataset_benchmark_spec.md`
  schema to `data/week1_demo_trajectories.json`.

**Action item for the team:** when there's network access, swap `MaterialAgent`
for a real Stockfish-UCI wrapper implementing the same `Agent` interface — this
is a drop-in replacement, not a redesign. Until then, regret values in any demo
output are relative to `MaterialAgent`'s own evaluation, not a strong reference,
so don't treat early numbers as meaningful — only the plumbing is being tested
this week.

Week 2 goal (per plan): replace the demo's ad hoc search with the actual
minimax/alpha-beta agent implementation and basic evaluation function.

## Week 2 status — complete

- `engine/evaluation.py` — material + standard piece-square tables + mobility,
  centipawn score from White's perspective.
- `engine/search.py` — iterative-deepening alpha-beta with capture-first move
  ordering (MVV-LVA style) and a time-budget cutoff.
- `engine/agents.py` — new `SearchAgent`, the actual "custom search-based
  agent" required by the MVP scope (`03_scope.md`). `MaterialAgent` is kept as
  a separate, deliberately simple, fixed reference for regret computation —
  the two are not the same agent.
- `engine/tests/test_search.py` — verifies `SearchAgent` finds a forced
  mate-in-1, takes an undefended free piece, and beats `RandomAgent` in
  self-play (4/4 games in the sanity run). Not a rigorous benchmark — that's
  Week 4.

## Week 3 status — complete, with one more noted deviation

**Deviation (Constitution Rule 7):** `07_dataset_benchmark_spec.md` specifies
Parquet for trajectory storage. Neither `pyarrow` nor `fastparquet` was
installable in this sandbox (no network). `experiments/data_store.py` writes
trajectories as JSON Lines instead — identical one-record-per-line structure,
convertible to Parquet later with one line of pandas once a parquet engine is
available (`pd.read_json(path, lines=True).to_parquet(...)`). SQLite metadata
tables (agents/benchmarks/experiments/runs) are exactly as planned, since
`sqlite3` is stdlib and needed no network.

**Also noted:** `experiments/benchmark.py` generates positions via self-play
between baseline agents rather than sampling real Lichess games/puzzles
(same network constraint as Week 1). It's tagged by game phase and split into
diagnostic/held-out sets per `07_dataset_benchmark_spec.md`, but is not yet
categorized into tactical/positional/defensive/endgame buckets the way a real
curated benchmark would be — that categorization needs real puzzle data.

**What's built and verified:**
- `experiments/benchmark.py` — self-play position sampling + stratified
  diagnostic/held-out split.
- `experiments/data_store.py` — SQLite schema (agents, benchmarks,
  experiments, runs) + JSONL trajectory read/write.
- `experiments/runner.py` — parallel experiment runner using
  `multiprocessing.Pool`, one worker per position, agent-agnostic (takes any
  `Agent` subclass + kwargs). This sandbox reports a single CPU core, so no
  real speedup is visible here, but the code is written to scale on a real
  multi-core machine, which is the point of building it this way now.
- `experiments/demo_week3.py` — full run: benchmark → split → parallel
  experiment → trajectory dataset + SQLite metadata. Verified: 16 positions
  generated, 12/4 diagnostic/held-out split, 12-position experiment run
  logged correctly across all four tables and the JSONL trajectory file.

**Action items for the team (when network access exists):**
1. Swap `MaterialAgent` for a real Stockfish-UCI wrapper (Week 1 note).
2. Replace `experiments/benchmark.py`'s self-play sampling with real curated
   Lichess positions, categorized per `07_dataset_benchmark_spec.md`.
3. Install `pyarrow` and switch `data_store.py` from JSONL to Parquet.

None of these require touching the `Agent` interface, the trajectory schema,
or the runner — they're drop-in replacements by design.

Week 4 goal (per plan): dataset → metrics → capability profile. GAMBIT should
function as a basic agent evaluation platform by the end of that week.

## Weeks 4-6 status — complete (reusing Week 3 data), with an honest caveat

Run via `run_weeks_4_6.py`, which deliberately reuses the Week 3 baseline
trajectory dataset (`data/trajectories_run_8d9b9e50.jsonl`, 12 diagnostic-set
positions, `SearchAgent(max_depth=3)`) rather than generating a fresh
benchmark — pulled directly from the SQLite run log by matching
`experiments.condition = 'depth_3'`, not by filename or "most recent run"
(which would otherwise pick up the diagnostic sub-experiments this same
script logs).

**Week 4 — capability profile.** `analysis/profile.py` + `analysis/report.py`
built the profile from the 12 existing records. Overall decision accuracy
25.0%. Endgame bucket has 0 positions (this self-play sample happened not to
reach one) — reported as `n/a`, not a fabricated number.

**Week 5 — weakness + hypotheses.** Weakest measured dimension: **Middlegame
play** (0.0% decision accuracy, n=7, mean regret -1.603). `diagnosis/
hypotheses.py` generated the standard three candidates (H1 search depth, H2
evaluation quality, H3 time budget) against it.

**Week 6 — diagnostic experiments.** All three hypotheses tested as paired
A/B comparisons on the 7 middlegame positions:

| Hypothesis | Condition A → B | mean \|regret\| | Improvement | p-value |
|---|---|---|---|---|
| H2 evaluation quality | material-only → material+PST | 2.313 → 2.313 | 0% | n/a (identical) |
| H1 search depth | depth=2 → depth=4 | 2.329 → 2.466 | -5.9% | 1.0 |
| H3 time budget | 100ms → 600ms | 2.176 → 2.314 | -6.4% | 0.5 |

**Honest read of this result, not a dressed-up one:** none of these are
statistically significant (p ≥ 0.5 throughout), and two of the three
"improvements" are actually negative — deeper search / more time budget
correlated with *higher* regret relative to the fixed-depth-2 reference
evaluator on this tiny sample. With only 7 positions this is not strong
evidence for or against any hypothesis; it's exactly the kind of
inconclusive result the protocol in `09_experiment_protocol.md` is designed
to surface honestly rather than paper over. The negative direction is also
plausibly an artifact of the reference evaluator itself being shallow
(depth 2) — a deeper search agent diverging further from a shallow reference
isn't necessarily "worse," just more different, which is a real limitation
of not having Stockfish as a strong, stable reference (see Week 1 deviation).

**Action item:** re-run this diagnostic suite once (a) a larger benchmark is
available and (b) a real Stockfish reference replaces `MaterialAgent`, before
drawing any conclusion about which hypothesis to act on in Week 7.

Full machine-readable output: `data/week4_6_summary.json`.

Week 7 (intervention) and Week 8 (dashboard) were not run in this pass —
scoped to Weeks 4-6 only, per this request.

## Week 7-8 status — complete, another honest null result

Run via `run_weeks_7_8.py`. First recovered the held-out half of the
original Week 3 benchmark by regenerating it with the exact same seeds
(`n_games=16, seed=1`, split `seed=1`) and **verifying** the regenerated
diagnostic set matches the logged Week 3 run 12/12 positions before trusting
the held-out half — this is a genuine recovery of already-existing data, not
a new sample.

**Week 7 — intervention + held-out validation.** Applied
`AdaptiveSearchAgent` (base depth 3, boosted depth 5, complexity threshold
30) against the 4 held-out positions, comparing to fixed-depth
`SearchAgent(max_depth=3)`.

**Result: 0.0% change.** Inspected per-position: the deeper search chose the
*exact same move* as the shallower one on all 4 held-out positions (not just
coincidentally equal regret — literally identical chosen moves). All 4
held-out positions happened to have complexity ≥ the boost threshold, so the
intervention fired on every one of them; it simply didn't change any
decision. This is a legitimate outcome, not a bug, and it's consistent with
Week 6's finding that search depth wasn't a clearly supported hypothesis for
this agent on this data — reported as-is rather than adjusted to look more
successful.

**What this run does demonstrate:** the full mechanism — diagnose on the
diagnostic set, apply an intervention, validate once (no peeking/retuning)
on held-out data the diagnosis never touched — works correctly end to end.
What it does *not* demonstrate, on this small self-play dataset with a
depth-2 shallow reference evaluator, is that adaptive search depth is
actually the right fix for this agent. Re-running this with (a) a larger,
real benchmark and (b) Stockfish as reference is the natural next step
before drawing any real conclusion — same two action items already flagged
for Weeks 1/3/6.

**Week 8 — report.** `dashboard/report_html.py` generated
`data/gambit_report.html`: capability profile bars, primary weakness
callout, ranked diagnostic evidence table, and the before/after intervention
comparison, with all deviations from the original spec listed in the
footer rather than buried.

Full machine-readable output: `data/full_pipeline_summary.json`.

## MVP status: complete

All 15 MVP checklist items from `03_scope.md` now have working code that's
been run at least once against real (self-play-generated) data:
chess environment ✅, agent interface ✅, custom search agent + baseline ✅,
benchmark + reference eval + regret ✅, trajectory logging + capability
profiling + failure detection ✅, hypothesis generation/ranking ✅, diagnostic
experiment ✅, intervention ✅, held-out validation ✅, report/dashboard ✅.

The results themselves are inconclusive/null at every diagnostic and
intervention step — expected and reportable given the dataset is ~16
self-play positions and the reference evaluator is a shallow placeholder,
not Stockfish. The three standing action items (Stockfish integration, a
real curated benchmark, pyarrow for Parquet) are what would need to happen
before GAMBIT's actual findings mean anything beyond "the loop runs."
