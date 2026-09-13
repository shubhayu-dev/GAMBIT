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
- Week 0 (design): complete.
- Week 1 (chess environment): complete — pure-Python rules engine,
  perft-verified through depth 4, baseline agents, full-loop demo.
- Week 2 (search agent): complete — iterative-deepening alpha-beta with
  piece-square-table evaluation (`engine/search.py`, `engine/evaluation.py`);
  `SearchAgent` finds forced mates, takes free material, beats random play.
- Week 3 (experiment framework): complete — self-play benchmark generation,
  diagnostic/held-out split, parallel experiment runner
  (`experiments/runner.py`), SQLite + JSONL trajectory logging.

See `docs/10_development_plan.md` for full details and three noted
deviations, all caused by no network access in this sandbox: Stockfish and
python-chess (Week 1), and pyarrow/Parquet + real Lichess benchmark data
(Week 3). Each has a documented drop-in fix for whenever the team has network
access — none require touching the Agent interface or data schema.

**All 8 weeks complete.** Full loop (OBSERVE → EVALUATE → DIAGNOSE →
EXPERIMENT → INTERVENE → VALIDATE) runs end to end against real
(self-play-generated) data — see `docs/10_development_plan.md` for the
full week-by-week write-up, including two honest null/inconclusive results
(Week 6 diagnostic experiments, Week 7 intervention validation) that are
reported as-is rather than dressed up. Final report: `data/gambit_report.html`.

Five noted deviations, all caused by no network access in this sandbox
(Stockfish/python-chess, real Lichess benchmark data, pyarrow/Parquet,
Streamlit) are documented in `docs/10_development_plan.md` with concrete
swap-in instructions for each — none require touching the Agent interface
or trajectory schema.

**Integration addendum (scope change, flagged explicitly):** a neural (MLP)
value agent, classical-vs-neural disagreement analysis, IsolationForest
anomaly detection, and a real-but-not-live LLM reasoning layer were added —
see `docs/11_integration_addendum.md` for what changed and why it's still
within the constitution's spirit.

**Audit (do this before trusting anything above): `docs/12_audit_report.md`.**
Before scaling the dataset, the whole pipeline was audited for correctness
instead of extended further. Two big findings:
1. **The regret formula was bugged** — it never used the `reference_move`
   field it stored, and was proven wrong with constructed test positions
   (`diagnosis/test_regret_audit.py`). Fixed in `experiments/runner.py`.
   Re-running the exact same 12 positions with the fix **reversed** the
   Week 5 "middlegame is the weakness" finding — middlegame went from 0% to
   100% decision accuracy, and Opening is now the actual weak dimension.
2. **The neural value model is worse than simple hand-coded baselines**
   (MAE 2.957 vs. 2.163 for the plain classical evaluator) at predicting its
   own training target — it should not be trusted as an improvement over
   classical evaluation, including in the disagreement analysis.

Also confirmed explicitly for the record: **all data in this project so far
is self-play generated, zero positions come from Lichess** — getting real
Lichess data remains a standing, unaddressed action item.

One controlled experiment (search depth 2 vs 4, corrected metric, real node
counts) was run properly and found no support for the depth hypothesis — no
intervention was built on it, since doing so without supporting evidence
would defeat the point of the audit. Full next-steps for scaling are in
`docs/12_audit_report.md` section 14.

**Phase 2 of the audit (this round): real Lichess data + neural model fix.**
`lichess.org` and `raw.githubusercontent.com` both block automated fetching
(confirmed directly, `ROBOTS_DISALLOWED`) — but the official Lichess dataset
on HuggingFace doesn't, so 96 real, validated Lichess puzzle positions are
now integrated (`data/lichess_puzzles_sample.csv`,
`experiments/lichess_data.py`). Testing on real data immediately found a
SECOND bug: unclipped mate scores blew mean regret up to 14,584 on real
puzzles (self-play data never triggered this). Fixed the same way as the
neural model's training target. Separately, fixing the neural model's
train/test split to be game-level (not position-level) revealed the
original result was leaky: true test R² is **-0.074**, and the neural model
is now worse than every baseline including a constant predictor. Full
writeup: `docs/13_lichess_integration.md` and `docs/14_final_audit_summary.md`
(files changed, tests added, commands, dataset/model stats, and exact next
steps, as requested).

**Phase 3 (this round): mate-handling stress tests + root-causing the neural
failure + a READY/NOT READY call.** 10 constructed mate-score test cases
all pass (no mate / reference-exact / mate found or missed / mate-in-1 /
mate-in-2 via a real puzzle / both sides / White & Black to move), with
sign conventions verified consistent throughout — and an explicit statement
that clipped mate regret is a **bounded approximation**, not exact regret.
Root-caused *why* the neural model fails (not just that it does): overfitting
from 106,881 parameters on ~1,400 unique training examples is the primary
cause, on top of a target that's inherently hard to fit even for the
hand-coded baselines (a 2-ply-search target vs. 0-ply static evaluators).
Also found the `game_phase()` heuristic is systematically miscalibrated
(76% agreement with real Lichess tags, one-directional — always
under-detects endgames), and decided how to handle it going forward (store
both classifiers separately, don't silently prefer one).

**Formal decision: NOT READY to scale from 96 to 500-2000 positions yet.**
Two blocking issues first: the neural model needs to either shrink, get much
more data, or be shelved until a better target exists; and `game_phase()`
needs recalibrating against the real data now available as ground truth.
Full report, including the complete claims-and-confidence table and exact
next steps: `docs/15_audit_round3.md`.
