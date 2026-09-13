# diagnosis/
Owner: Member C.

## Week 5 — done
- `failure_detection.py` — picks the weakest profile dimension with enough
  samples to be meaningful.
- `hypotheses.py` — generates 3 candidate, testable hypotheses (search depth
  / evaluation quality / time budget) tied to `SearchAgent`'s actual
  controllable knobs.

## Week 6 — done
- `diagnostic_experiment.py` — runs each hypothesis as a paired A/B
  experiment on the weak-bucket positions only (never the held-out set),
  with a Wilcoxon signed-rank test where sample size allows. Ranks
  hypotheses by improvement magnitude.
- Real run against the Week 3 dataset came back **inconclusive** (p ≥ 0.5,
  n=7) — see `../docs/10_development_plan.md` for the honest write-up. This
  is a legitimate outcome of the protocol, not a bug.

## Week 7 — done
- `intervention.py` — applies `AdaptiveSearchAgent` (complexity-based depth,
  in `engine/agents.py`) and validates once on held-out positions recovered
  from the original Week 3 benchmark split.
- Real run showed **0% change** — the deeper search chose identical moves
  to the shallower one on all 4 held-out positions. Also reported as-is.

Run Weeks 4-6 (reusing existing data): `python3 ../run_weeks_4_6.py`
Run Weeks 7-8 (recovers held-out data + builds report): `python3 ../run_weeks_7_8.py`
Run the full Weeks 4-8 pipeline on a fresh benchmark: `python3 ../run_full_pipeline.py`

## Integration addendum — done
- `llm_reasoner.py` — evidence payload builder + real (but not live in this
  sandbox — no network/API key) Anthropic API call interface. See
  `../docs/11_integration_addendum.md` for how the actual reasoning step was
  performed, and `../data/llm_reasoning_output.json` for the real output.

Run the full integration demo (neural agent, disagreement, anomaly
detection, evidence payload) against the existing baseline dataset:
`python3 ../run_integration_demo.py`

## Audit — done (see docs/12_audit_report.md)
Before scaling the dataset, the full pipeline was audited for correctness
rather than extended. Headline finding: the regret formula was buggy (proven
with constructed test positions) and fixing it **reversed** the Week 5
weakness finding. The neural model was also shown to be worse than simple
hand-coded baselines at its own training target — it should not currently be
trusted in the disagreement analysis.
- `test_regret_audit.py` — constructed-position test suite proving the bug.
- `run_audit.py` — sections 5-7: disagreement (adds move disagreement, not
  just evaluation disagreement), anomaly detection audit, and direct
  investigation of the opening/middlegame contradiction.
- `run_audit_section9.py` — one controlled experiment (depth 2 vs 4 on the
  corrected weak bucket) with real node/time cost accounting. Result: no
  support for the search-depth hypothesis (regret got worse, cost went up
  10.7x in nodes) — reported as-is, no intervention built on it (see report
  section 12 for why).

Read `../docs/12_audit_report.md` before trusting or extending any prior
diagnostic/intervention result in this project.

## Audit round 3 — done (see docs/15_audit_round3.md)
- `test_mate_score_audit.py` — 10 constructed mate-handling test cases (no
  mate, reference-move-exact, mate found/missed, mate-in-1, mate-in-2 via a
  real Lichess puzzle, White/Black to move, both sides involving mate).
  All 10 pass; sign convention verified consistent throughout. Explicitly
  documents that clipped mate regret is a **bounded approximation**, not
  exact regret.
- Found a minor engine robustness gap along the way: `Board(fen)` doesn't
  validate FEN legality, so a malformed position (side not to move already
  in check) let a rook pseudo-legally "capture" the enemy king during
  search. Not fixed this round (audit only) — flagged as an action item.
- Root-caused the neural model's failure (not just "it's bad", but *why*):
  overfitting from 106,881 params on ~1,400 unique examples is the primary
  cause, compounded by a target (2-ply search) that even hand-coded
  baselines can only explain ~70-75% of variance in. Evidence and reasoning
  in `docs/15_audit_round3.md` section 8.
- **Decision: NOT READY to scale the dataset yet.** Two blocking issues:
  the neural model doesn't generalize at all, and `game_phase()` is
  measurably miscalibrated (76% agreement with real Lichess tags,
  systematically under-detecting endgames). Both need to be addressed
  before 96 → 500-2000 positions, per the report's own reasoning about not
  producing "2000 potentially wrong measurements instead of 12."
