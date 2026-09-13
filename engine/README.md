# engine/
Owner: Member A.

## Week 1 — done
- `environment.py` — chess rules engine: board repr (FEN in/out), legal move
  generation (all piece types, castling, en passant, promotion), state
  transition (`apply_move`), check/checkmate/stalemate detection,
  `game_phase()` classifier for benchmark tagging.
- `tests/test_environment.py` — perft correctness tests (depths 1-4, matches
  known-good values) plus targeted tests for castling, en passant, checkmate.

## Week 2 — done
- `evaluation.py` — material + piece-square tables + mobility, centipawn
  score from White's perspective.
- `search.py` — iterative-deepening alpha-beta with capture-first move
  ordering and a time-budget cutoff.
- `agents.py` — `Agent` interface (docs/06), `RandomAgent` baseline,
  `MaterialAgent` (fixed, simple — stands in as the reference evaluator until
  Stockfish is available), and `SearchAgent` (the real agent under test:
  iterative deepening + PST evaluation).
- `tests/test_search.py` — mate-in-1, free-piece capture, and a self-play
  sanity check against `RandomAgent`. All passing.
- `demo_week1.py` — full state → agent → action → transition loop demo.

Run tests: `python3 tests/test_environment.py && python3 tests/test_search.py`

## Next (Week 4)
Aggregate trajectory data (from `experiments/`) into the multidimensional
behavioral profile (tactical/positional/defensive/endgame/consistency/
efficiency) per `../docs/08_metrics_spec.md`.

## Integration addendum — done (see docs/11_integration_addendum.md)
- `features.py` — board -> 769-dim feature vector (12 one-hot piece planes +
  side-to-move), CNN-ready encoding even though the current model is an MLP.
- `self_play_data.py` — self-play game-outcome data generator (documented as
  not yet practical as the primary training target at this scale).
- `neural_agent.py` — `NeuralAgent`, trained via distillation from
  `MaterialAgent`'s depth-2 evaluation. Same alpha-beta search as
  `SearchAgent`, different evaluation source.
- `train_neural_agent.py` — trains + saves the model. Run:
  `python3 train_neural_agent.py` (~3 min; labels 2000 positions).

## Audit addition — done
- `search.py` now counts nodes visited per search (`agent_nodes_searched`,
  `agent_depth_reached` on trajectory records) — previously untracked, and
  needed for docs/12_audit_report.md section 11's cost/benefit reporting.
- `agents.py`'s `SearchAgent`/`AdaptiveSearchAgent` expose `last_search_info`
  so the runner can log it.

## Phase 2 — game-level train/test split fix
- `self_play_data.py` now supports `generate_value_training_data_with_game_ids`
  + `game_level_split` — fixes a real leakage gap (positions from the same
  self-play game could land in both train and test).
- `retrain_with_game_split.py` — retrained with the fix. Result: true test
  R² is **-0.074** (was 0.504 under the leaky split) and the neural model is
  now worse than every baseline, including constant prediction. See
  `docs/14_final_audit_summary.md`.
