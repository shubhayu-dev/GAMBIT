# CHANGE_LOG

Dated by when this round of work was done, not by individual commit —
these changes were made together in one working session.

## 2026-09-21

### Fixed — diagnosis pipeline (was non-functional as shipped)

- **`diagnosis/diagnostic_experiment.py`**: removed a hard dependency on
  `python-chess` (`import chess`) and a call to `agent.select_move(chess.Board(fen), ...)`
  — neither exists anywhere else in this codebase. Every agent in
  `engine/agents.py` exposes `get_move(board: environment.Board,
  time_budget_ms, seed) -> Move`, using this project's own `Board`, not
  `python-chess`'s. This crashed the internal-agent path immediately
  (`ModuleNotFoundError`, then `AttributeError` once `chess` was
  installed). Fixed to use `environment.Board`/`get_move` directly.
- **`diagnosis/diagnostic_experiment.py`**: `run_diagnostic_suite()` and
  `rank_hypotheses()` previously silently no-op'd (`pass` / `return []`).
  Now raise `NotImplementedError` pointing at their replacements
  (`run_empirical_investigation()`, `intervention.extract_dominant_fix()`)
  — a caller could otherwise believe an experiment ran when nothing
  happened.
- **`diagnosis/diagnostic_experiment.py`** / **`diagnosis/intervention.py`**:
  every case record now carries an explicit `validation_status` (e.g.
  `"single_position_match (unconfirmed -- see intervention.py for
  held-out validation)"`), since recovering the reference move on ONE
  position after a knob bump is a materially weaker claim than the
  project's earlier paired Wilcoxon test across a diagnostic set, and the
  previous version called this "proven" outright.
- **`diagnosis/hypotheses.py`**: fixed a `KeyError` — `universal_search_knobs`
  entries were missing the `"source"` key that `form_prioritized_hypotheses()`
  unconditionally indexed (`k["source"]`), crashing any blunder
  investigation that reached the eval-perturbation branch. Tagged the
  knobs and switched the filter to `.get()` defensively.
- **`diagnosis/intervention.py`**: fixed a metric mismatch —
  `evaluate_intervention()` previously compared a precomputed *continuous*
  regret value for the baseline against a *binary 0/1 match-proxy* for the
  intervention whenever no oracle function was supplied, feeding two
  different scales into one `regret_reduction_pct` subtraction. Now scores
  both sides with the same metric and reports which one was used
  (`regret_metric` field) instead of leaving it implicit.
- **`diagnosis/llm_reasoner.py`**: restored the "do not fabricate numbers,
  moves, or claims not present in the payload" instruction, dropped in the
  version that introduced the "Proven Ledger" schema. Added an explicit
  rule requiring hedged language ("candidate fix", "recovered the
  reference move on this position") for anything not marked validated,
  rather than assertive language ("resolved", "eliminated"). Reverted the
  default model from a silently-downgraded `gemini-2.5-flash` back to
  `gemini-3.6-flash`, consistent with `analysis/commentary.py`.
- **`run_full_pipeline.py`**: fixed four calls to functions that no longer
  exist (`generate_hypotheses`, `run_diagnostic_suite`, `rank_hypotheses`,
  `apply_intervention_and_validate`) — the diagnosis stage could not run
  at all before this. Rewired to `failure_detection.extract_blunders()` →
  `diagnostic_experiment.run_empirical_investigation()` →
  `intervention.extract_dominant_fix()` → `intervention.evaluate_intervention()`
  (run twice: in-sample on the diagnostic blunders, then genuinely
  held-out). Added an adapter translating the new result shapes into the
  field names `dashboard/report_html.py` already expects, so that file
  didn't need to change. No fabricated `p_value` — the new methodology
  has no significance test, so the report shows none rather than one that
  would look statistical but isn't.

*Verified by running the fixed modules directly against the real engine
(`SearchAgent`, `environment.Board`) — not just import/syntax checks.
`run_full_pipeline.py`'s `main()` itself could not be run end-to-end in
the development sandbox (missing `torch`, network-blocked Lichess fetch);
the import chain and the step 3-5 logic were verified with synthetic data
standing in for the blocked data source.*

### Added — move commentary system

New subsystem in `analysis/`, deliberately separate from
`diagnosis/llm_reasoner.py` (see `README.md`/`CONTRIBUTING.md` for the
scope-boundary rationale):

- `rule_based_commentary.py` — filters out moves that don't need an LLM:
  only-legal-move, forced check replies (below a configurable eval-swing
  threshold), simple recaptures, checkmate.
- `commentary.py` — batches whatever's left into a strict-JSON LLM call.
  Supports Gemini and a local Ollama model.
- `commentary_report.py` — renders rule-based + LLM results as a
  browser-openable HTML report (per-move board snapshot with from/to
  square highlighting, colored tag badge, eval line, search-coverage
  note), in the same dark/gold visual language as
  `dashboard/report_html.py`.
- `run_commentary_demo.py` — end-to-end runnable driver: plays a self-play
  game, runs the rule-based filter, calls the LLM for the rest, writes the
  HTML report.

### Added — live search-comparison capture (`engine/search.py`)

- `alpha_beta()` / `search_best_move()` can now record `root_trace`: every
  root move actually compared *in the same live, time-bounded search call*
  that produced the final move, each with its score and a `fully_searched`
  flag (false if the deadline cut that move's evaluation short).
  `search_best_move()`'s `info` dict also gains `n_legal_at_root`.
- This replaced an earlier approach (`evaluate_root_candidates()`, a fresh
  *unconstrained* re-search after the fact) which was found to produce a
  misleading comparison: since `SearchAgent` runs on a wall-clock time
  budget (not a fixed depth/node count), it frequently compares only a
  handful of legal moves before time runs out — observed as low as 3-4 of
  25-30 in the middlegame. An unconstrained reconstruction made the played
  move look like it lost to alternatives the live search never actually
  reached, which is a different and less honest claim than "what did the
  agent's real decision weigh." `evaluate_root_candidates()` is kept for
  the different, legitimate question "what would unlimited time have
  found" and is explicitly documented as not equivalent to `root_trace`.
- Verified across a 12-ply game: 9/12 plies mismatched (played move not
  top-ranked) using the reconstruction approach; 0/12 mismatched once
  `root_trace` was used instead.

### Added — Ollama (qwen2.5) support in `commentary.py`

- `call_commentary_llm(payload, provider="ollama"|"gemini", ...)` — Gemini
  path unchanged (`_call_gemini`); new `_call_ollama` posts to a local
  Ollama server's `/api/chat`.
- `num_ctx` (8192) and `num_predict` (4096) set explicitly, since Ollama's
  own defaults (2048 context, a small model-defined output cap) are too
  small for a multi-move batched payload and truncate silently otherwise.
- `annotate_game()` now detects when a batch response is missing entries
  for moves it was asked about, and automatically retries each missing
  move individually as a fallback. Root cause of the dropped-move behavior
  observed with `qwen2.5:7b-instruct`: NOT a token-budget issue (a 2-move
  batch reproduced it, well under any reasonable token limit) — smaller
  instruct models under JSON-mode constrained decoding can simply stop
  after one array entry regardless of chunk size. Verified the fallback
  recovers 5/5 moves in a batch that would otherwise have returned 1/5.

### Fixed — bugs found while building/testing the above

- **`run_commentary_demo.py`**: `.env` file was never actually loaded
  before the `GEMINI_API_KEY`/Ollama-reachability check ran —
  `load_dotenv()` lived inside `commentary.py`, which wasn't imported
  until after the check already needed to pass. Only a real shell
  `export` worked; a `.env` file was silently ignored. Fixed by loading
  it at the top of the script itself.
- **`commentary.py`**: `eval_before_cp`/`eval_after_cp`/`score_cp` fields
  were named as centipawns but never actually converted —
  `engine.evaluation.evaluate()` returns pawn-scale floats (a queen swing
  reads ~9.0, not ~900). Now genuinely converted (×100, rounded), and the
  prompt states the scale explicitly.
- **`commentary.py`**: closed a specific contradiction observed in real
  model output — an explanation asserted an alternative move "was
  significantly better" while also noting, in the same sentence, that it
  "did not have time to fully evaluate it." Added a hard rule naming this
  exact failure pattern: a `fully_searched: false` score may be mentioned
  as unreliable, never used to assert a comparison.
- **`run_commentary_demo.py`**: `N_PLIES` cap (originally 20, arbitrary)
  raised to 200 so a demo run plays to actual game termination
  (`is_terminal()`) rather than stopping mid-game.