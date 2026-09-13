# 14 — Final Summary: Making GAMBIT Scientifically Defensible

Direct response to the audit-response document's "Definition of Done" and
"Final output required" sections. No new AI features were added in this
phase — everything below is either a correctness fix, a validation, or a
real (if partial) external data integration.

---

## 1. Files changed

**Fixed (correctness):**
- `experiments/runner.py` — regret formula now implements the documented
  `R(s,a)=V(s,a*)-V(s,a)` (was previously a single-move value delta that
  never used `reference_move`); added mate-score clipping (±20 pawns)
  found necessary once real Lichess data exposed it; old formula kept as
  `single_move_eval_delta` for transparency.
- `engine/search.py`, `engine/agents.py` — added real node-count and
  depth-reached instrumentation (`SearchAgent.last_search_info`), previously
  entirely absent.
- `engine/self_play_data.py` — added `generate_value_training_data_with_game_ids`
  and `game_level_split` to fix the neural model's train/test leakage gap
  (positions from the same self-play game could previously land in both
  splits).

**Added (audit tooling, real data, tests):**
- `diagnosis/test_regret_audit.py` — constructed-position test suite proving
  the regret bug and validating the fix.
- `diagnosis/run_audit.py` — sections 5-7 of the audit: disagreement
  (adds move disagreement, not just evaluation disagreement), anomaly
  detection audit, opening/middlegame contradiction investigation.
- `diagnosis/run_audit_section9.py` — one controlled experiment (depth 2 vs
  4) with real cost accounting.
- `engine/retrain_with_game_split.py` — retrains the neural model with a
  leakage-free game-level split; reports train/val/test/baseline comparison.
- `experiments/lichess_data.py` — loads and validates real Lichess puzzle
  data against our own engine.
- `data/lichess_puzzles_sample.csv` — 96 validated real Lichess puzzles.
- `docs/12_audit_report.md`, `docs/13_lichess_integration.md`,
  `docs/14_final_audit_summary.md` (this file).

---

## 2. Tests added

- 6 constructed-position regret tests (`test_regret_audit.py`): reference
  move played exactly (White and Black), slightly worse move, blunder,
  mate-in-1 played, mate-in-1 avoided. All pass against the corrected formula.
- 3 assertion checks for zero game-ID overlap across train/val/test splits
  (`retrain_with_game_split.py`) — passed (175/37/38 games, zero overlap).
- Data validation as a de facto test: every one of the 99 transcribed
  Lichess rows checked against our own legal-move generator (96 passed, 3
  correctly rejected).

---

## 3. Commands used (all reproducible)

```
python3 diagnosis/test_regret_audit.py          # regret bug proof + fix validation
python3 diagnosis/run_audit.py                  # sections 5-7: disagreement, anomaly, contradiction
python3 diagnosis/run_audit_section9.py         # controlled depth experiment, corrected metric
python3 experiments/lichess_data.py             # load + validate real Lichess data
python3 engine/retrain_with_game_split.py       # leakage-free neural retrain + baselines
```

---

## 4. Dataset statistics

| Dataset | n | Source | Endgame coverage | Notes |
|---|---|---|---|---|
| Classical benchmark (Weeks 4-8) | 16 (12 diag / 4 held-out) | Self-play | 0 | Regret now corrected |
| Neural training pool | 7,474 (7,165 unique) | Self-play | 0 | Game-level split now available |
| Lichess sample | 96 (validated) | **Real, official Lichess** | 43 | See `docs/13` |

Zero FEN overlap confirmed by direct set intersection between the 16/4
benchmark and the full 7,474-position neural pool.

---

## 5. Model statistics

**Original (leaky, position-level split):** 769→128→64→1, 106,881 params,
1600 train / 400 test. Test MAE 2.957, R²=0.504.

**Corrected (game-level split, no leakage):** same architecture, 1400 train
/ 300 val / 300 test, verified zero game overlap.

| Split | n | MAE | RMSE | R² |
|---|---|---|---|---|
| Train | 1400 | 0.310 | 0.928 | 0.976 |
| Val | 300 | 4.011 | 5.355 | 0.099 |
| Test | 300 | 3.702 | 5.253 | **-0.074** |

**The leaky split was overstating generalization substantially** (R² 0.504
→ -0.074 once leakage is removed). This is a materially worse and more
honest result.

---

## 6. Baseline results (test split, game-level, no leakage)

| Predictor | MAE | RMSE |
|---|---|---|
| Constant | 3.596 | 5.137 |
| Material-only | 2.333 | 3.749 |
| **Classical `evaluate()`** | **2.234** | **3.463** |
| Neural MLP | 3.702 | 5.253 |

**The neural model is now worse than every baseline, including the
constant predictor.** This is a stronger and more damning negative result
than the original (leaky) comparison found. The neural model, as currently
built (this architecture, this amount of data, this distillation target),
does not generalize to unseen games at all.

---

## 7. Experiment results

- **Regret bug**: proven and fixed. Reversed the Week 5 weakness finding
  (middlegame 0%→100% decision accuracy; opening now the weak dimension).
- **Controlled depth experiment** (opening bucket, n=5, corrected metric):
  no support for the search-depth hypothesis — regret got worse (+0.134),
  cost went up 10.7× in nodes. No intervention was built on this (per audit
  Step 9 — hypothesis not supported).
- **Second mate-clipping bug**, found via real Lichess data: mean regret
  14,584 → 4.488 after clipping. The bug only manifested on real tactical
  puzzles; self-play data never triggered it.
- **Real vs. self-play agent performance**: 46.9% reference-move agreement
  on 96 real Lichess puzzles vs. much higher agreement on self-play data —
  the classical agent is measurably worse on adversarially-curated real
  puzzles than on random self-play positions.
- **Phase heuristic accuracy**: 76% agreement with real Lichess theme tags,
  with a one-directional bias (under-detects endgames).

---

## 8. Known limitations (unresolved, stated plainly)

1. Lichess integration is at 96 positions, not the 500-2000 target —
   mechanical to extend (more HTML pages), not yet done.
2. The neural model does not currently generalize at all (negative test R²)
   — needs either much more training data, a smaller/regularized
   architecture, or a better target (real game outcomes / Stockfish) before
   it's usable for anything beyond demonstrating the pipeline runs.
3. `game_phase()` heuristic is measurably miscalibrated (76% accuracy) and
   not yet fixed.
4. The classical/neural disagreement and anomaly-detection numbers in
   `docs/11` and `docs/12` were computed before this session's fixes and
   have not been re-run against the corrected regret or the real Lichess
   data — they should be treated as superseded, not re-validated.
5. Still no Stockfish, no PyTorch, no live LLM API access, no Parquet — all
   previously documented, all still open.
6. n=96 (Lichess) and n=16 (self-play benchmark) are both still small enough
   that no conclusion here should be treated as final.

---

## 9. Exact next step

**Do not scale further yet.** In order:
1. Extend the Lichess CSV toward 500+ positions (mechanical: more HF viewer
   pages).
2. Re-run the full corrected pipeline (regret, profile, disagreement,
   anomaly) on the combined self-play + real Lichess dataset once (1) is done.
3. Only then decide whether the neural model is worth iterating on (smaller
   architecture, more data, or a different target) or shelving until
   Stockfish/real game outcomes are available — building on a model with
   negative test R² is not worthwhile yet.
4. Recalibrate `game_phase()` against the real Lichess theme tags now
   available as ground truth.
