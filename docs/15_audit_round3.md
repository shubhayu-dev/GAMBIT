# 15 — Technical Audit Report (Round 3): Mate Handling, Real-Data Composition, Neural Validity

Scope: no new AI techniques, no neural model optimization, no manufactured
intervention. Purpose: establish whether current measurements are valid
before scaling 96 → 500-2000 positions.

---

## 1. Bugs found (this round)

None new in the core pipeline. One **test-authoring bug** was found and
fixed in this round's own test script (a broken conditional produced `None`
instead of an evaluation — caught immediately by a crash, fixed, not a
pipeline issue). One **engine robustness gap** was found while constructing
test positions: `Board(fen)` does not validate that a loaded FEN is a
legal/reachable chess position. A hand-built (invalid) test FEN — where the
side NOT to move was already in check, impossible from legal play — caused
a rook to pseudo-legally "capture" the enemy king during search, crashing
`is_checkmate()`. Not fixed in this pass (out of scope: audit, not
optimization); recommended fix: reject FENs where the side not to move is
in check.

## 2. Bugs fixed (carried in from prior rounds, re-verified here)

- Regret formula (`R(s,a)=V(s,a*)-V(s,a)`, not a single-move delta) — Round 1.
- Mate-score clipping in regret (±20 pawns) — Round 2, **now exhaustively
  tested** (10 constructed cases, section below): all pass, sign convention
  verified consistent (regret ≥ 0 always, exactly 0 iff chosen==reference,
  bounded ≤40 whenever mate is involved).
- Game-level neural train/test split — Round 2, **re-verified here**: zero
  game-ID overlap confirmed by direct set intersection (175/37/38 train/val/
  test games).

## 3. Tests added (this round)

`diagnosis/test_mate_score_audit.py` — 10 constructed cases: no mate, agent
plays reference exactly, reference leads to mate (agent misses it, two
variants), agent finds the mate, mate-in-1 (White and Black to move
explicitly), mate-in-2 (using a **real** validated Lichess puzzle position,
not a hand-built one, after the hand-built one exposed the FEN-validation
gap above), both sides' evaluations involving mate scores. All 10 pass.
Explicit finding documented in the test output: **clipped mate regret is a
bounded approximation, not exact regret** — it cannot distinguish "slightly
missed a forced mate" from "catastrophically missed one"; both saturate at
the same clip value. This is now stated in the code, not just implied.

---

## 4. Data provenance (96 positions)

Source: official `Lichess/chess-puzzles` dataset via HuggingFace's dataset
viewer (page 1 of the paginated HTML table — see `docs/13`). **This is
puzzle data: each position is the moment immediately after an opponent's
in-game mistake**, not an ordinary/representative game position. Puzzle
generation methodology means every position here was selected *because* a
forcing, usually decisive, continuation exists — themes confirm this
directly: 38/96 tagged "crushing", 26/96 tagged "mate" (14 mate-in-1,
11 mate-in-2).

**Unique underlying games: unknown.** The `GameId` column was not
transcribed into `data/lichess_puzzles_sample.csv` (only PuzzleId, FEN,
Moves, Rating, Themes were kept). PuzzleIds are sequential in the source
dataset ordering, which is suggestive of broad game diversity but **not
verified** — stated plainly rather than assumed.

**Do not use this data as a representative sample of ordinary chess.** It
should be used for **evaluation/failure-analysis testing** (does the agent
find forcing tactics that a real, curated source says are findable) — not
as neural training data implying it represents typical play, and not
folded into the self-play-derived capability profile without a label
distinguishing "puzzle-sourced" from "self-play-sourced" positions.

## 5. Dataset composition (96 positions)

| Property | Value |
|---|---|
| Game phase (Lichess tags) | 50 middlegame, 43 endgame, 3 opening |
| Rating | min 538, max 2869, mean 1488, median 1470, stdev 536 |
| Side to move | 53 White, 43 Black |
| Total material (pawns=1) | min 5, max 76, mean 45.1, stdev 19.0 |
| Branching factor | min 2, max 52, mean 29.9, stdev 11.2 |
| Top themes | middlegame(50), short(49), endgame(43), crushing(38), advantage(32), long(27), mate(26), master(15), mateIn1(14) |
| Duplicate FENs within the sample | 0 |
| Overlap with self-play benchmark (12/4) | 0 |
| Overlap with neural training pool (7,474) | 0 |

---

## 6. Mate-score methodology (formal statement)

Regret is computed as the difference between the position's evaluation
after the reference move and after the chosen move (sign-adjusted for side
to move), **with both evaluations clipped to ±20 pawns before differencing**.
When either evaluation is a genuine mate score (~±99999, meaning the
reference search found a forced mate), the clip converts an effectively
unbounded/terminal value into a finite pawn-equivalent. **This is a
practical bounded-regret approximation, not mathematically exact regret.**
It correctly preserves ranking (missing a mate always costs more regret
than a normal suboptimal move, verified: mate-involved cases in the test
suite ranged 10.0-24.85, non-mate cases 0.0-0.01) but cannot distinguish
degrees of missed-mate severity beyond the clip bound.

## 7. Regret validity

10/10 constructed mate-handling tests pass. Sign convention verified
consistent across White-to-move, Black-to-move, and all mate combinations:
regret is never negative, and is exactly 0 iff the agent's move matches the
reference move — in every one of the 10 cases, not just the ones checked in
Round 1.

**Regret distribution on the real 96-position sample:**

| Statistic | Value |
|---|---|
| mean | 4.488 |
| median | 0.000 |
| stdev | 8.256 |
| min / max | -8.170 / 27.540 |
| exact-zero (agent matched reference) | 45/96 (46.9%) |
| high regret (>3.0 pawns) | 31/96 (32.3%) |

**By phase (our heuristic classifier):** middlegame n=73, mean +4.829;
endgame n=23, mean +3.407. (Note: our heuristic found **zero** opening
positions in this sample despite Lichess tagging 3 as opening — consistent
with the known 76%-agreement miscalibration, section 9 below.)

**By side to move:** White n=53, mean +3.048; Black n=43, mean +6.263. **This
gap should be read as exploratory, not established** — with stdev=8.256
overall and n in the 40s-50s per group, a ~3-point gap is well within
plausible noise.

**By branching-factor tercile:** low 4.695, mid 4.924, high 3.845 — flat,
no clear monotonic relationship with regret at this sample size.

**Conclusion: no dimension here should be called a "weakness" with any
confidence.** The only clean, robust fact is the overall bimodal-ish shape
(47% exact matches, 32% notably high) — consistent with puzzle data's
forcing nature (you either find the tactic or miss it badly), not with a
smooth distribution.

## 8. Neural model validity — root cause analysis

**Question: insufficient data, target noise, representation, capacity,
distribution shift, or a bug?** Evidence gathered this round, systematically:

- **Not a bug**: manual inspection of 15 examples (target vs. material-only
  vs. classical vs. neural prediction) shows self-consistent, correctly
  signed values throughout; the training pipeline behaves as documented.
- **Not primarily distribution shift**: train/val/test splits have nearly
  identical phase composition (train 33%/67% opening/middlegame; val and
  test both ~33%/67% too) and similar mean branching factor (29.9 vs.
  33.2/33.3) — the game-level split did not accidentally create a skewed test set.
- **Partially target difficulty, evidenced directly**: on a fresh
  same-distribution sample, even the two static hand-coded baselines
  (material-only, classical) only reach R²≈0.69-0.75 against the training
  target — because the target is `MaterialAgent`'s **2-ply-searched**
  evaluation, and ~21% of positions have the search finding something (a
  capture sequence, a tactic) that is invisible to any 0-ply static
  evaluator by construction. This caps how well *any* static evaluator,
  neural or not, can fit this particular target.
- **Primarily overfitting (capacity vs. data), evidenced directly**: on that
  same near-training-distribution sample, the neural model reaches
  MAE=0.398, R²=0.958 — far exceeding what the target's inherent
  search-vs-static gap should allow, meaning it is fitting training-set-
  specific detail (106,881 parameters against ~1,400 unique training
  examples, a 76:1 ratio) rather than the generalizable component of the
  function. This is confirmed decisively by the actual held-out (different
  games) test result: **R²=-0.074**, MAE=3.702 — collapsing far below even
  the static baselines' ~0.7 R² ceiling.

**Conclusion: the dominant cause is overfitting from excess capacity
relative to data size, compounded by a target that is inherently harder
than the baselines' own ceiling suggests it should be.** More data alone
would help but is not guaranteed sufficient while a 769→128→64→1 network
this size is fit on order-1000 examples; a smaller/more regularized
architecture, or substantially more data (the original request's caution
against "more data will fix it" as a lazy answer is taken seriously here —
this is a capacity problem first, a data-volume problem second).

## 9. Game-phase validity

Investigated rather than assumed correct on either side. Our classifier
(`Board.game_phase()`) uses fullmove number (≤10) and total piece count
(≥28 = opening; ≤12 = endgame; else middlegame) — a simple, deterministic,
purely structural rule. Lichess's theme tags are **human/community-curated
labels** applied during puzzle creation, not a formal rule either — so
neither side is uncontestable ground truth.

**Disagreement (76% overall agreement, from `docs/13`) is concentrated and
one-directional**: our classifier calls positions "middlegame" that Lichess
calls "endgame" in every observed mismatch case, never the reverse. This
points to our piece-count threshold (≤12) being stricter than how chess
players/Lichess actually use the term "endgame" — many real endgames
(rook endings, minor-piece endings) retain more than 12 points of material
while still being tactically/strategically endgame-like (few pieces,
king activity mattering, etc.).

**Decision: Option C — store both independently.** Neither classifier
should silently replace the other. `game_phase` (ours, deterministic,
always available) and `lichess_phase` (real tag, only available for
Lichess-sourced positions) should both be recorded on every record where
available, letting downstream analysis choose or compare. **Not yet
implemented in the schema** — recorded here as the decision and as the
concrete next action item, not done in this pass (would be a schema/feature
change, out of scope for an audit-only round).

## 10. Anomaly / disagreement validity (exploratory only, per this round's real data)

**Disagreement**: value disagreement mean 3.647 (n=96); correlation with
|regret| = **0.026** — essentially zero, and far more reliable than the
earlier n=12 estimate (0.212) precisely because n=96 is larger. This is a
clean null result, not noise misread as signal: **at a reasonable sample
size, evaluation disagreement does not predict regret.** Move disagreement
(40%, n=30 subsample) remains low, consistent with earlier findings.

**Per the audit's own instruction — since the neural model failed its
validity audit (section 8), disagreement is not treated as a trustworthy
diagnostic signal.** It remains available for exploratory use only.

**Anomaly detection** (contamination=0.15): 15/96 flagged (15.6%, close to
the parameter as expected). Anomalous positions have modestly higher mean
|regret| (6.889 vs. 4.499 for normal) — a real if unproven correspondence
at this n, an improvement over the earlier (n=12) analysis where the
correspondence was noisier. Anomalies skew endgame (9/15, vs. endgame being
only 24% of the dataset) — **an unusual-ness signal, not a demonstrated
failure signal**, per the audit's explicit framing: anomaly means
statistically rare in the feature space, not necessarily a bad decision.

---

## 11. Current claims and confidence

| Claim | Evidence available? | Confidence |
|---|---|---|
| Agent makes measurable mistakes | Yes — 32.3% of real puzzle positions show >3-pawn regret | **High** (mate-clipping-corrected, real data) |
| Opening is a weakness | Only 3 real opening-tagged positions exist; our heuristic found zero | **None** — not enough data to say anything |
| Middlegame is a weakness | n=73 (real data), mean regret 4.829, similar to endgame's 3.407 | **Low** — no clear separation from endgame at this n |
| Endgame is a weakness | n=23 (real data), mean regret 3.407, actually lower than middlegame | **Low** — same caveat, and direction is opposite of "weakness" |
| Neural model is useful | MAE/R² worse than every baseline including constant, on real held-out test | **None — actively contradicted** |
| Neural model generalizes | Test R²=-0.074; train R²=0.976 (game-level split) | **None — actively contradicted, overfitting confirmed** |
| Classical/neural disagreement detects failures | Correlation with regret = 0.026 at n=96 | **None** — clean null result |
| Anomalies correspond to failures | Modest gap (6.89 vs 4.50 mean regret) at n=96, phase-skewed | **Low-moderate, exploratory only** |
| LLM can generate useful hypotheses | Format compliance verified (`data/llm_reasoning_audited.json`); no case yet where its hypothesis was later confirmed by adequately-powered evidence | **Untested** — mechanism works, no validated instance yet |
| Intervention improves the agent | No hypothesis has passed section 8-11-style validity checks yet | **None** — correctly not attempted (Week 7/audit Section 12 null results) |

---

## 12. Blocking issues, in priority order

1. **Neural model does not generalize** (R²=-0.074) — must not be used for
   disagreement analysis, training-signal purposes, or any claim of
   usefulness until re-architected (smaller/regularized) or given a
   fundamentally different target (real outcomes, Stockfish).
2. **`game_phase()` heuristic is measurably miscalibrated** (76% agreement,
   one-directional endgame under-detection) — should not be trusted alone
   for phase-based claims; Option C (store both) decided but not implemented.
3. **Lichess `GameId` not captured** — cannot verify unique-game count or
   game-level split safety for this data source specifically, only for the
   self-play data.
4. **Only 96 Lichess positions, and only 1 page's worth manually
   transcribed** — real but small; per-subgroup n's (3 opening, 23 endgame,
   73 middlegame) are too small for confident phase-based claims.
5. Standing from prior rounds: no Stockfish, no PyTorch, no live LLM API,
   no Parquet.

## 13. READY / NOT READY decision

# **NOT READY**

The measurement *pipeline itself* (regret formula, mate handling, node
counting, game-level splitting mechanics) is now solid — thoroughly tested,
bugs found and fixed at each prior round, and this round's 10-case mate
audit passed cleanly. **But two of the three main analytical components
built on top of it are currently non-functional or unvalidated**: the
neural model does not generalize at all, and the phase classifier is
measurably wrong in a known direction. Scaling the dataset now would
produce a larger pile of measurements built on a broken neural model and a
miscalibrated phase label — exactly the "2000 potentially wrong
measurements instead of 12" failure mode flagged earlier in this project's
own history.

**Blocking issues to resolve before scaling (in order):** items 1 and 2
above. Items 3-4 are not strictly blocking (they don't invalidate what's
already measured) but should be addressed in the same pass as scaling,
not after.

## 14. Exact next implementation step

1. Fix `game_phase()`: recalibrate the piece-count threshold against the 96
   real Lichess-tagged positions now available as ground truth (e.g. try a
   few threshold values, pick the one maximizing agreement, or move to a
   less naive rule — e.g. queens-off-the-board as a stronger endgame signal
   than raw piece count). Re-validate against a fresh page of real data if
   possible, not the same 96 used to tune it.
2. Do NOT scale the dataset until (1) is done and the neural model question
   is at least decided (either: try a substantially smaller network — e.g.
   769→32→1 — as a cheap test of the capacity-vs-data hypothesis from
   section 8, or explicitly shelve the neural agent until Stockfish/real
   outcomes are available, per section 8's diagnosis).
3. Add `GameId` to future Lichess transcription so game-level split safety
   can actually be verified for real data, not just self-play data.
4. Only after (1) and a decision on (2): proceed to the dataset-scaling
   plan from the prior round's Step 13 (500-2000 positions, diversity
   preserved, puzzle vs. ordinary-game provenance kept separate, both phase
   labels stored per Option C).
