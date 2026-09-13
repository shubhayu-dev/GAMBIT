# 12 — Audit Report: Validating GAMBIT's Experimental Methodology

**Scope of this audit, per the request that triggered it:** determine whether
the current measurements and pipeline are correct, BEFORE scaling the
dataset. No new features were added. One real bug was found and fixed (a
correctness repair the audit explicitly called for, not scope creep). No
dataset scaling was performed.

---

## 1. Current architecture

Unchanged from `docs/11_integration_addendum.md`: chess environment (custom,
perft-verified) → `SearchAgent` (classical, alpha-beta + PST eval) and
`NeuralAgent` (MLP value net) → experiment runner → trajectory data →
analysis/diagnosis/intervention → static HTML report. This audit did not
change the architecture, only the regret formula inside the experiment
runner and added node-count instrumentation to the search.

---

## 2. Data provenance — nailed down explicitly

**All positions used anywhere in this project so far are self-play
generated. None come from Lichess or any other real-game source.**

Specifically, traced through the actual code:
- The 16-position benchmark (12 diagnostic / 4 held-out) used for every
  Week 4-8 result and the integration demo comes from
  `experiments/benchmark.py:generate_benchmark(n_games=16, seed=1)` —
  `RandomAgent` vs `RandomAgent` self-play, one position sampled per game.
- The neural model's ~2000-position training set comes from
  `engine/self_play_data.py:generate_value_training_data(n_games=250, seed=42)`
  — also `RandomAgent` vs `RandomAgent` self-play, a **separate, independently
  seeded** batch from the 16-position benchmark.

This was always documented as a deviation (`docs/10`, `docs/11` — "no network
access to fetch Lichess data"), but the language in conversation had drifted
to calling it "the dataset" without restating this every time, which is what
prompted this check. To be unambiguous for a professor-facing writeup: **zero
Lichess positions have been used in this project as of this audit.**
`docs/07_dataset_benchmark_spec.md` and `docs/11_integration_addendum.md`
already list "get real Lichess data" as a standing action item; nothing
about this audit changes that item, it just confirms it's still fully open.

---

## 3. Dataset composition

- **16-position benchmark:** 12 diagnostic / 4 held-out. Phase composition:
  6 opening, 10 middlegame, **0 endgame** (confirmed by direct inspection —
  `RandomAgent` self-play at ≤50 plies rarely reduces material below the
  endgame piece-count threshold). Each position comes from a distinct
  self-play game (verified: `source_game` index 0-15, no duplicates).
- **~2000-position neural training set:** sampled from a 7474-position pool
  (7165 unique — 309 exact-duplicate FENs within the pool itself, plausible
  given early-game random-play convergence). Phase composition: 651 opening
  (32.5%), 1349 middlegame (67.5%), **0 endgame** — same structural gap as
  the smaller benchmark, for the same reason.
- **Train/test contamination check:** exact FEN-set intersection computed
  directly. **Zero overlap** between the 16-position benchmark (diagnostic +
  held-out) and the neural training pool (all 7474, and the actual sampled
  2000). No leakage.
- **Representativeness:** biased away from endgames entirely (structural,
  not sampling noise — see above), and skewed toward middlegame over opening
  by construction of `generate_benchmark`'s ply-count randomization
  (`rng.randint(6, 50)`, which spends more plies in middlegame territory than
  opening on average).

---

## 4. Neural model validity — the most important negative finding of this audit

**Verified facts, from direct inspection of the trained model:**
- Input: 769-dim (12 one-hot piece planes × 64 squares + side-to-move flag).
- Target: `MaterialAgent(depth=2)`'s minimax-searched evaluation (material +
  mobility only, no PST), **not** game outcome (tested first, abandoned —
  only ~12% of self-play games reach a decisive result within 150 plies,
  making outcome labels overwhelmingly "draw" and useless as a signal — see
  `docs/11_integration_addendum.md`), and **not** Stockfish (unavailable).
- **A real bug was caught and fixed during original training:** ~9/2000
  positions had mate scores (~99999) from the reference search, which
  dominated the MLP's squared-error loss before being clipped to ±20 pawns.
  Documented in `engine/neural_agent.py`.
- Normalization: **none** — inputs are already binary (0/1), so this is
  defensible, but was previously undocumented; now stated explicitly here.
- Split: 1600 train / 400 test (80/20, `random_state=7`). `MLPRegressor`'s
  `early_stopping=True` additionally carves an internal ~10% validation
  split out of the 1600 (sklearn default), so the effective fit set is
  smaller than "1600" suggests — roughly 1440.
- Positions from the same self-play game **can** appear across train/test
  (no game-level grouping was done in the split) — a real, if probably minor,
  leakage risk not previously flagged. Not fixed in this audit (would require
  re-splitting and re-training); flagged as an action item.
- **Architecture: 769 → 128 → 64 → 1, 106,881 trainable parameters**, fit on
  ~1600 (effectively ~1440) examples — a **66.8:1 parameter-to-example
  ratio**. This is a lot of capacity for very little data.
- Final training loss (sklearn internal, MSE-based): 0.092. Internal
  early-stopping validation score: 0.350. `n_iter_`: 42 (stopped early).

**Baseline comparison (the audit's central question: does the neural model
add information beyond a trivial baseline?) — computed on the actual 400-position
held-out test set:**

| Predictor | MAE | RMSE |
|---|---|---|
| Constant (mean of train target) | 4.466 | 6.189 |
| Material-only eval (no search, no PST) | 2.299 | 3.714 |
| **Classical `evaluate()` (PST + material + mobility, no search)** | **2.163** | **3.454** |
| Neural MLP | 2.957 | 4.359 |

**Answer: no, not against the baselines that matter.** The neural model beats
the constant baseline (real signal learned) but is **worse than both the
material-only baseline and the classical PST-based evaluation function** at
predicting the very target it was trained to approximate. Combined with the
66.8:1 parameter ratio and the train/test R² gap previously observed
(0.930 train / 0.504 test), this points to overfitting: the model memorized
idiosyncrasies of its 1600 training positions rather than learning a
generalizable evaluation. **The neural model should not currently be treated
as an improvement on the classical evaluator for any purpose** — including
the classical-vs-neural disagreement analysis in `docs/11`, which was
comparing a demonstrably-worse predictor against a demonstrably-better one,
not two comparably-strong independent opinions.

---

## 5. Classical agent validity

No new issues found beyond what `docs/10`/`docs/11` already documented
(`MaterialAgent` as a shallow, non-Stockfish reference; `SearchAgent`'s
alpha-beta + PST evaluation verified via perft and mate-finding tests in
Week 1/2). One clarification from this audit: `SearchAgent` (the agent under
test) and `MaterialAgent` (the reference) share very similar machinery —
both are alpha-beta search over material-based evaluation, differing mainly
in depth (agent's `max_depth` vs reference's fixed depth 2) and whether PST
terms are included. This is worth naming plainly: **the "classical vs
neural" framing in `docs/11` is really "two agents sharing one search
algorithm, differing only in evaluation source."** That's a legitimate,
useful controlled comparison (it isolates the evaluation function as a
variable) but should not be described more broadly as "classical AI vs
modern AI" — external reviewer feedback on this point was correct and is
adopted here.

---

## 6. Regret validity — the headline finding

**Confirmed bug, via `diagnosis/test_regret_audit.py`:** `experiments/runner.py`'s
regret formula did **not** implement the documented
`R(s,a) = V(s,a*) - V(s,a)` (docs/08_metrics_spec.md). It computed the change
in position value across the agent's own single move (before vs. after),
and **never used the `reference_move` field it stored in every trajectory
record.**

**Direct proof:** constructed positions where the agent is forced to play
exactly the reference move. The documented formula correctly returns regret
≈ 0 in every such case; the actual `runner.py` formula returned **+0.560**
(White, start position) and **−0.370** (Black) — nonzero, in cases where
regret should be exactly zero by definition. Full test suite and six
constructed cases (reference move / worse move / blunder / white-to-move /
black-to-move / mate-in-1) are in `diagnosis/test_regret_audit.py`, runnable
independently.

**Fix applied** (`experiments/runner.py`): regret is now computed by
comparing the resulting evaluation after the chosen move to the resulting
evaluation after the reference move, sign-corrected for side to move. The
old (broken) formula is preserved as a separate field, `single_move_eval_delta`,
for transparency and comparison — it is a real, if differently-named,
quantity (how much the position's static evaluation changed across one ply),
just not "regret" in the documented sense.

**Impact — this changes the previously reported findings, substantially.**
Re-running the identical 12 diagnostic positions (same self-play seed, same
agent, same everything except the bug fix):

| | Old (buggy) mean regret | Corrected mean regret |
|---|---|---|
| Overall | ≈ −1.6 (Week 4 report) | **−0.048** |
| Positions with regret = exactly 0 | 0/12 | **8/12** |

**The corrected capability profile inverts the Week 5 finding:**

| Dimension | Old (buggy) decision accuracy | Corrected decision accuracy |
|---|---|---|
| Opening (n=5) | 60.0% | 60.0% (unchanged) |
| Middlegame (n=7) | **0.0%** | **100.0%** |
| Overall | 25.0% | 83.3% |

**The "middlegame is the weakness" finding from Week 5 — and everything
built on it in Weeks 6, 7, and the integration demo's disagreement/anomaly
analysis — was an artifact of the regret bug.** With the corrected metric,
the weak dimension is actually **Opening** (60% accuracy, n=5), and
middlegame is the agent's *strongest* measured dimension (100%, though n=7
is still small). This is reported plainly, per the instruction not to hide
negative results: a substantial portion of prior work needs to be treated
as measuring the wrong thing, not as wrong conclusions about the agent.

---

## 7. Disagreement analysis — audited

Both requested signals now computed separately (evaluation disagreement was
already tracked; move disagreement was not, and is a genuine audit gap that's
now closed):

- **Evaluation disagreement** (n=12): mean 3.186, median 2.574, stdev 2.903,
  range 0.014–10.078. Correlation with |regret| (still the buggy metric at
  the time this was computed — see section 11 for why it wasn't worth
  re-running with corrected regret at n=12): 0.212. **Explicitly exploratory
  only at this sample size — not a claim of a real effect**, per the
  request's own instruction.
- **Move disagreement** (newly computed): classical `SearchAgent` and
  `NeuralAgent`, both actually searched (not just raw eval compared),
  agreed on the chosen move in only **2/12 positions (16.7%)**. Given
  section 4's finding that the neural evaluator is a *worse* predictor than
  the classical one, this low agreement rate should be read as "the weaker
  evaluator is steering search differently," not as "two comparably valid
  opinions disagree."

---

## 8. Anomaly analysis — audited

Exact inputs confirmed: `[regret (buggy formula, at the time this ran),
time_used, complexity (legal move count), disagreement, game_phase code]`,
**no feature scaling applied**, `contamination=0.25` (chosen arbitrarily,
not tuned), `seed=0`. 3/12 (25%) flagged — mechanically close to the
contamination parameter itself, which is expected and not informative on
its own at this n.

**Anomaly status does not track |regret| cleanly:** flagged positions had
mean |regret| 2.273 vs. 1.613 for non-flagged — a small, unreliable gap at
n=12 (3 vs. 9). **Correctly interpreted per the request's framing: anomaly
means "statistically unusual in this 5-feature space," not "the agent made
a bad decision."** Nothing here should be read as identifying failures.

---

## 9. Current contradictions and limitations

**The opening/middlegame contradiction, investigated directly** (Week 5's
weakness dimension vs. where anomalies clustered): the specific candidate
explanations were tested against the data:

- *"Opening positions dominate the dataset"* — **false**, directly
  contradicted by the phase counts (5 opening vs. 7 middlegame; middlegame
  is the majority class).
- *"Anomaly features aren't measuring decision quality"* — **supported**;
  anomaly status doesn't track |regret| cleanly (section 8).
- *"Middlegame weakness is based on a different metric"* — **supported, and
  now moot**: the middlegame weakness itself has been shown to be a bug
  artifact (section 6), so this "contradiction" partly dissolves once the
  regret bug is fixed — the two signals were never going to agree because
  one of them wasn't measuring anything real.
- *"Sample size too small"* — **supported and dominant**: n=12 total, single
  digits per phase/anomaly bucket, nowhere near enough to trust either
  signal independently, before or after the regret fix.

**What remains statistically unreliable, in full, after this audit:**
1. Every number from this 12/4-position dataset, including the corrected
   ones — n is simply too small for any of it to generalize.
2. The neural model's practical usefulness (shown worse than simple
   baselines — section 4).
3. The classical/neural disagreement correlation (0.212 at n=12).
4. The anomaly detection results (contamination-driven flag count,
   unscaled features, n barely above the method's own stated floor).
5. All Week 6/7 hypothesis test results — computed with the buggy regret
   metric and not yet re-run with the fix (see section 11 for why not, and
   what would need to happen first).

---

## 10. LLM reasoning validity

Redone in the stricter format the audit requires — each hypothesis given
explicit evidence-for, evidence-against, confidence, a proposed experiment,
and a measurable prediction; no hypothesis stated as fact. Full output:
`data/llm_reasoning_audited.json`. Headline: **all three hypotheses (search
depth, evaluation quality, time budget) are rated "very low" or "low"
confidence** — not because they're implausible, but because the instrument
used to test them (the regret metric) needed fixing, and even after fixing
it, n=12 is too small to distinguish any of them from noise. This matches
what sections 6-9 independently found through direct computation, which is
itself a small positive sign that the reasoning step is properly grounded in
the evidence rather than freelancing.

---

## 11. Controlled experiment (section 9's required deliverable)

**Hypothesis tested:** H1 (search depth) on the newly-correct weak bucket
(opening positions, n=5), using the **corrected** regret metric and real
node-count instrumentation (added during this audit — search previously had
no node counting at all).

**Control:** `SearchAgent(max_depth=2)`. **Treatment:** `SearchAgent(max_depth=4)`.
Same 5 positions, generous fixed 2000ms time budget on both so depth (not
time) is the isolated variable.

| | Control (depth=2) | Treatment (depth=4) |
|---|---|---|
| mean regret | +0.210 | −0.312 |
| mean \|regret\| | 0.222 | 0.356 |
| blunder rate (\|regret\|>1.0) | 1/5 (20%) | 1/5 (20%) |
| reference-move agreement | 3/5 (60%) | 2/5 (40%) |
| mean nodes searched | 180 | 1,933 (10.7×) |
| mean depth actually reached | 2.0 | 3.2 (not always 4 — time-budget-limited on some positions even at 2000ms) |
| mean runtime | 0.295s | 2.002s (6.8×) |

**Result: mean |regret| got *worse* with more depth (+0.134), reference
agreement dropped, and cost went up 10.7× in nodes / 6.8× in time.** At n=5
this is nowhere near statistically distinguishable from noise (Wilcoxon
isn't even computable meaningfully below n=6) — **this does not refute H1
either.** It simply provides no support for it, at real computational cost.
Full output and code: `diagnosis/run_audit_section9.py`.

---

## 12. Intervention and held-out validation

**Per the audit's own standard ("if the experiment supports the
hypothesis...") — section 11's experiment did not support H1.** The
methodologically correct action is therefore **not** to construct and
validate an intervention based on it — doing so would be exactly the
"manufacture evidence that GAMBIT works" failure mode this audit exists to
prevent. So section 12 is deliberately **not executed** in this pass.

For context: Week 7 already exercised the intervention + held-out-validation
*mechanism* (adaptive search depth, validated once on 4 held-out positions)
using the old buggy regret metric, and found a genuine null result (0%
change — the deeper search chose identical moves to the shallower one on
all 4 positions). That result showed the plumbing works; it was also,
in retrospect, evidence-free in the same way section 11 is. Section 12 will
be properly exercisable once a hypothesis is actually supported by
adequately-powered evidence (see section 14).

---

## 13. What remains statistically unreliable (consolidated)

Everything in this project's quantitative results, without exception, is
currently underpowered: n=12/4 for the classical pipeline, n=1600/400 for
the neural model (large in absolute terms, but with a 66.8:1
parameter-to-example ratio that produces real overfitting), and every
correlation, anomaly rate, and hypothesis test computed so far should be
read as "the mechanism runs and produces a number," not "this number
reflects a real, generalizable pattern in agent behavior." The one exception
is section 6's regret-bug finding itself, which is not a sample-size
question — it was proven with constructed positions, independent of n.

---

## 14. Exact next steps for scaling (only now, after the audit)

In order:

1. **Re-run Weeks 4-6 on the existing 12/4-position data with the corrected
   regret metric** (cheap — already done in this audit, see section 6/11) —
   confirmed the qualitative picture changes substantially. Any further work
   on this exact dataset should use the corrected metric from
   `trajectories_run_a56f3c50.jsonl` onward, not the original buggy file.
2. **Fix the train/test game-grouping gap in the neural model's split**
   (section 4) and retrain — not done in this audit, flagged as the next
   correctness item before trusting the neural model further.
3. **Only then**, scale the self-play benchmark from ~16 to 500-2000
   positions, per the original request, with the corrected regret metric
   built in from the start (not retrofitted after the fact) and explicit
   game-level grouping so no two positions from the same self-play game can
   land in different diagnostic/held-out splits or train/test splits.
4. Re-run the section 11 controlled experiment (and any others) at that
   scale, where Wilcoxon and correlation statistics actually mean something.
5. Only after (3) and (4): revisit whether an intervention (section 12) is
   actually warranted by adequately-powered evidence, and validate it
   properly on held-out data at that point — not before.
6. Standing action items from `docs/10`/`docs/11`, unchanged by this audit:
   real Lichess data, real Stockfish reference, PyTorch/CNN, live LLM API
   access, Parquet storage.

**Bottom line:** the audit found one real, proven, significant bug (the
regret formula) that changed the qualitative conclusion of the entire
diagnostic pipeline, one real negative result (the neural model doesn't beat
simple baselines), and confirmed that everything else is currently too
small-sample to trust either way. That is exactly the outcome an audit like
this is supposed to be able to produce — the goal was never to confirm
GAMBIT works, it was to find out whether it does, and where it doesn't yet.
