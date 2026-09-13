# 11 — Integration Addendum: Classical + Modern AI + LLM Reasoning

Per Constitution Rule 7: **this changes the current GAMBIT scope.** It's
documented here as an addendum rather than silently folded into the
original `docs/03_scope.md`, so the history of what was originally scoped
vs what was added, and why, stays visible.

## What changed and why it's compatible with the original constitution

The original `03_scope.md` listed "large neural network training" and
"LLM-based autonomous agent" as explicitly out of scope. What was actually
requested and built here is narrower than what that excluded:

- **Not** deep RL, **not** a giant network: a small `sklearn.MLPRegressor`
  value model (see deviation note below on why not a CNN).
- **Not** an LLM that plays moves or modifies the agent unrestricted: an LLM
  reasoning layer that only interprets already-computed quantitative
  evidence and proposes hypotheses — every hypothesis still has to pass
  through the existing controlled A/B experiment protocol
  (`diagnosis/diagnostic_experiment.py`) before anything is acted on. This
  matches the original constitution's Rule 3/4 ("more ML ≠ better research",
  "never add technology because it sounds advanced") by giving the LLM a
  narrowly scoped, falsifiable role rather than authority.

New MUST-HAVE components added to the MVP:
- A neural (MLP) value agent, evaluated with the same search machinery as
  the classical agent, so the two are comparable.
- Classical-vs-neural evaluation disagreement as a diagnostic signal.
- Unsupervised anomaly detection (IsolationForest) over trajectory features.
- A real, working LLM-reasoning-layer interface (`diagnosis/llm_reasoner.py`),
  with its actual reasoning step performed manually once, honestly labeled,
  since no network/API access exists in this build environment.

## New deviations (same honesty pattern as docs/10, all caused by no network access)

1. **No PyTorch/TensorFlow.** The integration spec calls for a CNN over
   8×8×12 feature planes. Neither library is installable here. The neural
   agent uses `sklearn.MLPRegressor` instead — a real trained neural
   network, just not convolutional. The *input encoding* is still the
   12-plane one-hot board representation the spec describes
   (`engine/features.py`), flattened to 769 dims for the MLP, so a real CNN
   could consume the exact same encoding (reshaped to 12×8×8) with no
   pipeline changes whenever torch is available.

2. **Training target: distillation, not game outcome.** The spec's
   preferred value target is self-play game outcome. Tested first
   (`engine/self_play_data.py`): with RandomAgent self-play at this
   sandbox's scale, only ~12% of games reach a decisive result within 150
   plies — the rest hit the ply cap undecided, making outcome labels
   overwhelmingly "draw" and useless as a training signal. The model was
   instead trained to approximate `MaterialAgent`'s fixed-depth evaluation
   on ~2000 self-play positions (`engine/train_neural_agent.py`) — a
   legitimate technique (distillation/function approximation, the same idea
   behind how NNUE-style evaluators are often bootstrapped), but it means
   the model cannot know anything the classical evaluator doesn't already
   encode. **Action item:** once real Lichess game outcomes are available
   (see docs/10's Week 3 action item), retrain on those instead.

3. **No live LLM call.** `diagnosis/llm_reasoner.py`'s `call_llm_reasoner()`
   is real, working Anthropic API code, but this sandbox has neither network
   access nor an API key. The actual reasoning step for this project's real
   evidence (`data/llm_evidence_payload.json`) was performed once, manually,
   by Claude in the session that built this integration, and saved to
   `data/llm_reasoning_output.json` with that clearly labeled in the file's
   `_meta` block. **Action item:** set `ANTHROPIC_API_KEY` and this becomes
   a live call with zero code changes.

## What was actually run, and the honest results

Run via `run_integration_demo.py`, against the **same** 12-position
baseline dataset used throughout Weeks 4-8 (no new data generated):

- **Neural value model fit** (on 2000 self-play positions, 80/20 split):
  train R²=0.930, test R²=0.504. Genuine signal, but the train/test gap
  shows real overfitting — 769 input dimensions on 1600 training examples is
  a lot of capacity for not much data. A bug was caught and fixed during
  training: a handful of mate-score outliers (~99999) from the reference
  evaluator were dominating the regression loss before being clipped to
  ±20 pawns; documented in `engine/neural_agent.py`.
- **Classical/neural disagreement:** mean 3.19 pawns across the 12
  positions; correlation with |regret| = **0.212** (n=12) — weakly positive,
  but not trustworthy at this sample size.
- **Anomaly detection:** 3/12 positions flagged (25%, mechanically close to
  the contamination parameter used — not independently informative at this
  n). Notably, the anomalies clustered mostly in the *opening* phase, not
  *middlegame* — the dimension Week 5 identified as the primary weakness.
  **These two signals currently disagree with each other**, and that
  mismatch is reported as-is rather than picked around.

Full manual LLM reasoning on this evidence is in
`data/llm_reasoning_output.json` — short version: no hypothesis is
well-supported yet, sample sizes throughout are too small to trust either
the correlation or the anomaly clustering, and the right next step is
re-running the same pipeline on a couple hundred positions before drawing
any conclusion.

## Team action items before treating any of this as a real finding

1. Scale the dataset from ~12-20 positions to hundreds, ideally with real
   Lichess data once network access exists.
2. Replace the distillation target with real game outcomes once enough
   decisive games (or real Lichess results) are available.
3. Swap `MLPRegressor` for a real CNN (`torch`) once installable — the
   feature encoding is already CNN-ready.
4. Wire `call_llm_reasoner()` to a real API key and re-run the reasoning
   step live, ideally at the larger dataset size from item 1.
