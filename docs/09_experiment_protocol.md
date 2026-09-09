# 09 — Experiment Protocol

## How we establish a diagnosis/intervention is actually valid (design decision)
GAMBIT follows: `EVIDENCE → HYPOTHESIS → EXPERIMENT → CONCLUSION`, never
`OBSERVATION → GUESS`. Concretely:

1. **Detect a pattern** on the diagnostic set only (e.g. "regret is
   disproportionately high in high-complexity defensive positions").
2. **Generate 2–4 competing hypotheses** for the cause (e.g. H1 insufficient
   search depth, H2 evaluation-function weakness, H3 time management, H4 weak
   defensive heuristics).
3. **Design a controlled experiment** with:
   - one independent variable per hypothesis (e.g. search depth: 4 / 6 / 8)
   - everything else held constant (same positions, same agent, same reference)
   - a defined dependent variable (regret, decision accuracy)
   - a minimum sample size per condition (avoid drawing conclusions from a
     handful of positions)
4. **Run it only on the diagnostic set.** The held-out set stays untouched during
   hypothesis testing and intervention tuning.
5. **Rank hypotheses** by effect size + statistical significance (e.g. paired
   Wilcoxon signed-rank test or paired t-test on per-position regret,
   significance threshold p < 0.05, plus effect size — not p-value alone).
6. **Apply the best-supported intervention** (e.g. adaptive search depth based on
   position complexity).
7. **Validate on the held-out set, once.** Compare before/after regret and
   decision accuracy. No re-tuning against held-out results — if the intervention
   doesn't generalize, that's a valid (and reportable) finding, not a reason to
   peek and retune.
8. **Report** observation → hypotheses considered → evidence → diagnosis →
   intervention → held-out result, distinguishing correlation from causal
   evidence throughout.

## Reproducibility requirements
Every experiment run is defined by: agent, benchmark subset, time limit, trial
count, opponent/reference, random seed, and config — logged so any run can be
re-executed and produce the same trajectories.

## Example (illustrative numbers only — not real results)
```
H1 Search limitation     61%
H2 Evaluation weakness   27%
H3 Time management       12%

Search depth 4 → 51.4%
Search depth 6 → 64.7%
Search depth 8 → 66.1%

Held-out before → 55.2%
Held-out after  → 68.7%   (+13.5%)
```
