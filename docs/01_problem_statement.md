# 01 — Problem Statement

## The gap
Standard AI-agent evaluation collapses behavior into a single performance number
(e.g. "82% win rate"). This tells you an agent is better or worse than another, but
not *why*, not *where specifically it fails*, and not *whether a proposed fix
actually helps* versus just fitting the test set.

## The problem GAMBIT addresses
There is no lightweight, reproducible framework that takes an arbitrary
decision-making agent through the full scientific loop:

1. Measure performance at the level of individual decisions, not just outcomes.
2. Build a multidimensional behavioral profile instead of one score.
3. Detect systematic (not one-off) failure patterns.
4. Generate falsifiable hypotheses about the cause of a failure pattern.
5. Run a controlled experiment that can distinguish between competing hypotheses.
6. Apply a targeted intervention based on the diagnosis.
7. Validate the intervention on data the diagnosis process never saw.

## Why this matters
Without step 7 in particular, "improvements" are unfalsifiable — you can always
tune a system until it does better on the same data you diagnosed it with. GAMBIT's
contribution is making the full loop, including held-out validation, a standard,
repeatable procedure rather than an ad hoc one-off analysis.

## Why Chess as the first environment
Chess gives clearly defined states/actions, deterministic transitions, measurable
outcomes, strong reference engines (for computing regret), and large public
datasets — making it possible to quantify "how good was this specific decision,"
which is the foundation the entire diagnostic loop is built on.
