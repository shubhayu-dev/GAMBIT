# 08 — Metrics Specification

## What mathematically constitutes a "good" or "bad" decision (design decision)
Primary unit-level metric: **decision regret**.
```
R(s, a) = V(s, a*) - V(s, a)
```
- `s` — current state (board position)
- `a` — action the agent selected
- `a*` — reference/best action (from reference engine at fixed depth/time)
- `V` — estimated value of the resulting state (centipawn or win-probability scale)

A decision is classified **good** if `R(s,a) ≤ ε` for a threshold `ε` chosen per
game phase (tactical positions tolerate less regret than quiet positional ones);
**bad** otherwise. `ε` is a tunable parameter, justified experimentally, not
hard-coded arbitrarily — log the value used with every experiment run.

## Aggregate / profile metrics
- **Decision accuracy** — % of decisions with `R(s,a)` below threshold
- **Evaluation loss** — mean regret across a position set
- **Win rate** — game-level outcome, kept as a sanity check against decision-level metrics
- **Consistency** — variance of regret across positions of similar category/complexity
- **Robustness** — regret stability across different reference engines/settings
- **Computational efficiency** — nodes searched per second, time-to-decision
- **Search depth** — depth actually reached within the time budget

## Behavioral profile dimensions (MVP)
Tactical, Positional, Defensive, Endgame, Consistency, Efficiency — each computed
as aggregate decision accuracy / regret restricted to that benchmark category. Do
not add a dimension unless it's backed by a measurable benchmark subset.

## Statistical reporting
For any comparison (agent vs. agent, before vs. after intervention): report mean
regret, variance, sample size, and the statistical test used (see
`09_experiment_protocol.md`) — never just a point estimate.
