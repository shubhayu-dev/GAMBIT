# 04 — Abstract Architecture

Independent of implementation language. Twelve conceptual components:

1. Environment
2. Agent Interface
3. Experiment Runner
4. Trajectory / Data Collection
5. Evaluation Engine
6. Behavioral Analysis
7. Diagnostic Engine
8. Hypothesis Generation
9. Experiment Designer
10. Intervention Mechanism
11. Validation Engine
12. Reporting / Visualization

## Conceptual flow
```
Agent
  ↓
Environment
  ↓
Trajectory
  ↓
Evaluation
  ↓
Behavioral Profile
  ↓
Failure Detection
  ↓
Hypotheses
  ↓
Diagnostic Experiment
  ↓
Diagnosis
  ↓
Intervention
  ↓
Held-out Evaluation
```

## Layer responsibilities

**Layer 1 — Environment (MVP: Chess).** State representation, legal actions, state
transitions, termination conditions, environment metadata.

**Layer 2 — Agent Interface.** Standardized channel: `state → agent → action`. Lets
any agent (custom, baseline, external engine) be evaluated by the same framework —
see `06_agent_interface.md`.

**Layer 3 — Experiment Runner.** Takes agent, benchmark, time limit, trial count,
opponent, random seed, config → runs reproducible experiments.

**Layer 4 — Trajectory Collection.** Every decision becomes a structured record
(state, chosen move, reference move, eval before/after, regret, search depth, time
used, game phase) — see `07_dataset_benchmark_spec.md`.

**Evaluation Engine.** Answers "how good was this decision?" via decision regret and
related metrics — see `08_metrics_spec.md`.

**Behavioral Analysis.** Aggregates decisions into a multidimensional capability
profile (tactics, positional, defense, endgame, consistency, efficiency), not a
single score.

**Diagnostic Engine / Hypothesis Generation.** Given an observed failure pattern,
proposes competing candidate explanations (e.g. search depth vs. eval weakness vs.
time management).

**Experiment Designer / Intervention.** Designs a controlled experiment to
distinguish hypotheses, then applies a targeted modification based on the winning
explanation — see `09_experiment_protocol.md`.

**Validation Engine.** Confirms whether the intervention generalizes, using a
held-out split the diagnosis process never touched.
