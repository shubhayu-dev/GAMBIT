# 02 — Research Questions

## Primary research question
Can a closed-loop framework — evaluate, diagnose, experiment, intervene, validate —
built first around chess, reliably identify a systematic behavioral weakness in an
arbitrary agent, isolate a plausible cause for it via controlled experimentation,
and demonstrate a measurable, held-out-validated improvement from a targeted
intervention?

## Supporting questions
1. **Measurement** — What is the right unit-level metric for "decision quality," and
   does it correlate with game-level outcomes (win/loss/draw)?
2. **Profiling** — Can raw regret/trajectory data be aggregated into a stable,
   interpretable multidimensional behavioral profile (tactics, positional, defense,
   endgame, consistency, efficiency)?
3. **Diagnosis** — Given an observed weakness, how many controlled experiments are
   typically needed to distinguish between competing hypotheses (search depth vs.
   evaluation weakness vs. time management vs. heuristic gaps)?
4. **Causality vs. correlation** — What statistical evidence threshold is required
   before GAMBIT treats a hypothesis as "supported" rather than merely "consistent
   with the data"?
5. **Intervention validity** — Does an intervention that improves diagnostic-set
   performance generalize to a held-out set, or does it overfit to the diagnosed
   failure cases?
6. **Generalization (future work)** — Do the same evaluate → diagnose → intervene
   mechanics transfer to a non-chess decision-making environment?

## Non-goals for this project cycle
This is not asking "how strong an engine can we build" — engine strength is
instrumental (it's what produces the decisions GAMBIT studies), not the outcome
being measured.
