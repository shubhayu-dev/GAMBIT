# GAMBIT — Master Project Context (Constitution)

Paste this file first whenever anyone asks an LLM about GAMBIT. It is the source of
truth. An LLM may optimize *how* GAMBIT is implemented — it may not redefine *what*
GAMBIT is trying to prove.

## 1. Core Project Definition
GAMBIT is a controlled experimental framework for evaluating AI agents, analyzing
their decision-making behavior, diagnosing systematic weaknesses, designing targeted
experiments to investigate those weaknesses, applying controlled interventions, and
validating whether those interventions actually improve agent performance.

The initial experimental domain is **Chess** — the first laboratory, not the
ultimate definition of GAMBIT.

GAMBIT **is**: an AI agent evaluation system, agentic AI research project,
decision-making analysis framework, behavioral diagnosis framework, experimental AI
system, controlled agent-improvement framework.

GAMBIT **is not**: a chess engine project, a chatbot, a RAG system, a generic LLM
application, a leaderboard, an RL-only project.

## 2. Immutable Core Objective
```
OBSERVE → EVALUATE → DIAGNOSE → EXPERIMENT → INTERVENE → VALIDATE
```
GAMBIT must attempt to answer: how well is the agent performing; what behavioral
capabilities does it show; where does it systematically fail; what patterns
characterize those failures; what are plausible causes; what controlled experiment
can distinguish between causes; can a targeted intervention address the weakness;
does the intervention produce measurable improvement on held-out data.

## 3. What GAMBIT Is Not (do not silently drift into these)
Building a world-class chess engine / recreating Stockfish / training a massive
neural net / a generic LLM chatbot / a RAG pipeline / processing the entire Lichess
database / a general-purpose AI scientist / a fully autonomous self-modifying AI / a
distributed cloud platform / RL "because it sounds advanced." These are optional
future extensions, never automatic scope.

## 4. Anti-Hallucination / Project Integrity Rules
1. Never silently change the core objective.
2. Never turn GAMBIT into a generic chatbot, RAG app, or plain chess engine.
3. Never add technology because it sounds advanced.
4. "More ML" ≠ "better research" by default.
5. Never invent benchmark scores, results, or findings.
6. Clearly distinguish: existing facts / design decisions / hypotheses /
   assumptions / experimental results.
7. If a proposed change alters the research question, say explicitly: "This
   changes the current GAMBIT scope" — don't incorporate it silently.
8. Future extensions are labeled FUTURE WORK.
9. When several valid technical approaches exist, compare them against the
   objective rather than redefining the project.
10. When uncertain, return to the core loop in Section 2.

## 5. Canonical Success Criteria
An agent enters GAMBIT → GAMBIT evaluates it → identifies a measurable behavioral
weakness → proposes hypotheses → runs a controlled diagnostic experiment →
identifies a plausible cause → applies a targeted intervention → evaluates the
modified agent on held-out data → reports whether measurable improvement occurred.
If this loop runs end to end, GAMBIT has met its MVP objective.

See `/docs` for the detailed specs this constitution governs.

## Addendum: classical + modern AI + LLM reasoning

`docs/11_integration_addendum.md` documents a scope change, flagged per Rule 7
below rather than silently absorbed: a neural (MLP) value agent, a
classical-vs-neural disagreement signal, unsupervised anomaly detection, and an
LLM reasoning layer (interpretation/hypothesis-proposal only — never move
selection, never unrestricted self-modification) were added as new MUST-HAVE
components. Read that addendum before assuming the original Section 5 "out of
scope" list still excludes all neural/LLM work — it excludes deep RL and
autonomous LLM agents specifically; a small supervised value net and a
narrowly-scoped LLM reasoning layer are not the same thing.
