# CONTRIBUTING

## Read this first

1. `PROJECT_CONSTITUTION.md` — the actual source of truth on scope. Paste it
   into any LLM session you use to work on this project.
2. `docs/` in order (01 → 15) for the design history and, importantly, the
   audit trail. This project has caught itself in real, non-trivial bugs
   multiple times (a regret formula that never used the field it stored, a
   leaky train/test split, a diagnosis pipeline whose imports didn't exist)
   by auditing rather than only extending. That habit is part of what this
   project is.

## The one rule that matters most: don't launder confidence

Several real bugs fixed in this codebase share a shape: something looked
more certain than it was, because a label, a metric, or a silent fallback
made it *seem* validated when it wasn't. Concretely, from this codebase's
own history:

- A `case_record["proven_cause"]` that was actually a single-position
  match, with no significance test, no held-out check — labeled `"proven"`
  anyway until fixed to carry an explicit `validation_status`.
- `intervention.py` comparing a continuous regret value against a binary
  0/1 proxy in the same subtraction, producing a `regret_reduction_pct`
  number that looked meaningful but wasn't measuring one consistent thing.
- `run_diagnostic_suite()`/`rank_hypotheses()` silently no-op'ing (`pass` /
  `return []`) instead of erroring when their replacements existed — a
  caller could believe an A/B test ran when nothing happened.
- An LLM commentary prompt that dropped its own "do not fabricate" clause
  in a rewrite, and separately, a model output that asserted a move was
  "significantly better" from a score it also admitted wasn't fully
  evaluated — the exact contradiction the prompt exists to prevent.

If you're adding a metric, a validation check, or an LLM-facing prompt: ask
whether a downstream reader (human or another LLM stage) could read your
output as more certain than the underlying evidence supports. If unsure,
label the uncertainty explicitly rather than rounding it off. A loud
"unresolved" or `NotImplementedError` beats a quiet, wrong-looking success.

## Before you open a PR

- **Test against the real engine, not just imports.** Several bugs in this
  codebase (the `chess`/`select_move` mismatch, the `hypotheses.py`
  `KeyError`, the `.env`-loading order bug) passed a syntax check and even
  an import check while still crashing the first time they actually ran.
  `python3 -m py_compile` is a floor, not a test. Run the actual function
  against `engine.environment.Board` / a real `Agent`, not a mock, before
  calling something fixed.
- **If you touch `diagnosis/`**, keep the scope boundary with
  `analysis/commentary.py` intact: `diagnosis/llm_reasoner.py` feeds the
  DIAGNOSE step of the OBSERVE→EVALUATE→DIAGNOSE→EXPERIMENT→INTERVENE→VALIDATE
  loop; it is not the place for human-facing move-by-move commentary, and
  vice versa. If a change seems to need both, that's a sign it should be
  two changes.
- **If you touch move commentary (`analysis/commentary.py`,
  `rule_based_commentary.py`, `engine/search.py`'s `root_trace`)**: the
  live-vs-reconstructed distinction is load-bearing. Alternatives shown to
  an LLM (or a human) as "what the agent considered" must come from the
  *same, time-bounded search call* that produced the move
  (`info["root_trace"]`), not a fresh unconstrained re-search
  (`evaluate_root_candidates()`) — the two answer different questions, and
  conflating them previously produced a comparison where the played move
  looked like it lost to options the live search never even reached.
- **Update the field's `description`/docstring when you change what a
  function returns**, especially field names implying units (a real,
  fixed instance of this: a payload field named `eval_before_cp` that
  wasn't actually in centipawns — `evaluate()` returns pawn-scale floats,
  and nothing converted it before the "_cp" suffix was added).
- **Don't let a batch/retry layer hide partial failure.** If your code
  asks for N things and can get back fewer than N with no error (an LLM
  batch call is the concrete case in this repo), check for the gap and
  surface it — see `commentary.py`'s `annotate_game()` for the pattern
  (explicit missing-ply detection + single-item retry, not a silent
  `return whatever we got`).

## Code style notes (as practiced in this repo so far)

- Prefer a comment explaining *why* a fix was necessary over just making
  the change silently — future readers (including future LLM sessions
  working on this repo) benefit far more from "this crashed because X, so
  now it does Y" than from a diff with no narrative.
- Type hints and dataclasses for structured return values
  (`RuleBasedResult` in `rule_based_commentary.py` is the pattern to
  follow) over bare dicts where the shape matters.
- Mutable list "out-parameters" (`node_counter: Optional[list]`,
  `root_trace: Optional[list]`) are an established pattern in
  `engine/search.py` for threading optional instrumentation through
  recursive calls without changing return signatures — match it rather
  than inventing a new convention for similar needs.

## Questions this doesn't answer

If PROJECT_CONSTITUTION.md, the docs, and this file don't cover your
situation, that's worth raising explicitly rather than guessing — several
of the audits in `docs/` exist precisely because an earlier assumption
went unquestioned for longer than it should have.