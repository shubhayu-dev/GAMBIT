# CONTRIBUTING

## Read this first

1. `PROJECT_CONSTITUTION.md` — paste it into any LLM session working on
   this project.
2. `README.md`, all the way through — the Phase 1-5 history at the bottom
   is not changelog filler, it's load-bearing context. Three independent
   measurement bugs (a regret formula that never used the field it stored,
   unclipped mate scores, and worker-count-driven CPU contention corrupting
   wall-clock time budgets) were each found by auditing already-shipped
   results rather than only extending forward. That habit is part of what
   this project is, not an afterthought.

## The one rule that matters most: don't launder confidence

Every real bug listed in `CHANGE_LOG.md` and `README.md`'s phase history
shares a shape: something looked more certain than it was, because a
label, a metric, or a silent fallback made it *seem* validated when it
wasn't. Concretely:

- A regret number that looked fine until someone checked whether the
  formula actually used the field it claimed to (`docs/12_audit_report.md`).
- A neural model's R² that looked good until the train/test split was
  checked at the game level instead of the position level
  (`docs/14_final_audit_summary.md`).
- `n_workers=4` quietly corrupting wall-clock time budgets on a
  single-core sandbox — the numbers were internally consistent, just
  measuring something other than what they claimed to
  (`docs/16_worker_contention_finding.md`).
- Move-commentary alternatives reconstructed with an unconstrained,
  deadline-free re-search — technically real numbers, but answering a
  different, less honest question than "what did the agent's real
  decision weigh" (see `README.md`'s Phase 5 section).

If you're adding a metric, a validation check, or an LLM-facing prompt: ask
whether a downstream reader (human or another LLM stage) could read your
output as more certain than the underlying evidence supports. Label the
uncertainty explicitly rather than rounding it off.

## Before you open a PR

- **Test against the real engine, not just imports.** `python3 -m
  py_compile` is a floor, not a test. Run the function against a real
  `environment.Board` / `Agent`, not a mock.
- **This repo currently has two diagnosis pipelines** (`run_full_pipeline.py`'s
  H1/H2/H3 system, and `run_blunder_diagnosis_pipeline.py`'s Proven Ledger
  system) — unifying them is a real, open item, not something to assume is
  already done or to silently attempt as a side effect of an unrelated
  change.
- **Keep the diagnosis/commentary scope boundary intact.**
  `diagnosis/llm_reasoner.py` feeds the DIAGNOSE step for a whole blunder
  cohort; `analysis/commentary.py` explains one move for a human reading
  one game. If a change seems to need both, that's a sign it should be two
  changes.
- **`analysis/rule_based_commentary.py` and `analysis/move_commentary_rules.py`
  are different modules that used to share a name** — a real near-miss
  during the most recent integration (the incoming file would have
  silently overwritten the diagnosis fallback). Before adding a new file
  anywhere in this repo, check whether the name already means something
  else, especially in `analysis/`.
- **For anything in `engine/search.py`**: alternatives/comparisons
  presented as "what the agent considered" must come from the same
  time-bounded call that produced the move (`info["root_trace"]`), never a
  separate reconstruction — see the Phase 5 README section for the
  measured difference this makes (9/12 mismatched plies vs. 0/12).
- **If your code asks an LLM for N things and can silently get back
  fewer with no error, fix that** — see `commentary.py`'s `annotate_game()`
  for the pattern (explicit missing-item detection + retry), not a
  `diagnosis/llm_reasoner.py`-style "trust what came back."
- **Name fields by what they actually contain.** A payload field named
  `eval_before_cp` that was never actually converted from pawn-scale floats
  is a real bug that shipped in this repo's own history — check the unit,
  not just the name.

## Code style notes (as practiced here)

- A comment explaining *why* a fix was necessary, not just the diff itself
  — every phase in `README.md` and every entry in `CHANGE_LOG.md` follows
  this; match it.
- Type hints and small dataclasses for structured returns over bare dicts
  where the shape matters.
- Mutable list "out-parameters" (`node_counter`, `root_trace`) are the
  established pattern in `engine/search.py` for threading optional
  instrumentation through recursive calls without changing return
  signatures.