# CHANGE_LOG

For the project's own multi-week/multi-phase history (Weeks 0-8, Phases
1-4: the regret-formula bug, mate-score clipping, the neural model audit,
the worker-contention finding), see `README.md` — that's the authoritative,
detailed record and this file doesn't repeat it. This log covers the most
recent integration round specifically: merging a separately-developed
move-commentary subsystem into this repo.

## This round: move-commentary integration

### Context
A per-move human-readable commentary system (explaining *why the agent
played this move*, for a human reading one game) was developed in a
separate working session, against an export of this repo that turned out
to differ from the actual current state in two important ways: its
`diagnosis/` implementation was a different (and buggier) reimplementation
of ideas this repo had already solved better and natively, and its
`engine/search.py` had unrelated additions (GPU sibling-batching) not
present here. Integrating required reconciling these differences, not just
copying files over.

### Kept from this repo as authoritative, not touched
- `diagnosis/` — the real `FailureCode`/Proven Ledger architecture,
  `intervention.py`'s held-out validation, both LLM prompts' existing
  anti-fabrication language (already present, checked, not missing).
- `experiments/runner.py`'s worker-contention fix.
- All Week 0-8 docs, audit scripts, weekly demo scripts, and real data.

### Added (new, no equivalent existed here)
- `analysis/commentary.py`, `analysis/commentary_report.py`,
  `analysis/run_commentary_demo.py` — the move-commentary system itself.
  Supports Gemini and a local Ollama model (`qwen2.5:7b-instruct` default).
- `engine/search.py`: `root_trace` capture (`alpha_beta`'s new
  `is_root`/`root_trace` parameters, `search_best_move()`'s new
  `info["root_trace"]`/`info["n_legal_at_root"]`) and PV extraction
  (`info["pv"]`, not tracked before at all). Ported as a targeted addition
  to this repo's actual (simpler) search implementation, not a wholesale
  replacement — the unrelated GPU-batching code from the other export was
  deliberately left out; it isn't part of this repo's real search.py and
  wasn't asked for.
  - Verified against the real, unmodified `agents.py`/`environment.py`:
    0/8 mismatches between the played move and the top-scored move in its
    own `root_trace`, across a real self-play game.
  - Confirms directly what was suspected: `SearchAgent`'s wall-clock time
    budget means it often compares only a fraction of legal moves before
    choosing — as low as 2 of 22 observed in one mid-game position.
- `SETUP.md`, `CONTRIBUTING.md` — new.

### Fixed during integration
- **Naming collision**: the incoming commentary system's rule-based move
  filter was also named `rule_based_commentary.py`, which already exists
  here as an unrelated diagnosis post-mortem template fallback. Renamed
  the incoming file to `move_commentary_rules.py` and updated its internal
  cross-references (`commentary.py`, `commentary_report.py`,
  `run_commentary_demo.py`) before this could silently overwrite the
  existing diagnosis fallback.
- **Stale troubleshooting assumption caught before shipping**: a draft of
  `SETUP.md` initially carried over a note about `engine/neural_agent.py`
  needing `torch` and crashing without a guard — checked against this
  repo's actual file and found false: this repo's `neural_agent.py` uses
  `scikit-learn`'s `MLPRegressor`, not PyTorch, and has no such bug.
  Corrected before including it.

### Verified end-to-end (not just import-checked)
- Full commentary pipeline (`run_commentary_demo.py`) run against this
  merged repo: rule-based filtering, live `root_trace` capture, HTML report
  generation all confirmed working together against the real engine.
  Ollama itself was not available in the environment this integration was
  done in, so the actual LLM call (`commentary.py`'s `_call_ollama`) was
  not re-verified here — it was verified separately, in the session where
  it was originally built, against a mock server matching Ollama's
  documented request/response shape.

### Explicitly not done in this round
- The two diagnosis pipelines (`run_full_pipeline.py`'s H1/H2/H3 system vs.
  `run_blunder_diagnosis_pipeline.py`'s Proven Ledger system) were not
  unified — both still exist, independently. See `CONTRIBUTING.md`.
- No file outside `engine/search.py`, `analysis/`, and the three root
  `.md` docs listed above was modified. Audit scripts, weekly demo
  scripts, and data files were carried over as-is and were not
  individually re-reviewed as part of this integration.