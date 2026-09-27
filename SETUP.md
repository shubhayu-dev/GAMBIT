# SETUP

## 1. Prerequisites

- Python 3.10+
- A virtualenv: `python3 -m venv .venv && source .venv/bin/activate`

No `requirements.txt` yet — install what each piece below needs directly.

## 2. Core engine — zero external dependencies

`engine/environment.py`, `search.py`, `evaluation.py`, `agents.py` (the
`SearchAgent`/`MaterialAgent`/`RandomAgent` path) need nothing beyond the
standard library, deliberately (see `PROJECT_CONSTITUTION.md`).

```bash
cd GAMBIT
PYTHONPATH=engine python3 -c "
from environment import Board, START_FEN
from agents import SearchAgent
move = SearchAgent(max_depth=3).get_move(Board(START_FEN), time_budget_ms=200)
print(move.uci())
"
```

`engine/neural_agent.py` additionally needs `scikit-learn`
(`pip install scikit-learn`) for `MLPRegressor` — optional, only import this
module if you're using the neural value agent or its training data.

## 3. Diagnosis pipeline

Two parallel systems currently coexist here — see `README.md`'s Phase 4
section and `CONTRIBUTING.md` for why, and note that unifying them is an
open item, not something this setup (or the recent integration) resolved:

- **`run_full_pipeline.py`** — the older paired-hypothesis (H1/H2/H3) A/B
  system.
- **`run_blunder_diagnosis_pipeline.py`** — the newer `FailureCode` /
  "Proven Ledger" empirical system.

Both need:

```bash
pip install google-genai python-dotenv zstandard
```

`.env` in the repo root:

```
GEMINI_API_KEY=your_key_here
```

`diagnosis/llm_reasoner.py` also has a local-LLM path
(`LOCAL_SYSTEM_PROMPT`/`call_local_reasoner`) with a tested rule-based
fallback (`analysis/rule_based_commentary.py` — NOT the same file as
`analysis/move_commentary_rules.py`, see section 4) for when no local
endpoint is reachable; real Lichess data also needs network access this
sandbox didn't have (`docs/13_lichess_integration.md` covers the HuggingFace
workaround already integrated).

## 4. Move commentary — needs a local Ollama model (or Gemini)

Separate from the diagnosis pipeline above — explains individual moves in
plain language, for a human reading one game, rather than feeding the
DIAGNOSE step for a whole blunder cohort. Default provider is local:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b-instruct
# smaller/faster: qwen2.5:3b-instruct or 1.5b-instruct
# bigger/better with more VRAM: qwen2.5:14b-instruct

curl http://localhost:11434/api/tags   # confirm it's reachable
# if not: ollama serve

pip install requests python-dotenv
```

If your pulled tag differs from `qwen2.5:7b-instruct` (`ollama list` to
check), update `model_name=` in `analysis/run_commentary_demo.py`'s
`annotate_game(...)` call.

Run it:

```bash
cd GAMBIT
PYTHONPATH=engine:analysis python3 analysis/run_commentary_demo.py
```

Writes a browser-openable report to `data/commentary_report.html` (path and
`file://` link printed at the end).

**To use Gemini instead** (e.g. to compare quality against the local
model), pass `provider="gemini"` in the same `annotate_game(...)` call and
set `GEMINI_API_KEY` as in section 3.

### Tuning notes for local models

- `chunk_size` (moves per LLM call) defaults to 15 here vs. 40 for Gemini.
  Smaller instruct models can stop after one array entry in a multi-item
  JSON response regardless of chunk size — `annotate_game()` automatically
  retries whatever a batch drops, one move at a time, so this degrades
  gracefully rather than silently losing moves.
- `num_ctx` (8192) and `num_predict` (4096) are set explicitly in
  `commentary.py`'s `_call_ollama()` — Ollama's own defaults (2048 context,
  a small model-defined output cap) truncate a multi-move batched payload
  silently otherwise.

## 5. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ImportError`/`AttributeError` about `chess` or `select_move` anywhere | You're looking at a different, non-authoritative branch's code — this repo's real agents use `environment.Board`/`get_move()` throughout; there is no `python-chess` dependency here. |
| `ModuleNotFoundError: No module named 'engine'` | `PYTHONPATH=engine:...` puts the *contents* of `engine/` on the path, not a package named `engine`. Use `from environment import ...`, not `from engine.environment import ...`. |
| Confusing which `rule_based_commentary`-ish file does what | `analysis/rule_based_commentary.py` = diagnosis post-mortem template fallback. `analysis/move_commentary_rules.py` = per-move commentary rule filter. Same original name, different jobs — renamed on integration specifically to stop this confusion from becoming a silent file collision. |
| Gemini `503 UNAVAILABLE` | Real server-side demand spike — both LLM call sites already retry with backoff; if it survives that, wait a few minutes. |
| Ollama `ConnectionError` | Server not running — `curl http://localhost:11434/api/tags`, then `ollama serve`. |
| Regret/timing numbers look inconsistent with older runs | Check whether they predate the worker-contention fix in `experiments/runner.py` (`docs/16_worker_contention_finding.md`) — `n_workers` above your real core count silently corrupted wall-clock budgets before this was found and fixed. |
| `.env` seemingly ignored | Confirm `load_dotenv()` runs before any `os.environ.get(...)` check in whichever script you're running. |