# SETUP

This covers getting the repo running end to end: the core engine (no
external dependencies), the diagnosis pipeline (needs Gemini), and the move
commentary system (needs a local Ollama model).

## 1. Prerequisites

- Python 3.10+
- A virtualenv (recommended): `python3 -m venv .venv && source .venv/bin/activate`

There is no `requirements.txt` in the repo yet — install what each piece
below needs directly. If you're setting this up repeatedly, consider
freezing one (`pip freeze > requirements.txt`) once your environment works.

## 2. Core engine — zero setup

`engine/`, `experiments/`, and the rules/search/evaluation code have no
external dependencies (deliberately — see `engine/README.md` and
`PROJECT_CONSTITUTION.md`: no `python-chess`, no Stockfish). If all you want
is to run self-play games or the search engine directly:

```bash
cd GAMBIT
PYTHONPATH=engine python3 -c "
from environment import Board, START_FEN
from agents import SearchAgent
board = Board(START_FEN)
move = SearchAgent(max_depth=3).get_move(board, time_budget_ms=200)
print(move.uci())
"
```

**Known gap, not fixed by this setup:** `engine/neural_agent.py` defines
`class ChessValueNet(nn.Module)` unconditionally even when its own
`try/except ImportError` for `torch` fails, so importing anything that
pulls in `neural_agent.py` (including `run_full_pipeline.py`) will crash
with `NameError: name 'nn' is not defined` if `torch` isn't installed.
Install `torch` if you need that path, or expect this to fail until
`neural_agent.py` itself is fixed to guard the class definition too.

## 3. Diagnosis pipeline (`run_full_pipeline.py`) — needs Gemini

```bash
pip install google-genai python-dotenv zstandard
```

Create a `.env` file in the repo root (same directory you run scripts from):

```
GEMINI_API_KEY=your_key_here
```

Also needs real Lichess puzzle data (`experiments/lichess_data.py` pulls
from HuggingFace, not lichess.org directly — see README for why) and, for
the neural-agent code path specifically, `torch` (see the gap noted above).

Run it:

```bash
PYTHONPATH=engine:experiments:analysis:diagnosis:dashboard python3 run_full_pipeline.py
```

## 4. Move commentary (`analysis/run_commentary_demo.py`) — needs Ollama

This is a separate LLM call from the diagnosis pipeline (see README's "Move
commentary system" section for why it's a distinct module). Default
provider is a local Ollama model — no API key, but the server needs to be
running.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model. 7b-instruct is a reasonable default for a laptop.
ollama pull qwen2.5:7b-instruct
# smaller/faster: qwen2.5:3b-instruct or qwen2.5:1.5b-instruct
# bigger/better if you have the VRAM: qwen2.5:14b-instruct

# Confirm it's reachable (installs as a systemd service and should
# already be running; if this fails, run `ollama serve` manually):
curl http://localhost:11434/api/tags

# Python deps for this path
pip install requests python-dotenv
```

If the model tag you pulled differs from `qwen2.5:7b-instruct` (check with
`ollama list`), update `model_name=` in the `annotate_game(...)` call near
the bottom of `analysis/run_commentary_demo.py`'s `main()`.

Run it:

```bash
cd GAMBIT
PYTHONPATH=engine:analysis python3 analysis/run_commentary_demo.py
```

Output: terminal progress, plus a browser-openable report written to
`data/commentary_report.html` (the script prints the exact path and a
`file://` link when it finishes).

**If you'd rather use Gemini for commentary too** (e.g. to compare quality
against the local model), pass `provider="gemini"` instead of
`provider="ollama"` in the `annotate_game(...)` call, and follow the
`GEMINI_API_KEY` setup from section 3.

### Tuning notes specific to local models

- `chunk_size` (how many moves go into one LLM call) defaults to 15 for
  Ollama in the demo script, vs. 40 for Gemini. Smaller local models
  sometimes stop after generating one item in a multi-item JSON array
  regardless of chunk size — `commentary.py`'s `annotate_game()`
  automatically retries any move a batch dropped, one at a time, as a
  fallback, so this degrades gracefully rather than silently losing moves.
- `num_ctx` (8192) and `num_predict` (4096) are set explicitly in
  `commentary.py`'s `_call_ollama()` — Ollama's own defaults (2048 context,
  a small model-defined output cap) are too small for a multi-move batched
  payload and will truncate silently otherwise.

## 5. Troubleshooting quick reference

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: No module named 'engine'` | You imported `engine.X` somewhere, but `PYTHONPATH=engine:...` puts the *contents* of `engine/` on the path, not a package named `engine`. Use bare `from environment import ...`, not `from engine.environment import ...` — check your editor isn't auto-rewriting this on save. |
| `ImportError: cannot import name 'classify_game' ... circular import` | Your local copy of a file has extra/duplicated content merged into it. Re-copy the file fresh rather than hand-editing. |
| Gemini `503 UNAVAILABLE` | Real server-side demand spike, not your setup — `commentary.py`/`llm_reasoner.py` already retry with backoff; if it survives that, wait a few minutes and retry. |
| Ollama `ConnectionError` | Server not running — `curl http://localhost:11434/api/tags` to check, `ollama serve` to start it. |
| `NameError: name 'nn' is not defined` importing `neural_agent` | `torch` isn't installed — see section 2. |
| `.env` file seemingly ignored | Confirm `load_dotenv()` is called *before* any `os.environ.get(...)` check in whichever script you're running — this bit us once already in `run_commentary_demo.py` (fixed), but double-check any new entry point you add. |