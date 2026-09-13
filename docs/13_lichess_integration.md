# 13 — Lichess Data Integration

## What was attempted and why it looks the way it does

Per the audit response's Step 4, the goal was a reproducible pipeline
pulling 500-2000 real Lichess positions. Two independent blockers were hit
and confirmed directly, not assumed:

1. **`lichess.org` itself** (both the main API and `database.lichess.org`
   bulk dumps) — no network access at all from the bash sandbox (confirmed
   in every prior doc), and `web_fetch` (which does have network access,
   separate from the sandbox) returned `ROBOTS_DISALLOWED` when tried
   directly against `lichess.org/api/games/user/...`.
2. **`raw.githubusercontent.com`** (a third-party mirror repo,
   `mcognetta/lichess-combined-puzzle-game-db`, containing a real 100-puzzle
   CC0 sample with full games and Stockfish evals) — also `ROBOTS_DISALLOWED`
   when fetched directly.

**What worked:** `huggingface.co/datasets/Lichess/chess-puzzles` — the
**official Lichess-maintained** puzzle dataset (6M+ puzzles), viewable
through HuggingFace's dataset viewer HTML pages. This is not blocked by
robots.txt. Each page of the viewer shows ~100 rows as an HTML table
(`PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays,
Themes, OpeningTags`).

**The catch:** this is a paginated HTML *preview*, not a bulk file download.
Getting more data means fetching more pages and transcribing each page's
table into structured data by hand (there is no API endpoint available to
this tool that returns raw rows as JSON without robots restrictions).
Five pages (~460 rows) were fetched during this work; **one page (99 rows)
was transcribed** into `data/lichess_puzzles_sample.csv` — transcribing
all five pages by hand was not a good use of effort for this pass. This is
explicitly **less than the 500-2000 target**, and is reported as such rather
than padded.

## Data quality — validated, not assumed

`experiments/lichess_data.py` loads the CSV and validates every row against
**our own engine**: the puzzle's setup move and reference move must both be
legal per our move generator, or the row is rejected and logged (not
silently dropped or guessed at).

**Result: 3 of 99 transcribed rows failed validation** — genuine manual
transcription errors (e.g. a mistyped FEN character, `6G1K` instead of
`6K1` — `G` isn't a legal piece letter). These 3 rows were removed from the
CSV rather than corrected by guessing. **96/96 remaining rows validate
cleanly.** This validation step is itself worth keeping permanently for any
future manually- or automatically-transcribed external data.

## What this real data revealed that self-play data couldn't

1. **Real phase coverage.** The 96-position Lichess sample has 43 genuine
   endgame positions (from Lichess's own theme tags) — the self-play data,
   across both the 16-position benchmark and the ~7500-position neural
   training pool, has **zero**. This confirms the "endgame absent" gap noted
   in `docs/12_audit_report.md` was a self-play sampling artifact, not a
   fundamental limitation of the environment.
2. **Our `game_phase()` heuristic is only 76% accurate** against Lichess's
   real theme tags (73/96 agreement), and the errors are one-directional:
   our heuristic systematically calls positions "middlegame" that Lichess
   calls "endgame" — its piece-count threshold (≤12) is stricter than how
   the term is actually used. Not fixed in this pass (would need
   re-calibrating and re-validating against more real data); flagged as an
   action item.
3. **A second mate-score clipping bug**, found only because real puzzles
   include genuine forced mates that self-play almost never produces:
   running the corrected (Step 6) regret formula on this real data gave a
   mean regret of **14,584** — driven by 14/96 positions where the reference
   search legitimately found a mate score (~99999) and the agent didn't.
   Fixed by clipping evaluations to ±20 pawns before differencing (same
   pattern as the neural model's training-target fix). After the fix: mean
   regret 4.488, median 0.000, max |regret| 27.5. See `experiments/runner.py`.
4. **The classical agent performs meaningfully worse on real tactical
   puzzles than on self-play middlegames**: 46.9% exact reference-move
   agreement on the 96 real puzzles, versus much higher agreement on the
   self-play weak-bucket positions audited in `docs/12`. This is intuitive
   (Lichess puzzles are adversarially curated to be hard) but is now a real,
   measured number rather than an assumption.

## Honest bottom line on Step 4

Real Lichess data is now integrated, validated, and has already surfaced
two genuine bugs/gaps (mate-score clipping in regret, phase-heuristic
miscalibration) that the self-play data never would have exposed. It is
**not yet at the 500-2000 position target** — getting there means either
transcribing more HuggingFace viewer pages (mechanical, ~4-5x more of the
same manual work already done) or finding a fetchable bulk source this
tool's `web_fetch` can actually reach (not found in this session).
