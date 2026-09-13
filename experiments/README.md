# experiments/
Owner: Member B.

## Week 3 — done
- `benchmark.py` — position generation (self-play sampling — see
  `../docs/10_development_plan.md` for the noted deviation from real Lichess
  data) + stratified diagnostic/held-out split per
  `../docs/07_dataset_benchmark_spec.md`.
- `data_store.py` — SQLite schema (agents, benchmarks, experiments, runs) +
  JSONL trajectory read/write (JSONL stands in for Parquet — see deviation
  note; swap is one pandas line once `pyarrow` is installable).
- `runner.py` — parallel experiment runner (`multiprocessing.Pool`, one
  worker per position), agent-agnostic: works with any `Agent` subclass from
  `engine/agents.py`.
- `demo_week3.py` — full run: benchmark → split → parallel experiment →
  trajectory dataset + SQLite metadata. Verified working end to end.

Run demo: `python3 demo_week3.py`

## Next (Week 4+)
Wire `analysis/` to read from `data/gambit.db` + the JSONL trajectory files
this module produces.

## Audit fix — done (see docs/12_audit_report.md section 6)
`runner.py`'s regret formula was proven incorrect (didn't match the
documented `R(s,a)=V(s,a*)-V(s,a)`, never used the stored `reference_move`)
and has been fixed. The old formula is preserved as `single_move_eval_delta`
on every trajectory record for transparency/comparison — it is NOT regret,
despite being the value the whole project reported as regret through the
integration addendum. Re-running Weeks 4-6 on the same 12 positions with the
fix **inverted** the Week 5 weakness finding (middlegame went from 0% to
100% decision accuracy) — see the audit report for full numbers.

## Phase 2 — real Lichess data + a second regret bug found
- `lichess_data.py` — loads and validates real Lichess puzzle data against
  our own engine (96/99 transcribed rows passed; 3 transcription errors
  caught and removed, not guessed at).
- Testing the corrected regret formula against REAL data (rather than
  self-play) immediately found a second bug: unclipped mate scores blew
  mean regret up to 14,584. Fixed with the same clipping pattern used for
  the neural model's training target. See `docs/13_lichess_integration.md`.
