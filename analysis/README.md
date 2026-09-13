# analysis/
Owner: Member C.

## Week 4 — done
- `metrics.py` — decision accuracy, mean/abs regret, regret stdev, mean time
  used, computed from trajectory records.
- `profile.py` — multidimensional capability profile (opening/middlegame/
  endgame from exact game phase; tactical/positional/defensive as documented
  proxies — see the module docstring for why, given no curated benchmark is
  available yet).
- `report.py` — ASCII-style report formatting (same numbers the Week 8
  dashboard renders as HTML).

Verified on the real Week 3 dataset: `python3 ../run_weeks_4_6.py`

## Integration addendum — done
- `disagreement.py` — classical-vs-neural evaluation disagreement per
  position, correlated against decision regret.
- `anomaly.py` — IsolationForest anomaly detection over trajectory features.
