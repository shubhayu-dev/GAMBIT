# 10 — 8-Week Development Plan

**Week 0 — Design (this document set).**
Problem statement, research questions, scope, abstract + technical architecture,
agent interface, dataset spec, metrics spec, benchmark design, git repo. ✅

**Week 1 — Chess environment.** Position → legal moves → state transition.
Recommend prototyping in Python (`python-chess`) before porting to C++.

**Week 2 — Agent.** Position → search → decision. Minimax, alpha-beta, basic
evaluation. Reference agent = Stockfish via UCI wrapper implementing the same
interface.

**Week 3 — Experiment framework.** Agent → benchmark → trajectory dataset.
Experiment runner, parallel execution (independent experiments, multiprocessing),
data logging to Parquet/SQLite.

**Week 4 — Evaluation.** Dataset → metrics → capability profile. GAMBIT should
function as a basic agent evaluation platform by end of this week.

**Week 5 — Behavioral diagnosis.** Failure patterns → hypotheses (2–4 candidates
per identified weakness).

**Week 6 — Diagnostic experiments.** Hypothesis → controlled experiment →
evidence, on the diagnostic set only.

**Week 7 — Intervention.** Diagnosis → targeted modification → held-out
evaluation, single pass, no re-tuning against held-out results.

**Week 8 — Integration.** Working system + dashboard (Streamlit) + report +
presentation.

## Team responsibility map
| Member | Owns | Learns |
|---|---|---|
| A — Core AI / Chess Engine | board repr, move gen, search, eval, agent, agent interface | C++, bitboards, minimax, alpha-beta |
| B — Experimentation & Systems | experiment runner, benchmark execution, parallelization, data storage | C++, threading/multiprocessing, reproducibility, data formats |
| C — AI Research & Diagnostics | metrics, statistical analysis, behavior analysis, failure detection, hypothesis engine | NumPy/Pandas/SciPy, statistics, experimental design |
| D — Integration & Visualization | dashboard, visualization, system integration, reports | Streamlit, viz, API integration |

**Team rule:** the four roles are not four independent mini-projects — every
member should understand the full pipeline (Environment → Agent → Experiment →
Data → Analysis → Diagnosis → Intervention → Validation), and interfaces between
components (agent interface, trajectory schema) must be frozen before parallel
implementation starts.
