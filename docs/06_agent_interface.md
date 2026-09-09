# 06 — Agent Interface Specification

## What constitutes an "agent" (design decision)
For GAMBIT, an agent is anything that implements the interface below. It must be:
- **Stateless across calls at the interface level** — all context needed to choose
  a move comes from the `state` argument, not hidden mutable globals (internal
  search state within a single call is fine).
- **Deterministic given the same state, time budget, and random seed** — required
  for reproducible experiments.
- Agnostic to whether it's a custom engine, Stockfish, a random baseline, or a
  neural agent — GAMBIT never depends on one implementation.

## Reference interface (prototype in Python first, port to C++ later)
```python
class Agent:
    def name(self) -> str:
        """Stable identifier used in trajectory records and reports."""
        ...

    def get_move(self, fen: str, time_budget_ms: int, seed: int | None = None) -> str:
        """
        fen: board state in FEN notation
        time_budget_ms: max thinking time for this decision
        seed: optional RNG seed for reproducibility
        returns: move in UCI notation (e.g. 'e2e4')
        """
        ...

    def config(self) -> dict:
        """Hyperparameters relevant to the experiment (search depth, eval weights,
        etc.) — logged alongside every trajectory record for this agent."""
        ...
```

## C++ equivalent (once ported)
```cpp
class Agent {
public:
    virtual std::string name() const = 0;
    virtual Move getMove(const Board& state, int timeBudgetMs, uint64_t seed = 0) = 0;
    virtual nlohmann::json config() const = 0;
    virtual ~Agent() = default;
};
```

## Contract
- `state → agent → action`: `S_t → Agent → A_t → Environment → S_{t+1}`.
- The environment (not the agent) enforces legality; an illegal move returned by an
  agent is a recorded failure, not a crash.
- Any wrapper around an external engine (e.g. Stockfish via UCI) must implement
  this same interface so it's interchangeable with custom agents in the runner.
