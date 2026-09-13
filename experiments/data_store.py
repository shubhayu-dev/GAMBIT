"""
Data layer (docs/07_dataset_benchmark_spec.md, Layer 4 in docs/04).

Deviation, flagged per Constitution Rule 7: the spec calls for trajectories in
Parquet. Neither `pyarrow` nor `fastparquet` could be installed in this
sandbox (no network). Trajectories are stored as JSON Lines (.jsonl) instead —
same one-record-per-line structure, trivially convertible to Parquet later via
`pandas.read_json(..., lines=True).to_parquet(...)` once a parquet engine is
available. SQLite (stdlib, no network needed) holds the structured metadata
exactly as planned: agents, experiments, benchmarks, runs, configs.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "gambit.db"


def init_db(db_path: Path = DB_PATH):
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            config_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS benchmarks (
            id TEXT PRIMARY KEY,
            description TEXT,
            n_positions INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY,
            agent_id TEXT REFERENCES agents(id),
            benchmark_id TEXT REFERENCES benchmarks(id),
            condition TEXT,
            time_budget_ms INTEGER,
            reference_depth INTEGER,
            seed INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            experiment_id TEXT REFERENCES experiments(id),
            n_positions_run INTEGER,
            mean_regret REAL,
            trajectory_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    return conn


def register_agent(conn, agent_id: str, name: str, config: dict):
    conn.execute(
        "INSERT OR REPLACE INTO agents (id, name, config_json) VALUES (?, ?, ?)",
        (agent_id, name, json.dumps(config)),
    )
    conn.commit()


def register_benchmark(conn, benchmark_id: str, description: str, n_positions: int):
    conn.execute(
        "INSERT OR REPLACE INTO benchmarks (id, description, n_positions) VALUES (?, ?, ?)",
        (benchmark_id, description, n_positions),
    )
    conn.commit()


def register_experiment(conn, experiment_id: str, agent_id: str, benchmark_id: str,
                         condition: str, time_budget_ms: int, reference_depth: int, seed: int):
    conn.execute(
        """INSERT OR REPLACE INTO experiments
           (id, agent_id, benchmark_id, condition, time_budget_ms, reference_depth, seed)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (experiment_id, agent_id, benchmark_id, condition, time_budget_ms, reference_depth, seed),
    )
    conn.commit()


def register_run(conn, run_id: str, experiment_id: str, n_positions_run: int,
                  mean_regret: float, trajectory_path: str):
    conn.execute(
        """INSERT OR REPLACE INTO runs
           (id, experiment_id, n_positions_run, mean_regret, trajectory_path)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, experiment_id, n_positions_run, mean_regret, trajectory_path),
    )
    conn.commit()


def write_trajectory_jsonl(records: List[Dict], path: Path):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def read_trajectory_jsonl(path: Path) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
