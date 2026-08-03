"""
Persistent metrics store for GPU router learning.
Extends in-memory MetricsStore with SQLite backing for durability.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("/home/scott/.hermes/state/gpu-router-metrics.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS lane_metrics (
    lane TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    success INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    PRIMARY KEY (lane, model, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_lane_model ON lane_metrics(lane, model);
CREATE TABLE IF NOT EXISTS router_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


@contextmanager
def get_conn(db_path: Path = DEFAULT_DB):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path = DEFAULT_DB) -> None:
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA)


def record_latency(lane: str, model: str, latency_ms: float, success: bool, db_path: Path = DEFAULT_DB) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO lane_metrics (lane, model, latency_ms, success, timestamp) VALUES (?, ?, ?, ?, ?)",
            (lane, model, latency_ms, 1 if success else 0, time.time()),
        )


def get_lane_stats(lane: str, model: str, window_s: float = 3600, db_path: Path = DEFAULT_DB) -> dict[str, Any]:
    """Get aggregated stats for a lane+model within time window."""
    cutoff = time.time() - window_s
    with get_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT latency_ms, success FROM lane_metrics WHERE lane = ? AND model = ? AND timestamp > ?",
            (lane, model, cutoff),
        ).fetchall()

    if not rows:
        return {"avg_latency_ms": None, "success_rate": None, "samples": 0}

    latencies = [r["latency_ms"] for r in rows]
    successes = [r["success"] for r in rows]
    return {
        "avg_latency_ms": sum(latencies) / len(latencies),
        "success_rate": sum(successes) / len(successes),
        "samples": len(rows),
    }


def get_all_lane_models(db_path: Path = DEFAULT_DB) -> list[tuple[str, str]]:
    with get_conn(db_path) as conn:
        return [(r["lane"], r["model"]) for r in conn.execute("SELECT DISTINCT lane, model FROM lane_metrics").fetchall()]


def set_state(key: str, value: Any, db_path: Path = DEFAULT_DB) -> None:
    with get_conn(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO router_state (key, value, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(value), time.time()),
        )


def get_state(key: str, default: Any = None, db_path: Path = DEFAULT_DB) -> Any:
    with get_conn(db_path) as conn:
        row = conn.execute("SELECT value FROM router_state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default