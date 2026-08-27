"""Minimal SQLite-backed job queue for local-agent-forge Phase 3 jobs.

Each job run gets one row: status, timing, and full input/output logged as JSON,
so a job's history is fully auditable and partial failures are visible rather
than swallowed. Deliberately not a general task-scheduling framework -- systemd
timers own scheduling, this module just owns logging + status for a single run.
"""

import json
import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    scheduled_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    input TEXT,
    output TEXT,
    latency_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    error TEXT
);
"""


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.commit()
    conn.close()


@contextmanager
def _connect(db_path):
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def create_job(db_path, job_type, input_data=None):
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO jobs (job_type, status, scheduled_at, input) VALUES (?, 'pending', ?, ?)",
            (job_type, time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps(input_data)),
        )
        conn.commit()
        return cur.lastrowid


def start_job(db_path, job_id):
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ?",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), job_id),
        )
        conn.commit()
    return time.monotonic()


def finish_job(db_path, job_id, start_time, output_data):
    latency_ms = int((time.monotonic() - start_time) * 1000)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'done', finished_at = ?, output = ?, latency_ms = ? WHERE id = ?",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps(output_data), latency_ms, job_id),
        )
        conn.commit()


def fail_job(db_path, job_id, start_time, error, output_data=None):
    latency_ms = int((time.monotonic() - start_time) * 1000)
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'failed', finished_at = ?, output = ?, latency_ms = ?, error = ? WHERE id = ?",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), json.dumps(output_data), latency_ms, str(error), job_id),
        )
        conn.commit()
