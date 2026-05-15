"""SQLite job database.

Schema covers jobs, stage cache, and benchmark runs. WAL journal mode lets the GUI's
queue model read-while-write safely against the broker.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from aep.constants import FILE_DB
from aep.util.paths import runtime_dir

log = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    output_path TEXT,
    preset_id TEXT NOT NULL,
    state TEXT NOT NULL,        -- queued|running|paused|completed|failed|cancelled
    progress REAL NOT NULL DEFAULT 0,
    error TEXT,
    current_stage TEXT,
    last_failed_stage TEXT,
    resume_from_stage TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    plan_json TEXT,             -- frozen JobPlan
    probe_json TEXT,            -- cached MediaInfo for quick redisplay
    preset_overrides_json TEXT  -- sparse per-job preset overrides (or NULL)
);

CREATE INDEX IF NOT EXISTS ix_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS ix_jobs_created ON jobs(created_at);

CREATE TABLE IF NOT EXISTS stage_cache (
    job_id TEXT NOT NULL,
    stage_name TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    output_dir TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    PRIMARY KEY(job_id, stage_name)
);

CREATE TABLE IF NOT EXISTS bench_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hardware_fingerprint TEXT NOT NULL,
    preset_id TEXT NOT NULL,
    fps REAL NOT NULL,
    vram_peak_mb INTEGER,
    duration_s REAL,
    notes TEXT,
    created_at TEXT NOT NULL
);
"""


_lock = threading.Lock()


def db_path() -> Path:
    return runtime_dir() / FILE_DB


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Open a thread-safe connection. Uses WAL for concurrent reads.

    Trade-off: we open per-call rather than holding a long-lived shared connection. SQLite
    on Windows is finicky with multi-threaded connections; per-call open + WAL gives us
    correctness without locking overhead at our scale (handful of jobs/sec at most).
    """
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), isolation_level=None, timeout=15.0, check_same_thread=False)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        yield conn
    finally:
        conn.close()


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r["name"] == column for r in rows)


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply additive schema migrations for users upgrading from older DBs.

    SQLite's `ADD COLUMN` is the only safe in-place mutation we use; for
    anything more invasive we'd rebuild the table. So far every migration
    has been a column addition with a NULL default.
    """
    if not _column_exists(conn, "jobs", "preset_overrides_json"):
        conn.execute("ALTER TABLE jobs ADD COLUMN preset_overrides_json TEXT")
        log.info("migrated jobs table: added preset_overrides_json")
    if not _column_exists(conn, "jobs", "current_stage"):
        conn.execute("ALTER TABLE jobs ADD COLUMN current_stage TEXT")
        log.info("migrated jobs table: added current_stage")
    if not _column_exists(conn, "jobs", "last_failed_stage"):
        conn.execute("ALTER TABLE jobs ADD COLUMN last_failed_stage TEXT")
        log.info("migrated jobs table: added last_failed_stage")
    if not _column_exists(conn, "jobs", "resume_from_stage"):
        conn.execute("ALTER TABLE jobs ADD COLUMN resume_from_stage TEXT")
        log.info("migrated jobs table: added resume_from_stage")
    if not _column_exists(conn, "jobs", "retry_count"):
        conn.execute("ALTER TABLE jobs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0")
        log.info("migrated jobs table: added retry_count")


def init_db() -> None:
    with _lock, connect() as conn:
        conn.executescript(SCHEMA_SQL)
        _migrate(conn)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES (?, ?)",
            ("schema_version", "3"),
        )
    log.info("sqlite initialized at %s", db_path())
