"""Job DAO + simple queue ordering.

The queue is just "jobs in QUEUED state, ordered by created_at." The broker pulls one at
a time, up to HardwareSettings.max_concurrent_jobs, capped by the broker's hard limit.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from aep.jobs.models import Job, JobState
from aep.persist.db import connect


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def insert_job(job: Job) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs(id, source_path, output_path, preset_id, state, progress, "
            "error, current_stage, last_failed_stage, resume_from_stage, retry_count, "
            "created_at, updated_at, started_at, finished_at, plan_json, probe_json, "
            "preset_overrides_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job.id, job.source_path, job.output_path, job.preset_id,
                job.state.value, job.progress, job.error,
                job.current_stage, job.last_failed_stage, job.resume_from_stage, job.retry_count,
                job.created_at, job.updated_at, job.started_at, job.finished_at,
                json.dumps(job.plan), json.dumps(job.probe) if job.probe else None,
                json.dumps(job.preset_overrides) if job.preset_overrides else None,
            ),
        )


def update_job(job: Job) -> None:
    job.updated_at = _now()
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET source_path=?, output_path=?, preset_id=?, state=?, progress=?, "
            "error=?, current_stage=?, last_failed_stage=?, resume_from_stage=?, retry_count=?, "
            "updated_at=?, started_at=?, finished_at=?, plan_json=?, probe_json=?, preset_overrides_json=? "
            "WHERE id=?",
            (
                job.source_path, job.output_path, job.preset_id, job.state.value,
                job.progress, job.error, job.current_stage, job.last_failed_stage,
                job.resume_from_stage, job.retry_count, job.updated_at, job.started_at, job.finished_at,
                json.dumps(job.plan), json.dumps(job.probe) if job.probe else None,
                json.dumps(job.preset_overrides) if job.preset_overrides else None,
                job.id,
            ),
        )


def _row_to_job(row: dict) -> Job:
    # `row.keys()` here is the cursor description from a `SELECT *`. After a
    # mid-life ALTER TABLE the new column is present on fresh DBs but might
    # be absent on rows from older test fixtures, so use `.get()`-style
    # access via membership check instead of subscripting.
    overrides_json = row.get("preset_overrides_json") if hasattr(row, "get") else (
        row["preset_overrides_json"] if "preset_overrides_json" in row.keys() else None
    )
    current_stage = row.get("current_stage") if hasattr(row, "get") else (
        row["current_stage"] if "current_stage" in row.keys() else None
    )
    last_failed_stage = row.get("last_failed_stage") if hasattr(row, "get") else (
        row["last_failed_stage"] if "last_failed_stage" in row.keys() else None
    )
    resume_from_stage = row.get("resume_from_stage") if hasattr(row, "get") else (
        row["resume_from_stage"] if "resume_from_stage" in row.keys() else None
    )
    retry_count = row.get("retry_count") if hasattr(row, "get") else (
        row["retry_count"] if "retry_count" in row.keys() else 0
    )
    return Job(
        id=row["id"],
        source_path=row["source_path"],
        output_path=row["output_path"],
        preset_id=row["preset_id"],
        state=JobState(row["state"]),
        progress=row["progress"] or 0.0,
        error=row["error"],
        current_stage=current_stage,
        last_failed_stage=last_failed_stage,
        resume_from_stage=resume_from_stage,
        retry_count=int(retry_count or 0),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        plan=json.loads(row["plan_json"]) if row["plan_json"] else {},
        probe=json.loads(row["probe_json"]) if row["probe_json"] else None,
        preset_overrides=json.loads(overrides_json) if overrides_json else None,
    )


def list_jobs() -> list[Job]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY created_at ASC").fetchall()
    return [_row_to_job(dict(r)) for r in rows]


def next_queued() -> Job | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE state=? ORDER BY created_at ASC LIMIT 1",
            (JobState.QUEUED.value,),
        ).fetchone()
    if not row:
        return None
    return _row_to_job(dict(row))


def get_job(job_id: str) -> Job | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return None
    return _row_to_job(dict(row))


def delete_job(job_id: str) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        conn.execute("DELETE FROM stage_cache WHERE job_id=?", (job_id,))
