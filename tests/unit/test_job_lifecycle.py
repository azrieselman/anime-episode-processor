from __future__ import annotations

from pathlib import Path

from aep.jobs.cleanup import cleanup_job_artifacts
from aep.jobs.models import Job, JobState
from aep.jobs.queue import get_job, insert_job, update_job
from aep.persist.db import init_db


def test_job_retry_metadata_roundtrip(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.current_stage = "05_upscale"
    job.last_failed_stage = "05_upscale"
    job.resume_from_stage = "05_upscale"
    job.retry_count = 2
    insert_job(job)
    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.current_stage == "05_upscale"
    assert loaded.last_failed_stage == "05_upscale"
    assert loaded.resume_from_stage == "05_upscale"
    assert loaded.retry_count == 2


def test_cleanup_job_artifacts_removes_workdir_and_ramdisk(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    job_id = "cleanupjob"
    workdir = tmp_runtime / "jobs" / job_id
    ramdisk = tmp_path / "ramdisk"
    (workdir / "x").mkdir(parents=True, exist_ok=True)
    (workdir / "x" / "a.txt").write_text("x", encoding="utf-8")
    (ramdisk / job_id / "y").mkdir(parents=True, exist_ok=True)
    (ramdisk / job_id / "y" / "b.txt").write_text("y", encoding="utf-8")
    cleanup_job_artifacts(job_id, ramdisk_path=ramdisk)
    assert not workdir.exists()
    assert not (ramdisk / job_id).exists()


def test_retry_fields_can_be_updated(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    job = Job(source_path=str(tmp_path / "in.mkv"))
    insert_job(job)
    job.state = JobState.FAILED
    job.last_failed_stage = "06_interpolate"
    job.resume_from_stage = "06_interpolate"
    job.retry_count = 1
    update_job(job)
    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.FAILED
    assert loaded.last_failed_stage == "06_interpolate"
    assert loaded.resume_from_stage == "06_interpolate"
    assert loaded.retry_count == 1
