from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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


def test_broker_retry_failed_when_last_failed_stage_null(tmp_runtime: Path, tmp_path: Path) -> None:
    """Failures before any stage sets ``current_stage`` leave ``last_failed_stage`` null — retry must still work."""
    init_db()
    from aep.jobs.broker import JobBroker

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.state = JobState.FAILED
    job.last_failed_stage = None
    job.error = "RAM-disk gate"
    insert_job(job)

    broker = JobBroker()
    ret = broker.retry_failed(job.id)
    assert ret is not None
    assert ret.state == JobState.QUEUED
    assert ret.resume_from_stage is None
    assert ret.retry_count == 1
    assert ret.error is None

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.QUEUED


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


def test_broker_cancel_queued_resets_blank_state(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    from aep.jobs.broker import JobBroker

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.plan = {"batch_progress": {"done": 1, "total": 4}}
    job.progress = 0.4
    job.error = "x"
    job.started_at = "2020-01-01T00:00:00+00:00"
    job.current_stage = "04_decode_serve"
    job.last_failed_stage = "03_scene_detect"
    job.resume_from_stage = "02_sample_bench"
    job.retry_count = 3
    job.probe = {"a": 1}
    job.preset_overrides = {"decode": {"hwaccel": "x"}}
    orig_created = job.created_at
    insert_job(job)

    broker = JobBroker()
    broker.cancel(job.id)

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.QUEUED
    assert loaded.progress == 0.0
    assert loaded.plan == {}
    assert loaded.error is None
    assert loaded.started_at is None
    assert loaded.finished_at is None
    assert loaded.current_stage is None
    assert loaded.last_failed_stage is None
    assert loaded.resume_from_stage is None
    assert loaded.retry_count == 0
    assert loaded.probe is None
    assert loaded.preset_overrides == {"decode": {"hwaccel": "x"}}
    assert loaded.created_at == orig_created


def test_broker_cancel_paused_bumps_created_at(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    from aep.jobs.broker import JobBroker

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.state = JobState.PAUSED
    job.created_at = "2000-01-01T00:00:00+00:00"
    insert_job(job)

    broker = JobBroker()
    with patch("aep.jobs.broker._now", return_value="2099-01-01T00:00:00+00:00"):
        broker.cancel(job.id)

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.QUEUED
    assert loaded.created_at == "2099-01-01T00:00:00+00:00"


def test_broker_cancel_completed_is_noop(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    from aep.jobs.broker import JobBroker

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.state = JobState.COMPLETED
    job.finished_at = "2020-01-01T00:00:00+00:00"
    insert_job(job)

    broker = JobBroker()
    broker.cancel(job.id)

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.COMPLETED
    assert loaded.finished_at == "2020-01-01T00:00:00+00:00"
