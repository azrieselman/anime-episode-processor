from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from aep.jobs.cleanup import cleanup_job_artifacts
from aep.jobs.models import Job, JobState
from aep.jobs.queue import get_job, insert_job, update_job
from aep.persist.db import connect, init_db
from aep.persist.settings import AppSettings
from aep.pipeline.cache import lookup as cache_lookup
from aep.pipeline.cache import record as cache_record


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


def test_cleanup_job_artifacts_clears_stage_cache(tmp_runtime: Path, tmp_path: Path) -> None:
    """Without this, a re-queued job hits the cache for 00_probe but the
    rehydration of ctx.media_info silently no-ops (probe.json was deleted with
    the workdir), and 01_plan then explodes with
    'requires 00_probe to have populated ctx.media_info'.
    """
    init_db()
    job_id = "cleanup_cache_job"
    workdir = tmp_runtime / "jobs" / job_id
    probe_dir = workdir / "00_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "probe.json").write_text("{}", encoding="utf-8")
    cache_record(job_id, "00_probe", "deadbeef", probe_dir)
    cache_record(job_id, "01_plan", "cafebabe", workdir / "01_plan")
    other_job = "other_job"
    cache_record(other_job, "00_probe", "feedface", tmp_path / "other_probe")

    assert cache_lookup(job_id, "00_probe") is not None
    cleanup_job_artifacts(job_id)
    assert cache_lookup(job_id, "00_probe") is None
    assert cache_lookup(job_id, "01_plan") is None
    assert cache_lookup(other_job, "00_probe") is not None

    with connect() as conn:
        rows = conn.execute(
            "SELECT job_id FROM stage_cache WHERE job_id=?", (job_id,)
        ).fetchall()
    assert rows == []


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


def test_auto_retry_failed_enabled_requeues_and_increments(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    from aep.jobs.broker import JobBroker

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.state = JobState.FAILED
    job.last_failed_stage = "06_interpolate"
    insert_job(job)

    broker = JobBroker()
    settings = AppSettings()
    settings.general.auto_retry_failed_jobs = True
    settings.general.auto_retry_failed_job_attempts = 2

    did_retry = broker._auto_retry_failed_if_enabled(job, settings)
    assert did_retry is True
    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.QUEUED
    assert loaded.resume_from_stage == "06_interpolate"
    assert loaded.retry_count == 1


def test_auto_retry_failed_stops_at_max_attempts(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    from aep.jobs.broker import JobBroker

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.state = JobState.FAILED
    job.last_failed_stage = "05_upscale"
    job.retry_count = 2
    insert_job(job)

    broker = JobBroker()
    settings = AppSettings()
    settings.general.auto_retry_failed_jobs = True
    settings.general.auto_retry_failed_job_attempts = 2

    did_retry = broker._auto_retry_failed_if_enabled(job, settings)
    assert did_retry is False
    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.FAILED
    assert loaded.retry_count == 2


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


def test_recover_orphaned_batched_job_restarts_interrupted_batch(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    """After a crash mid-batch, resume from 04_decode_serve at the interrupted batch."""
    init_db()
    from aep.jobs.broker import recover_orphaned_running_job
    from aep.util.paths import jobs_dir

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.state = JobState.RUNNING
    job.current_stage = "06_interpolate"
    insert_job(job)

    workdir = jobs_dir() / job.id
    plan_dir = workdir / "01_plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / "plan.json").write_text(
        json.dumps({
            "batches": [
                {"index": 0, "start_pts": 0.0, "end_pts": 30.0},
                {"index": 1, "start_pts": 30.0, "end_pts": 60.0},
            ],
        }),
        encoding="utf-8",
    )
    seg_dir = workdir / "batch_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "segment_00.mkv").write_bytes(b"done")
    # No segment_01 yet — batch 1 was interrupted mid-pipeline.

    ramdisk = tmp_path / "ramdisk"
    batch_dir = ramdisk / job.id / "batch_01" / "06_interpolate"
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "frame.txt").write_text("x", encoding="utf-8")

    with patch(
        "aep.jobs.broker.load_settings",
        return_value=AppSettings.model_validate({
            "paths": {"ramdisk_path": str(ramdisk)},
        }),
    ):
        recover_orphaned_running_job(job)
        update_job(job)

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.QUEUED
    assert loaded.resume_from_stage == "04_decode_serve"
    assert loaded.current_stage is None
    assert loaded.plan.get("batch_progress") == {"done": 1, "total": 2}
    assert not (seg_dir / "segment_01.mkv").exists()
    assert not (ramdisk / job.id / "batch_01").exists()


def test_recover_orphaned_non_batched_job_resumes_current_stage(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    init_db()
    from aep.jobs.broker import recover_orphaned_running_job

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.state = JobState.RUNNING
    job.current_stage = "05_upscale"
    insert_job(job)

    recover_orphaned_running_job(job)
    update_job(job)

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.QUEUED
    assert loaded.resume_from_stage == "05_upscale"


def test_broker_start_sweeps_orphaned_running_jobs(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    init_db()
    import threading
    import time
    from unittest.mock import patch

    from aep.jobs.broker import JobBroker

    job = Job(source_path=str(tmp_path / "in.mkv"))
    job.state = JobState.RUNNING
    job.current_stage = "01_plan"
    insert_job(job)

    done = threading.Event()

    def fake_run_one(self, j):
        done.set()
        time.sleep(0.05)

    broker = JobBroker()
    settings = AppSettings()
    settings.general.auto_start_jobs = True
    with patch("aep.jobs.broker.load_settings", return_value=settings), patch.object(
        JobBroker, "_run_one", fake_run_one,
    ):
        broker.start()
        try:
            assert done.wait(timeout=5.0), "recovered job was not dispatched"
        finally:
            broker.stop(timeout=2.0)

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state in (JobState.RUNNING, JobState.COMPLETED)
