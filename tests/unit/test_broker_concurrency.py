"""Tests for the JobBroker dispatcher's max_concurrent_jobs honoring.

These tests stub out `_run_one` so we can verify the dispatcher caps
in-flight work correctly without spinning up real pipelines. We also
verify two queued jobs don't get claimed by two workers at once.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from aep.jobs.broker import (
    _MAX_CONCURRENCY_HARD_CAP,
    JobBroker,
    _runtime_mark_resumed,
)
from aep.jobs.models import Job, JobState
from aep.jobs.queue import get_job, insert_job, update_job
from aep.persist.db import init_db
from aep.persist.settings import AppSettings
from aep.pipeline.context import PipelineContext


@pytest.fixture(autouse=True)
def _db(tmp_runtime):
    # tmp_runtime sets AEP_RUNTIME_DIR + busts caches; we just need to
    # initialize the schema in the resulting empty runtime dir.
    init_db()
    yield


def _settings_with_concurrency(n: int) -> AppSettings:
    s = AppSettings()
    s.hardware.max_concurrent_jobs = n
    # These tests pre-date the M6.5 manual-start gate. They assert the
    # dispatcher's concurrency contract, not the queue-pause contract —
    # so flip auto_start_jobs True to keep the legacy fire-and-forget
    # behavior they were written against.
    s.general.auto_start_jobs = True
    return s


def test_broker_caps_inflight_at_max_concurrent_jobs() -> None:
    # With max=2 and 3 queued jobs, only 2 should be running at once.
    from aep.jobs.queue import insert_job

    inflight = 0
    peak_inflight = 0
    lock = threading.Lock()
    finished = threading.Event()
    finish_count = [0]
    total_jobs = 3

    def fake_run_one(self, job):
        nonlocal inflight, peak_inflight
        with lock:
            inflight += 1
            peak_inflight = max(peak_inflight, inflight)
        # Hold the slot long enough for the dispatcher to attempt to dispatch
        # a third job; if the cap is broken, peak_inflight will hit 3.
        time.sleep(0.15)
        with lock:
            inflight -= 1
            finish_count[0] += 1
            if finish_count[0] >= total_jobs:
                finished.set()

    broker = JobBroker()
    with patch(
        "aep.jobs.broker.load_settings",
        return_value=_settings_with_concurrency(2),
    ), patch.object(JobBroker, "_run_one", fake_run_one):
        for i in range(total_jobs):
            j = Job(
                source_path=f"/data/x{i}.mkv",
                output_path=None,
                preset_id="anime_balanced",
            )
            insert_job(j)
        broker.start()
        try:
            assert finished.wait(timeout=5.0), "jobs did not all complete in time"
        finally:
            broker.stop(timeout=2.0)

    assert peak_inflight == 2, f"expected peak 2, saw {peak_inflight}"


def test_concurrency_clamped_to_hard_cap() -> None:
    # Garbage-large settings shouldn't spawn 100 worker threads.
    broker = JobBroker()
    with patch(
        "aep.jobs.broker.load_settings",
        return_value=_settings_with_concurrency(99),
    ):
        broker.start()
        try:
            assert broker._pool is not None
            assert broker._pool._max_workers == _MAX_CONCURRENCY_HARD_CAP
        finally:
            broker.stop(timeout=1.0)


def test_each_job_dispatched_exactly_once() -> None:
    # Two workers must not both claim the same DB row.
    from aep.jobs.queue import insert_job

    seen_ids: list[str] = []
    seen_lock = threading.Lock()
    done = threading.Event()
    target = 5

    def fake_run_one(self, job):
        with seen_lock:
            seen_ids.append(job.id)
            if len(seen_ids) >= target:
                done.set()
        time.sleep(0.05)

    broker = JobBroker()
    with patch(
        "aep.jobs.broker.load_settings",
        return_value=_settings_with_concurrency(3),
    ), patch.object(JobBroker, "_run_one", fake_run_one):
        ids = []
        for i in range(target):
            j = Job(
                source_path=f"/data/y{i}.mkv",
                output_path=None,
                preset_id="anime_balanced",
            )
            insert_job(j)
            ids.append(j.id)
        broker.start()
        try:
            assert done.wait(timeout=5.0)
        finally:
            broker.stop(timeout=2.0)

    assert sorted(seen_ids) == sorted(ids)
    # No duplicate dispatches.
    assert len(seen_ids) == len(set(seen_ids))


# ---------- M6.5: queue-level pause/start gate ----------


def _settings_default_paused() -> AppSettings:
    """Default settings with auto_start_jobs=False (the M6.5 default)."""
    s = AppSettings()
    s.hardware.max_concurrent_jobs = 2
    # auto_start_jobs defaults to False; spell it out for clarity.
    s.general.auto_start_jobs = False
    return s


def test_broker_starts_paused_by_default_and_does_not_dispatch() -> None:
    """With default settings, queued jobs sit untouched until start_queue()."""
    from aep.jobs.queue import insert_job

    seen: list[str] = []
    seen_lock = threading.Lock()

    def fake_run_one(self, job):
        with seen_lock:
            seen.append(job.id)

    broker = JobBroker()
    with patch(
        "aep.jobs.broker.load_settings",
        return_value=_settings_default_paused(),
    ), patch.object(JobBroker, "_run_one", fake_run_one):
        for i in range(3):
            insert_job(Job(
                source_path=f"/data/p{i}.mkv",
                output_path=None,
                preset_id="anime_balanced",
            ))
        broker.start()
        try:
            # Give the dispatcher loop a few iterations — it should NOT pick
            # anything up because the queue is paused.
            assert broker.is_queue_paused() is True
            time.sleep(0.5)
            with seen_lock:
                assert seen == [], f"paused queue dispatched jobs: {seen}"
        finally:
            broker.stop(timeout=2.0)


def test_start_queue_releases_pause_and_dispatch_proceeds() -> None:
    """After start_queue() the dispatcher claims queued jobs."""
    from aep.jobs.queue import insert_job

    done = threading.Event()
    seen: list[str] = []
    seen_lock = threading.Lock()
    target = 2

    def fake_run_one(self, job):
        with seen_lock:
            seen.append(job.id)
            if len(seen) >= target:
                done.set()
        # Mirror real workers: terminal DB row + auto-pause when nothing left.
        db_job = get_job(job.id)
        assert db_job is not None
        db_job.state = JobState.COMPLETED
        db_job.current_stage = None
        db_job.finished_at = datetime.now(UTC).isoformat(timespec="seconds")
        update_job(db_job)
        self._maybe_auto_pause_queue_if_idle()

    broker = JobBroker()
    with patch(
        "aep.jobs.broker.load_settings",
        return_value=_settings_default_paused(),
    ), patch.object(JobBroker, "_run_one", fake_run_one):
        for i in range(target):
            insert_job(Job(
                source_path=f"/data/r{i}.mkv",
                output_path=None,
                preset_id="anime_balanced",
            ))
        broker.start()
        try:
            assert broker.is_queue_paused() is True
            broker.start_queue()
            assert broker.is_queue_paused() is False
            assert done.wait(timeout=5.0), "unpaused queue did not dispatch"
        finally:
            broker.stop(timeout=2.0)

    assert len(seen) == target
    assert broker.is_queue_paused() is True


def test_maybe_auto_pause_queue_if_idle_pauses_when_no_queued_or_running() -> None:
    """When dispatch is active but every row is terminal, pause until start_queue()."""
    broker = JobBroker()
    j = Job(
        source_path="/data/done.mkv",
        output_path=None,
        preset_id="anime_balanced",
        state=JobState.COMPLETED,
    )
    insert_job(j)
    with patch(
        "aep.jobs.broker.load_settings",
        return_value=_settings_default_paused(),
    ):
        broker.start()
        try:
            broker.start_queue()
            assert broker.is_queue_paused() is False
            broker._maybe_auto_pause_queue_if_idle()
            assert broker.is_queue_paused() is True
        finally:
            broker.stop(timeout=2.0)


def test_maybe_auto_pause_queue_if_idle_keeps_running_when_jobs_still_queued() -> None:
    broker = JobBroker()
    insert_job(
        Job(
            source_path="/data/wait.mkv",
            output_path=None,
            preset_id="anime_balanced",
        ),
    )
    with patch(
        "aep.jobs.broker.load_settings",
        return_value=_settings_default_paused(),
    ):
        broker.start()
        try:
            broker.start_queue()
            assert broker.is_queue_paused() is False
            broker._maybe_auto_pause_queue_if_idle()
            assert broker.is_queue_paused() is False
        finally:
            broker.stop(timeout=2.0)


def test_pause_queue_halts_further_dispatch_but_keeps_inflight() -> None:
    """pause_queue() prevents new claims; in-flight job runs to completion."""
    from aep.jobs.queue import insert_job

    enter_first = threading.Event()
    release_first = threading.Event()
    seen: list[str] = []
    seen_lock = threading.Lock()

    def fake_run_one(self, job):
        with seen_lock:
            seen.append(job.id)
            n = len(seen)
        if n == 1:
            enter_first.set()
            # Wait inside the first job so we have a chance to pause and
            # observe that the second job doesn't get claimed.
            release_first.wait(timeout=3.0)

    broker = JobBroker()
    settings = _settings_default_paused()
    settings.hardware.max_concurrent_jobs = 1  # serialize dispatch
    with patch(
        "aep.jobs.broker.load_settings",
        return_value=settings,
    ), patch.object(JobBroker, "_run_one", fake_run_one):
        for i in range(2):
            insert_job(Job(
                source_path=f"/data/h{i}.mkv",
                output_path=None,
                preset_id="anime_balanced",
            ))
        broker.start()
        try:
            broker.start_queue()
            assert enter_first.wait(timeout=3.0), "first job never started"
            broker.pause_queue()
            # Let the second job's potential claim window elapse.
            time.sleep(0.5)
            with seen_lock:
                assert len(seen) == 1, f"pause_queue didn't halt dispatch: {seen}"
            release_first.set()
            time.sleep(0.3)
            with seen_lock:
                # Still only one — second job stays queued because we're paused.
                assert len(seen) == 1
        finally:
            release_first.set()
            broker.stop(timeout=2.0)


def test_auto_start_jobs_true_boots_unpaused() -> None:
    """general.auto_start_jobs=True makes the broker boot un-paused."""
    broker = JobBroker()
    s = AppSettings()
    s.hardware.max_concurrent_jobs = 1
    s.general.auto_start_jobs = True
    with patch("aep.jobs.broker.load_settings", return_value=s):
        broker.start()
        try:
            assert broker.is_queue_paused() is False
        finally:
            broker.stop(timeout=1.0)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="seconds")


def test_queue_active_elapsed_excludes_current_pause_window() -> None:
    broker = JobBroker()
    broker._queue_started_at_s = 100.0
    broker._queue_paused_accum_s = 20.0
    broker._queue_pause_started_at_s = 150.0
    with patch("aep.jobs.broker.time.time", return_value=170.0):
        elapsed = broker.get_queue_active_elapsed_s()
    assert elapsed == pytest.approx(30.0)


def test_job_active_elapsed_subtracts_job_pause_windows() -> None:
    job = Job(
        source_path="/data/overlap.mkv",
        output_path=None,
        preset_id="anime_balanced",
        started_at=_iso(1000.0),
        finished_at=_iso(1020.0),
    )
    job.plan = {
        "__runtime": {
            "paused_accum_s": 7.0,
            "pause_started_at": _iso(1018.0),
        },
    }
    insert_job(job)
    # insert_job writes timestamps/rows at creation time; persist updated fields.
    update_job(job)

    broker = JobBroker()
    elapsed = broker.get_job_active_elapsed_s(job.id)
    # Total wall: 20s. Runtime metadata says 7s paused + 2s active pause window.
    assert elapsed == pytest.approx(11.0)


def test_resume_inactive_paused_job_clears_pause_metadata_before_queue() -> None:
    """Resuming a PAUSED row (worker already exited) must not leave pause_started_at set.

    Otherwise the dispatcher's later _runtime_mark_resumed would count queue wait
    time as paused time.
    """
    job = Job(
        source_path="/data/paused.mkv",
        output_path=None,
        preset_id="anime_balanced",
        state=JobState.PAUSED,
        started_at=_iso(1000.0),
        current_stage="05_upscale",
    )
    job.plan = {
        "__runtime": {
            "paused_accum_s": 0.0,
            "pause_started_at": _iso(1010.0),
        },
    }
    insert_job(job)
    update_job(job)

    broker = JobBroker()
    with patch("aep.jobs.broker._now", return_value=_iso(1015.0)):
        broker.resume(job.id)

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.QUEUED
    assert loaded.resume_from_stage == "05_upscale"
    runtime = loaded.plan["__runtime"]
    assert runtime["pause_started_at"] is None
    assert runtime["paused_accum_s"] == pytest.approx(5.0)

    # Dispatcher transition must not add queue-wait time to paused_accum_s.
    _runtime_mark_resumed(loaded, _iso(1050.0))
    update_job(loaded)
    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.plan["__runtime"]["paused_accum_s"] == pytest.approx(5.0)


def test_job_active_elapsed_returns_none_before_job_start() -> None:
    job = Job(
        source_path="/data/not-started.mkv",
        output_path=None,
        preset_id="anime_balanced",
    )
    insert_job(job)
    broker = JobBroker()
    assert broker.get_job_active_elapsed_s(job.id) is None


def test_job_active_elapsed_excludes_failed_idle_after_retry() -> None:
    job = Job(
        source_path="/data/failed.mkv",
        output_path=None,
        preset_id="anime_balanced",
        state=JobState.FAILED,
        started_at=_iso(1000.0),
        finished_at=_iso(1010.0),
    )
    insert_job(job)
    update_job(job)

    broker = JobBroker()
    assert broker.get_job_active_elapsed_s(job.id) == pytest.approx(10.0)

    with patch("aep.jobs.broker._now", return_value=_iso(1110.0)):
        broker.retry_failed(job.id)

    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.QUEUED
    assert loaded.finished_at is None
    assert loaded.plan["__runtime"]["paused_accum_s"] == pytest.approx(100.0)

    with patch("aep.jobs.broker.time.time", return_value=1110.0):
        assert broker.get_job_active_elapsed_s(job.id) == pytest.approx(10.0)


def test_pause_sets_pause_event_without_premature_db_paused_row(tmp_path: Path) -> None:
    """pause(job_id) cooperates with the worker; DB stays RUNNING until PausedError."""
    job = Job(source_path=str(tmp_path / "a.mkv"))
    job.state = JobState.RUNNING
    insert_job(job)

    broker = JobBroker()
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True, exist_ok=True)
    ctx = PipelineContext(
        job_id=job.id,
        source_path=tmp_path / "a.mkv",
        workdir=workdir,
        output_path=tmp_path / "out.mkv",
        preset_id="anime_balanced",
        preset_data={},
    )
    assert not ctx.pause_event.is_set()
    with broker._active_lock:
        broker._active[job.id] = ctx

    broker.pause(job.id)

    assert ctx.pause_event.is_set()
    loaded = get_job(job.id)
    assert loaded is not None
    assert loaded.state == JobState.RUNNING
