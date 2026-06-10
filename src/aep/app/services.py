"""Application service layer.

The GUI layer never imports from `aep.persist.*`, `aep.jobs.queue`, or pipeline modules
directly. It talks to these services. This is the seam we'll need anyway when we move
from in-process broker to a worker-process broker with IPC.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Collection
from pathlib import Path
from typing import Any, Callable

from aep.bench.models import BenchmarkRequest, BenchmarkResult
from aep.bench.runner import BenchmarkRunner, delete_benchmark_run
from aep.jobs.broker import JobBroker
from aep.jobs.cleanup import cleanup_job_artifacts
from aep.jobs.models import Job, JobState
from aep.jobs.queue import QueuedDispatchOrder
from aep.jobs.queue import delete_job as _delete_job
from aep.jobs.queue import get_job as _get_job
from aep.jobs.queue import update_job as _update_job
from aep.media.ffprobe import FfprobeAnalyzer
from aep.media.models import MediaInfo
from aep.persist.presets import (
    Preset,
    delete_user_preset,
    list_presets,
    load_preset,
    save_user_preset,
)
from aep.persist.settings import AppSettings, load_settings, save_settings
from aep.pipeline.events import StageEvent

log = logging.getLogger(__name__)


class SettingsService:
    def __init__(self) -> None:
        self._cached: AppSettings | None = None

    def get(self) -> AppSettings:
        if self._cached is None:
            self._cached = load_settings()
        return self._cached

    def update(self, settings: AppSettings) -> None:
        save_settings(settings)
        self._cached = settings


class PresetService:
    def list(self) -> list[Preset]:
        return list_presets()

    def get(self, preset_id: str) -> Preset:
        return load_preset(preset_id)

    def save(self, preset: Preset) -> Path:
        return save_user_preset(preset)

    def delete_user(self, preset_id: str) -> bool:
        return delete_user_preset(preset_id)


class MediaService:
    def __init__(self) -> None:
        self._analyzer = FfprobeAnalyzer()

    def analyze(self, path: Path) -> MediaInfo:
        return self._analyzer.analyze(path)


class JobService:
    """Thin wrapper around JobBroker; the GUI binds to this."""

    def __init__(self, broker: JobBroker) -> None:
        self._broker = broker

    def enqueue(
        self,
        source: Path,
        preset_id: str,
        *,
        output: Path | None = None,
        preset_overrides: dict[str, Any] | None = None,
    ) -> Job:
        return self._broker.enqueue(
            source, preset_id, output_path=output, preset_overrides=preset_overrides,
        )

    def preview_output_path(
        self, source: Path, preset_id: str, *, output: Path | None = None,
    ) -> Path:
        return self._broker.preview_output_path(source, preset_id, output_path=output)

    def cancel(self, job_id: str) -> None:
        self._broker.cancel(job_id)

    def pause(self, job_id: str) -> None:
        self._broker.pause(job_id)

    def any_job_in_ids_running(self, job_ids: Collection[str]) -> bool:
        """True if any of the given jobs still has pipeline state ``RUNNING``."""
        if not job_ids:
            return False
        want = frozenset(job_ids)
        return any(j.state == JobState.RUNNING for j in self.list_jobs() if j.id in want)

    def resume(self, job_id: str) -> None:
        self._broker.resume(job_id)

    # ----- queue-level pause/start --------------------------------------

    def start_queue(self) -> None:
        self._broker.start_queue()

    def set_queued_dispatch_order(self, order: QueuedDispatchOrder) -> None:
        self._broker.set_queued_dispatch_order(order)

    def pause_queue(self) -> None:
        self._broker.pause_queue()

    def is_queue_paused(self) -> bool:
        return self._broker.is_queue_paused()

    def get_queue_active_elapsed_s(self) -> float:
        return self._broker.get_queue_active_elapsed_s()

    def get_job_active_elapsed_s(self, job_id: str) -> float | None:
        return self._broker.get_job_active_elapsed_s(job_id)

    def remove(self, job_id: str) -> None:
        job = _get_job(job_id)
        self._broker.cancel(job_id)
        if job is not None:
            ramdisk = self.settings().paths.ramdisk_path
            cleanup_job_artifacts(
                job_id,
                ramdisk_path=Path(ramdisk) if ramdisk else None,
            )
        _delete_job(job_id)

    def clear_queue(self) -> None:
        """Remove every job like repeated ``remove()`` (cancel, artifacts, DB row)."""
        for jid in [j.id for j in self.list_jobs()]:
            self.remove(jid)

    def retry_failed(self, job_id: str) -> Job | None:
        return self._broker.retry_failed(job_id)

    @staticmethod
    def settings() -> AppSettings:
        return load_settings()

    def list_jobs(self) -> list[Job]:
        return self._broker.jobs()

    def get(self, job_id: str) -> Job | None:
        return _get_job(job_id)

    def set_preset_overrides(
        self, job_id: str, overrides: dict[str, Any] | None,
    ) -> Job | None:
        """Persist sparse preset overrides for a queued job.

        We refuse to mutate jobs that have already advanced past QUEUED:
        once the broker has loaded the preset and built a PipelineContext,
        the new config wouldn't be picked up without restarting the job,
        and silently dropping the change is worse than refusing it.
        """
        job = _get_job(job_id)
        if job is None:
            return None
        if job.state != JobState.QUEUED:
            log.warning(
                "refusing to update preset_overrides on job %s in state %s; "
                "only queued jobs can be edited",
                job_id, job.state.value,
            )
            return job
        # Treat empty dict the same as None so we don't write `{}` to the DB,
        # which would later look like "the user explicitly overrode nothing."
        job.preset_overrides = overrides if overrides else None
        _update_job(job)
        log.info("job %s preset_overrides updated: %s", job_id, job.preset_overrides)
        return job

    def subscribe(self, cb) -> None:  # type: ignore[no-untyped-def]
        self._broker.subscribe(cb)


class BenchmarkService:
    def __init__(self) -> None:
        self._runner = BenchmarkRunner()

    def run(
        self,
        request: BenchmarkRequest,
        *,
        cancel_event: threading.Event | None = None,
        on_event: Callable[[StageEvent], None] | None = None,
    ) -> BenchmarkResult:
        return self._runner.run(
            request,
            cancel_event=cancel_event,
            on_event=on_event,
        )

    def export_result(self, result: BenchmarkResult, path: Path) -> None:
        path.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )

    def delete_run_data(self, result: BenchmarkResult) -> None:
        delete_benchmark_run(result)

    @staticmethod
    def probe_vmaf_available() -> bool:
        from aep.bench.vmaf import is_libvmaf_available

        return is_libvmaf_available()


class AppServices:
    """Aggregate. Lives for the lifetime of the GUI."""

    def __init__(self) -> None:
        self.settings = SettingsService()
        self.presets = PresetService()
        self.media = MediaService()
        self.broker = JobBroker()
        self.jobs = JobService(self.broker)
        self.benchmark = BenchmarkService()

    def start(self) -> None:
        self.broker.start()

    def stop(self) -> None:
        self.broker.stop()
