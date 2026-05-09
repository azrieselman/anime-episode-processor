"""Job broker.

Runs jobs in the same process on a dedicated worker thread pool; this keeps the GUI
responsive (the main thread never executes pipeline code) and lets us test the event
flow without IPC plumbing. The public API is broker-agnostic so a subprocess worker
backend can drop in later without GUI changes.

Public surface:
* enqueue(source_path, preset_id, output_path=None) -> Job
* cancel(job_id)
* pause(job_id) / resume(job_id)
* subscribe(callback) — receives StageEvents AND job state changes
* start() / stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from aep.errors import AEPError, CancelledError, PausedError
from aep.jobs.cleanup import cleanup_job_artifacts
from aep.jobs.models import Job, JobState
from aep.jobs.queue import get_job, insert_job, list_jobs, next_queued, update_job
from aep.logging_setup import attach_job_log_handler, detach_job_log_handler
from aep.media.models import MediaInfo
from aep.persist.presets import Preset, load_preset
from aep.persist.settings import load_settings
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.runner import PipelineRunner, build_default_stages
from aep.util.paths import jobs_dir

log = logging.getLogger(__name__)


BrokerCallback = Callable[[object], None]   # receives StageEvent or Job


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _parse_timestamp_s(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


# Hard upper bound on concurrent jobs regardless of what settings say.
# A single job easily saturates a 10GB GPU; allowing 8+ would just thrash.
_MAX_CONCURRENCY_HARD_CAP = 4


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursive merge of `overrides` onto a copy of `base`.

    Lists are replaced wholesale (no element-level merge) because every list
    in our preset schema is semantically atomic -- e.g. encoder.extra_args
    is a complete arg vector, not a partial fragment. Non-dict scalars at
    leaves are simply overwritten.
    """
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# Sentinel payloads published on subscribe() callbacks for queue-state
# transitions. Subscribers may switch on isinstance(payload, QueueStateEvent)
# and inspect .paused. Kept as a tiny dataclass-style class (not @dataclass to
# avoid import noise) because the GUI needs to distinguish these from Job and
# StageEvent payloads.
class QueueStateEvent:
    __slots__ = ("paused",)

    def __init__(self, *, paused: bool) -> None:
        self.paused = paused

    def __repr__(self) -> str:  # pragma: no cover — debugging only
        return f"QueueStateEvent(paused={self.paused})"


class JobBroker:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._subs: list[BrokerCallback] = []
        self._active_lock = threading.Lock()
        self._active: dict[str, PipelineContext] = {}
        # Pool + semaphore are created lazily on start() so we read the
        # latest settings.hardware.max_concurrent_jobs at startup time.
        self._pool: ThreadPoolExecutor | None = None
        self._slots: threading.Semaphore | None = None
        # Queue-level pause flag. When True, the dispatcher loop will not
        # claim queued jobs even if slots are available. Default True so the
        # GUI flow is "drop files → configure → click Start Queue". Settings
        # can opt back into the old auto-start behavior via
        # general.auto_start_jobs (read once on start()).
        self._queue_paused: bool = True
        self._queue_timing_lock = threading.Lock()
        self._queue_started_at_s: float | None = None
        self._queue_pause_started_at_s: float | None = None
        self._queue_paused_accum_s: float = 0.0
        self._queue_pause_windows: list[tuple[float, float | None]] = []

    # ----- subscription -------------------------------------------------

    def subscribe(self, cb: BrokerCallback) -> None:
        self._subs.append(cb)

    def _publish(self, payload: object) -> None:
        for cb in self._subs:
            try:
                cb(payload)
            except Exception:
                log.exception("broker subscriber raised; continuing")

    # ----- public API ---------------------------------------------------

    def enqueue(
        self,
        source_path: Path | str,
        preset_id: str,
        *,
        output_path: Path | str | None = None,
    ) -> Job:
        src = Path(source_path).resolve()
        job = Job(
            source_path=str(src),
            output_path=str(output_path) if output_path else None,
            preset_id=preset_id,
        )
        insert_job(job)
        log.info("enqueued job %s for %s preset=%s", job.id, src, preset_id)
        self._publish(job)
        self._wake.set()
        return job

    def preview_output_path(
        self,
        source_path: Path | str,
        preset_id: str,
        *,
        output_path: Path | str | None = None,
    ) -> Path:
        """Return the resolved output path that ``enqueue`` would produce.

        Used by the GUI to honor ``general.confirm_overwrite`` before adding a
        job — this returns the same path ``_run_one`` would derive at start.
        Falls back to the explicit ``output_path`` when set.
        """
        if output_path:
            return Path(output_path)
        src = Path(source_path).resolve()
        preset = load_preset(preset_id)
        job = Job(source_path=str(src), preset_id=preset_id)
        return self._derive_output(job, preset)

    def cancel(self, job_id: str) -> None:
        with self._active_lock:
            ctx = self._active.get(job_id)
        if ctx:
            ctx.cancel_event.set()
        # Update DB optimistically; the worker loop will mark it cancelled if currently
        # running, otherwise we persist the state directly.
        job = get_job(job_id)
        if job and not job.is_terminal():
            if job.state == JobState.QUEUED:
                job.state = JobState.CANCELLED
                job.finished_at = _now()
                update_job(job)
                self._publish(job)
                settings = load_settings()
                ramdisk_path = Path(settings.paths.ramdisk_path) if settings.paths.ramdisk_path else None
                cleanup_job_artifacts(job_id, ramdisk_path=ramdisk_path)

    def retry_failed(self, job_id: str) -> Job | None:
        job = get_job(job_id)
        if job is None or job.state != JobState.FAILED:
            return None
        if not job.last_failed_stage:
            return None
        job.state = JobState.QUEUED
        job.error = None
        job.finished_at = None
        job.current_stage = None
        job.resume_from_stage = job.last_failed_stage
        job.retry_count += 1
        update_job(job)
        self._publish(job)
        self._wake.set()
        return job

    def pause(self, job_id: str) -> None:
        with self._active_lock:
            ctx = self._active.get(job_id)
        if ctx:
            ctx.pause_event.set()
            job = get_job(job_id)
            if job:
                job.state = JobState.PAUSED
                update_job(job)
                self._publish(job)

    # ----- queue-level pause/start --------------------------------------

    def start_queue(self) -> None:
        """Release the queue-level pause so the dispatcher can claim jobs.

        Idempotent. Wakes the dispatcher in case it was sleeping on _wake.
        Does NOT affect per-job pause state (that's pause()/resume()).
        """
        if not self._queue_paused:
            return
        self._queue_paused = False
        with self._queue_timing_lock:
            now_s = time.time()
            if self._queue_pause_started_at_s is not None:
                self._queue_paused_accum_s += max(0.0, now_s - self._queue_pause_started_at_s)
                self._queue_pause_started_at_s = None
            if self._queue_pause_windows and self._queue_pause_windows[-1][1] is None:
                start_s, _ = self._queue_pause_windows[-1]
                self._queue_pause_windows[-1] = (start_s, now_s)
        log.info("queue started (dispatch unpaused)")
        self._publish(QueueStateEvent(paused=False))
        self._wake.set()

    def pause_queue(self) -> None:
        """Halt the dispatcher from claiming additional queued jobs.

        Already-running jobs continue to completion (use cancel()/pause()
        for those). Idempotent.
        """
        if self._queue_paused:
            return
        self._queue_paused = True
        with self._queue_timing_lock:
            now_s = time.time()
            if self._queue_pause_started_at_s is None:
                self._queue_pause_started_at_s = now_s
                self._queue_pause_windows.append((now_s, None))
        log.info("queue paused (dispatch halted)")
        self._publish(QueueStateEvent(paused=True))

    def is_queue_paused(self) -> bool:
        return self._queue_paused

    def get_queue_active_elapsed_s(self) -> float:
        with self._queue_timing_lock:
            if self._queue_started_at_s is None:
                return 0.0
            now_s = time.time()
            paused_s = self._queue_paused_accum_s
            if self._queue_pause_started_at_s is not None:
                paused_s += max(0.0, now_s - self._queue_pause_started_at_s)
            elapsed_s = max(0.0, now_s - self._queue_started_at_s - paused_s)
        return elapsed_s

    def get_job_active_elapsed_s(self, job_id: str) -> float | None:
        job = get_job(job_id)
        if job is None:
            return None
        start_s = _parse_timestamp_s(job.started_at)
        if start_s is None:
            return None
        terminal_end_s = _parse_timestamp_s(job.finished_at)
        end_s = terminal_end_s if terminal_end_s is not None else time.time()
        if end_s <= start_s:
            return 0.0
        with self._queue_timing_lock:
            pause_windows = list(self._queue_pause_windows)
        paused_overlap_s = 0.0
        for pause_start_s, pause_end_s in pause_windows:
            overlap_start_s = max(start_s, pause_start_s)
            overlap_end_s = min(end_s, pause_end_s if pause_end_s is not None else end_s)
            if overlap_end_s > overlap_start_s:
                paused_overlap_s += overlap_end_s - overlap_start_s
        return max(0.0, end_s - start_s - paused_overlap_s)

    def resume(self, job_id: str) -> None:
        with self._active_lock:
            ctx = self._active.get(job_id)
        if ctx:
            ctx.pause_event.clear()
            job = get_job(job_id)
            if job:
                job.state = JobState.RUNNING
                update_job(job)
                self._publish(job)
            return
        job = get_job(job_id)
        if not job or job.state != JobState.PAUSED:
            return
        if job.current_stage:
            job.resume_from_stage = job.current_stage
        job.state = JobState.QUEUED
        job.error = None
        job.finished_at = None
        update_job(job)
        self._publish(job)
        self._wake.set()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        # Read max_concurrent_jobs from settings; clamp to hard cap. The
        # setting is read once at start() so changing it later requires a
        # broker restart — this matches the behavior the user gets when
        # they save settings (the message box prompts a restart).
        try:
            n = load_settings().hardware.max_concurrent_jobs
        except Exception:
            n = 1
        n = max(1, min(int(n), _MAX_CONCURRENCY_HARD_CAP))
        self._slots = threading.Semaphore(n)
        self._pool = ThreadPoolExecutor(
            max_workers=n,
            thread_name_prefix="aep-job",
        )
        # Honor general.auto_start_jobs: if True, queue boots un-paused so
        # any pre-existing QUEUED jobs run immediately at startup.
        try:
            auto_start = bool(load_settings().general.auto_start_jobs)
        except Exception:
            auto_start = False
        self._queue_paused = not auto_start
        with self._queue_timing_lock:
            now_s = time.time()
            self._queue_started_at_s = now_s
            self._queue_pause_started_at_s = now_s if self._queue_paused else None
            self._queue_paused_accum_s = 0.0
            self._queue_pause_windows = [(now_s, None)] if self._queue_paused else []
        log.info(
            "broker starting with max_concurrent_jobs=%d queue_paused=%s",
            n, self._queue_paused,
        )
        self._thread = threading.Thread(target=self._loop, name="aep-broker", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        with self._active_lock:
            for ctx in self._active.values():
                ctx.cancel_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)
        if self._pool is not None:
            # Don't cancel pending futures — in-flight jobs already saw their
            # ctx.cancel_event above and will exit cleanly.
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None
            self._slots = None

    def jobs(self) -> list[Job]:
        return list_jobs()

    # ----- worker loop --------------------------------------------------

    def _loop(self) -> None:
        log.info("broker loop started")
        # Dispatcher pattern: this thread claims one queued job at a time and
        # hands it to the worker pool. The semaphore caps in-flight work at
        # max_concurrent_jobs; the slot is released by the worker on exit.
        # Claiming is serialized through this single thread, so two workers
        # can't pick up the same DB row even though next_queued() itself is
        # not transactional.
        while not self._stop.is_set():
            assert self._slots is not None and self._pool is not None
            # Block until a slot frees, but check _stop frequently so shutdown
            # doesn't have to wait for an in-flight job.
            if not self._slots.acquire(timeout=0.5):
                continue
            if self._stop.is_set():
                self._slots.release()
                break
            # Queue-level pause gate. We hold the slot during this check and
            # release it before sleeping so other dispatchers (none today,
            # but the semaphore is the API) can't be blocked by us.
            if self._queue_paused:
                self._slots.release()
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            job = next_queued()
            if not job:
                self._slots.release()
                self._wake.wait(timeout=1.0)
                self._wake.clear()
                continue
            # Transition to RUNNING immediately so the next loop iteration's
            # next_queued() won't pick this same row up again.
            job.state = JobState.RUNNING
            job.started_at = _now()
            job.error = None
            job.current_stage = None
            update_job(job)
            self._publish(job)
            try:
                self._pool.submit(self._worker_entry, job)
            except RuntimeError:
                # Pool was shut down between our checks — release slot and
                # exit cleanly; the job stays in RUNNING state but will be
                # picked back up next start (TODO: add a startup sweeper).
                self._slots.release()
                break
        log.info("broker loop exited")

    def _worker_entry(self, job: Job) -> None:
        """Pool entry point. Always releases its slot, even on failure."""
        try:
            self._run_one(job)
        except Exception:
            log.exception("worker hit an unexpected error for job %s", job.id)
        finally:
            if self._slots is not None:
                self._slots.release()

    def _run_one(self, job: Job) -> None:
        # State has already been set to RUNNING by the dispatcher loop; we
        # don't double-update here so we avoid an extra DB round-trip and
        # an extra publish event for the same transition.

        # Per-job log: a thread-filtered rotating handler that writes to
        # <workdir>/job.log. We attach this *before* preset/context load so
        # any failure there is captured in the per-job file too. Thread
        # filtering keeps concurrent jobs' logs from bleeding together.
        workdir = jobs_dir() / job.id
        workdir.mkdir(parents=True, exist_ok=True)
        job_log_path = workdir / "job.log"
        job_log_handler = attach_job_log_handler(
            job_log_path,
            thread_id=threading.get_ident(),
        )
        log.info("job %s started; log: %s", job.id, job_log_path)

        # Build context. Per-job preset_overrides are deep-merged onto the
        # loaded preset's JSON dump and re-validated so the planner and every
        # downstream stage see the user's GUI overrides (otherwise the broker
        # would silently use the on-disk preset).
        settings = load_settings()
        preset = load_preset(job.preset_id)
        resolved_preset_data = preset.model_dump(mode="json")
        resolved_preset_data.setdefault("decode", {})
        resolved_preset_data["decode"]["hwaccel"] = settings.hardware.decode_hwaccel
        if job.preset_overrides:
            resolved_preset_data = _deep_merge(resolved_preset_data, job.preset_overrides)
        preset = Preset.model_validate(resolved_preset_data)
        output = Path(job.output_path) if job.output_path else self._derive_output(job, preset)
        # Pull ramdisk_path from app settings; estimator runs inside 01_plan and
        # writes ctx.ramdisk_estimate_bytes, which the routing decision in
        # PipelineContext.stage_dir() consults alongside this path.
        ramdisk_path = (
            Path(settings.paths.ramdisk_path)
            if settings.paths.ramdisk_path
            else None
        )
        ctx = PipelineContext(
            job_id=job.id,
            source_path=Path(job.source_path),
            workdir=workdir,
            output_path=output,
            preset_id=job.preset_id,
            preset_data=resolved_preset_data,
            ramdisk_path=ramdisk_path,
        )
        ctx.extras["resume_from_stage"] = job.resume_from_stage
        ctx.extras["pipeline_order"] = settings.pipeline.order
        if job.resume_from_stage:
            plan_path = workdir / "01_plan" / "plan.json"
            if plan_path.is_file():
                try:
                    loaded_plan = json.loads(plan_path.read_text(encoding="utf-8"))
                    if isinstance(loaded_plan, dict):
                        ctx.plan = loaded_plan
                except Exception as exc:
                    log.warning(
                        "job %s: failed to rehydrate ctx.plan from %s: %s",
                        job.id, plan_path, exc,
                    )
            cuts_path = workdir / "03_scene_detect" / "scene_cuts.json"
            if cuts_path.is_file():
                try:
                    cuts_doc = json.loads(cuts_path.read_text(encoding="utf-8"))
                    cuts = cuts_doc.get("frame_indices", [])
                    if isinstance(cuts, list):
                        ctx.scene_cuts = [int(v) for v in cuts]
                except Exception as exc:
                    log.warning(
                        "job %s: failed to rehydrate scene cuts from %s: %s",
                        job.id, cuts_path, exc,
                    )
            probe_path = workdir / "00_probe" / "probe.json"
            if probe_path.is_file():
                try:
                    probe_doc = json.loads(probe_path.read_text(encoding="utf-8"))
                    ctx.media_info = MediaInfo.model_validate(probe_doc)
                except Exception as exc:
                    log.warning(
                        "job %s: failed to rehydrate media_info from %s: %s",
                        job.id, probe_path, exc,
                    )
        with self._active_lock:
            self._active[job.id] = ctx

        events = EventSink()
        events.subscribe(lambda ev: self._on_stage_event(job, ev))

        runner = PipelineRunner(build_default_stages(order=settings.pipeline.order))

        try:
            runner.run(ctx, events)
            job.state = JobState.COMPLETED
            job.progress = 1.0
            job.finished_at = _now()
            job.resume_from_stage = None
            if ctx.media_info:
                job.probe = ctx.media_info.model_dump(mode="json")
            update_job(job)
            self._publish(job)
        except PausedError:
            job.state = JobState.PAUSED
            job.error = None
            job.resume_from_stage = job.current_stage
            update_job(job)
            self._publish(job)
        except CancelledError:
            job.state = JobState.CANCELLED
            job.finished_at = _now()
            job.resume_from_stage = None
            update_job(job)
            self._publish(job)
        except AEPError as exc:
            job.state = JobState.FAILED
            job.error = f"{type(exc).__name__}: {exc}"
            job.finished_at = _now()
            job.last_failed_stage = job.current_stage
            update_job(job)
            self._publish(job)
            log.error("job %s failed: %s", job.id, job.error)
        except Exception as exc:
            job.state = JobState.FAILED
            job.error = f"Unhandled: {type(exc).__name__}: {exc}"
            job.finished_at = _now()
            job.last_failed_stage = job.current_stage
            update_job(job)
            self._publish(job)
            log.exception("job %s crashed", job.id)
        finally:
            with self._active_lock:
                self._active.pop(job.id, None)
            # Always detach the per-job log handler, even on crash, so the
            # next job on this thread doesn't accidentally inherit it.
            detach_job_log_handler(job_log_handler)
            if job.state == JobState.CANCELLED:
                cleanup_job_artifacts(job.id, ramdisk_path=ramdisk_path)

    def _on_stage_event(self, job: Job, ev: StageEvent) -> None:
        # Coalesce all of the per-event mutations into a single ``update_job``
        # call. The previous implementation could fire 4 SQLite writes for one
        # StageEvent (batch progress + current_stage + progress + completion
        # bump), each of which round-trips through WAL. Most events only need
        # to mutate one or two fields; we accumulate the dirty flag locally
        # and persist exactly once if anything changed.
        dirty = False

        if ev.kind in {"started", "completed", "skipped"}:
            with self._active_lock:
                ctx = self._active.get(job.id)
            if ctx is not None and self._update_batch_progress(job, ev, ctx):
                dirty = True

        if ev.kind == "started":
            job.current_stage = ev.stage
            dirty = True

        total_stages = 11
        if ev.progress is not None:
            base = self._stage_index(ev.stage) / total_stages
            job.progress = min(0.999, base + ev.progress / total_stages)
            dirty = True

        if ev.kind in {"completed", "skipped"}:
            stage_finished_progress = (self._stage_index(ev.stage) + 1) / total_stages
            if job.progress < stage_finished_progress:
                job.progress = stage_finished_progress
                dirty = True

        if dirty:
            update_job(job)
        self._publish(ev)

    @staticmethod
    def _update_batch_progress(job: Job, ev: StageEvent, ctx: PipelineContext) -> bool:
        """Persist user-facing batch progress as done/total into job.plan."""
        raw_batches = (ctx.plan or {}).get("batches")
        total = len(raw_batches) if isinstance(raw_batches, list) and raw_batches else 1

        done = 0
        if total > 1:
            segments_dir = ctx.workdir / "batch_segments"
            if segments_dir.is_dir():
                done = len(list(segments_dir.glob("segment_*.mkv")))
            if (
                ev.stage == "08_encode"
                and ev.kind in {"completed", "skipped"}
                and ctx._active_batch_idx is not None
            ):
                # 08 complete fires before the segment copy; include current batch.
                done = max(done, int(ctx._active_batch_idx) + 1)
            done = max(0, min(done, total))
        elif ev.stage == "10_validate" and ev.kind in {"completed", "skipped"}:
            done = 1

        plan = dict(job.plan or {})
        batch_progress = plan.get("batch_progress")
        prev_done = None
        prev_total = None
        if isinstance(batch_progress, dict):
            prev_done = batch_progress.get("done")
            prev_total = batch_progress.get("total")

        if prev_done == done and prev_total == total:
            return False

        plan["batch_progress"] = {"done": done, "total": total}
        job.plan = plan
        return True

    @staticmethod
    def _stage_index(name: str) -> int:
        order = [
            "00_probe", "01_plan", "02_sample_bench", "03_scene_detect", "04_decode_serve",
            "05_upscale", "06_interpolate", "07_postprocess", "08_encode", "09_mux",
            "10_validate",
        ]
        try:
            return order.index(name)
        except ValueError:
            return 0

    @staticmethod
    def _derive_output(job: Job, preset) -> Path:  # type: ignore[no-untyped-def]
        """Derive an output path by formatting the user's naming template.

        Variables (locked at 1.0 to keep the surface small):
            {source_stem}  base name of the input file (no extension)
            {height}       target output height as a string, e.g. "1080";
                           "src" when the preset preserves source resolution
            {fps}          target output fps rounded to int as a string, e.g.
                           "60"; "src" when neither target_fps nor multiplier
                           is set on the preset

        We resolve {height}/{fps} from the preset alone because this runs
        *before* the pipeline plan exists — the plan stage doesn't run until
        a job starts. Tokens that cannot be resolved fall back to "src".

        If the template is missing or formatting fails (unknown var, bad
        characters), fall back to the legacy "<stem>.aep.<ext>" form so the
        broker never refuses to start a job over cosmetic config issues.

        Output directory:
            * if general.output_dir is set, write there;
            * otherwise write next to the source file.
        """
        src = Path(job.source_path)
        ext = "mkv" if preset.container == "mkv" else "mp4"

        # Pull naming template + output_dir from settings (best-effort — we
        # don't want a bad settings file to make the broker explode).
        try:
            settings = load_settings()
            template = (settings.general.output_naming_template or "").strip()
            out_dir = settings.general.output_dir or None
        except Exception:
            template = ""
            out_dir = None

        # Resolve template vars from the preset.
        height = JobBroker._height_from_preset(preset)
        fps = JobBroker._fps_from_preset(preset)

        filename: str | None = None
        if template:
            try:
                filename = template.format(
                    source_stem=src.stem,
                    height=height,
                    fps=fps,
                )
            except (KeyError, IndexError, ValueError) as exc:
                log.warning(
                    "output_naming_template %r failed to format (%s); falling back",
                    template, exc,
                )
                filename = None

        if not filename:
            filename = f"{src.stem}.aep.{ext}"
        # If the template forgot a recognized container extension, append it.
        # Path.suffix is unreliable here because anime filenames often contain
        # multiple dots ("Show.S01E01"), so we check explicitly for .mkv/.mp4
        # at the end (case-insensitive).
        if not filename.lower().endswith((".mkv", ".mp4")):
            filename = f"{filename}.{ext}"

        parent = Path(out_dir).expanduser() if out_dir else src.parent
        return parent / filename

    @staticmethod
    def _height_from_preset(preset) -> str:  # type: ignore[no-untyped-def]
        """Best-effort target height for the naming template; "src" if unknown."""
        tr = preset.target_resolution
        if tr.mode == "explicit" and tr.height:
            return str(int(tr.height))
        if tr.mode == "named" and tr.named:
            # "1080p" -> "1080"; tr.named is constrained by the schema.
            return tr.named.rstrip("p")
        return "src"

    @staticmethod
    def _fps_from_preset(preset) -> str:  # type: ignore[no-untyped-def]
        """Best-effort target fps for the naming template; "src" if unknown.

        We prefer interpolation.target_fps when interpolation is enabled; if
        only multiplier is set we cannot compute a number without the source
        rate, so we fall back to "src". When interpolation is disabled the
        output rides the source rate — also "src".
        """
        ic = preset.interpolation
        if not getattr(ic, "enabled", False):
            return "src"
        if ic.target_fps:
            # Drop trailing zeros for integer-valued fps (60.0 → "60").
            f = float(ic.target_fps)
            return str(int(f)) if f.is_integer() else f"{f:g}"
        return "src"
