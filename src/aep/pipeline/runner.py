"""Pipeline runner.

Executes a list of Stages against a PipelineContext, handling:
* Per-stage timing
* Cache hits (stage skipping)
* Cancellation
* Pause/resume (cooperative — stages call `ctx.check_cancel()` and may also poll
  `ctx.pause_event`; the runner blocks between stages while paused)
* Error capture and rollback of the failed stage

Stages later than the failed one are NOT run. Earlier stages are not rolled back; their
outputs remain in the workdir so a future re-run can pick them up via the cache.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

from aep.errors import CancelledError, PausedError, PipelineError, StageError
from aep.media.models import MediaInfo
from aep.persist.settings import PipelineOrder
from aep.pipeline.batches import BatchSpec
from aep.pipeline.cache import (
    lookup as cache_lookup,
)
from aep.pipeline.cache import (
    read_stage_manifest,
    write_stage_manifest,
)
from aep.pipeline.cache import (
    record as cache_record,
)
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import Stage, StageResult
from aep.util.proc import proc_stats_scope

log = logging.getLogger(__name__)


def _rehydrate_plan_from_cached_stage_dir(
    ctx: PipelineContext, stage_name: str, stage_output_dir: Path
) -> None:
    """After a cache hit, `run()` is skipped — restore ctx fields from disk."""
    if stage_name == "00_probe":
        probe_path = stage_output_dir / "probe.json"
        if probe_path.is_file():
            try:
                doc = json.loads(probe_path.read_text(encoding="utf-8"))
                ctx.media_info = MediaInfo.model_validate(doc)
            except Exception:
                log.warning(
                    "rehydrate ctx.media_info from cached %s failed",
                    stage_output_dir,
                    exc_info=True,
                )
        return

    manifest = read_stage_manifest(stage_output_dir)
    if not manifest:
        return
    artifacts = manifest.get("artifacts") or {}
    metrics = manifest.get("metrics") or {}
    params = manifest.get("params") or {}
    try:
        if stage_name == "04_decode_serve":
            ctx.plan.setdefault("decode", {})
            fd = artifacts.get("frames_dir")
            if fd:
                ctx.plan["decode"]["dir"] = fd
            if "frames" in metrics:
                ctx.plan["decode"]["count"] = metrics["frames"]
            fmt = params.get("frame_format")
            if isinstance(fmt, str) and fmt:
                ctx.plan["decode"]["frame_format"] = fmt
        elif stage_name == "05_upscale":
            ctx.plan.setdefault("upscale", {})
            fd = artifacts.get("frames_dir")
            if fd:
                ctx.plan["upscale"]["dir"] = fd
        elif stage_name == "06_interpolate":
            ctx.plan.setdefault("interpolate", {})
            fd = artifacts.get("frames_dir")
            if fd:
                ctx.plan["interpolate"]["dir"] = fd
        elif stage_name == "07_postprocess":
            ctx.plan.setdefault("postprocess", {})
            fd = artifacts.get("frames_dir")
            if fd:
                ctx.plan["postprocess"]["dir"] = fd
    except Exception:
        log.warning(
            "rehydrate ctx.plan from cached %s (%s) failed",
            stage_name,
            stage_output_dir,
            exc_info=True,
        )


# M6.5: stages that run once per batch in batched mode. Order is preserved
# from the canonical stage list. The runner partitions stages into
# pre-batch (00-03), per-batch (04-08), and post-batch (09-10).
_PER_BATCH_STAGES: frozenset[str] = frozenset({
    "04_decode_serve",
    "05_upscale",
    "06_interpolate",
    "07_postprocess",
    "08_encode",
})


class PipelineRunner:
    def __init__(self, stages: Iterable[Stage]) -> None:
        self._stages: list[Stage] = list(stages)

    def run(self, ctx: PipelineContext, events: EventSink) -> dict[str, StageResult]:
        results: dict[str, StageResult] = {}
        resume_from_stage = str(ctx.extras.get("resume_from_stage") or "").strip() or None
        resume_index = self._stage_start_index(resume_from_stage)

        # M6.5: detect batched mode from the plan and partition the stage list.
        # 01_plan populates ctx.plan["batches"] (list of dicts) when the active
        # preset has batching enabled AND the source has a known duration.
        # We can't read it before 01_plan runs, so we run pre-batch stages
        # (00-03) first, then re-check.
        pre_batch: list[Stage] = []
        per_batch: list[Stage] = []
        post_batch: list[Stage] = []
        in_per_batch = False
        for stage in self._stages:
            if stage.name in _PER_BATCH_STAGES:
                per_batch.append(stage)
                in_per_batch = True
            elif in_per_batch:
                post_batch.append(stage)
            else:
                pre_batch.append(stage)

        # Phase 1: run pre-batch stages (00-03). After this, ctx.plan["batches"]
        # is populated if the planner decided to batch.
        for stage in pre_batch:
            if resume_index is not None and self._stage_index(stage.name) < resume_index:
                continue
            self._run_one_stage(ctx, stage, events, results)

        batches = self._extract_batches(ctx)
        if batches:
            self._run_batched(
                ctx,
                per_batch,
                batches,
                events,
                results,
                resume_index=resume_index,
            )
        else:
            # Single-pass: run per-batch stages exactly once with no active
            # batch index. Identical to pre-M6.5 behavior.
            for stage in per_batch:
                if resume_index is not None and self._stage_index(stage.name) < resume_index:
                    continue
                self._run_one_stage(ctx, stage, events, results)

        # Phase 3: post-batch stages (09 mux, 10 validate) run once.
        for stage in post_batch:
            if resume_index is not None and self._stage_index(stage.name) < resume_index:
                continue
            self._run_one_stage(ctx, stage, events, results)

        self._write_perf_profile(ctx, results)
        return results

    # ---------------------------------------------------------------- batched

    def _extract_batches(self, ctx: PipelineContext) -> list[BatchSpec]:
        """Reconstruct BatchSpec objects from the plan, if the planner emitted them."""
        raw = (ctx.plan or {}).get("batches") or []
        if not raw:
            return []
        batches: list[BatchSpec] = []
        for d in raw:
            try:
                # BatchSpec.duration_s is a derived @property — do not pass it
                # to the constructor. The plan dict carries it for convenience
                # of downstream consumers but the dataclass recomputes it.
                batches.append(BatchSpec(
                    index=int(d["index"]),
                    start_pts=float(d["start_pts"]),
                    end_pts=float(d["end_pts"]),
                    frame_count_estimate=int(d.get("frame_count_estimate", 0)),
                    est_bytes=int(d.get("est_bytes", 0)),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise PipelineError(
                    f"runner: malformed batch in plan: {d!r}",
                ) from exc
        return batches

    def _run_batched(
        self,
        ctx: PipelineContext,
        per_batch: list[Stage],
        batches: list[BatchSpec],
        events: EventSink,
        results: dict[str, StageResult],
        *,
        resume_index: int | None = None,
    ) -> None:
        """Iterate batches: set window, run 04-08, copy segment, cleanup.

        Per-batch StageResults are recorded into `results` under
        ``"<stage>__batch_<NN>"`` to keep them all visible. The most-recent
        batch's result is also written under the bare stage name so existing
        callers (e.g. mux fallback) keep working.
        """
        ctx.plan.setdefault("decode", {})
        segments_dir = ctx.workdir / "batch_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)
        per_batch_to_run = [
            s for s in per_batch
            if resume_index is None or self._stage_index(s.name) >= resume_index
        ]
        if not per_batch_to_run:
            # Resume may target a post-batch stage (09_mux / 10_validate). In that
            # case there is nothing left to execute per-batch; attempting to copy a
            # fresh encoded segment would fail because 08_encode was intentionally
            # skipped. Rehydrate any persisted batch segments for stage 09 and return.
            recovered_segments = sorted(
                p
                for p in segments_dir.glob("segment_*.mkv")
                if p.is_file() and p.stat().st_size > 0
            )
            ctx.encoded_segments = recovered_segments
            events.emit(StageEvent(
                ctx.job_id,
                "runner",
                "log",
                message=(
                    "per-batch stages already complete for this resume point; "
                    f"reusing {len(recovered_segments)} persisted segment(s)"
                ),
            ))
            return
        start_batch_idx = 0
        if (
            resume_index is not None
            and resume_index > self._stage_index("04_decode_serve")
            and not any(s.name == "04_decode_serve" for s in per_batch_to_run)
        ):
            start_batch_idx = self._infer_resume_batch_index(ctx)

        for batch in batches:
            if batch.index < start_batch_idx:
                continue
            ctx.check_cancel()
            batch_stages_to_run = list(per_batch_to_run)
            if (
                resume_index is not None
                and batch.index > start_batch_idx
                and not any(s.name == "04_decode_serve" for s in per_batch_to_run)
            ):
                # Resume stage narrowing should apply only to the resumed batch.
                # Subsequent batches must run the full per-batch chain.
                batch_stages_to_run = list(per_batch)
            has_frame_input = any(
                bool((ctx.plan.get(k, {}) or {}).get("dir"))
                and Path(str((ctx.plan.get(k, {}) or {}).get("dir"))).is_dir()
                for k in ("postprocess", "interpolate", "upscale", "decode")
            )
            if (
                any(s.name == "08_encode" for s in per_batch_to_run)
                and not any(s.name in {"04_decode_serve", "05_upscale", "06_interpolate", "07_postprocess"} for s in per_batch_to_run)
                and not has_frame_input
            ):
                batch_stages_to_run = [s for s in per_batch if self._stage_index(s.name) >= self._stage_index("04_decode_serve")]
            # Gate RAM-disk space only when this batch will decode fresh frames.
            if any(s.name == "04_decode_serve" for s in batch_stages_to_run):
                ctx.assert_ramdisk_has_room_for(batch)

            ctx._active_batch_idx = batch.index
            ctx.plan["decode"]["pts_window"] = (batch.start_pts, batch.end_pts)
            if not any(s.name == "04_decode_serve" for s in per_batch_to_run):
                decode_stage_dir = ctx.batch_dir(batch.index, "04_decode_serve")
                _rehydrate_plan_from_cached_stage_dir(ctx, "04_decode_serve", decode_stage_dir)
            if not any(s.name == "05_upscale" for s in per_batch_to_run):
                upscale_stage_dir = ctx.batch_dir(batch.index, "05_upscale")
                _rehydrate_plan_from_cached_stage_dir(ctx, "05_upscale", upscale_stage_dir)
            if not any(s.name == "06_interpolate" for s in per_batch_to_run):
                interpolate_stage_dir = ctx.batch_dir(batch.index, "06_interpolate")
                _rehydrate_plan_from_cached_stage_dir(ctx, "06_interpolate", interpolate_stage_dir)
            if not any(s.name == "07_postprocess" for s in per_batch_to_run):
                postprocess_stage_dir = ctx.batch_dir(batch.index, "07_postprocess")
                _rehydrate_plan_from_cached_stage_dir(ctx, "07_postprocess", postprocess_stage_dir)
            events.emit(StageEvent(
                ctx.job_id, "runner", "log",
                message=(
                    f"batch {batch.index:02d}/{len(batches):02d} "
                    f"[{batch.start_pts:.3f}s→{batch.end_pts:.3f}s, "
                    f"~{batch.frame_count_estimate} frames]"
                ),
            ))

            completed_stage_names: list[str] = []
            try:
                for stage in batch_stages_to_run:
                    self._run_one_stage(
                        ctx, stage, events, results,
                        result_alias=f"{stage.name}__batch_{batch.index:02d}",
                    )
                    completed_stage_names.append(stage.name)

                # Copy the encoded segment out of the RAM-disk before cleanup.
                # We always read it back from the actual stage_dir() to handle
                # the case where 08_encode was a cache hit.
                encode_dir = ctx.stage_dir("08_encode")
                src_segment = encode_dir / "video.mkv"
                if not src_segment.exists() or src_segment.stat().st_size == 0:
                    raise PipelineError(
                        f"batch {batch.index:02d}: encoded segment missing or empty",
                        context={"src": str(src_segment)},
                    )
                dst_segment = segments_dir / f"segment_{batch.index:02d}.mkv"
                _link_or_copy(src_segment, dst_segment)
                ctx.encoded_segments.append(dst_segment)
            finally:
                # Always clear active batch + window so a failure surfaces
                # cleanly without leaving stale state on the context.
                ctx._active_batch_idx = None
                ctx.plan.get("decode", {}).pop("pts_window", None)

            # Cleanup runs only after the segment is durably written.
            ctx.cleanup_batch_dir(batch.index)

    def _infer_resume_batch_index(self, ctx: PipelineContext) -> int:
        """Best-effort resume batch detection from surviving RAM-disk dirs.

        Completed batches are cleaned after their segment is copied out, so on a
        paused run the earliest remaining batch directory corresponds to the
        in-progress batch we should resume from.
        """
        if ctx.ramdisk_path is None:
            return 0
        root = ctx.ramdisk_path / ctx.job_id
        if not root.is_dir():
            return 0
        batch_indices: list[int] = []
        for d in root.glob("batch_*"):
            if not d.is_dir():
                continue
            name = d.name
            try:
                idx = int(name.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            batch_indices.append(idx)
        inferred = min(batch_indices) if batch_indices else 0
        return inferred

    # -------------------------------------------------------- single-stage run

    def _run_one_stage(
        self,
        ctx: PipelineContext,
        stage: Stage,
        events: EventSink,
        results: dict[str, StageResult],
        *,
        result_alias: str | None = None,
    ) -> None:
        """Run one stage with full caching/cancellation/error semantics.

        ``result_alias`` lets the batched path record per-batch results under
        unique keys (``08_encode__batch_03``) while still exposing the most
        recent run under the bare stage name for downstream callers.
        """
        ctx.check_cancel()
        self._wait_if_paused(ctx)

        t0 = time.monotonic()
        plan = stage.plan(ctx)

        def _record(result: StageResult) -> None:
            results[stage.name] = result
            ctx.stage_results[stage.name] = result
            if result_alias is not None:
                results[result_alias] = result

        if stage.can_skip(ctx):
            events.emit(StageEvent(ctx.job_id, stage.name, "skipped",
                                   message="stage marked skippable"))
            _record(StageResult(stage.name, success=True, duration_s=0.0,
                                skipped=True, notes=["can_skip()=True"]))
            return

        # Cache lookup is keyed only on (job_id, stage_name). In batched mode
        # multiple batches share a stage name, so we suppress the cache lookup
        # for per-batch stages — each batch's plan.cache_key already differs
        # (pts_window is in params), but the persistent cache_record API can't
        # distinguish them. The runtime cost is small because batches are
        # small; correctness wins.
        skip_cache = stage.name in _PER_BATCH_STAGES and ctx._active_batch_idx is not None
        if not skip_cache:
            cache_hit = cache_lookup(ctx.job_id, stage.name)
            if cache_hit and cache_hit[0] == plan.cache_key:
                _rehydrate_plan_from_cached_stage_dir(
                    ctx, stage.name, cache_hit[1],
                )
                events.emit(StageEvent(ctx.job_id, stage.name, "skipped",
                                       message=f"cache hit ({plan.cache_key[:8]})"))
                _record(StageResult(stage.name, success=True, duration_s=0.0,
                                    cached=True, notes=[f"cache_key={plan.cache_key}"]))
                return

        events.emit(StageEvent(ctx.job_id, stage.name, "started",
                               message=f"plan={plan.cache_key[:8]}"))
        try:
            with proc_stats_scope() as proc_stats:
                result = stage.run(ctx, plan, events)
        except CancelledError:
            events.emit(StageEvent(ctx.job_id, stage.name, "warning",
                                   message="cancelled mid-stage; rolling back"))
            stage.rollback(ctx, plan)
            raise
        except PausedError:
            events.emit(StageEvent(ctx.job_id, stage.name, "warning",
                                   message="paused at safe checkpoint"))
            raise
        except Exception as exc:
            events.emit(StageEvent(ctx.job_id, stage.name, "error",
                                   message=str(exc)))
            log.exception("stage %s failed", stage.name)
            stage.rollback(ctx, plan)
            raise StageError(f"stage {stage.name} failed: {exc}",
                             context={"stage": stage.name}) from exc

        result.duration_s = time.monotonic() - t0
        frame_manifest_stats = dict(ctx.frame_manifest_stats)
        result.metrics.setdefault("perf", {})
        result.metrics["perf"].update({
            "proc_calls": int(proc_stats.get("calls", 0.0)),
            "proc_streaming_calls": int(proc_stats.get("streaming_calls", 0.0)),
            "proc_capture_calls": int(proc_stats.get("capture_calls", 0.0)),
            "proc_wall_s": round(float(proc_stats.get("wall_s", 0.0)), 3),
            "frame_manifest_scans_total": int(frame_manifest_stats.get("scans", 0)),
            "frame_manifest_cache_hits_total": int(frame_manifest_stats.get("cache_hits", 0)),
        })
        _record(result)

        stage_dir = ctx.stage_dir(stage.name)
        write_stage_manifest(stage_dir, {
            "stage": stage.name,
            "cache_key": plan.cache_key,
            "params": plan.params,
            "duration_s": result.duration_s,
            "metrics": result.metrics,
            "artifacts": {k: str(v) for k, v in result.artifacts.items()},
            "notes": result.notes,
        })

        if result.success:
            if not skip_cache:
                cache_record(ctx.job_id, stage.name, plan.cache_key, stage_dir)
            events.emit(StageEvent(ctx.job_id, stage.name, "completed",
                                   message=f"{result.duration_s:.1f}s"))
        else:
            raise PipelineError(f"stage {stage.name} returned success=False")

    def _write_perf_profile(self, ctx: PipelineContext, results: dict[str, StageResult]) -> None:
        by_stage: dict[str, dict[str, object]] = {}
        stage_has_batch_runs: set[str] = set()
        for name in results:
            if "__batch_" in name:
                stage_has_batch_runs.add(name.split("__batch_", 1)[0])
        total_proc_calls = 0
        total_proc_wall_s = 0.0
        for name, result in results.items():
            is_batch_alias = "__batch_" in name
            stage_name = name.split("__batch_", 1)[0] if is_batch_alias else name
            # In batched jobs, the bare stage key contains only the most-recent
            # batch result. Skip it to avoid counting one batch twice.
            if not is_batch_alias and stage_name in stage_has_batch_runs:
                continue
            perf = (result.metrics or {}).get("perf", {})
            if isinstance(perf, dict):
                total_proc_calls += int(perf.get("proc_calls", 0) or 0)
                total_proc_wall_s += float(perf.get("proc_wall_s", 0.0) or 0.0)
            stage_entry = by_stage.setdefault(stage_name, {
                "duration_s": 0.0,
                "runs": 0,
                "metrics": {},
            })
            stage_entry["duration_s"] = round(
                float(stage_entry.get("duration_s", 0.0)) + float(result.duration_s),
                3,
            )
            stage_entry["runs"] = int(stage_entry.get("runs", 0)) + 1
            if isinstance(perf, dict):
                aggregated_perf = stage_entry["metrics"].setdefault("perf", {})
                if isinstance(aggregated_perf, dict):
                    aggregated_perf["proc_calls"] = int(aggregated_perf.get("proc_calls", 0)) + int(perf.get("proc_calls", 0) or 0)
                    aggregated_perf["proc_streaming_calls"] = int(aggregated_perf.get("proc_streaming_calls", 0)) + int(perf.get("proc_streaming_calls", 0) or 0)
                    aggregated_perf["proc_capture_calls"] = int(aggregated_perf.get("proc_capture_calls", 0)) + int(perf.get("proc_capture_calls", 0) or 0)
                    aggregated_perf["proc_wall_s"] = round(
                        float(aggregated_perf.get("proc_wall_s", 0.0)) + float(perf.get("proc_wall_s", 0.0) or 0.0),
                        3,
                    )
                    # Frame-manifest stats are cumulative counters sampled per-stage;
                    # preserve the highest observed totals for easier interpretation.
                    aggregated_perf["frame_manifest_scans_total"] = max(
                        int(aggregated_perf.get("frame_manifest_scans_total", 0)),
                        int(perf.get("frame_manifest_scans_total", 0) or 0),
                    )
                    aggregated_perf["frame_manifest_cache_hits_total"] = max(
                        int(aggregated_perf.get("frame_manifest_cache_hits_total", 0)),
                        int(perf.get("frame_manifest_cache_hits_total", 0) or 0),
                    )
        payload = {
            "job_id": ctx.job_id,
            "generated_at_epoch_ms": int(time.time() * 1000),
            "total_stages": len(by_stage),
            "total_proc_calls": total_proc_calls,
            "total_proc_wall_s": round(total_proc_wall_s, 3),
            "frame_manifest_stats": dict(ctx.frame_manifest_stats),
            "stages": by_stage,
        }
        out_path = ctx.workdir / "perf_profile.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _wait_if_paused(self, ctx: PipelineContext) -> None:
        if ctx.pause_event.is_set():
            log.info("pipeline paused for job %s", ctx.job_id)
            while ctx.pause_event.is_set():
                if ctx.cancel_event.wait(timeout=0.5):
                    raise CancelledError("cancelled while paused")
            log.info("pipeline resumed for job %s", ctx.job_id)

    def stage_names(self) -> list[str]:
        return [s.name for s in self._stages]

    def _stage_index(self, stage_name: str) -> int:
        try:
            return self.stage_names().index(stage_name)
        except ValueError:
            return -1

    def _stage_start_index(self, stage_name: str | None) -> int | None:
        if not stage_name:
            return None
        idx = self._stage_index(stage_name)
        return idx if idx >= 0 else None


def build_default_stages(
    *,
    order: PipelineOrder = "interpolate_first",
) -> list[Stage]:
    """Construct the canonical stage list.

    The upscaler / interpolation / postprocess stages no-op cleanly when their
    plan subdicts have ``active``/``enabled`` set False. Batched presets still use
    the frame path (decode → encode per chunk). 02_sample_bench is a placeholder until
    live perf benchmarking against the source is implemented.

    ``order`` controls whether ``05_upscale`` runs before ``06_interpolate`` or after;
    stage *names* stay fixed for cache keys and resume.
    """
    from aep.pipeline.stages.placeholder import PlaceholderStage
    from aep.pipeline.stages.s00_probe import ProbeStage
    from aep.pipeline.stages.s01_plan import PlanStage
    from aep.pipeline.stages.s03_scene_detect import SceneDetectStage
    from aep.pipeline.stages.s04_decode_serve import DecodeServeStage
    from aep.pipeline.stages.s05_upscale import UpscaleStage
    from aep.pipeline.stages.s06_interpolate import InterpolateStage
    from aep.pipeline.stages.s07_postprocess import PostprocessStage
    from aep.pipeline.stages.s08_encode import EncodeStage
    from aep.pipeline.stages.s09_mux import MuxStage
    from aep.pipeline.stages.s10_validate import ValidateStage

    upscale_stage = UpscaleStage()
    interpolate_stage = InterpolateStage()
    if order == "interpolate_first":
        frame_stages = [interpolate_stage, upscale_stage]
    else:
        frame_stages = [upscale_stage, interpolate_stage]

    return [
        ProbeStage(),
        PlanStage(),
        PlaceholderStage("02_sample_bench"),
        SceneDetectStage(),
        DecodeServeStage(),
        *frame_stages,
        PostprocessStage(),
        EncodeStage(),
        MuxStage(),
        ValidateStage(),
    ]


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
        return
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
