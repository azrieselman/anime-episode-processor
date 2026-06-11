"""Stage 06: interpolate.

Runs RIFE against the upscaled frames produced by stage 05 (or the decoded
frames produced by stage 04 if upscaling was disabled). Honors scene-cut
boundaries from stage 03 by invoking RIFE once on the full batch and then
overwriting the morphed frames at each cut with hardlinks to the boundary
frame.

Why one RIFE pass per batch (instead of per-run sub-batching)?
RIFE has no scene-cut hint, and a vanilla pass will interpolate across hard
cuts, producing the morphing-transition look that makes Waifu2x-Extension-GUI's
RIFE output recognizable. The previous implementation split the batch into
per-cut runs and invoked RIFE separately on each one, which (a) spawned the
binary K+1 times per batch, (b) inserted M-1 extra duplicate frames per cut
and so drifted timing slightly, and (c) crashed when a cut produced a
single-frame run. The consolidated path runs RIFE once on all `L` input
frames (yielding `L*M` outputs) and then replaces, for each scene cut at
input frame `c`, the M-1 morphed output frames at indices ``(c-1)*M + 2 ..
c*M`` with hardlinks to the preserved RIFE output of input `c-1`. Output
length stays exactly `L*M` so timing matches a non-cut-aware RIFE pass.

Cut indices in ``ctx.scene_cuts`` are 0-based source-frame indices; in
batched mode they're translated to batch-local input indices using the
batch's PTS window and the source FPS.

Behavior:
* When ``interpolate.active=False``, no-op and downstream falls back to the
  prior stage's frames.
* When ``multiplier <= 1``, also no-op (interpolation produces no extra frames).
* Otherwise: stage all input frames, run RIFE once, post-process morphed
  frames at each scene cut.

Reads:    ctx.plan, ctx.scene_cuts, ctx.media_info, stage_05 frames (or stage_04 frames)
Writes:   ctx.plan["interpolate"] = {count, dir, multiplier, cuts_applied}
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from aep.adapters.ncnn_base import empty_dir, stderr_indicates_vulkan_gpu_fault
from aep.adapters.rife import (
    RifeAdapter,
    RifeJob,
    expected_output_count,
    local_cuts_from_global,
    morphed_output_range,
    replace_with_boundary_dup,
)
from aep.constants import DEFAULT_RIFE_THREADS
from aep.errors import CancelledError, PausedError, PipelineError, StageError
from aep.persist.settings import load_settings
from aep.pipeline.batch_timing import (
    assert_frame_dir_count,
    count_numeric_frames_in_dir,
    drop_rife_output_prefix,
    resolve_batch_frame_plan,
)
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.frame_dedupe import (
    decode_batch_frame_offset,
    expand_rife_output_dir,
    local_cuts_compact_from_full,
)
from aep.util.proc import ProcError, ProcInterrupted, run_streaming

log = logging.getLogger(__name__)

# RIFE may log vkQueueSubmit failures without exiting; retry the full invocation.
_MAX_RIFE_GPU_FAULT_ATTEMPTS = 3


class InterpolateStage(BaseStage):
    name = "06_interpolate"

    def __init__(self, rife: RifeAdapter | None = None) -> None:
        self._rife = rife

    # --------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        if not ctx.plan:
            raise PipelineError(f"{self.name} requires 01_plan to have populated ctx.plan")
        cfg = ctx.plan.get("interpolate", {}) or {}
        active = bool(cfg.get("active", False))
        in_src = str(cfg.get("input_source", "upscale"))
        rife_threads = self._rife_threads()
        upstream_dir = ""
        if active:
            upstream_dir = str(
                ctx.plan.get(in_src, {}).get("dir")
                or ctx.plan.get("decode", {}).get("dir")
                or "",
            )
        params: dict[str, object] = {
            "active": active,
            "version": cfg.get("version"),
            "multiplier": cfg.get("multiplier", 1),
            "scene_cut_count": len(ctx.scene_cuts) if active else 0,
            "duplicate_on_scene_cut": cfg.get("duplicate_on_scene_cut", True),
            "fp16": cfg.get("fp16", True),
            "frame_format": cfg.get("frame_format", "png"),
            "rife_threads": rife_threads,
            "input_source": in_src,  # "upscale" or "decode"
            "upstream_frames_dir": upstream_dir,
        }
        fd = ctx.plan.get("frame_dedupe") or {}
        params["frame_dedupe"] = {
            "active": bool(fd.get("active")),
            "full_decode_count": fd.get("full_decode_count"),
            "compact_decode_count": fd.get("compact_decode_count"),
            "pipeline_order": ctx.plan.get("pipeline_order", "interpolate_first"),
        }
        tool_version = "skipped"
        if active:
            try:
                tool_version = self._adapter().version
            except Exception:
                tool_version = "unknown"
        cache_key = compute_cache_key(
            source_fingerprint=str(ctx.source_path),
            stage_name=self.name,
            tool_versions={"rife-ncnn-vulkan": tool_version},
            params=params,
        )
        out_dir = ctx.stage_dir(self.name) / "frames"
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            outputs=[out_dir],
        )

    # --------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        active = bool(plan.params.get("active", False))
        out_dir: Path = plan.outputs[0]
        multiplier = int(plan.params.get("multiplier", 1) or 1)

        if not active or multiplier <= 1:
            events.emit(StageEvent(
                ctx.job_id, self.name, "log",
                message=f"interpolation skipped (active={active}, multiplier={multiplier})",
            ))
            return StageResult(stage_name=self.name, success=True,
                               duration_s=time.monotonic() - t0,
                               metrics={"skipped": True, "multiplier": multiplier})

        # Resolve the input directory: prefer upscale, else decode.
        in_source = str(plan.params.get("input_source", "upscale"))
        in_dir_str = ctx.plan.get(in_source, {}).get("dir") or ctx.plan.get("decode", {}).get("dir")
        if not in_dir_str:
            raise StageError(f"{self.name}: cannot resolve input dir from plan ({in_source})")
        in_dir = Path(in_dir_str)
        if not in_dir.is_dir():
            raise StageError(f"{self.name}: input frames dir missing: {in_dir}")

        frame_format = str(plan.params.get("frame_format", "png"))
        in_manifest = ctx.get_frame_manifest(in_dir, format=frame_format)
        in_count = in_manifest["count"]
        if in_count == 0:
            raise StageError(f"{self.name}: no input frames in {in_dir}")
        empty_dir(out_dir)

        fd = ctx.plan.get("frame_dedupe") or {}
        pipeline_order = str(ctx.plan.get("pipeline_order") or "interpolate_first")
        full_decode_n = fd.get("full_decode_count")
        if not isinstance(full_decode_n, int) or full_decode_n <= 0:
            full_decode_n = in_count
        use_compact_rife = (
            bool(fd.get("active"))
            and pipeline_order == "interpolate_first"
            and isinstance(fd.get("kept_order"), list)
            and full_decode_n > in_count
        )
        kept_order: list[int] = list(fd["kept_order"]) if use_compact_rife else []

        batch_frame_plan = resolve_batch_frame_plan(ctx)
        rife_input_base = (
            batch_frame_plan.rife_input_base
            if batch_frame_plan is not None
            else decode_batch_frame_offset(ctx)
        )
        rife_input_count = (
            batch_frame_plan.rife_input_count
            if batch_frame_plan is not None
            else in_count
        )
        if batch_frame_plan is not None:
            numeric_in = count_numeric_frames_in_dir(in_dir, frame_format=frame_format)
            if numeric_in > 0:
                in_count = numeric_in
            if in_count != rife_input_count:
                raise StageError(
                    f"{self.name}: upstream produced {in_count} frames, expected "
                    f"{rife_input_count} for RIFE input (batch "
                    f"[{batch_frame_plan.start_pts:.3f}s, "
                    f"{batch_frame_plan.end_pts:.3f}s))",
                    context={
                        "in_dir": str(in_dir),
                        "batch_index": ctx._active_batch_idx,
                    },
                )

        # Translate global scene cuts into batch-local 1-based RIFE input indices.
        batch_offset = decode_batch_frame_offset(ctx)
        if use_compact_rife:
            local_cuts_full = local_cuts_from_global(
                ctx.scene_cuts,
                rife_input_base=rife_input_base,
                in_count=full_decode_n,
            )
            local_cuts = local_cuts_compact_from_full(
                local_cuts_full,
                kept_order,
                l_prime=in_count,
            )
        else:
            local_cuts = local_cuts_from_global(
                ctx.scene_cuts,
                rife_input_base=rife_input_base,
                in_count=rife_input_count,
            )
        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=(
                f"interpolating {in_count} frames at {multiplier}x "
                f"(rife_input_base={rife_input_base}, content_offset={batch_offset}, "
                f"full_decode={full_decode_n}, "
                f"{len(local_cuts)} scene cut(s) inside this batch "
                f"of {len(ctx.scene_cuts)} global)"
            ),
        ))

        if ctx.pause_event.is_set():
            ctx.extras["pause_checkpoint"] = {"stage": self.name}
            raise PausedError("paused before RIFE invocation")

        version = str(plan.params.get("version") or "v4.22-lite")

        # Single interpolation invocation across the entire batch.
        self._run_rife(
            adapter=self._adapter(),
            version=version,
            multiplier=multiplier,
            input_dir=in_dir,
            output_dir=out_dir,
            frame_format=frame_format,
            threads=str(plan.params.get("rife_threads") or DEFAULT_RIFE_THREADS),
            ctx=ctx,
            events=events,
        )

        out_manifest = ctx.get_frame_manifest(out_dir, format=frame_format)
        rife_out_count = out_manifest["count"]
        expected_rife_total = expected_output_count(rife_input_count, multiplier)
        if rife_out_count != expected_rife_total:
            raise StageError(
                f"{self.name}: RIFE produced {rife_out_count} frames, expected "
                f"{expected_rife_total} (in={rife_input_count}, M={multiplier})",
            )

        events.emit(StageEvent(
            ctx.job_id, self.name, "progress",
            progress=0.85,
            message=f"RIFE produced {rife_out_count} frames; applying scene-cut fixups",
        ))

        # Post-process: at each batch-local cut, overwrite the M-1 morphed
        # frames between input c-1 and input c with hardlinks to input c-1's
        # preserved RIFE output (output index (c-2)*M + 1).
        cuts_applied = 0
        for c in local_cuts:
            first_morph, count = morphed_output_range(c, multiplier)
            if count <= 0:
                continue
            boundary_idx = (c - 2) * multiplier + 1
            replace_with_boundary_dup(
                out_dir,
                boundary_idx=boundary_idx,
                start_idx=first_morph,
                count=count,
                format=frame_format,
            )
            cuts_applied += 1

        rife_skip = (
            batch_frame_plan.rife_output_skip
            if batch_frame_plan is not None
            else 0
        )
        if rife_skip > 0:
            drop_rife_output_prefix(
                out_dir,
                frame_format=frame_format,
                drop_count=rife_skip,
            )
            cache_key = f"{out_dir.resolve()}|{frame_format}"
            ctx.frame_manifests.pop(cache_key, None)
            events.emit(StageEvent(
                ctx.job_id, self.name, "log",
                message=(
                    f"dropped {rife_skip} RIFE output frame(s) from overlap context "
                    f"at batch boundary"
                ),
            ))

        if use_compact_rife:
            expand_root = out_dir.parent / "_rife_expand_tmp"
            empty_dir(expand_root)
            try:
                expand_rife_output_dir(
                    compact_rife_dir=out_dir,
                    dest_dir=expand_root,
                    kept_order=kept_order,
                    full_count=full_decode_n,
                    multiplier=multiplier,
                    frame_format=frame_format,
                )
            except OSError as exc:
                raise StageError(
                    f"{self.name}: frame dedupe expansion failed: {exc}",
                    context={"out_dir": str(out_dir)},
                ) from exc
            for p in list(out_dir.iterdir()):
                if p.is_file():
                    p.unlink()
            for p in sorted(expand_root.iterdir()):
                if p.is_file():
                    shutil.move(str(p), str(out_dir / p.name))
            shutil.rmtree(expand_root, ignore_errors=True)

        if use_compact_rife:
            expected_total = full_decode_n * multiplier
        elif batch_frame_plan is not None:
            expected_total = batch_frame_plan.expected_output_frames
        else:
            expected_total = expected_output_count(in_count, multiplier)

        # Sanity: total frame count is unchanged by the post-process step
        # (it overwrites in place rather than inserting frames).
        produced_manifest = ctx.get_frame_manifest(out_dir, format=frame_format)
        produced = produced_manifest["count"]
        if produced != expected_total:
            raise StageError(
                f"{self.name}: produced {produced} frames, expected {expected_total} "
                f"(in={in_count}, M={multiplier}, cuts_applied={cuts_applied})",
            )
        assert_frame_dir_count(
            out_dir,
            frame_format=frame_format,
            expected=expected_total,
            label=self.name,
        )

        ctx.plan.setdefault("interpolate", {})
        ctx.plan["interpolate"]["count"] = produced
        ctx.plan["interpolate"]["dir"] = str(out_dir)
        ctx.plan["interpolate"]["multiplier"] = multiplier
        ctx.plan["interpolate"]["cuts_applied"] = cuts_applied

        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=(
                f"interpolated to {produced} frames; "
                f"replaced morphs at {cuts_applied} scene cut(s)"
            ),
        ))
        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"frames_dir": out_dir},
            metrics={
                "frames": produced,
                "multiplier": multiplier,
                "cuts_applied": cuts_applied,
                "scene_cuts_global": len(ctx.scene_cuts),
                "input_bytes": in_manifest["bytes"],
                "output_bytes": produced_manifest["bytes"],
            },
        )

    # --------------------------------------------------------------- internals

    def _adapter(self) -> RifeAdapter:
        if self._rife is None:
            self._rife = RifeAdapter()
        return self._rife

    def _rife_threads(self) -> str:
        try:
            return load_settings().hardware.rife_threads
        except Exception as exc:
            log.debug("settings load failed at plan time; using default RIFE threads: %s", exc)
            return DEFAULT_RIFE_THREADS

    def _run_rife(
        self,
        *,
        adapter: RifeAdapter,
        version: str,
        multiplier: int,
        input_dir: Path,
        output_dir: Path,
        frame_format: str,
        threads: str,
        ctx: PipelineContext,
        events: EventSink,
    ) -> None:
        job = RifeJob(
            input_dir=input_dir, output_dir=output_dir,
            version=version, multiplier=multiplier,
            frame_format=frame_format,
            threads=threads,
        )
        argv = adapter.build_rife_argv(job)
        last_stderr = ""
        for attempt in range(1, _MAX_RIFE_GPU_FAULT_ATTEMPTS + 1):
            if attempt > 1:
                empty_dir(output_dir)
            gpu_fault_seen: list[bool] = [False]
            stderr_lines: list[str] = []

            def should_interrupt(
                _gpu_fault_seen: list[bool] = gpu_fault_seen,
            ) -> str | None:
                if ctx.cancel_event.is_set():
                    return "cancel"
                if ctx.pause_event.is_set():
                    return "pause"
                if _gpu_fault_seen[0]:
                    return "gpu_fault"
                return None

            try:
                for stream, line in run_streaming(argv, should_interrupt=should_interrupt):
                    if stream == "stderr":
                        stderr_lines.append(line)
                        if stderr_indicates_vulkan_gpu_fault(line):
                            gpu_fault_seen[0] = True
                        if line.strip():
                            events.emit(
                                StageEvent(ctx.job_id, self.name, "log", message=line.strip()),
                            )
            except ProcError as exc:
                last_stderr = exc.result.stderr
                if (
                    stderr_indicates_vulkan_gpu_fault(last_stderr)
                    and attempt < _MAX_RIFE_GPU_FAULT_ATTEMPTS
                ):
                    events.emit(StageEvent(
                        ctx.job_id, self.name, "log",
                        message=(
                            f"Vulkan GPU fault during RIFE (attempt {attempt}/"
                            f"{_MAX_RIFE_GPU_FAULT_ATTEMPTS}); retrying interpolation"
                        ),
                    ))
                    continue
                raise StageError(
                    "rife-ncnn-vulkan failed",
                    context={"stderr_tail": last_stderr[-2000:], "attempt": attempt},
                ) from exc
            except ProcInterrupted as exc:
                if exc.reason == "cancel":
                    raise CancelledError("cancelled during interpolation") from exc
                if exc.reason == "pause":
                    raise PausedError("paused during interpolation") from exc
                if exc.reason == "gpu_fault":
                    last_stderr = exc.result.stderr
                    if attempt < _MAX_RIFE_GPU_FAULT_ATTEMPTS:
                        events.emit(StageEvent(
                            ctx.job_id, self.name, "log",
                            message=(
                                f"Vulkan GPU fault during RIFE (attempt {attempt}/"
                                f"{_MAX_RIFE_GPU_FAULT_ATTEMPTS}); retrying interpolation"
                            ),
                        ))
                        continue
                    raise StageError(
                        "rife-ncnn-vulkan: Vulkan GPU fault persisted after retries",
                        context={
                            "stderr_tail": last_stderr[-2000:],
                            "attempts": attempt,
                        },
                    ) from exc
                raise StageError(
                    f"rife-ncnn-vulkan interrupted ({exc.reason})",
                    context={"stderr_tail": exc.result.stderr[-2000:]},
                ) from exc

            last_stderr = "".join(stderr_lines)
            if stderr_indicates_vulkan_gpu_fault(last_stderr):
                if attempt < _MAX_RIFE_GPU_FAULT_ATTEMPTS:
                    events.emit(StageEvent(
                        ctx.job_id, self.name, "log",
                        message=(
                            f"Vulkan GPU fault during RIFE (attempt {attempt}/"
                            f"{_MAX_RIFE_GPU_FAULT_ATTEMPTS}); retrying interpolation"
                        ),
                    ))
                    continue
                raise StageError(
                    "rife-ncnn-vulkan: Vulkan GPU fault persisted after retries",
                    context={
                        "stderr_tail": last_stderr[-2000:],
                        "attempts": attempt,
                    },
                )
            return

