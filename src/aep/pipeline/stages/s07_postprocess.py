"""Stage 07: postprocess.

Optional. When enabled and at least one sub-filter is on (deband / deblock /
grain_addback>0), runs FFmpeg with the configured ``-vf`` chain over the
frames produced by stage 06 (or stage 05 if interpolation is disabled, or
stage 04 if upscaling is also disabled). Output is again a directory of
numbered frames so the encode stage doesn't need to know whether postprocess
ran or not.

When postprocess is disabled or all sub-filters are off, this stage no-ops
and stage 08 reads from the prior stage directly. We do NOT short-circuit by
hardlinking — that would couple stage 07 and 08 cache keys together.

Reads:    ctx.plan, prior stage's frames dir
Writes:   <stage>/frames/, ctx.plan["postprocess"] = {active, dir}
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aep.adapters.ffmpeg import FFmpegAdapter, raise_if_failed
from aep.adapters.ncnn_base import empty_dir
from aep.constants import PNG_COMPRESSION_LEVEL
from aep.encode.postprocess import build_postprocess_chain
from aep.errors import PipelineError, StageError
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.proc import ProcError, run_capture

log = logging.getLogger(__name__)


class PostprocessStage(BaseStage):
    name = "07_postprocess"

    def __init__(self, ffmpeg: FFmpegAdapter | None = None) -> None:
        self._ffmpeg = ffmpeg or FFmpegAdapter()

    # --------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        if not ctx.plan:
            raise PipelineError(f"{self.name} requires 01_plan to have populated ctx.plan")
        cfg = ctx.plan.get("postprocess", {}) or {}
        chain = build_postprocess_chain(
            enabled=bool(cfg.get("enabled", False)),
            deband=bool(cfg.get("deband", False)),
            deblock=bool(cfg.get("deblock", False)),
            grain_addback=int(cfg.get("grain_addback", 0) or 0),
        )
        active = chain.vf is not None
        params: dict[str, object] = {
            "active": active,
            "vf": chain.vf or "",
            "input_source": cfg.get("input_source", "interpolate"),
            "frame_format": cfg.get("frame_format", "png"),
        }
        cache_key = compute_cache_key(
            source_fingerprint=str(ctx.source_path),
            stage_name=self.name,
            tool_versions={"ffmpeg": _safe_version(self._ffmpeg) if active else "skipped"},
            params=params,
        )
        out_dir = ctx.stage_dir(self.name) / "frames"
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            outputs=[out_dir],
            notes=chain.rationale,
        )

    # --------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        active = bool(plan.params.get("active", False))
        out_dir: Path = plan.outputs[0]
        frame_format = str(plan.params.get("frame_format", "png"))

        if not active:
            events.emit(StageEvent(ctx.job_id, self.name, "log",
                                   message="postprocess inactive (no filters enabled)"))
            return StageResult(stage_name=self.name, success=True,
                               duration_s=time.monotonic() - t0,
                               metrics={"skipped": True})

        # Resolve input dir.
        in_source = str(plan.params.get("input_source", "interpolate"))
        in_dir_str = (
            ctx.plan.get(in_source, {}).get("dir")
            or ctx.plan.get("upscale", {}).get("dir")
            or ctx.plan.get("decode", {}).get("dir")
        )
        if not in_dir_str:
            raise StageError(f"{self.name}: cannot resolve input dir from plan")
        in_dir = Path(in_dir_str)
        if not in_dir.is_dir():
            raise StageError(f"{self.name}: input frames dir missing: {in_dir}")

        in_manifest = ctx.get_frame_manifest(in_dir, format=frame_format)
        in_count = in_manifest["count"]
        if in_count == 0:
            raise StageError(f"{self.name}: no input frames in {in_dir}")
        empty_dir(out_dir)

        # We use FFmpeg's image2 demux/mux. Doing it through FFmpeg (rather than
        # rolling our own filter pipeline) keeps the filters consistent with
        # what the encode stage expects when we may apply postprocess
        # inline at encode time.
        vf = str(plan.params.get("vf"))
        cmd: list[str | object] = [
            self._ffmpeg.path,
            "-hide_banner", "-nostdin", "-loglevel", "error",
            "-y",
            "-start_number", "1",
            "-i", str(in_dir / f"%08d.{frame_format}"),
            "-map", "0:v:0",
            "-vf", vf,
            "-an", "-sn", "-dn",
        ]
        if frame_format == "png":
            # M6.5: PNG_COMPRESSION_LEVEL=6 (libpng default) keeps RAM-disk
            # footprint reasonable for batched mode without taxing CPU encode.
            cmd += ["-c:v", "png", "-compression_level", str(PNG_COMPRESSION_LEVEL)]
        else:
            cmd += ["-c:v", "libwebp", "-lossless", "1", "-compression_level", "6"]
        cmd += ["-start_number", "1", str(out_dir / f"%08d.{frame_format}")]

        events.emit(StageEvent(
            ctx.job_id, self.name, "started",
            message=f"postprocess vf=\"{vf}\" over {in_count} frames",
        ))
        try:
            result = run_capture(cmd, timeout=24 * 3600.0, check=False)
        except ProcError as exc:
            raise StageError(
                "postprocess: ffmpeg invocation failed",
                context={"stderr": exc.result.stderr[:2000]},
            ) from exc
        raise_if_failed(result.returncode, result.stderr)

        out_manifest = ctx.get_frame_manifest(out_dir, format=frame_format)
        out_count = out_manifest["count"]
        if out_count != in_count:
            raise StageError(
                f"{self.name}: produced {out_count} frames, expected {in_count}",
            )

        ctx.plan.setdefault("postprocess", {})
        ctx.plan["postprocess"]["count"] = out_count
        ctx.plan["postprocess"]["dir"] = str(out_dir)

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"frames_dir": out_dir},
            metrics={
                "frames": out_count,
                "vf": vf,
                "input_bytes": in_manifest["bytes"],
                "output_bytes": out_manifest["bytes"],
            },
        )


def _safe_version(adapter: FFmpegAdapter) -> str:
    try:
        return adapter.version
    except Exception:
        return "unknown"
