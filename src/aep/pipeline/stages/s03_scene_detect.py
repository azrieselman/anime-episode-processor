"""Stage 03: scene_detect.

Detects scene cuts on the source's primary video stream (PySceneDetect or FFmpeg
``scdet``). Writes ``ctx.scene_cuts`` as a sorted list of 0-based source-frame indices,
which stage 06 (interpolate) consumes to avoid morphing across cuts.

When the preset disables interpolation (``preset.interpolation.enabled=False``)
this stage is a fast no-op — we emit an empty list and skip detection
entirely. The cache key reflects this so toggling the flag invalidates
correctly downstream.

Reads:    ctx.source_path, ctx.media_info, ctx.preset_data
Writes:   ctx.scene_cuts, <stage>/scene_cuts.json
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from fractions import Fraction
from pathlib import Path

from aep.adapters.ffmpeg import FFmpegAdapter
from aep.encode.scene_detect import (
    cuts_to_frame_indices,
    detect_scene_cuts,
    detect_scene_cuts_ffmpeg_scdet,
)
from aep.errors import PipelineError
from aep.persist.presets import Preset
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.fps import parse_rational
from aep.util.fps import total_frames as fps_total_frames


class SceneDetectStage(BaseStage):
    name = "03_scene_detect"

    def __init__(self, ffmpeg: FFmpegAdapter | None = None) -> None:
        self._ffmpeg = ffmpeg or FFmpegAdapter()

    # --------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        if ctx.media_info is None:
            raise PipelineError(f"{self.name} requires probe info")
        preset = Preset.model_validate(ctx.preset_data)
        threshold = preset.interpolation.scene_cut_threshold
        active = preset.interpolation.enabled
        scene_backend = str(preset.interpolation.scene_detect_backend)
        ffmpeg_pct = float(preset.interpolation.scene_change_threshold_percent)
        ffmpeg_scdet_w = int(preset.interpolation.ffmpeg_scdet_scale_width)
        params: dict[str, object] = {
            "threshold": threshold,
            "scene_detect_backend": scene_backend,
            "scene_change_threshold_percent": ffmpeg_pct,
            "ffmpeg_scdet_scale_width": ffmpeg_scdet_w,
            "active": active,
            "decode_hwaccel": str((ctx.plan.get("decode", {}) or {}).get("hwaccel", "off")),
        }
        cache_key = compute_cache_key(
            source_fingerprint=ctx.media_info.fmt.filename,
            stage_name=self.name,
            tool_versions=_scene_detect_tool_versions(active, scene_backend, self._ffmpeg),
            params=params,
        )
        out = ctx.stage_dir(self.name) / "scene_cuts.json"
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            inputs=[ctx.source_path],
            outputs=[out],
        )

    # --------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        media = ctx.media_info
        if media is None:
            raise PipelineError(f"{self.name}: ctx.media_info missing")
        primary = media.primary_video
        active = bool(plan.params.get("active", True))
        threshold = float(plan.params.get("threshold", 0.4))
        backend = str(plan.params.get("scene_detect_backend", "pyscenedetect"))
        ffmpeg_pct = float(plan.params.get("scene_change_threshold_percent", 10.0))
        ffmpeg_scdet_w = int(plan.params.get("ffmpeg_scdet_scale_width", 320))

        if not active or primary is None:
            ctx.scene_cuts = []
            _write_report(
                plan.outputs[0],
                cuts=[],
                threshold=threshold,
                fps=None,
                total_frames=None,
                raw_count=0,
                active=active,
                scene_detect_backend=backend,
            )
            events.emit(StageEvent(
                ctx.job_id, self.name, "log",
                message="scene detection disabled by preset" if not active else "no video stream",
            ))
            return StageResult(
                stage_name=self.name,
                success=True,
                duration_s=time.monotonic() - t0,
                artifacts={"scene_cuts_json": plan.outputs[0]},
                metrics={"cuts": 0, "skipped": True},
            )

        fps = parse_rational(primary.avg_frame_rate) or parse_rational(primary.r_frame_rate) or Fraction(24, 1)
        duration = primary.duration_s or media.fmt.duration_s or 0.0
        nb = primary.nb_frames or fps_total_frames(fps, duration)

        if backend == "ffmpeg_scdet":
            scale_note = f"scale_width={ffmpeg_scdet_w}" if ffmpeg_scdet_w > 0 else "scale_width=0 (full-res)"
            events.emit(StageEvent(
                ctx.job_id, self.name, "started",
                message=f"scene detect backend=ffmpeg_scdet threshold_percent={ffmpeg_pct} {scale_note}",
            ))
            try:
                raw_cuts = detect_scene_cuts_ffmpeg_scdet(
                    ctx.source_path,
                    ffmpeg_executable=self._ffmpeg.path,
                    video_stream_index=primary.index,
                    threshold_percent=ffmpeg_pct,
                    fps=fps,
                    scale_width=ffmpeg_scdet_w,
                )
            except Exception as exc:
                raise PipelineError(f"{self.name}: ffmpeg scene detection failed: {exc}") from exc
            remapped_threshold = ffmpeg_pct
        else:
            remapped_threshold = _map_legacy_threshold(threshold)
            events.emit(StageEvent(
                ctx.job_id, self.name, "started",
                message=f"scene detect backend=pyscenedetect threshold={threshold} mapped={remapped_threshold}",
            ))
            try:
                raw_cuts = detect_scene_cuts(str(ctx.source_path), threshold=remapped_threshold)
            except Exception as exc:
                raise PipelineError(f"{self.name}: scene detection failed: {exc}") from exc

        cuts_idx = cuts_to_frame_indices(raw_cuts, total_frames=nb)
        ctx.scene_cuts = cuts_idx

        _write_report(
            plan.outputs[0],
            cuts=cuts_idx,
            threshold=threshold,
            fps=str(fps),
            total_frames=nb,
            raw_count=len(raw_cuts),
            active=True,
            scene_detect_backend=backend,
            scene_change_threshold_percent=ffmpeg_pct if backend == "ffmpeg_scdet" else None,
            ffmpeg_scdet_scale_width=ffmpeg_scdet_w if backend == "ffmpeg_scdet" else None,
        )

        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=f"detected {len(cuts_idx)} scene cuts (raw {len(raw_cuts)}; "
                    f"fps {float(fps):.3f}; duration {duration:.1f}s; backend={backend})",
        ))
        metrics: dict[str, object] = {
            "cuts": len(cuts_idx),
            "raw_cuts": len(raw_cuts),
            "threshold": threshold,
            "mapped_threshold": remapped_threshold,
            "fps": str(fps),
            "scene_detect_backend": backend,
        }
        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"scene_cuts_json": plan.outputs[0]},
            metrics=metrics,
        )


def _scene_detect_tool_versions(active: bool, backend: str, ff: FFmpegAdapter) -> dict[str, str]:
    if not active:
        return {"pyscenedetect": "skipped", "ffmpeg": "skipped"}
    if backend == "ffmpeg_scdet":
        return {"pyscenedetect": "unused", "ffmpeg": _ffmpeg_adapter_version(ff)}
    return {"pyscenedetect": _safe_pyscenedetect_version(), "ffmpeg": "unused"}


def _safe_pyscenedetect_version() -> str:
    try:
        return importlib.metadata.version("scenedetect")
    except Exception:
        return "unknown"


def _ffmpeg_adapter_version(adapter: FFmpegAdapter) -> str:
    try:
        return adapter.version
    except Exception:
        return "unknown"


def _map_legacy_threshold(value: float) -> float:
    """Map legacy FFmpeg scene score threshold [0..1] into ContentDetector scale."""
    # Preserve current preset UX: 0.4 maps to the PySceneDetect default of 27.
    if value <= 1.0:
        return max(0.0, min(1.0, value)) * 67.5
    return max(0.0, value)


def _write_report(
    path: Path,
    *,
    cuts: list[int],
    threshold: float,
    fps: str | None,
    total_frames: int | None,
    raw_count: int,
    active: bool,
    scene_detect_backend: str | None = None,
    scene_change_threshold_percent: float | None = None,
    ffmpeg_scdet_scale_width: int | None = None,
) -> None:
    report: dict[str, object] = {
        "active": active,
        "threshold": threshold,
        "fps": fps,
        "total_frames": total_frames,
        "raw_cut_count": raw_count,
        "frame_indices": cuts,
    }
    if scene_detect_backend is not None:
        report["scene_detect_backend"] = scene_detect_backend
    if scene_change_threshold_percent is not None:
        report["scene_change_threshold_percent"] = scene_change_threshold_percent
    if ffmpeg_scdet_scale_width is not None:
        report["ffmpeg_scdet_scale_width"] = ffmpeg_scdet_scale_width
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
