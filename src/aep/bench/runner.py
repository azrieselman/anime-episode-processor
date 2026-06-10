"""Standalone benchmark runner built on top of the pipeline."""

from __future__ import annotations

import json
import logging
import shutil
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from aep.bench.models import BenchmarkRequest, BenchmarkResult
from aep.bench.vmaf import compute_vmaf_for_segment, is_libvmaf_available
from aep.jobs.broker import merge_preset_data_for_job
from aep.persist.presets import Preset, load_preset
from aep.persist.settings import load_settings
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.runner import PipelineRunner, build_default_stages
from aep.pipeline.stage import StageResult
from aep.util.paths import bench_dir

log = logging.getLogger(__name__)

BenchmarkEventCallback = Callable[[StageEvent], None]

_FULL_SCOPE_STAGES = frozenset({
    "00_probe",
    "01_plan",
    "03_scene_detect",
    "04_decode_serve",
    "05_upscale",
    "06_interpolate",
    "07_postprocess",
    "08_encode",
})
_ENCODE_ONLY_STAGES = frozenset({"00_probe", "01_plan", "08_encode"})


class BenchmarkRunner:
    def run(
        self,
        request: BenchmarkRequest,
        *,
        cancel_event: threading.Event | None = None,
        on_event: BenchmarkEventCallback | None = None,
    ) -> BenchmarkResult:
        settings = load_settings()
        preset = load_preset(request.preset_id)
        resolved_preset_data = merge_preset_data_for_job(
            preset.model_dump(mode="json"),
            request.preset_overrides,
            settings_decode_hwaccel=settings.hardware.decode_hwaccel,
        )
        resolved_preset_data = _apply_scope_overrides(
            resolved_preset_data,
            scope=request.scope,
        )
        Preset.model_validate(resolved_preset_data)

        run_id = _new_run_id()
        workdir = bench_dir() / run_id
        workdir.mkdir(parents=True, exist_ok=True)
        ffmpeg_log_path = workdir / "ffmpeg.log"
        ffmpeg_log_path.write_text("", encoding="utf-8")

        ramdisk_path = (
            Path(settings.paths.ramdisk_path)
            if settings.paths.ramdisk_path
            else None
        )
        ctx = PipelineContext(
            job_id=run_id,
            source_path=request.source_path,
            workdir=workdir,
            output_path=workdir / "benchmark_output.mkv",
            preset_id=request.preset_id,
            preset_data=resolved_preset_data,
            ramdisk_path=ramdisk_path,
            cancel_event=cancel_event or threading.Event(),
        )
        ctx.extras["pipeline_order"] = settings.pipeline.order
        ctx.extras["prefer_hardware_encoder"] = settings.hardware.prefer_hardware_encoder
        ctx.extras["benchmark"] = request.benchmark_extra()
        ctx.extras["benchmark_encode_samples"] = []
        ctx.extras["benchmark_ffmpeg_log_path"] = str(ffmpeg_log_path)

        events = EventSink()
        events.subscribe(_event_logger(ffmpeg_log_path, on_event))

        stage_names = _FULL_SCOPE_STAGES if request.scope == "full" else _ENCODE_ONLY_STAGES
        all_stages = build_default_stages(order=settings.pipeline.order)
        stages = [stage for stage in all_stages if stage.name in stage_names]
        runner = PipelineRunner(stages)
        runner.run(ctx, events)

        perf_profile_path = workdir / "perf_profile.json"
        perf_profile = _read_json_file(perf_profile_path) if perf_profile_path.is_file() else {}
        encoded_video = _resolve_benchmark_encoded_video(ctx, workdir)
        hw = ctx.extras.get("hardware_profile")
        hw_fp = hw.fingerprint() if hasattr(hw, "fingerprint") else None
        warnings: list[str] = []
        vmaf = None
        if request.compute_vmaf:
            if encoded_video is not None and is_libvmaf_available():
                try:
                    vmaf = compute_vmaf_for_segment(
                        source_path=request.source_path,
                        encoded_path=encoded_video,
                        start_s=request.start_s,
                        duration_s=request.duration_s,
                        log_path=workdir / "vmaf.json",
                    )
                except Exception as exc:
                    warnings.append(f"VMAF failed: {exc}")
            elif encoded_video is None:
                warnings.append("VMAF skipped: encoded segment missing")
            else:
                warnings.append("VMAF skipped: ffmpeg libvmaf filter unavailable")

        return BenchmarkResult(
            run_id=run_id,
            request=request,
            workdir=workdir,
            ffmpeg_log_path=ffmpeg_log_path,
            encoded_video_path=encoded_video,
            perf_profile=perf_profile,
            encode_samples=list(ctx.extras.get("benchmark_encode_samples") or []),
            vmaf=vmaf,
            hardware_fingerprint=hw_fp,
            completed_at=datetime.now(timezone.utc).isoformat(),
            warnings=warnings,
        )


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"bench_{stamp}_{uuid4().hex[:8]}"


def delete_benchmark_run(result: BenchmarkResult) -> None:
    """Remove a benchmark run's on-disk workdir under ``bench_dir()``."""
    run_id = result.run_id
    if not run_id.startswith("bench_"):
        raise ValueError(f"refusing to delete unexpected benchmark run id: {run_id}")

    root = bench_dir().resolve()
    target = result.workdir.resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"refusing to delete benchmark path outside bench dir: {target}")
    if target.name != run_id:
        raise ValueError(
            f"refusing to delete benchmark path whose name does not match run id: {target}",
        )
    if not target.is_dir():
        log.info("benchmark run %s workdir already absent: %s", run_id, target)
        return
    shutil.rmtree(target)
    log.info("deleted benchmark run workdir: %s", target)


def _read_json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_nonempty_video(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _resolve_benchmark_encoded_video(ctx: PipelineContext, workdir: Path) -> Path | None:
    """Locate the encoded benchmark clip after the pipeline finishes.

    Full-scope frame benchmarks run through batched encode, which copies the
    segment to ``batch_segments/`` and clears the RAM-disk ``08_encode`` dir.
    Encode-only runs write the canonical ``08_encode/video.mkv`` instead.
    """
    segments = [
        p for p in ctx.encoded_segments
        if isinstance(p, Path) and _is_nonempty_video(p)
    ]
    if len(segments) == 1:
        return segments[0]
    if len(segments) > 1:
        concat_path = workdir / "batch_segments" / "video_concat.mkv"
        if _is_nonempty_video(concat_path):
            return concat_path

    seg_dir = workdir / "batch_segments"
    if seg_dir.is_dir():
        batch_segments = sorted(
            p for p in seg_dir.glob("segment_*.mkv")
            if _is_nonempty_video(p)
        )
        if len(batch_segments) == 1:
            return batch_segments[0]
        if len(batch_segments) > 1:
            concat_path = seg_dir / "video_concat.mkv"
            if _is_nonempty_video(concat_path):
                return concat_path

    encode_result = ctx.stage_results.get("08_encode")
    if isinstance(encode_result, StageResult):
        artifact = encode_result.artifacts.get("video_only")
        if isinstance(artifact, Path) and _is_nonempty_video(artifact):
            return artifact

    canonical = workdir / "08_encode" / "video.mkv"
    if _is_nonempty_video(canonical):
        return canonical
    return None


def _event_logger(
    ffmpeg_log_path: Path,
    on_event: BenchmarkEventCallback | None,
) -> BenchmarkEventCallback:
    def _callback(event: StageEvent) -> None:
        extra = event.extra if isinstance(event.extra, dict) else {}
        line = extra.get("ffmpeg_line")
        with ffmpeg_log_path.open("a", encoding="utf-8") as fh:
            if isinstance(line, str) and line:
                fh.write(f"{line}\n")
            elif event.message:
                fh.write(f"[{event.stage}][{event.kind}] {event.message}\n")
        if on_event is not None:
            on_event(event)

    return _callback


def _apply_scope_overrides(
    preset_data: dict,
    *,
    scope: str,
) -> dict:
    out = dict(preset_data)
    if scope != "encode_only":
        return out
    out["upscaler"] = {**dict(out.get("upscaler") or {}), "enabled": False, "engine": "none"}
    out["interpolation"] = {
        **dict(out.get("interpolation") or {}),
        "enabled": False,
        "engine": "none",
    }
    out["postprocess"] = {
        **dict(out.get("postprocess") or {}),
        "enabled": False,
        "deband": False,
        "deblock": False,
        "grain_addback": 0,
    }
    out["batching"] = {
        **dict(out.get("batching") or {}),
        "mode": "manual",
        "enabled": False,
    }
    out["frame_dedupe"] = {**dict(out.get("frame_dedupe") or {}), "enabled": False}
    return out
