"""Stage 01: plan.

Produces a frozen JobPlan dict that captures everything later stages need to be
deterministic:

  * the active preset (full dump)
  * the probed MediaInfo summary (selected fields — full info already lives in stage_00)
  * the resolved HardwareProfile fingerprint
  * the encoder recommendation (final encoder name + cfg + rationale + warnings)
  * the stream mapping plan (which audio/subs survive, in which order)
  * the mux tool decision

Reads:    ctx.media_info, ctx.preset_data
Writes:   ctx.plan, ctx.extras["hardware_profile"], <stage>/plan.json

Why freeze a plan instead of recomputing per stage?
  * Reproducibility — the plan is written to disk; a re-run with the same plan
    produces the same output even if hardware changes mid-week.
  * Cache integrity — if the plan changes, downstream cache keys change; if it
    doesn't, stages skip cleanly.
  * Debuggability — one JSON document tells the whole "why" of a job.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from aep.adapters.anime4kcpp import Anime4kcppAdapter
from aep.adapters.anime4kcpp_vs import Anime4kcppVsAdapter
from aep.adapters.ffprobe import FFProbeAdapter
from aep.adapters.realcugan import RealCuganAdapter
from aep.adapters.realesrgan import RealesrganAdapter
from aep.adapters.rife import RifeAdapter
from aep.adapters.waifu2x import Waifu2xAdapter
from aep.bench.hardware import HardwareProfile, probe_hardware
from aep.encode.recommender import recommend
from aep.errors import PipelineError
from aep.mux.mapping import decide_mux_tool, plan_streams
from aep.persist.presets import Preset
from aep.persist.settings import PipelineOrder
from aep.pipeline.batches import BatchSpec, plan_batches
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.fps import derive_target_fps, parse_rational, to_num_den

log = logging.getLogger(__name__)


class PlanStage(BaseStage):
    name = "01_plan"

    def __init__(self, hardware: HardwareProfile | None = None) -> None:
        # Hardware can be injected (tests); otherwise probed at run time.
        self._hardware = hardware

    # --------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        if ctx.media_info is None:
            raise PipelineError("01_plan requires 00_probe to have populated ctx.media_info")
        # Cache key uses media fingerprint + preset id + hardware fingerprint so that
        # changing presets or upgrading the GPU correctly invalidates downstream stages.
        hw = self._get_hardware()
        params: dict[str, object] = {
            "preset_id": ctx.preset_id,
            "hardware_fp": hw.fingerprint(),
            "container": ctx.preset_data.get("container", "mkv"),
        }
        cache_key = compute_cache_key(
            source_fingerprint=ctx.media_info.fmt.filename,
            stage_name=self.name,
            tool_versions={"ffmpeg": hw.ffmpeg_version or "none"},
            params=params,
        )
        out = ctx.stage_dir(self.name) / "plan.json"
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            outputs=[out],
        )

    # --------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        media = ctx.media_info
        if media is None:  # defensive — plan() already validated this
            raise PipelineError("01_plan: ctx.media_info missing")

        preset = Preset.model_validate(ctx.preset_data)
        hw = self._get_hardware()
        ctx.extras["hardware_profile"] = hw

        # 1. encoder recommendation
        primary = media.primary_video
        rec = recommend(
            preset,
            hardware=hw,
            source_codec=primary.codec_name if primary else None,
            source_pix_fmt=primary.pix_fmt if primary else None,
            goal="auto",
        )
        for w in rec.warnings:
            events.emit(StageEvent(ctx.job_id, self.name, "warning", message=w))
        for r in rec.rationale:
            events.emit(StageEvent(ctx.job_id, self.name, "log", message=f"encoder: {r}"))

        # 2. stream mapping plan
        mapping = plan_streams(
            media,
            preset.streams,
            container=preset.container,
        )
        for w in mapping.warnings:
            events.emit(StageEvent(ctx.job_id, self.name, "warning", message=w))

        # 3. mux tool decision
        decision = decide_mux_tool(media, preset.streams, container=preset.container)
        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=f"mux tool: {decision.tool} — {decision.reason}",
        ))

        # 4. resolved target geometry (None = preserve source)
        target_w, target_h = _resolve_target_geometry(preset, media)

        # 5. M3: upscaler / interpolation / decode / output-fps planning
        resolved_decode_hwaccel = _resolve_decode_hwaccel(preset.decode.hwaccel)
        events.emit(StageEvent(
            ctx.job_id,
            self.name,
            "log",
            message=f"decode hwaccel resolved to {resolved_decode_hwaccel}",
        ))
        pipeline_order_raw = ctx.extras.get("pipeline_order")
        pipeline_order: PipelineOrder = (
            pipeline_order_raw
            if pipeline_order_raw in ("interpolate_first", "upscale_first")
            else "interpolate_first"
        )
        m3_plan, m3_warnings, m3_rationale = _plan_m3_video_path(
            preset,
            media,
            primary,
            decode_hwaccel=resolved_decode_hwaccel,
            pipeline_order=pipeline_order,
        )
        for w in m3_warnings:
            events.emit(StageEvent(ctx.job_id, self.name, "warning", message=w))
        for r in m3_rationale:
            events.emit(StageEvent(ctx.job_id, self.name, "log", message=f"video path: {r}"))

        # 6. Estimate peak frame-storage bytes for ramdisk free-space guard.
        # Computed here because the planner is the only stage that knows both the
        # final geometry and the interpolation multiplier; the broker writes the
        # value back onto the context so stages 04-07 can decide ramdisk routing.
        ramdisk_estimate = _estimate_frame_bytes(
            media=media,
            target_w=target_w,
            target_h=target_h,
            m3_plan=m3_plan,
        )
        ctx.ramdisk_estimate_bytes = ramdisk_estimate

        # 6b. Batch plan. We only materialize batches when the preset has
        # batching enabled AND the source duration exceeds one chunk —
        # short clips don't benefit from chunking and would just add overhead.
        # Stage 04+ check `ctx.plan["batches"]` non-empty to decide whether
        # to iterate or run end-to-end. The planner is responsible for
        # producing a contiguous list covering the whole source.
        batches = _plan_video_batches(
            preset=preset,
            media=media,
            primary=primary,
            target_w=target_w,
            target_h=target_h,
            m3_plan=m3_plan,
        )
        for b in batches:
            events.emit(StageEvent(
                ctx.job_id, self.name, "log",
                message=(
                    f"batch {b.index:02d}: {b.start_pts:.3f}…{b.end_pts:.3f}s "
                    f"(≈{b.frame_count_estimate} frames, ~{b.est_bytes // (1024 * 1024)} MiB)"
                ),
            ))

        plan_doc: dict[str, Any] = {
            "preset_id": preset.meta.id,
            "preset": preset.model_dump(mode="json"),
            "container": preset.container,
            "media_summary": {
                "duration_s": media.fmt.duration_s,
                "video_streams": len(media.video_streams),
                "audio_streams": len(media.audio_streams),
                "subtitle_streams": len(media.subtitle_streams),
                "attachments": len(media.attachments),
                "chapters": len(media.chapters),
                "is_matroska": media.is_matroska,
                "primary_video": (
                    primary.model_dump(mode="json") if primary else None
                ),
            },
            "hardware": {
                "fingerprint": hw.fingerprint(),
                "ffmpeg_version": hw.ffmpeg_version,
                "encoders": hw.ffmpeg_encoders,
                "gpu": {
                    "has_nvidia": hw.gpu.has_nvidia,
                    "arch": hw.gpu.arch,
                    "vram_total_mib": hw.gpu.vram_total_mib,
                    "nvenc": {
                        "h264": hw.gpu.nvenc_h264,
                        "hevc": hw.gpu.nvenc_hevc,
                        "av1": hw.gpu.nvenc_av1,
                    },
                },
            },
            "encoder": {
                "final_name": rec.encoder.name,
                "cfg": rec.encoder.model_dump(mode="json"),
                "rationale": rec.rationale,
                "warnings": rec.warnings,
            },
            "stream_mapping": {
                "container": mapping.container,
                "audio": [
                    {
                        "source_index": m.source_index,
                        "out_idx": m.out_idx,
                        "codec_name": m.codec_name,
                        "language": m.language,
                        "title": m.title,
                    }
                    for m in mapping.audio_streams
                ],
                "subtitles": [
                    {
                        "source_index": m.source_index,
                        "out_idx": m.out_idx,
                        "codec_name": m.codec_name,
                        "language": m.language,
                        "title": m.title,
                    }
                    for m in mapping.subtitle_streams
                ],
                "skipped": [{"source_index": i, "reason": r} for i, r in mapping.skipped_streams],
                "copy_chapters": mapping.copy_chapters,
                "rationale": mapping.rationale,
                "warnings": mapping.warnings,
            },
            "mux": asdict(decision),
            "target_geometry": {
                "width": target_w,
                "height": target_h,
                "preserved": target_w is None and target_h is None,
            },
            # M3 sections — each stage reads its own subdict.
            "decode": m3_plan["decode"],
            "upscale": m3_plan["upscale"],
            "interpolate": m3_plan["interpolate"],
            "postprocess": m3_plan["postprocess"],
            "encode_input_mode": m3_plan["encode_input_mode"],
            "output_fps": m3_plan["output_fps"],
            "encode_input_source": m3_plan["encode_input_source"],
            "pipeline_order": m3_plan["pipeline_order"],
            "video_path": {
                "warnings": m3_warnings,
                "rationale": m3_rationale,
            },
            "ramdisk_estimate_bytes": ramdisk_estimate,
            "batches": [b.to_dict() for b in batches],
            "batching": {
                "enabled": preset.batching.enabled,
                "chunk_seconds": preset.batching.chunk_seconds,
                "boundary_policy": preset.batching.boundary_policy,
                "count": len(batches),
            },
        }
        ctx.plan = plan_doc

        out_path: Path = plan.outputs[0]
        out_path.write_text(json.dumps(plan_doc, indent=2, default=str), encoding="utf-8")

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"plan_json": out_path},
            metrics={
                "final_encoder": rec.encoder.name,
                "audio_kept": len(mapping.audio_streams),
                "subs_kept": len(mapping.subtitle_streams),
                "subs_skipped": len(mapping.skipped_streams),
                "mux_tool": decision.tool,
                "warnings": len(rec.warnings) + len(mapping.warnings) + len(m3_warnings),
                "encode_input_mode": m3_plan["encode_input_mode"],
                "output_fps": m3_plan["output_fps"],
                "upscale_active": m3_plan["upscale"]["active"],
                "interpolate_active": m3_plan["interpolate"]["active"],
                "batches": len(batches),
            },
        )

    # --------------------------------------------------------------- helpers

    def _get_hardware(self) -> HardwareProfile:
        if self._hardware is None:
            self._hardware = probe_hardware()
        return self._hardware


# ---------- target resolution helpers --------------------------------------


def _resolve_target_geometry(preset: Preset, media) -> tuple[int | None, int | None]:
    """Return (width, height) the encoder should target, or (None, None) to preserve.

    The upscaler stage owns real geometry once active; the planner honors the preset's
    `target_resolution.named` for the encode-only path.
    """
    tr = preset.target_resolution
    if tr.mode == "scale_only":
        return None, None
    if tr.mode == "explicit":
        return tr.width, tr.height
    # named
    name_to_size = {
        "720p": (1280, 720),
        "1080p": (1920, 1080),
        "1440p": (2560, 1440),
        "2160p": (3840, 2160),
    }
    return name_to_size.get(tr.named or "", (None, None))


# ---------- frame-byte estimator (drives ramdisk free-space guard) ---------


# Bytes per pixel we assume an on-disk PNG/WebP frame occupies in practice.
# Real anime frames vary widely (50%-150% of raw RGB depending on detail),
# but we want a *conservative* number so the ramdisk guard never under-budgets
# and ENOSPCs mid-stage. 4 bytes/px (raw RGBA) is the worst case for ncnn
# tools that also write a small alpha channel; for typical anime PNGs the real
# bytes-per-frame lands at ~1.5-2.5 bytes/px. We keep 4 to stay safe.
_BYTES_PER_PIXEL = 4

def _estimate_frame_bytes(
    *,
    media,
    target_w: int | None,
    target_h: int | None,
    m3_plan: dict[str, Any],
) -> int:
    """Best-effort estimate of peak on-disk frame storage in bytes.

    Used by `PipelineContext.stage_dir()` to decide whether the configured
    ramdisk has enough free space for the planner's byte estimate (no extra
    multiplier — ``_BYTES_PER_PIXEL`` is already conservative).
    A return of 0 means "unknown" — the routing layer treats this as
    "trust the user" and uses the ramdisk if it's writable.

    Computation:
        frames        = duration_s * source_fps   (or media.primary.nb_frames)
        out_frames    = frames * interpolate_multiplier
        out_w, out_h  = target geometry, falling back to source geometry
        peak_bytes    = out_frames * out_w * out_h * _BYTES_PER_PIXEL

    Returns 0 if any required input is missing.
    """
    primary = media.primary_video if hasattr(media, "primary_video") else None
    if primary is None:
        return 0

    # Frame count: prefer probed nb_frames, else duration*fps.
    frames: int | None = primary.nb_frames
    if not frames:
        dur = media.fmt.duration_s if media.fmt else None
        fps = parse_rational(primary.r_frame_rate) or parse_rational(primary.avg_frame_rate)
        if dur and fps and fps > 0:
            frames = int(float(fps) * float(dur))
    if not frames or frames <= 0:
        return 0

    # Output geometry: target if planner committed to one; else source.
    out_w = target_w or primary.width or 0
    out_h = target_h or primary.height or 0
    if out_w <= 0 or out_h <= 0:
        return 0

    # Apply scale factor if upscaler is active and target geometry wasn't pinned.
    if (
        target_w is None
        and target_h is None
        and m3_plan.get("upscale", {}).get("active")
    ):
        scale = m3_plan["upscale"].get("scale") or 1
        try:
            scale = int(scale)
        except (TypeError, ValueError):
            scale = 1
        out_w *= scale
        out_h *= scale

    # Apply interpolation multiplier to frame count.
    multiplier = m3_plan.get("interpolate", {}).get("multiplier") or 1
    try:
        multiplier = int(multiplier)
    except (TypeError, ValueError):
        multiplier = 1
    out_frames = frames * max(1, multiplier)

    bytes_per_frame = out_w * out_h * _BYTES_PER_PIXEL
    peak = int(out_frames * bytes_per_frame)
    return max(0, peak)


# ---------- batch planner glue --------------------------------------------


def _plan_video_batches(
    *,
    preset: Preset,
    media,
    primary,
    target_w: int | None,
    target_h: int | None,
    m3_plan: dict[str, Any],
) -> list[BatchSpec]:
    """Resolve preset.batching + source duration into a concrete batch list.

    Returns an empty list when batching is disabled OR when the source duration
    is unknown / not positive — stages 04+ treat empty-list as "unbatched mode,
    process the whole source in one pass."
    """
    bcfg = preset.batching
    if not bcfg.enabled:
        return []
    duration = media.fmt.duration_s if media.fmt else None
    if not duration or duration <= 0:
        log.warning("batch planner: source duration unknown; falling back to unbatched mode")
        return []

    # Resolve output_fps from the M3 plan: planner already wrote a "num/den"
    # string. Numeric form is what plan_batches() needs for frame counts.
    output_fps = _output_fps_numeric(m3_plan)
    bytes_per_frame = _bytes_per_output_frame(
        primary=primary,
        target_w=target_w,
        target_h=target_h,
        m3_plan=m3_plan,
    )

    # Keyframe list — only probed when the preset asks for keyframe snapping.
    # Probing is fast (~seconds for a 24-min episode) but on a paused queue
    # the user is waiting for the planner to return, so skip it when not needed.
    keyframes: list[float] = []
    if bcfg.boundary_policy == "keyframe":
        try:
            kf_adapter = FFProbeAdapter()
            keyframes = kf_adapter.list_video_keyframes(media.fmt.filename)
            log.info("batch planner: probed %d keyframes", len(keyframes))
        except Exception as exc:
            log.warning(
                "batch planner: keyframe probe failed (%s); using exact boundaries",
                exc,
            )
            keyframes = []

    return plan_batches(
        duration_s=float(duration),
        chunk_seconds=int(bcfg.chunk_seconds),
        boundary_policy=bcfg.boundary_policy,
        keyframes=keyframes if keyframes else None,
        output_fps=output_fps,
        bytes_per_output_frame=bytes_per_frame,
    )


def _output_fps_numeric(m3_plan: dict[str, Any]) -> float | None:
    """Convert plan['output_fps'] ("num/den" or "") to a float, or None."""
    raw = m3_plan.get("output_fps") or ""
    if not raw or "/" not in raw:
        return None
    try:
        n_str, d_str = raw.split("/", 1)
        n = float(n_str)
        d = float(d_str)
        if d <= 0:
            return None
        return n / d
    except ValueError:
        return None


def _bytes_per_output_frame(
    *,
    primary,
    target_w: int | None,
    target_h: int | None,
    m3_plan: dict[str, Any],
) -> int:
    """Per-frame byte budget at output geometry, used by the RAM-disk gate.

    Mirrors the logic in `_estimate_frame_bytes` but for a single frame so
    callers can multiply by per-batch frame counts. 4 bytes/px (worst-case
    RGBA PNG) keeps the gate conservative.
    """
    if primary is None:
        return 0
    out_w = target_w or primary.width or 0
    out_h = target_h or primary.height or 0
    if out_w <= 0 or out_h <= 0:
        return 0
    if (
        target_w is None
        and target_h is None
        and m3_plan.get("upscale", {}).get("active")
    ):
        scale = m3_plan["upscale"].get("scale") or 1
        try:
            scale = int(scale)
        except (TypeError, ValueError):
            scale = 1
        out_w *= scale
        out_h *= scale
    return out_w * out_h * _BYTES_PER_PIXEL


# ---------- M3 video-path planner ------------------------------------------


_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
_TEN_BIT_TOKENS = ("p10", "p012", "p10le", "p10be", "p12", "p012le", "p12le", "p12be")


def _detect_hdr(primary) -> tuple[bool, bool, list[str]]:
    """Return (is_high_bit_depth, is_hdr_transfer, notes).

    is_high_bit_depth: pix_fmt indicates 10/12-bit (yuv420p10le, etc.)
    is_hdr_transfer:   color_transfer is PQ (smpte2084) or HLG (arib-std-b67)
    """
    notes: list[str] = []
    if primary is None:
        return False, False, notes
    pix = (primary.pix_fmt or "").lower()
    high_bit = any(tok in pix for tok in _TEN_BIT_TOKENS)
    transfer = (primary.color_transfer or "").lower()
    is_hdr = transfer in _HDR_TRANSFERS
    if high_bit:
        notes.append(f"source pix_fmt={primary.pix_fmt} (high bit depth)")
    if is_hdr:
        notes.append(f"source color_transfer={transfer} (HDR)")
    return high_bit, is_hdr, notes


def _plan_m3_video_path(
    preset: Preset,
    media,
    primary,
    *,
    decode_hwaccel: str = "off",
    pipeline_order: PipelineOrder = "interpolate_first",
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Build the M3 plan subtree: decode/upscale/interpolate/postprocess + mode + fps.

    Returns (m3_plan_dict, warnings, rationale).

    The mode ("frames" vs "source") drives whether the encode stage takes a frame
    sequence (M3 path) or the original source file (M2 passthrough). We pick
    "frames" if any of upscale/interpolate/postprocess will actually run, or if
    the preset enables time-based batching (chunk decode → intermediate stages
    → encode → concat). Encode-only, non-batched presets keep the source fast path.
    """
    warnings: list[str] = []
    rationale: list[str] = []
    rationale.append(f"pipeline_order={pipeline_order}")

    # ----- source fps --------------------------------------------------
    source_rate: Fraction | None = None
    if primary is not None:
        source_rate = (
            parse_rational(primary.avg_frame_rate)
            or parse_rational(primary.r_frame_rate)
        )
    if source_rate is None:
        rationale.append("source fps unknown; output_fps will mirror multiplier×unknown")

    # ----- HDR / bit-depth detection ----------------------------------
    high_bit, is_hdr, hdr_notes = _detect_hdr(primary)
    rationale.extend(hdr_notes)

    # ----- upscaler decision ------------------------------------------
    up_cfg = preset.upscaler
    upscale_active = bool(up_cfg.enabled) and up_cfg.engine != "none"

    # NCNN binaries are 8-bit sRGB only. Honor hdr_policy.
    if upscale_active and (high_bit or is_hdr):
        if up_cfg.hdr_policy == "skip":
            warnings.append(
                "upscaler disabled: source is high-bit-depth/HDR and hdr_policy=skip"
            )
            rationale.append("upscale.active forced False by hdr_policy=skip")
            upscale_active = False
        else:  # allow_8bit_roundtrip
            warnings.append(
                "upscaler will round-trip through 8-bit BT.709; HDR/wide-gamut info lost"
            )
            rationale.append("upscale proceeding under hdr_policy=allow_8bit_roundtrip")

    # Engine-specific combination validation (model × scale × denoise).
    if upscale_active and up_cfg.engine == "realcugan-ncnn-vulkan":
        for note in RealCuganAdapter.validate_combination(
            up_cfg.model, up_cfg.scale, up_cfg.denoise,
        ):
            warnings.append(f"realcugan: {note}")
    elif upscale_active and up_cfg.engine == "realesrgan-ncnn-vulkan":
        for note in RealesrganAdapter.validate_combination(up_cfg.model, up_cfg.scale):
            warnings.append(f"realesrgan: {note}")
    elif upscale_active and up_cfg.engine == "waifu2x-ncnn-vulkan":
        for note in Waifu2xAdapter.validate_combination(
            up_cfg.model, up_cfg.scale, up_cfg.denoise,
        ):
            warnings.append(f"waifu2x: {note}")
    elif upscale_active and up_cfg.engine == "anime4kcpp":
        for note in Anime4kcppAdapter.validate_combination(
            up_cfg.model, up_cfg.scale, up_cfg.denoise,
        ):
            warnings.append(f"anime4kcpp: {note}")
    elif upscale_active and up_cfg.engine == "anime4kcpp-vs":
        for note in Anime4kcppVsAdapter.validate_combination(
            up_cfg.model, up_cfg.scale, up_cfg.denoise,
        ):
            warnings.append(f"anime4kcpp-vs: {note}")

    # ----- interpolation decision -------------------------------------
    in_cfg = preset.interpolation
    interp_active = bool(in_cfg.enabled) and in_cfg.engine == "rife-ncnn-vulkan"
    if interp_active:
        for note in RifeAdapter.validate_version(in_cfg.version):
            warnings.append(f"rife: {note}")

    effective_rate, multiplier, fps_notes = derive_target_fps(
        source_rate,
        target_fps=in_cfg.target_fps if interp_active else None,
        multiplier=in_cfg.multiplier if interp_active else None,
    )
    for n in fps_notes:
        warnings.append(f"fps: {n}")

    # If interpolation is enabled but the multiplier collapsed to 1, it's a no-op.
    if interp_active and (multiplier or 1) <= 1:
        rationale.append("interpolation requested but multiplier=1; treating as inactive")
        interp_active = False

    # output_fps: the rate the encoder will tag the output with. If interp inactive
    # and source rate known, use source. Otherwise use effective_rate.
    out_rate = effective_rate if interp_active else source_rate
    if out_rate is not None:
        n, d = to_num_den(out_rate)
        output_fps = f"{n}/{d}"
    else:
        output_fps = ""

    # ----- postprocess decision ---------------------------------------
    pp_cfg = preset.postprocess
    pp_active = bool(pp_cfg.enabled) and (
        pp_cfg.deband or pp_cfg.deblock or pp_cfg.grain_addback > 0
    )

    # ----- frames vs source mode --------------------------------------
    batching_requested = bool(preset.batching.enabled)
    encode_input_mode = (
        "frames"
        if (upscale_active or interp_active or pp_active or batching_requested)
        else "source"
    )
    rationale.append(f"encode_input_mode={encode_input_mode}")
    if batching_requested and encode_input_mode == "frames":
        rationale.append("batching: using frame pipeline per chunk (decode → … → encode → concat)")

    # ----- decode plan -------------------------------------------------
    # Only when encode is from frames does decode-serve actually run.
    decode_target_w: int | None = None
    decode_target_h: int | None = None
    if encode_input_mode == "frames" and not upscale_active:
        # No upscaler in the chain — let decode pre-resize to target so the encoder
        # gets the final geometry directly.
        tw, th = _resolve_target_geometry(preset, media)
        if primary is not None and (tw, th) != (primary.width, primary.height):
            decode_target_w, decode_target_h = tw, th

    # Frame format is governed by the upscaler config; even if upscaler is off the
    # downstream stages share a format so cache keys agree.
    frame_format = up_cfg.intermediate_format

    decode_plan = {
        "active": encode_input_mode == "frames",
        "frame_format": frame_format,
        "bt709_normalize": True,
        "target_w": decode_target_w,
        "target_h": decode_target_h,
        "hwaccel": decode_hwaccel,
    }

    if pipeline_order == "interpolate_first":
        interpolate_input_source: str = "decode"
        upscale_input_source: str = "interpolate" if interp_active else "decode"
    else:
        interpolate_input_source = "upscale" if upscale_active else "decode"
        upscale_input_source = "decode"

    if pipeline_order == "interpolate_first":
        if upscale_active:
            postprocess_input_source = "upscale"
        elif interp_active:
            postprocess_input_source = "interpolate"
        else:
            postprocess_input_source = "decode"
    elif interp_active:
        postprocess_input_source = "interpolate"
    elif upscale_active:
        postprocess_input_source = "upscale"
    else:
        postprocess_input_source = "decode"

    if pp_active:
        encode_input_source = "postprocess"
    elif pipeline_order == "interpolate_first":
        encode_input_source = (
            "upscale" if upscale_active
            else "interpolate" if interp_active
            else "decode"
        )
    else:
        encode_input_source = (
            "interpolate" if interp_active
            else "upscale" if upscale_active
            else "decode"
        )

    upscale_plan = {
        "active": upscale_active,
        "engine": up_cfg.engine,
        "model": up_cfg.model,
        "scale": up_cfg.scale,
        "denoise": up_cfg.denoise,
        "tile_size": up_cfg.tile_size,
        "tta": up_cfg.tta,
        "fp16": up_cfg.fp16,
        "frame_format": frame_format,
        "input_source": upscale_input_source,
    }

    # ----- HDR / bit-depth subtree ------------------------------------
    # Records what we detected on the source and the policy decision so that
    # validate stage knows which transfer-characteristic regressions are
    # *expected* (e.g. SMPTE2084 → BT.709 under allow_8bit_roundtrip) vs
    # accidental (e.g. transfer dropped when it should have been preserved).
    if upscale_active and (high_bit or is_hdr) and up_cfg.hdr_policy == "allow_8bit_roundtrip":
        # NCNN frame path is hard-clamped to 8-bit BT.709; encoder will emit 8-bit.
        target_pix_fmt = "yuv420p"
        target_transfer = "bt709"
        roundtripped = True
    else:
        # Either upscale skipped or no HDR/high-bit-depth involvement; the
        # encoder picks pix_fmt from the source's bit depth in encoders.py.
        target_pix_fmt = None  # encoder-derived
        target_transfer = (primary.color_transfer or None) if primary else None
        roundtripped = False

    hdr_plan = {
        "was_10bit": bool(high_bit),
        "was_hdr_transfer": bool(is_hdr),
        "source_pix_fmt": (primary.pix_fmt if primary else None),
        "source_color_transfer": (primary.color_transfer if primary else None),
        "policy": up_cfg.hdr_policy if (high_bit or is_hdr) else "n/a",
        "target_pix_fmt": target_pix_fmt,
        "target_color_transfer": target_transfer,
        "roundtripped_to_8bit": roundtripped,
    }

    interpolate_plan = {
        "active": interp_active,
        "engine": in_cfg.engine,
        "version": in_cfg.version,
        "multiplier": multiplier or 1,
        "duplicate_on_scene_cut": in_cfg.duplicate_on_scene_cut,
        "fp16": in_cfg.fp16,
        "frame_format": frame_format,
        # Which earlier stage's frames RIFE consumes. The stage resolves the path
        # itself; this is recorded for plan diffability and debug.
        "input_source": interpolate_input_source,
    }

    postprocess_plan = {
        "enabled": pp_active,
        "deband": pp_cfg.deband,
        "deblock": pp_cfg.deblock,
        "grain_addback": pp_cfg.grain_addback,
        "input_source": postprocess_input_source,
        "frame_format": frame_format,
    }

    return (
        {
            "decode": decode_plan,
            "upscale": upscale_plan,
            "interpolate": interpolate_plan,
            "postprocess": postprocess_plan,
            "hdr": hdr_plan,
            "encode_input_mode": encode_input_mode,
            "output_fps": output_fps,
            "encode_input_source": encode_input_source,
            "pipeline_order": pipeline_order,
        },
        warnings,
        rationale,
    )


def _resolve_decode_hwaccel(mode: str) -> str:
    m = (mode or "auto").lower()
    if m == "off":
        return "off"
    if m == "d3d11va":
        return "d3d11va"
    return "d3d11va" if os.name == "nt" else "off"
