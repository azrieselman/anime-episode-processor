"""Stage 08: encode.

Two execution modes, selected by the plan:

* ``frames``: encode a directory of numbered PNG/WebP frames (produced by
  stage 04, 05, 06, or 07 depending on which were active) at the plan's
  target fps. This is the default for any preset that touches the video frames.
* ``source``: re-encode the source's primary video stream directly. The
  behavior, retained as a fallback for presets that disable upscaler AND
  interpolation AND postprocess (rare; mostly useful for codec-only conversions
  or HDR sources where the frame path is skipped per ``hdr_policy=skip``).

Both modes produce a video-only intermediate; stage 09 (mux) combines it with
source audio/subs/chapters/attachments. Output extension is always .mkv.

Reads:    ctx.plan, ctx.source_path (source mode) or stage_NN/frames (frames mode)
Writes:   <stage>/video.mkv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aep.adapters.ffmpeg import FFmpegAdapter, decode_hwaccel_has_sw_fallback, raise_if_failed
from aep.encode.encoders import EncodeBuildResult, build_encoder_args, _is_10bit_pix_fmt
from aep.errors import CancelledError, EncodeError, PausedError, PipelineError
from aep.persist.presets import EncoderCfg
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.proc import ProcError, ProcInterrupted, ProcResult, run_capture, run_streaming

log = logging.getLogger(__name__)


def nvenc_relaxed_strategies(
    cfg: EncoderCfg, *, source_is_10bit: bool,
) -> list[tuple[EncoderCfg, str | None]]:
    """Ordered NVENC attempts: preset first, then safer multipass/AQ, then 8-bit.

    Some driver/GPU combinations reject ``-multipass fullres``, temporal AQ,
    or 10-bit Main10 output; later entries relax those knobs while keeping the
    same codec (``hevc_nvenc`` / ``h264_nvenc`` / ``av1_nvenc``).

    The second tuple element, when set, overrides ``source_pix_fmt`` passed to
    ``build_encoder_args`` (e.g. ``\"yuv420p\"`` forces 8-bit output).
    """
    if not cfg.name.endswith("_nvenc"):
        return [(cfg, None)]

    strategies: list[tuple[EncoderCfg, str | None]] = []
    seen: set[tuple[object, ...]] = set()

    def _fingerprint(c: EncoderCfg, pix_override: str | None) -> tuple[object, ...]:
        return (
            c.name,
            c.nvenc_multipass,
            int(c.nvenc_temporal_aq),
            int(c.nvenc_spatial_aq),
            c.nvenc_b_ref_mode,
            c.nvenc_bframes,
            pix_override,
        )

    def push(c: EncoderCfg, pix_override: str | None) -> None:
        key = _fingerprint(c, pix_override)
        if key not in seen:
            seen.add(key)
            strategies.append((c, pix_override))

    push(cfg, None)
    if cfg.nvenc_multipass == "fullres":
        push(cfg.model_copy(update={"nvenc_multipass": "qres"}), None)
    if cfg.nvenc_multipass in ("fullres", "qres"):
        push(cfg.model_copy(update={"nvenc_multipass": "disabled"}), None)
    if cfg.nvenc_temporal_aq:
        push(
            cfg.model_copy(update={"nvenc_multipass": "disabled", "nvenc_temporal_aq": False}),
            None,
        )
    if source_is_10bit and cfg.name in ("hevc_nvenc", "av1_nvenc"):
        push(
            cfg.model_copy(update={"nvenc_multipass": "disabled", "nvenc_temporal_aq": False}),
            "yuv420p",
        )
    return strategies


class EncodeStage(BaseStage):
    name = "08_encode"

    def __init__(self, ffmpeg: FFmpegAdapter | None = None) -> None:
        self._ffmpeg = ffmpeg or FFmpegAdapter()

    # --------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        if not ctx.plan:
            raise PipelineError("08_encode requires 01_plan to have populated ctx.plan")

        encoder_name = str(ctx.plan["encoder"]["final_name"])
        encoder_cfg = ctx.plan["encoder"]["cfg"]
        target = ctx.plan.get("target_geometry", {}) or {}
        media = ctx.media_info
        if media is None:
            raise PipelineError("08_encode requires ctx.media_info")
        primary = media.primary_video

        mode = str(ctx.plan.get("encode_input_mode", "source"))
        params: dict[str, object] = {
            "encoder": encoder_name,
            "encoder_cfg": encoder_cfg,
            "target_w": target.get("width"),
            "target_h": target.get("height"),
            "source_pix_fmt": primary.pix_fmt if primary else None,
            "mode": mode,
            "output_fps": ctx.plan.get("output_fps"),  # "num/den" or None
            "frame_format": ctx.plan.get("decode", {}).get("frame_format", "png"),
            "decode_hwaccel": ctx.plan.get("decode", {}).get("hwaccel", "off"),
        }
        cache_key = compute_cache_key(
            source_fingerprint=str(ctx.source_path),
            stage_name=self.name,
            tool_versions={"ffmpeg": _safe_version(self._ffmpeg)},
            params=params,
        )
        out = ctx.stage_dir(self.name) / "video.mkv"
        # Inputs differ by mode for cache invalidation:
        if mode == "frames":
            in_dir = self._resolve_frame_dir(ctx)
            inputs = [in_dir] if in_dir else [ctx.source_path]
        else:
            inputs = [ctx.source_path]
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            inputs=inputs,
            outputs=[out],
        )

    # --------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        media = ctx.media_info
        if media is None:
            raise PipelineError("08_encode: ctx.media_info missing")
        primary = media.primary_video

        cfg = EncoderCfg.model_validate(ctx.plan["encoder"]["cfg"])
        target_w = plan.params.get("target_w")
        target_h = plan.params.get("target_h")
        if isinstance(target_w, int) and target_w <= 0:
            target_w = None
        if isinstance(target_h, int) and target_h <= 0:
            target_h = None

        out_path: Path = plan.outputs[0]
        mode = str(plan.params.get("mode", "source"))
        decode_hwaccel = str(plan.params.get("decode_hwaccel", "off"))

        # M6.5: in batched mode, both frame and source paths must respect the
        # active batch's PTS window. Frame mode already does this implicitly
        # because s04 only decoded that window's frames; source mode needs the
        # window applied to the encoder ffmpeg invocation directly.
        pts_window = ctx.plan.get("decode", {}).get("pts_window")
        start_pts: float | None = None
        end_pts: float | None = None
        if pts_window:
            try:
                start_pts = float(pts_window[0])
                end_pts = float(pts_window[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise EncodeError(
                    f"08_encode: invalid pts_window {pts_window!r}"
                ) from exc

        tw = target_w if isinstance(target_w, int) else None
        th = target_h if isinstance(target_h, int) else None
        source_pix = primary.pix_fmt if primary else None
        source_is_10bit = _is_10bit_pix_fmt(source_pix)
        strategies = nvenc_relaxed_strategies(cfg, source_is_10bit=source_is_10bit)

        events.emit(StageEvent(
            ctx.job_id, self.name, "started",
            message=f"encoding ({mode}) with {cfg.name} → {out_path.name}",
        ))

        final_build: EncodeBuildResult | None = None
        result: ProcResult | None = None

        for strat_idx, (attempt_cfg, pix_override) in enumerate(strategies):
            eff_pix = pix_override if pix_override is not None else source_pix
            build = build_encoder_args(
                attempt_cfg,
                target_width=tw,
                target_height=th,
                source_pix_fmt=eff_pix,
                fps_mode="passthrough",
            )
            if strat_idx == 0:
                for r in build.rationale:
                    events.emit(StageEvent(ctx.job_id, self.name, "log", message=r))
            gp = list(build.global_prefix) if build.global_prefix else None
            if mode == "frames":
                cmd = self._build_frames_cmd(
                    ctx, plan, out_path, build_args=build.args, global_prefix=gp,
                )
            else:
                cmd = self._ffmpeg.build_passthrough_video_encode(
                    source=ctx.source_path,
                    video_only_out=out_path,
                    encoder_args=build.args,
                    decode_hwaccel=decode_hwaccel,
                    progress=False,
                    allow_overwrite=True,
                    start_pts=start_pts,
                    end_pts=end_pts,
                    global_prefix=gp,
                )

            try:
                stderr_lines: list[str] = []
                for stream, line in run_streaming(
                    cmd,
                    should_interrupt=lambda: "cancel" if ctx.cancel_event.is_set() else (
                        "pause" if ctx.pause_event.is_set() else None
                    ),
                ):
                    if stream == "stderr":
                        stderr_lines.append(line)
                result = ProcResult([str(c) for c in cmd], 0, "", "\n".join(stderr_lines))
            except ProcError as exc:
                if mode == "source" and decode_hwaccel_has_sw_fallback(decode_hwaccel):
                    events.emit(StageEvent(
                        ctx.job_id,
                        self.name,
                        "warning",
                        message=(
                            f"encode: hardware decode ({decode_hwaccel}) failed; "
                            "retrying with software decode"
                        ),
                    ))
                    fallback_cmd = self._ffmpeg.build_passthrough_video_encode(
                        source=ctx.source_path,
                        video_only_out=out_path,
                        encoder_args=build.args,
                        decode_hwaccel="off",
                        progress=False,
                        allow_overwrite=True,
                        start_pts=start_pts,
                        end_pts=end_pts,
                        global_prefix=gp,
                    )
                    result = run_capture(fallback_cmd, timeout=24 * 3600.0, check=False)
                else:
                    result = exc.result
            except ProcInterrupted as exc:
                if exc.reason == "cancel":
                    raise CancelledError("cancelled during encode") from exc
                ctx.extras["pause_checkpoint"] = {"stage": self.name, "status": "interrupted"}
                raise PausedError("paused during encode") from exc

            assert result is not None
            if result.returncode == 0:
                final_build = build
                if strat_idx > 0:
                    events.emit(StageEvent(
                        ctx.job_id,
                        self.name,
                        "warning",
                        message=(
                            "encode: NVENC failed with the preset's quality knobs; "
                            "succeeded after relaxing multipass / AQ / bit depth"
                        ),
                    ))
                break

            if strat_idx < len(strategies) - 1:
                events.emit(StageEvent(
                    ctx.job_id,
                    self.name,
                    "warning",
                    message=(
                        "encode: NVENC failed; retrying with safer encoder settings "
                        f"(stderr tail: {result.stderr[-400:]!r})"
                    ),
                ))
                continue

            raise EncodeError(
                "ffmpeg encode failed",
                context={"stderr": result.stderr[:2000]},
            )

        assert final_build is not None and result is not None
        raise_if_failed(result.returncode, result.stderr)

        if not out_path.exists() or out_path.stat().st_size == 0:
            raise EncodeError(
                "encode produced empty/missing output",
                context={"output": str(out_path), "stderr": result.stderr[-2000:]},
            )

        size_mib = out_path.stat().st_size / (1024 * 1024)
        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=f"encoded video: {size_mib:.1f} MiB",
        ))

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"video_only": out_path},
            metrics={
                "encoder": cfg.name,
                "size_bytes": out_path.stat().st_size,
                "pix_fmt": final_build.pix_fmt,
                "mode": mode,
            },
        )

    # --------------------------------------------------------------- internals

    def _resolve_frame_dir(self, ctx: PipelineContext) -> Path | None:
        """Pick the frames dir the encoder should read from.

        Prefer ``plan.encode_input_source`` when set (01_plan). Otherwise walk
        stage keys in an order consistent with ``pipeline_order`` (legacy plans
        without those keys keep the historical upscale-first walk).
        """
        plan = ctx.plan or {}
        preferred = plan.get("encode_input_source")
        if isinstance(preferred, str) and preferred:
            sub = plan.get(preferred) or {}
            if isinstance(sub, dict):
                d = sub.get("dir")
                if d and Path(str(d)).is_dir():
                    return Path(str(d))

        po = plan.get("pipeline_order")
        if po == "interpolate_first":
            keys = ("postprocess", "upscale", "interpolate", "decode")
        else:
            keys = ("postprocess", "interpolate", "upscale", "decode")

        for key in keys:
            sub = plan.get(key) or {}
            if not isinstance(sub, dict):
                continue
            d = sub.get("dir")
            if d and Path(str(d)).is_dir():
                return Path(str(d))
        return None

    def _build_frames_cmd(
        self,
        ctx: PipelineContext,
        plan: StagePlan,
        out_path: Path,
        *,
        build_args: list[str],
        global_prefix: list[str | Path] | None = None,
    ) -> list[str | Path]:
        in_dir = self._resolve_frame_dir(ctx)
        if not in_dir:
            raise EncodeError("08_encode (frames mode): no frames dir found in plan")
        frame_format = str(plan.params.get("frame_format", "png"))
        manifest = ctx.get_frame_manifest(in_dir, format=frame_format)
        n_frames = manifest["count"]
        if n_frames == 0:
            raise EncodeError(f"08_encode (frames mode): zero {frame_format} frames in {in_dir}")

        # output_fps comes through as "num/den" for fractional rates.
        fps_str = str(ctx.plan.get("output_fps") or "24000/1001")
        if "/" in fps_str:
            num_s, den_s = fps_str.split("/", 1)
            try:
                fps_num, fps_den = int(num_s), int(den_s)
            except ValueError as exc:
                raise EncodeError(f"08_encode: malformed output_fps {fps_str!r}") from exc
        else:
            try:
                fps_num, fps_den = int(round(float(fps_str) * 1000)), 1000
            except ValueError as exc:
                raise EncodeError(f"08_encode: malformed output_fps {fps_str!r}") from exc

        return self._ffmpeg.build_encode_from_frames(
            frame_dir=in_dir,
            frame_format=frame_format,
            fps_num=fps_num,
            fps_den=fps_den,
            video_only_out=out_path,
            encoder_args=build_args,
            allow_overwrite=True,
            progress=False,
            global_prefix=global_prefix,
        )


def _safe_version(adapter: FFmpegAdapter) -> str:
    try:
        return adapter.version
    except Exception:
        return "unknown"
