"""Stage 04: decode_serve.

Decodes the source's primary video stream to a directory of numbered frames
that the upscale stage (05) can feed to NCNN-Vulkan binaries directly.

Behavior is gated by the active plan:

* If the plan says ``encode_input_mode == "source"`` (no upscaler, no RIFE),
  this stage is a no-op. Stage 08 will read from the original source.
* Otherwise we decode every frame to ``<stage>/frames/<N>.<format>`` using
  the preset's ``intermediate_format`` (PNG default, WebP-lossless option).

Color handling: NCNN-Vulkan binaries are 8-bit sRGB. We force a colorspace
normalization to BT.709 limited 8-bit RGB unless the plan says we should
skip the upscaler entirely (HDR + ``hdr_policy=skip``). The plan stage owns
that decision; stage 04 just executes.

Reads:    ctx.source_path, ctx.plan
Writes:   ctx.plan["decode"] = {format, count, dir}, <stage>/frames/*.{png|webp}
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from aep.adapters.ffmpeg import FFmpegAdapter, raise_if_failed
from aep.errors import CancelledError, PausedError, PipelineError, StageError
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.proc import ProcError, ProcInterrupted, ProcResult, run_capture, run_streaming

log = logging.getLogger(__name__)


class DecodeServeStage(BaseStage):
    name = "04_decode_serve"

    def __init__(self, ffmpeg: FFmpegAdapter | None = None) -> None:
        self._ffmpeg = ffmpeg or FFmpegAdapter()

    # --------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        if not ctx.plan:
            raise PipelineError(f"{self.name} requires 01_plan to have populated ctx.plan")
        decode_cfg = ctx.plan.get("decode", {}) or {}
        active = bool(decode_cfg.get("active", False))
        frame_format = str(decode_cfg.get("frame_format", "png"))
        bt709 = bool(decode_cfg.get("bt709_normalize", True))
        hdr = ctx.plan.get("hdr") or {}
        pts_window = decode_cfg.get("pts_window")
        params: dict[str, object] = {
            "active": active,
            "frame_format": frame_format,
            "bt709_normalize": bt709,
            "decode_hwaccel": str(decode_cfg.get("hwaccel", "off")),
            "encode_input_mode": ctx.plan.get("encode_input_mode", "source"),
            # Batching / zscale path must bust the cache when they change.
            "pts_window": list(pts_window)
            if isinstance(pts_window, (list, tuple))
            else pts_window,
            "use_zscale": bool(hdr.get("was_10bit") or hdr.get("was_hdr_transfer")),
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
            inputs=[ctx.source_path],
            outputs=[out_dir],
        )

    # --------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        active = bool(plan.params.get("active", False))
        out_dir: Path = plan.outputs[0]
        frame_format = str(plan.params.get("frame_format", "png"))
        decode_hwaccel = str(plan.params.get("decode_hwaccel", "off"))

        if not active:
            events.emit(StageEvent(
                ctx.job_id, self.name, "log",
                message="decode_serve skipped (encode_input_mode=source)",
            ))
            return StageResult(
                stage_name=self.name,
                success=True,
                duration_s=time.monotonic() - t0,
                metrics={"skipped": True},
            )

        # Wipe and recreate output dir so a partial prior run doesn't leak.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # We honor the plan's target geometry only for the upscale-disabled
        # branch. When upscaling is active, decode at SOURCE resolution and let
        # the upscaler+downscale path determine the final pixel size. The plan
        # records the right values in ``decode.target_w/h`` (None when source-
        # res decoding is required).
        decode_cfg = ctx.plan.get("decode", {})
        tgt_w = decode_cfg.get("target_w")
        tgt_h = decode_cfg.get("target_h")
        if isinstance(tgt_w, int) and tgt_w <= 0:
            tgt_w = None
        if isinstance(tgt_h, int) and tgt_h <= 0:
            tgt_h = None

        # M6.5: when the runner is iterating batches, it sets a (start, end) PTS
        # window so this stage decodes only the current chunk. Single-pass jobs
        # leave it unset and we decode the full source.
        pts_window = decode_cfg.get("pts_window")
        start_pts: float | None = None
        end_pts: float | None = None
        if pts_window:
            try:
                start_pts = float(pts_window[0])
                end_pts = float(pts_window[1])
            except (TypeError, ValueError, IndexError) as exc:
                raise StageError(
                    f"decode_serve: invalid pts_window {pts_window!r}",
                ) from exc

        hdr = ctx.plan.get("hdr") or {}
        use_zscale = bool(hdr.get("was_10bit") or hdr.get("was_hdr_transfer"))

        cmd = self._ffmpeg.build_decode_to_frames(
            source=ctx.source_path,
            out_dir=out_dir,
            frame_format=frame_format,
            target_width=tgt_w if isinstance(tgt_w, int) else None,
            target_height=tgt_h if isinstance(tgt_h, int) else None,
            bt709_normalize=bool(plan.params.get("bt709_normalize", True)),
            use_zscale=use_zscale,
            decode_hwaccel=decode_hwaccel,
            start_pts=start_pts,
            end_pts=end_pts,
        )
        events.emit(StageEvent(
            ctx.job_id, self.name, "started",
            message=f"decoding to {frame_format} frames in {out_dir.name}",
        ))

        result, used_fallback = self._run_decode_with_hwaccel_fallback(
            ctx=ctx,
            events=events,
            primary_cmd=cmd,
            decode_hwaccel=decode_hwaccel,
            out_dir=out_dir,
            frame_format=frame_format,
            tgt_w=tgt_w,
            tgt_h=tgt_h,
            bt709_normalize=bool(plan.params.get("bt709_normalize", True)),
            use_zscale=use_zscale,
            start_pts=start_pts,
            end_pts=end_pts,
        )
        if result.returncode != 0:
            log.error(
                "decode_serve: ffmpeg failed (use_zscale=%s, fallback=%s): %s",
                use_zscale,
                used_fallback,
                (result.stderr or "")[-4000:],
            )
        raise_if_failed(result.returncode, result.stderr)

        out_manifest = ctx.get_frame_manifest(out_dir, format=frame_format)
        n = out_manifest["count"]
        if n == 0:
            raise StageError(
                "decode_serve produced zero frames",
                context={"out_dir": str(out_dir), "stderr": result.stderr[-2000:]},
            )

        # Persist accounting back into ctx.plan for downstream stages.
        ctx.plan.setdefault("decode", {})
        ctx.plan["decode"]["count"] = n
        ctx.plan["decode"]["dir"] = str(out_dir)
        ctx.plan["decode"]["frame_format"] = frame_format

        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=f"decoded {n} frames",
        ))
        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"frames_dir": out_dir},
            metrics={"frames": n, "format": frame_format, "output_bytes": out_manifest["bytes"]},
        )

    def rollback(self, ctx: PipelineContext, plan: StagePlan) -> None:
        # Frames are content-addressed via the cache; deleting them is safe but
        # we leave them in place so a re-run can hit the cache.
        return None

    # ----------------------------------------------------- hwaccel fallback

    def _run_decode_with_hwaccel_fallback(
        self,
        *,
        ctx: PipelineContext,
        events: EventSink,
        primary_cmd: list,
        decode_hwaccel: str,
        out_dir: Path,
        frame_format: str,
        tgt_w: int | None,
        tgt_h: int | None,
        bt709_normalize: bool,
        use_zscale: bool,
        start_pts: float | None,
        end_pts: float | None,
    ) -> tuple[ProcResult, bool]:
        """Run the primary decode command; on D3D11VA failure, retry without hwaccel.

        Returns ``(result, used_fallback)``. ProcInterrupted is propagated so
        cancel/pause keeps working; non-d3d11va failures bubble up unchanged.
        """
        try:
            result = _run_capture_via_streaming(primary_cmd, ctx)
            return result, False
        except ProcError as exc:
            if decode_hwaccel != "d3d11va":
                raise StageError(
                    "decode_serve: ffmpeg invocation failed",
                    context={"stderr": exc.result.stderr[:2000]},
                ) from exc
            events.emit(StageEvent(
                ctx.job_id,
                self.name,
                "warning",
                message="decode_serve: D3D11VA decode failed; retrying with software decode",
            ))
        except ProcInterrupted as exc:
            if exc.reason == "cancel":
                raise CancelledError("cancelled during decode") from exc
            ctx.extras["pause_checkpoint"] = {"stage": self.name, "status": "interrupted"}
            raise PausedError("paused during decode") from exc

        fallback_cmd = self._ffmpeg.build_decode_to_frames(
            source=ctx.source_path,
            out_dir=out_dir,
            frame_format=frame_format,
            target_width=tgt_w if isinstance(tgt_w, int) else None,
            target_height=tgt_h if isinstance(tgt_h, int) else None,
            bt709_normalize=bt709_normalize,
            use_zscale=use_zscale,
            decode_hwaccel="off",
            start_pts=start_pts,
            end_pts=end_pts,
        )
        result = run_capture(fallback_cmd, timeout=24 * 3600.0, check=False)
        return result, True


def _run_capture_via_streaming(cmd: list, ctx: PipelineContext) -> ProcResult:
    """Drain ``run_streaming`` and return a ProcResult for a successful run.

    Raises ``ProcError`` on non-zero exit (preserving the wrapped result) and
    ``ProcInterrupted`` on cancel/pause.
    """
    stderr_lines: list[str] = []
    for stream, line in run_streaming(
        cmd,
        should_interrupt=lambda: "cancel" if ctx.cancel_event.is_set() else (
            "pause" if ctx.pause_event.is_set() else None
        ),
    ):
        if stream == "stderr":
            stderr_lines.append(line)
    return ProcResult([str(c) for c in cmd], 0, "", "\n".join(stderr_lines))


def _safe_version(adapter: FFmpegAdapter) -> str:
    try:
        return adapter.version
    except Exception:
        return "unknown"
