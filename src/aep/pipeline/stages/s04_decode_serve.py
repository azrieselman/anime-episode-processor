"""Stage 04: decode_serve.

Decodes the source's primary video stream to a directory of numbered frames
that the upscale stage (05) can feed to NCNN-Vulkan binaries directly.

Behavior is gated by the active plan:

* If the plan says ``encode_input_mode == "source"`` (no upscaler, no RIFE),
  this stage is a no-op. Stage 08 will read from the original source.
* Otherwise we decode every frame to ``<stage>/frames/<N>.<format>`` using
  the preset's ``intermediate_format`` and ``decode.png_intermediate_codec``
  when the extension is png (MJPEG-in-.png vs true PNG).

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

from aep.adapters.ffmpeg import (
    FFmpegAdapter,
    decode_hwaccel_fallback_chain,
    decode_hwaccel_uses_hardware_decode,
    raise_if_failed,
)
from aep.errors import CancelledError, PausedError, PipelineError, StageError
from aep.pipeline.batch_timing import (
    decode_time_pad_s,
    merge_batch_frame_plan_into_decode,
    reconcile_batch_decode_outputs,
    resolve_batch_frame_plan,
)
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent, emit_tool_log
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.frame_dedupe import (
    SCENE_SCORE_META_BASENAME,
    compact_decode_directory,
    decode_batch_frame_offset,
    load_scene_score_scan_results,
    merge_dedupe_state_into_plan,
    scene_cut_protect_indices,
    skip_indices_from_scores,
    write_dedupe_map,
)
from aep.util.proc import ProcError, ProcInterrupted, ProcResult, run_capture, run_streaming

log = logging.getLogger(__name__)


def _normalize_png_intermediate_codec(raw: object) -> str:
    if raw is None or raw == "":
        return "mjpeg"
    s = str(raw).lower().strip()
    return s if s in ("mjpeg", "libpng") else "mjpeg"


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
            "png_intermediate_codec": _normalize_png_intermediate_codec(
                decode_cfg.get("png_intermediate_codec"),
            ),
            "bt709_normalize": bt709,
            "decode_hwaccel": str(decode_cfg.get("hwaccel", "off")),
            "encode_input_mode": ctx.plan.get("encode_input_mode", "source"),
            # Batching / zscale path must bust the cache when they change.
            "pts_window": list(pts_window)
            if isinstance(pts_window, (list, tuple))
            else pts_window,
            "use_zscale": bool(hdr.get("was_10bit") or hdr.get("was_hdr_transfer")),
        }
        fd = ctx.plan.get("frame_dedupe") or {}
        params["frame_dedupe"] = {
            "active": bool(fd.get("active")),
            "threshold": fd.get("threshold", 0.02),
            "protect_scene_cuts": fd.get("protect_scene_cuts", True),
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
        png_intermediate_codec = _normalize_png_intermediate_codec(
            plan.params.get("png_intermediate_codec"),
        )

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

        batch_frame_plan = resolve_batch_frame_plan(ctx)
        decode_start_pts = start_pts
        decode_end_pts = end_pts
        decode_time_pad = 0.0
        if batch_frame_plan is not None:
            merge_batch_frame_plan_into_decode(ctx, batch_frame_plan)
            decode_start_pts = batch_frame_plan.decode_start_pts
            decode_end_pts = batch_frame_plan.end_pts
            decode_time_pad = decode_time_pad_s(batch_frame_plan.source_fps)

        hdr = ctx.plan.get("hdr") or {}
        use_zscale = bool(hdr.get("was_10bit") or hdr.get("was_hdr_transfer"))
        bt709_normalize = bool(plan.params.get("bt709_normalize", True))
        decode_loglevel = "verbose" if _benchmark_verbose_enabled(ctx) else "error"

        fd_plan = ctx.plan.get("frame_dedupe") or {}
        fd_active = bool(fd_plan.get("active"))
        precomputed_scores: dict[int, float] | None = None
        dedupe_decode_hw_fallback = False

        events.emit(StageEvent(
            ctx.job_id, self.name, "started",
            message=f"decoding to {frame_format} frames in {out_dir.name}",
        ))

        if fd_active:
            scan_meta = out_dir.parent / SCENE_SCORE_META_BASENAME
            try:
                if scan_meta.exists():
                    scan_meta.unlink()
            except OSError:
                pass
            scan_cwd = out_dir.parent
            # Fused decode+scene scores shares one filter graph with the frame encoder branch.
            # With NVDEC/D3D11VA that tends to serialize extra work on the decode path and can
            # hurt throughput; keep fusion for software decode only and run a separate scan pass
            # after hardware decode (same as the former fused-failure fallback).
            if decode_hwaccel_uses_hardware_decode(decode_hwaccel):
                events.emit(StageEvent(
                    ctx.job_id, self.name, "log",
                    message=(
                        "decode_serve: frame dedupe — hardware decode: separate decode "
                        "and scene score scan (fusion disabled)"
                    ),
                ))
                cmd = self._ffmpeg.build_decode_to_frames(
                    source=ctx.source_path,
                    out_dir=out_dir,
                    frame_format=frame_format,
                    png_intermediate_codec=png_intermediate_codec,
                    target_width=tgt_w if isinstance(tgt_w, int) else None,
                    target_height=tgt_h if isinstance(tgt_h, int) else None,
                    bt709_normalize=bt709_normalize,
                    use_zscale=use_zscale,
                    decode_hwaccel=decode_hwaccel,
                    start_pts=decode_start_pts,
                    end_pts=decode_end_pts,
                    time_pad_s=decode_time_pad,
                    loglevel=decode_loglevel,
                )
                result, used_fallback = self._run_decode_with_hwaccel_fallback(
                    ctx=ctx,
                    events=events,
                    primary_cmd=cmd,
                    decode_hwaccel=decode_hwaccel,
                    out_dir=out_dir,
                    frame_format=frame_format,
                    png_intermediate_codec=png_intermediate_codec,
                    tgt_w=tgt_w,
                    tgt_h=tgt_h,
                    bt709_normalize=bt709_normalize,
                    use_zscale=use_zscale,
                    start_pts=decode_start_pts,
                    end_pts=decode_end_pts,
                    time_pad_s=decode_time_pad,
                    loglevel=decode_loglevel,
                )
            else:
                fused_cmd = self._ffmpeg.build_decode_to_frames_with_scene_metadata_fused(
                    source=ctx.source_path,
                    out_dir=out_dir,
                    metadata_out=scan_meta,
                    frame_format=frame_format,
                    png_intermediate_codec=png_intermediate_codec,
                    target_width=tgt_w if isinstance(tgt_w, int) else None,
                    target_height=tgt_h if isinstance(tgt_h, int) else None,
                    bt709_normalize=bt709_normalize,
                    use_zscale=use_zscale,
                    decode_hwaccel=decode_hwaccel,
                    start_pts=decode_start_pts,
                    end_pts=decode_end_pts,
                    time_pad_s=decode_time_pad,
                    loglevel=decode_loglevel,
                )
                events.emit(StageEvent(
                    ctx.job_id, self.name, "log",
                    message="decode_serve: frame dedupe — trying single-pass decode+scene metadata",
                ))
                fused_res, fused_hw_fallback = self._run_fused_decode_with_hwaccel_fallback(
                    ctx=ctx,
                    events=events,
                    primary_cmd=fused_cmd,
                    decode_hwaccel=decode_hwaccel,
                    out_dir=out_dir,
                    scan_cwd=scan_cwd,
                    scan_meta=scan_meta,
                    frame_format=frame_format,
                    png_intermediate_codec=png_intermediate_codec,
                    tgt_w=tgt_w,
                    tgt_h=tgt_h,
                    bt709_normalize=bt709_normalize,
                    use_zscale=use_zscale,
                    start_pts=decode_start_pts,
                    end_pts=decode_end_pts,
                    time_pad_s=decode_time_pad,
                    loglevel=decode_loglevel,
                )
                out_manifest_try = ctx.get_frame_manifest(out_dir, format=frame_format)
                n_try = int(out_manifest_try["count"])
                fused_ok = fused_res.returncode == 0 and n_try > 0
                if fused_ok:
                    precomputed_scores = load_scene_score_scan_results(
                        meta_path=scan_meta,
                        stderr=fused_res.stderr or "",
                    )
                    try:
                        if scan_meta.exists():
                            scan_meta.unlink()
                    except OSError:
                        pass
                    result, used_fallback = fused_res, fused_hw_fallback
                    dedupe_decode_hw_fallback = fused_hw_fallback
                else:
                    events.emit(StageEvent(
                        ctx.job_id, self.name, "warning",
                        message=(
                            "decode_serve: fused decode+scene metadata failed or produced zero frames "
                            f"(rc={fused_res.returncode}); falling back to decode then scan"
                        ),
                    ))
                    log.warning(
                        "decode_serve: fused decode failed (rc=%s, frames=%s); stderr_tail=%s",
                        fused_res.returncode,
                        n_try,
                        (fused_res.stderr or "")[-2000:],
                    )
                    try:
                        shutil.rmtree(out_dir)
                    except OSError:
                        pass
                    out_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        if scan_meta.exists():
                            scan_meta.unlink()
                    except OSError:
                        pass
                    cmd = self._ffmpeg.build_decode_to_frames(
                        source=ctx.source_path,
                        out_dir=out_dir,
                        frame_format=frame_format,
                        png_intermediate_codec=png_intermediate_codec,
                        target_width=tgt_w if isinstance(tgt_w, int) else None,
                        target_height=tgt_h if isinstance(tgt_h, int) else None,
                        bt709_normalize=bt709_normalize,
                        use_zscale=use_zscale,
                        decode_hwaccel=decode_hwaccel,
                        start_pts=decode_start_pts,
                        end_pts=decode_end_pts,
                        time_pad_s=decode_time_pad,
                        loglevel=decode_loglevel,
                    )
                    result, used_fallback = self._run_decode_with_hwaccel_fallback(
                        ctx=ctx,
                        events=events,
                        primary_cmd=cmd,
                        decode_hwaccel=decode_hwaccel,
                        out_dir=out_dir,
                        frame_format=frame_format,
                        png_intermediate_codec=png_intermediate_codec,
                        tgt_w=tgt_w,
                        tgt_h=tgt_h,
                        bt709_normalize=bt709_normalize,
                        use_zscale=use_zscale,
                        start_pts=decode_start_pts,
                        end_pts=decode_end_pts,
                        time_pad_s=decode_time_pad,
                        loglevel=decode_loglevel,
                    )
        else:
            cmd = self._ffmpeg.build_decode_to_frames(
                source=ctx.source_path,
                out_dir=out_dir,
                frame_format=frame_format,
                png_intermediate_codec=png_intermediate_codec,
                target_width=tgt_w if isinstance(tgt_w, int) else None,
                target_height=tgt_h if isinstance(tgt_h, int) else None,
                bt709_normalize=bt709_normalize,
                use_zscale=use_zscale,
                decode_hwaccel=decode_hwaccel,
                start_pts=decode_start_pts,
                end_pts=decode_end_pts,
                time_pad_s=decode_time_pad,
                loglevel=decode_loglevel,
            )
            result, used_fallback = self._run_decode_with_hwaccel_fallback(
                ctx=ctx,
                events=events,
                primary_cmd=cmd,
                decode_hwaccel=decode_hwaccel,
                out_dir=out_dir,
                frame_format=frame_format,
                png_intermediate_codec=png_intermediate_codec,
                tgt_w=tgt_w,
                tgt_h=tgt_h,
                bt709_normalize=bt709_normalize,
                use_zscale=use_zscale,
                start_pts=decode_start_pts,
                end_pts=decode_end_pts,
                time_pad_s=decode_time_pad,
                loglevel=decode_loglevel,
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

        trim_removed = 0
        batch_reconciled_shortfall = False
        if batch_frame_plan is not None:
            n, trim_removed, batch_frame_plan, batch_reconciled_shortfall = (
                reconcile_batch_decode_outputs(
                    ctx,
                    out_dir=out_dir,
                    frame_format=frame_format,
                    plan=batch_frame_plan,
                )
            )
            if trim_removed:
                events.emit(StageEvent(
                    ctx.job_id, self.name, "log",
                    message=(
                        f"trimmed {trim_removed} excess decode frame(s) "
                        f"(target {batch_frame_plan.expected_decode_frames} for "
                        f"[{batch_frame_plan.start_pts:.3f}s, "
                        f"{batch_frame_plan.end_pts:.3f}s))"
                    ),
                ))
            elif batch_reconciled_shortfall:
                events.emit(StageEvent(
                    ctx.job_id, self.name, "warning",
                    message=(
                        f"decode_serve: reconciled batch frame plan to {n} frames "
                        f"(window [{batch_frame_plan.start_pts:.3f}s, "
                        f"{batch_frame_plan.end_pts:.3f}s))"
                    ),
                ))

        # Persist accounting back into ctx.plan for downstream stages.
        ctx.plan.setdefault("decode", {})
        ctx.plan["decode"]["count"] = n
        ctx.plan["decode"]["dir"] = str(out_dir)
        ctx.plan["decode"]["frame_format"] = frame_format
        ctx.plan["decode"]["png_intermediate_codec"] = png_intermediate_codec

        dedupe_metrics: dict[str, int | float] = {}
        n, dedupe_metrics = self._apply_frame_dedupe_if_needed(
            ctx=ctx,
            events=events,
            out_dir=out_dir,
            frame_format=frame_format,
            decode_hwaccel=decode_hwaccel,
            use_zscale=use_zscale,
            start_pts=decode_start_pts,
            end_pts=decode_end_pts,
            full_count=n,
            precomputed_scores=precomputed_scores,
            decode_hwaccel_fallback_for_scores=dedupe_decode_hw_fallback,
        )
        ctx.plan["decode"]["count"] = n
        out_manifest = ctx.get_frame_manifest(out_dir, format=frame_format)

        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=f"decoded {n} frames",
        ))
        metrics: dict[str, object] = {
            "frames": n,
            "format": frame_format,
            "png_intermediate_codec": png_intermediate_codec,
            "output_bytes": out_manifest["bytes"],
            "batch_trim_removed": trim_removed,
        }
        if batch_frame_plan is not None:
            metrics["batch_expected_content_frames"] = batch_frame_plan.expected_content_frames
            metrics["batch_overlap_source_frames"] = batch_frame_plan.overlap_source_frames
        metrics.update(dedupe_metrics)
        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"frames_dir": out_dir},
            metrics=metrics,
        )

    def _apply_frame_dedupe_if_needed(
        self,
        ctx: PipelineContext,
        events: EventSink,
        *,
        out_dir: Path,
        frame_format: str,
        decode_hwaccel: str,
        use_zscale: bool,
        start_pts: float | None,
        end_pts: float | None,
        full_count: int,
        precomputed_scores: dict[int, float] | None = None,
        decode_hwaccel_fallback_for_scores: bool = False,
    ) -> tuple[int, dict[str, int | float]]:
        fd = ctx.plan.get("frame_dedupe") or {}
        if not fd.get("active"):
            return full_count, {}
        ctx.check_cancel()

        if precomputed_scores is not None:
            events.emit(StageEvent(
                ctx.job_id, self.name, "log",
                message="frame dedupe: using scene scores from fused decode (no second pass)",
            ))
            scores = precomputed_scores
            metrics: dict[str, int | float] = {
                "dedupe_scan_returncode": 0,
                "dedupe_scores_parsed": len(scores),
                "dedupe_fused_single_pass": 1,
                "dedupe_hwaccel_fallback": int(decode_hwaccel_fallback_for_scores),
            }
            used_fallback = decode_hwaccel_fallback_for_scores
        else:
            events.emit(StageEvent(
                ctx.job_id, self.name, "log",
                message="frame dedupe: scanning scene scores (ffmpeg)",
            ))
            scan_meta = out_dir.parent / SCENE_SCORE_META_BASENAME
            try:
                if scan_meta.exists():
                    scan_meta.unlink()
            except OSError:
                pass
            scan_cwd = out_dir.parent
            scan_cmd = self._ffmpeg.build_scene_score_scan(
                source=ctx.source_path,
                metadata_out=scan_meta,
                decode_hwaccel=decode_hwaccel,
                use_zscale=use_zscale,
                start_pts=start_pts,
                end_pts=end_pts,
            )
            scan_res = run_capture(scan_cmd, cwd=scan_cwd, timeout=24 * 3600.0, check=False)
            used_fallback = False
            fallback_modes = decode_hwaccel_fallback_chain(decode_hwaccel)
            if scan_res.returncode != 0 and fallback_modes:
                used_fallback = True
                for fb_mode in fallback_modes:
                    events.emit(StageEvent(
                        ctx.job_id, self.name, "warning",
                        message=(
                            f"frame dedupe: hardware decode ({decode_hwaccel}) scan failed; "
                            f"retrying with {fb_mode} decode"
                        ),
                    ))
                    try:
                        if scan_meta.exists():
                            scan_meta.unlink()
                    except OSError:
                        pass
                    scan_cmd = self._ffmpeg.build_scene_score_scan(
                        source=ctx.source_path,
                        metadata_out=scan_meta,
                        decode_hwaccel=fb_mode,
                        use_zscale=use_zscale,
                        start_pts=start_pts,
                        end_pts=end_pts,
                    )
                    scan_res = run_capture(scan_cmd, cwd=scan_cwd, timeout=24 * 3600.0, check=False)
                    if scan_res.returncode == 0:
                        break
            scores = load_scene_score_scan_results(
                meta_path=scan_meta,
                stderr=scan_res.stderr or "",
            )
            try:
                if scan_meta.exists():
                    scan_meta.unlink()
            except OSError:
                pass
            metrics = {
                "dedupe_scan_returncode": scan_res.returncode,
                "dedupe_scores_parsed": len(scores),
            }
            if scan_res.returncode != 0:
                log.warning(
                    "frame dedupe: scene scan ffmpeg rc=%s stderr_tail=%s",
                    scan_res.returncode,
                    (scan_res.stderr or "")[-800:],
                )

        if not scores:
            events.emit(StageEvent(
                ctx.job_id, self.name, "warning",
                message="frame dedupe: no scene scores parsed; skipping compaction",
            ))
            self._write_identity_dedupe_map(
                ctx, out_dir, full_count, frame_format, fd, used_fallback,
            )
            return full_count, metrics

        threshold = float(fd.get("threshold", 0.02))
        protect = bool(fd.get("protect_scene_cuts", True))
        protected: set[int] = set()
        if protect:
            protected = scene_cut_protect_indices(
                ctx.scene_cuts,
                batch_offset=decode_batch_frame_offset(ctx),
                full_count=full_count,
            )
        skip = skip_indices_from_scores(
            scores,
            full_count=full_count,
            threshold=threshold,
            protected=protected,
        )
        # Guard: bogus parse / threshold combo can mark almost everything duplicate.
        if full_count > 2 and len(skip) >= full_count - 1:
            events.emit(StageEvent(
                ctx.job_id, self.name, "warning",
                message=(
                    "frame dedupe: would skip all but one frame (likely score parse mismatch "
                    f"or threshold={threshold} too strict for this source); disabling compaction"
                ),
            ))
            self._write_identity_dedupe_map(
                ctx, out_dir, full_count, frame_format, fd, used_fallback,
            )
            return full_count, metrics
        kept_order = [i for i in range(1, full_count + 1) if i not in skip]
        metrics["dedupe_skipped"] = len(skip)
        metrics["dedupe_hwaccel_fallback"] = int(used_fallback)
        if not skip:
            events.emit(StageEvent(
                ctx.job_id, self.name, "log",
                message="frame dedupe: no duplicate frames under threshold",
            ))
            self._write_identity_dedupe_map(
                ctx, out_dir, full_count, frame_format, fd, used_fallback,
            )
            return full_count, metrics

        try:
            kept_order, _ded_dir = compact_decode_directory(
                frames_dir=out_dir,
                full_count=full_count,
                skip=skip,
                frame_format=frame_format,
            )
        except (OSError, ValueError) as exc:
            raise StageError(
                f"frame dedupe: compaction failed: {exc}",
                context={"out_dir": str(out_dir)},
            ) from exc

        compact_count = len(kept_order)
        doc = {
            "full_decode_count": full_count,
            "compact_decode_count": compact_count,
            "kept_order": kept_order,
            "skipped": sorted(skip),
            "threshold": threshold,
            "protect_scene_cuts": protect,
            "frame_format": frame_format,
            "scan_hwaccel_fallback": used_fallback,
        }
        merge_dedupe_state_into_plan(ctx, doc)
        write_dedupe_map(ctx.stage_dir(self.name), doc)
        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=(
                f"frame dedupe: compacted {full_count} → {compact_count} frames "
                f"(threshold={threshold})"
            ),
        ))
        return compact_count, metrics

    def _write_identity_dedupe_map(
        self,
        ctx: PipelineContext,
        _out_dir: Path,
        full_count: int,
        frame_format: str,
        fd: dict,
        scan_hwaccel_fallback: bool,
    ) -> None:
        kept_order = list(range(1, full_count + 1))
        doc = {
            "full_decode_count": full_count,
            "compact_decode_count": full_count,
            "kept_order": kept_order,
            "skipped": [],
            "threshold": float(fd.get("threshold", 0.02)),
            "protect_scene_cuts": bool(fd.get("protect_scene_cuts", True)),
            "frame_format": frame_format,
            "scan_hwaccel_fallback": scan_hwaccel_fallback,
        }
        merge_dedupe_state_into_plan(ctx, doc)
        write_dedupe_map(ctx.stage_dir(self.name), doc)

    def rollback(self, ctx: PipelineContext, plan: StagePlan) -> None:
        # Frames are content-addressed via the cache; deleting them is safe but
        # we leave them in place so a re-run can hit the cache.
        return None

    # ----------------------------------------------------- hwaccel fallback

    def _run_fused_decode_with_hwaccel_fallback(
        self,
        *,
        ctx: PipelineContext,
        events: EventSink,
        primary_cmd: list,
        decode_hwaccel: str,
        out_dir: Path,
        scan_cwd: Path,
        scan_meta: Path,
        frame_format: str,
        png_intermediate_codec: str,
        tgt_w: int | None,
        tgt_h: int | None,
        bt709_normalize: bool,
        use_zscale: bool,
        start_pts: float | None,
        end_pts: float | None,
        time_pad_s: float = 0.0,
        loglevel: str = "error",
    ) -> tuple[ProcResult, bool]:
        """Run fused decode+metadata; on hwaccel failure retry using fallback decode modes.

        Uses ``run_capture`` (not streaming) so ``cwd`` can place the metadata sidecar
        next to ``out_dir`` without a second full decode when fusion succeeds.
        """
        result = run_capture(
            primary_cmd,
            cwd=scan_cwd,
            timeout=24 * 3600.0,
            check=False,
        )
        fallback_modes = decode_hwaccel_fallback_chain(decode_hwaccel)
        if result.returncode == 0 or not fallback_modes:
            return result, False

        result2 = result
        for fb_mode in fallback_modes:
            events.emit(StageEvent(
                ctx.job_id,
                self.name,
                "warning",
                message=(
                    f"decode_serve: hardware decode ({decode_hwaccel}) fused decode failed; "
                    f"retrying fusion with {fb_mode} decode"
                ),
            ))
            try:
                shutil.rmtree(out_dir)
            except OSError:
                pass
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                if scan_meta.exists():
                    scan_meta.unlink()
            except OSError:
                pass
            fb_cmd = self._ffmpeg.build_decode_to_frames_with_scene_metadata_fused(
                source=ctx.source_path,
                out_dir=out_dir,
                metadata_out=scan_meta,
                frame_format=frame_format,
                png_intermediate_codec=png_intermediate_codec,
                target_width=tgt_w if isinstance(tgt_w, int) else None,
                target_height=tgt_h if isinstance(tgt_h, int) else None,
                bt709_normalize=bt709_normalize,
                use_zscale=use_zscale,
                decode_hwaccel=fb_mode,
                start_pts=start_pts,
                end_pts=end_pts,
                time_pad_s=time_pad_s,
                loglevel=loglevel,
            )
            result2 = run_capture(fb_cmd, cwd=scan_cwd, timeout=24 * 3600.0, check=False)
            if result2.returncode == 0:
                break
        return result2, True

    def _run_decode_with_hwaccel_fallback(
        self,
        *,
        ctx: PipelineContext,
        events: EventSink,
        primary_cmd: list,
        decode_hwaccel: str,
        out_dir: Path,
        frame_format: str,
        png_intermediate_codec: str,
        tgt_w: int | None,
        tgt_h: int | None,
        bt709_normalize: bool,
        use_zscale: bool,
        start_pts: float | None,
        end_pts: float | None,
        time_pad_s: float = 0.0,
        loglevel: str = "error",
    ) -> tuple[ProcResult, bool]:
        """Run the primary decode command; on hwaccel failure, retry using fallback modes.

        Returns ``(result, used_fallback)``. ProcInterrupted is propagated so
        cancel/pause keeps working; modes without hwaccel fallback bubble up unchanged.
        """
        try:
            result = _run_capture_via_streaming(
                primary_cmd, ctx, events=events, stage=self.name,
            )
            return result, False
        except ProcError as exc:
            fallback_modes = decode_hwaccel_fallback_chain(decode_hwaccel)
            if not fallback_modes:
                raise StageError(
                    "decode_serve: ffmpeg invocation failed",
                    context={"stderr": exc.result.stderr[:2000]},
                ) from exc
            result = exc.result
        except ProcInterrupted as exc:
            if exc.reason == "cancel":
                raise CancelledError("cancelled during decode") from exc
            ctx.extras["pause_checkpoint"] = {"stage": self.name, "status": "interrupted"}
            raise PausedError("paused during decode") from exc

        fallback_modes = decode_hwaccel_fallback_chain(decode_hwaccel)
        for fb_mode in fallback_modes:
            events.emit(StageEvent(
                ctx.job_id,
                self.name,
                "warning",
                message=(
                    f"decode_serve: hardware decode ({decode_hwaccel}) failed; "
                    f"retrying with {fb_mode} decode"
                ),
            ))
            # Drop any partial frames from the failed hw attempt so we never mix two
            # decodes in one directory (e.g. odd sizes or duplicate indices).
            try:
                shutil.rmtree(out_dir)
            except OSError:
                pass
            out_dir.mkdir(parents=True, exist_ok=True)

            fallback_cmd = self._ffmpeg.build_decode_to_frames(
                source=ctx.source_path,
                out_dir=out_dir,
                frame_format=frame_format,
                png_intermediate_codec=png_intermediate_codec,
                target_width=tgt_w if isinstance(tgt_w, int) else None,
                target_height=tgt_h if isinstance(tgt_h, int) else None,
                bt709_normalize=bt709_normalize,
                use_zscale=use_zscale,
                decode_hwaccel=fb_mode,
                start_pts=start_pts,
                end_pts=end_pts,
                time_pad_s=time_pad_s,
                loglevel=loglevel,
            )
            result = run_capture(fallback_cmd, timeout=24 * 3600.0, check=False)
            if result.returncode == 0:
                break
        return result, True


def _run_capture_via_streaming(
    cmd: list,
    ctx: PipelineContext,
    *,
    events: EventSink | None = None,
    stage: str | None = None,
) -> ProcResult:
    """Drain ``run_streaming`` and return a ProcResult for a successful run.

    Raises ``ProcError`` on non-zero exit (preserving the wrapped result) and
    ``ProcInterrupted`` on cancel/pause.
    """
    emit_tool = (
        events is not None
        and stage is not None
        and logging.getLogger().isEnabledFor(logging.DEBUG)
    )
    stderr_lines: list[str] = []
    for stream, line in run_streaming(
        cmd,
        should_interrupt=lambda: "cancel" if ctx.cancel_event.is_set() else (
            "pause" if ctx.pause_event.is_set() else None
        ),
    ):
        if stream == "stderr":
            stderr_lines.append(line)
            if emit_tool:
                emit_tool_log(events, ctx.job_id, stage, line)
    return ProcResult([str(c) for c in cmd], 0, "", "\n".join(stderr_lines))


def _safe_version(adapter: FFmpegAdapter) -> str:
    try:
        return adapter.version
    except Exception:
        return "unknown"


def _benchmark_verbose_enabled(ctx: PipelineContext) -> bool:
    raw = ctx.extras.get("benchmark")
    return isinstance(raw, dict) and bool(raw.get("verbose_ffmpeg"))
