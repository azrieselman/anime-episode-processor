"""Stage 09: mux.

Combines the encoded video from stage 08 with selected audio/subtitle/chapter/
attachment streams from the original source. Routes to either ffmpeg-mux or
mkvmerge-mux based on the decision frozen in stage 01's plan.

Reads:    ctx.plan, ctx.source_path, stage_results['08_encode'].artifacts['video_only']
Writes:   ctx.output_path

The output goes directly to `ctx.output_path` — this is the user-visible artifact, so
we mux to ``<stem>.partial<suffix>`` (e.g. ``out.partial.mkv``) then rename. Using a
final suffix of ``.mkv`` keeps ffmpeg's format autodetection happy on Windows.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from aep.adapters.ffmpeg import FFmpegAdapter
from aep.adapters.mkvtoolnix import MkvmergeAdapter, MkvpropeditAdapter
from aep.errors import MuxError, PipelineError
from aep.mux.ffmpeg_mux import run_ffmpeg_mux
from aep.mux.mapping import MuxToolDecision, decide_mux_tool, plan_streams
from aep.mux.mkvtoolnix_mux import run_mkvmerge_mux
from aep.persist.presets import Preset
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.proc import ProcError, run_capture

log = logging.getLogger(__name__)


class MuxStage(BaseStage):
    name = "09_mux"

    def __init__(
        self,
        *,
        ffmpeg: FFmpegAdapter | None = None,
        mkvmerge: MkvmergeAdapter | None = None,
        mkvpropedit: MkvpropeditAdapter | None = None,
    ) -> None:
        self._ffmpeg = ffmpeg or FFmpegAdapter()
        self._mkvmerge = mkvmerge or MkvmergeAdapter()
        self._mkvpropedit = mkvpropedit or MkvpropeditAdapter()

    # ------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        encoded_video = self._encoded_video_path(ctx)
        params: dict[str, object] = {
            "preset_id": ctx.preset_id,
            "encoded_video": str(encoded_video),
            "container": ctx.plan.get("container", "mkv") if ctx.plan else "mkv",
        }
        # Tools that participate in the mux step.
        tool_versions = {
            "ffmpeg": _safe_version(self._ffmpeg),
            "mkvmerge": _safe_version(self._mkvmerge),
            "mkvpropedit": _safe_version(self._mkvpropedit),
        }
        cache_key = compute_cache_key(
            source_fingerprint=str(ctx.source_path),
            stage_name=self.name,
            tool_versions=tool_versions,
            params=params,
        )
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            inputs=[encoded_video, ctx.source_path],
            outputs=[ctx.output_path],
        )

    # ------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        if ctx.media_info is None or not ctx.plan:
            raise PipelineError("09_mux requires 00_probe and 01_plan to have run")

        preset = Preset.model_validate(ctx.plan["preset"])
        media = ctx.media_info
        encoded_video = self._encoded_video_path(ctx)

        # Stream map is recomputed (it includes rich dataclasses), but mux
        # backend decision is reused from the frozen plan when available.
        mapping = plan_streams(media, preset.streams, container=preset.container)
        mux_cfg = (ctx.plan or {}).get("mux") or {}
        if isinstance(mux_cfg, dict) and mux_cfg.get("tool") in {"ffmpeg", "mkvmerge"}:
            decision = MuxToolDecision(
                tool=str(mux_cfg["tool"]),
                reason=str(mux_cfg.get("reason") or "frozen in stage 01 plan"),
                needs_propedit_pass=bool(mux_cfg.get("needs_propedit_pass", False)),
            )
        else:
            decision = decide_mux_tool(media, preset.streams, container=preset.container)
        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=f"mux backend: {decision.tool} ({decision.reason})",
        ))

        # Atomic write: mux to a temp path, then rename. The name must still end
        # with the real container suffix (e.g. .mkv): ``with_suffix(.mkv.partial)``
        # yields ``*.mkv.partial``, which ffmpeg treats as an unknown format.
        ctx.output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = ctx.output_path.parent / (
            f"{ctx.output_path.stem}.partial{ctx.output_path.suffix}"
        )
        if partial.exists():
            partial.unlink()

        try:
            if decision.tool == "mkvmerge":
                result = run_mkvmerge_mux(
                    encoded_video=encoded_video,
                    source=ctx.source_path,
                    output=partial,
                    plan=mapping,
                    cfg=preset.streams,
                    mkvmerge=self._mkvmerge,
                    mkvpropedit=self._mkvpropedit,
                )
            else:
                result = run_ffmpeg_mux(
                    encoded_video=encoded_video,
                    source=ctx.source_path,
                    output=partial,
                    plan=mapping,
                    ffmpeg=self._ffmpeg,
                    mkvpropedit=self._mkvpropedit,
                    apply_propedit=decision.needs_propedit_pass,
                    allow_overwrite=False,  # we just unlinked it
                )
        except MuxError:
            if partial.exists():
                try:
                    partial.unlink()
                except OSError:
                    log.warning("failed to clean up partial file: %s", partial)
            raise

        # Atomic rename — on Windows os.replace is atomic for same-volume targets.
        if ctx.output_path.exists():
            ctx.output_path.unlink()
        os.replace(partial, ctx.output_path)

        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=f"output written: {ctx.output_path.name} (tool={result.used_tool})",
        ))

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"output": ctx.output_path},
            metrics={
                "mux_tool": result.used_tool,
                "propedit_applied": result.propedit_applied,
                "audio_streams": len(mapping.audio_streams),
                "subtitle_streams": len(mapping.subtitle_streams),
                "subs_skipped": len(mapping.skipped_streams),
                "size_bytes": (
                    ctx.output_path.stat().st_size if ctx.output_path.exists() else 0
                ),
            },
        )

    # ------------------------------------------------------------- helpers

    def _encoded_video_path(self, ctx: PipelineContext) -> Path:
        cached = ctx.extras.get("mux_encoded_video_path")
        if isinstance(cached, str):
            cached_path = Path(cached)
            if cached_path.is_file() and cached_path.stat().st_size > 0:
                return cached_path
        # M6.5: when the runner produced per-batch encoded segments, we have to
        # concatenate them into a single intermediate video file before mux.
        # This runs at most once per job (the concatenated file is reused if
        # this method is called again — e.g. by plan() after run()).
        if ctx.encoded_segments:
            path = self._concat_segments(ctx)
            ctx.extras["mux_encoded_video_path"] = str(path)
            return path

        encode_result = ctx.stage_results.get("08_encode")
        if encode_result and "video_only" in encode_result.artifacts:
            path = encode_result.artifacts["video_only"]
            ctx.extras["mux_encoded_video_path"] = str(path)
            return path
        # Fallback: look in the canonical location.
        for ext in (".mkv", ".mp4"):
            candidate = ctx.workdir / "08_encode" / f"video{ext}"
            if candidate.exists():
                ctx.extras["mux_encoded_video_path"] = str(candidate)
                return candidate
        raise PipelineError(
            "09_mux: encoded video not found. Run 08_encode first."
        )

    def _concat_segments(self, ctx: PipelineContext) -> Path:
        """Concatenate per-batch segments into a single video file via ffmpeg's concat demuxer.

        All segments share encoder configuration (same preset), so stream copy
        is safe — no re-encoding penalty. The concat list file lives next to
        the segments in `<workdir>/batch_segments/`.

        Idempotent: if the concatenated file already exists and is at least as
        new as the newest segment, we reuse it. This matters because
        `_encoded_video_path()` is called from both plan() (cache-key build)
        and run() (actual mux); we don't want to redo concat work each time.
        """
        segments_dir = ctx.workdir / "batch_segments"
        list_path = segments_dir / "concat.txt"
        out_path = segments_dir / "video_concat.mkv"

        # Reuse if up to date.
        if out_path.exists():
            try:
                out_mtime = out_path.stat().st_mtime
                newest = max((s.stat().st_mtime for s in ctx.encoded_segments), default=0.0)
                if out_mtime >= newest:
                    return out_path
            except OSError:
                pass  # fall through and rebuild

        # Write the concat-demuxer list. ffmpeg requires Unix-style 'file ' lines
        # with single quotes around each path; backslashes and single quotes in
        # paths are escaped per the demuxer spec.
        lines = ["# auto-generated by AEP M6.5 batched mux"]
        for seg in ctx.encoded_segments:
            # Use forward slashes — ffmpeg accepts them on Windows and avoids
            # the need to escape backslashes inside the single-quoted token.
            posix = seg.as_posix().replace("'", r"'\''")
            lines.append(f"file '{posix}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = [
            self._ffmpeg.command_executable(),
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-fflags", "+genpts",
            "-avoid_negative_ts", "make_zero",
            "-map", "0:v:0",
            "-c", "copy",
            "-map_metadata", "-1",
            "-map_chapters", "-1",
            str(out_path),
        ]
        log.info("concat demux: joining %d segments → %s", len(ctx.encoded_segments), out_path)
        try:
            result = run_capture(cmd, timeout=3600.0, check=False)
        except ProcError as exc:
            raise MuxError(
                "09_mux concat demuxer failed (run_capture)",
                context={"stderr": exc.result.stderr[:2000]},
            ) from exc
        if result.returncode != 0:
            raise MuxError(
                f"09_mux concat demuxer exited {result.returncode}",
                context={"stderr": result.stderr[-2000:], "list": str(list_path)},
            )
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise MuxError(
                "09_mux concat produced empty output",
                context={"out": str(out_path)},
            )
        return out_path


def _safe_version(adapter) -> str:
    try:
        return adapter.version
    except Exception:
        return "unknown"
