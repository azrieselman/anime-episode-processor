"""mkvmerge-based mux executor.

Used when `decide_mux_tool` selects mkvmerge — typically when the source has
attachments (fonts) or complex chapter editions that ffmpeg's matroska writer is
historically bad at preserving.

mkvmerge's track model is "include-by-default, exclude-by-flag", which is the inverse
of ffmpeg's. The adapter (`MkvmergeAdapter.build_mkv_mux`) already speaks that model;
this executor just wires it to the StreamMappingPlan's preferences.

NOTE: with mkvmerge we don't have the same per-stream `-disposition` and
`-metadata:s` levers — track flags ride along with the source by default. After mux
we still optionally run mkvpropedit if the plan has explicit overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path

from aep.adapters.mkvtoolnix import MkvmergeAdapter, MkvpropeditAdapter
from aep.errors import MuxError
from aep.mux.ffmpeg_mux import MuxResult
from aep.mux.mapping import StreamMappingPlan
from aep.persist.presets import StreamMappingCfg
from aep.util.proc import ProcError, run_capture

log = logging.getLogger(__name__)


def run_mkvmerge_mux(
    *,
    encoded_video: Path,
    source: Path,
    output: Path,
    plan: StreamMappingPlan,
    cfg: StreamMappingCfg,
    mkvmerge: MkvmergeAdapter,
    mkvpropedit: MkvpropeditAdapter | None = None,
    excluded_audio_track_ids: list[int] | None = None,
) -> MuxResult:
    """Execute the mkvmerge mux for a `StreamMappingPlan`.

    `cfg` is needed because mkvmerge's track-selection model is whole-class flags
    (`--no-audio`, `--no-subtitles`); we don't render per-stream maps the way ffmpeg
    does. The plan's audio/subtitle counts are still emitted as rationale.
    """
    if output.suffix.lower() not in {".mkv", ".webm"}:
        raise MuxError(
            f"mkvmerge can only write Matroska/WebM containers; got {output.suffix!r}",
            context={"output": str(output)},
        )

    cmd = mkvmerge.build_mkv_mux(
        encoded_video=encoded_video,
        source=source,
        output=output,
        copy_audio=cfg.copy_audio,
        copy_subtitles=cfg.copy_subtitles,
        copy_chapters=cfg.copy_chapters,
        copy_attachments=cfg.copy_attachments,
        copy_global_tags=cfg.copy_global_metadata,
        excluded_track_ids=excluded_audio_track_ids,
    )
    try:
        result = run_capture(cmd, timeout=3600.0, check=False)
    except ProcError as exc:
        raise MuxError("mkvmerge failed",
                       context={"stderr": exc.result.stderr[:2000]}) from exc
    if result.returncode not in (0, 1):
        # 0=ok, 1=warnings, 2+=errors
        raise MuxError(
            f"mkvmerge exited with code {result.returncode}",
            context={"stderr": result.stderr[-2000:]},
        )

    propedit_applied = False
    if (
        mkvpropedit is not None
        and (plan.mkvpropedit_track_edits or plan.mkvpropedit_segment_title)
    ):
        try:
            mkvpropedit.apply_edits(
                output,
                plan.mkvpropedit_track_edits,
                segment_title=plan.mkvpropedit_segment_title,
            )
            propedit_applied = True
        except MuxError as exc:
            log.warning("mkvpropedit correction pass failed (non-fatal): %s", exc)

    return MuxResult(output_path=output, used_tool="mkvmerge", propedit_applied=propedit_applied)
