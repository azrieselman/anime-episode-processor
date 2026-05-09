"""FFmpeg-based mux executor.

Takes a `StreamMappingPlan` plus the encoded video and source paths, and produces a
final muxed file. Handles both the mux invocation and the optional mkvpropedit
correction pass for Matroska outputs.

Stage 09 calls this when `decide_mux_tool` returns `tool="ffmpeg"`. The other backend
(`mkvtoolnix_mux`) handles the mkvmerge path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.ffmpeg import FFmpegAdapter, raise_if_failed
from aep.adapters.mkvtoolnix import MkvpropeditAdapter
from aep.errors import MuxError
from aep.mux.mapping import StreamMappingPlan
from aep.util.proc import ProcError, run_capture

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MuxResult:
    output_path: Path
    used_tool: str          # "ffmpeg" | "mkvmerge"
    propedit_applied: bool


def run_ffmpeg_mux(
    *,
    encoded_video: Path,
    source: Path,
    output: Path,
    plan: StreamMappingPlan,
    ffmpeg: FFmpegAdapter,
    mkvpropedit: MkvpropeditAdapter | None = None,
    apply_propedit: bool = False,
    allow_overwrite: bool = False,
) -> MuxResult:
    """Execute the ffmpeg mux for a `StreamMappingPlan`.

    `apply_propedit` is honored only when `output` is .mkv and a MkvpropeditAdapter is
    provided. We swallow propedit failures (they're cosmetic) and log a warning.
    """
    cmd = ffmpeg.build_remux_with_streams(
        encoded_video=encoded_video,
        source=source,
        output=output,
        map_args=plan.map_args,
        copy_args=[*plan.copy_args, *plan.metadata_args, *plan.disposition_args],
        global_metadata=plan.copy_global_metadata,
        chapters=plan.copy_chapters,
        allow_overwrite=allow_overwrite,
    )
    try:
        result = run_capture(cmd, timeout=3600.0, check=False)
    except ProcError as exc:
        raise MuxError("ffmpeg mux failed",
                       context={"stderr": exc.result.stderr[:2000]}) from exc
    raise_if_failed(result.returncode, result.stderr)

    propedit_applied = False
    if (
        apply_propedit
        and output.suffix.lower() == ".mkv"
        and mkvpropedit is not None
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
            # Cosmetic — the file is still playable. Log and continue.
            log.warning("mkvpropedit correction pass failed (non-fatal): %s", exc)

    return MuxResult(output_path=output, used_tool="ffmpeg", propedit_applied=propedit_applied)
