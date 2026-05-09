"""ffprobe-based media analyzer.

Why ffprobe (and not direct libav bindings):
* Single static binary; no Python C-extension dependency hell.
* Stable JSON contract.
* Identical behavior between dev and packaged builds.

Why we still call MKVToolNix elsewhere: ffprobe under-reports Matroska attachment names
in some builds, and mkvmerge -J gives us authoritative track UIDs (needed for any later
mkvpropedit operations).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aep.adapters.ffprobe import FFProbeAdapter
from aep.errors import ProbeError
from aep.media.models import (
    Chapter,
    Disposition,
    FormatInfo,
    MediaInfo,
    StreamInfo,
    StreamKind,
)

log = logging.getLogger(__name__)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip() in {"1", "true", "True", "yes"}
    return False


def _stream_kind(codec_type: str | None) -> StreamKind:
    if codec_type in {"video", "audio", "subtitle", "attachment", "data"}:
        return codec_type  # type: ignore[return-value]
    return "unknown"


def _parse_disposition(d: dict[str, Any] | None) -> Disposition:
    d = d or {}
    return Disposition(
        default=_bool(d.get("default")),
        forced=_bool(d.get("forced")),
        attached_pic=_bool(d.get("attached_pic")),
        hearing_impaired=_bool(d.get("hearing_impaired")),
        visual_impaired=_bool(d.get("visual_impaired")),
        captions=_bool(d.get("captions")),
        descriptions=_bool(d.get("descriptions")),
        original=_bool(d.get("original")),
        comment=_bool(d.get("comment")),
        dub=_bool(d.get("dub")),
    )


def _parse_stream(raw: dict[str, Any]) -> StreamInfo:
    tags = {str(k): str(v) for k, v in (raw.get("tags") or {}).items()}
    kind = _stream_kind(raw.get("codec_type"))
    return StreamInfo(
        index=int(raw.get("index", 0)),
        kind=kind,
        codec_name=raw.get("codec_name"),
        codec_long_name=raw.get("codec_long_name"),
        profile=raw.get("profile"),
        language=tags.get("language") or raw.get("tags", {}).get("LANGUAGE"),
        title=tags.get("title") or tags.get("TITLE"),
        disposition=_parse_disposition(raw.get("disposition")),
        bit_rate=_to_int(raw.get("bit_rate")),
        duration_s=_to_float(raw.get("duration")),
        tags=tags,
        width=_to_int(raw.get("width")),
        height=_to_int(raw.get("height")),
        pix_fmt=raw.get("pix_fmt"),
        color_space=raw.get("color_space"),
        color_primaries=raw.get("color_primaries"),
        color_transfer=raw.get("color_transfer"),
        color_range=raw.get("color_range"),
        field_order=raw.get("field_order"),
        avg_frame_rate=raw.get("avg_frame_rate"),
        r_frame_rate=raw.get("r_frame_rate"),
        nb_frames=_to_int(raw.get("nb_frames")),
        sar=raw.get("sample_aspect_ratio"),
        dar=raw.get("display_aspect_ratio"),
        bits_per_raw_sample=_to_int(raw.get("bits_per_raw_sample")),
        sample_rate=_to_int(raw.get("sample_rate")),
        channels=_to_int(raw.get("channels")),
        channel_layout=raw.get("channel_layout"),
        filename=tags.get("filename") or tags.get("FILENAME"),
        mimetype=tags.get("mimetype") or tags.get("MIMETYPE"),
    )


def _parse_chapter(raw: dict[str, Any]) -> Chapter:
    tags = {str(k): str(v) for k, v in (raw.get("tags") or {}).items()}
    start = _to_float(raw.get("start_time")) or 0.0
    end = _to_float(raw.get("end_time")) or 0.0
    return Chapter(
        id=int(raw.get("id", 0)),
        start_time_s=start,
        end_time_s=end,
        title=tags.get("title") or tags.get("TITLE"),
        tags=tags,
    )


def _parse_format(raw: dict[str, Any], source_path: Path) -> FormatInfo:
    tags = {str(k): str(v) for k, v in (raw.get("tags") or {}).items()}
    return FormatInfo(
        filename=str(raw.get("filename") or source_path),
        format_name=raw.get("format_name", ""),
        format_long_name=raw.get("format_long_name"),
        duration_s=_to_float(raw.get("duration")),
        bit_rate=_to_int(raw.get("bit_rate")),
        size_bytes=_to_int(raw.get("size")),
        tags=tags,
    )


class FfprobeAnalyzer:
    """Runs ffprobe and returns a normalized MediaInfo.

    Construction is cheap (just stores the adapter); call `.analyze(path)` per file.
    """

    def __init__(self, ffprobe: FFProbeAdapter | None = None) -> None:
        self._ffprobe = ffprobe or FFProbeAdapter()

    def analyze(self, source: Path) -> MediaInfo:
        source = Path(source).resolve()
        if not source.is_file():
            raise ProbeError(f"not a file: {source}")

        try:
            raw_text = self._ffprobe.probe_full_json(source)
        except Exception as exc:
            raise ProbeError(f"ffprobe failed: {source}", context={"reason": str(exc)}) from exc

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ProbeError("ffprobe produced invalid JSON", context={"reason": str(exc)}) from exc

        fmt = _parse_format(data.get("format", {}), source)
        all_streams = [_parse_stream(s) for s in data.get("streams", [])]
        attachments = [s for s in all_streams if s.kind == "attachment"]
        non_attachment = [s for s in all_streams if s.kind != "attachment"]
        chapters = [_parse_chapter(c) for c in data.get("chapters", [])]

        is_mkv = "matroska" in fmt.format_name.lower()

        info = MediaInfo(
            source_path=str(source),
            fmt=fmt,
            streams=non_attachment,
            chapters=chapters,
            attachments=attachments,
            is_matroska=is_mkv,
            likely_anime=None,  # filled in by classify.py downstream
        )

        # Notes that the GUI surfaces in the inspector
        if not info.video_streams:
            info.notes.append("No video stream detected.")
        elif len(info.video_streams) > 1:
            info.notes.append(
                f"{len(info.video_streams)} video streams; the first will be processed."
            )
        if info.is_matroska and not info.attachments:
            info.notes.append("No attachments (fonts) found in this MKV.")
        for s in info.subtitle_streams:
            if s.codec_name == "hdmv_pgs_subtitle":
                info.notes.append(
                    f"Stream #{s.index} is PGS (image) subtitles — fine to copy in MKV, "
                    "but cannot be remuxed into MP4."
                )
            elif s.codec_name == "dvd_subtitle":
                info.notes.append(
                    f"Stream #{s.index} is VobSub — keeps in MKV, will not move to MP4."
                )

        log.info(
            "probed %s: %d streams, %d attachments, %d chapters, mkv=%s",
            source.name, len(info.streams), len(info.attachments),
            len(info.chapters), info.is_matroska,
        )
        return info
