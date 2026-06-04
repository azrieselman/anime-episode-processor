"""Demux-based video extent probing and MediaInfo enrichment."""

from __future__ import annotations

import logging
from pathlib import Path

from aep.adapters.ffprobe import FFProbeAdapter, VideoPacketExtent
from aep.media.models import MediaInfo, StreamInfo
from aep.util.fps import parse_rational

log = logging.getLogger(__name__)

# When packet lines omit duration, extend past the last PTS by one frame period.
_FRAME_PERIOD_PAD_S = 0.05


def _video_stream_ordinal(primary: StreamInfo, media: MediaInfo) -> int:
    """Map a video ``StreamInfo`` to ffmpeg ``v:N`` (ordinal among video streams)."""
    ordinal = 0
    for stream in media.streams:
        if stream.kind != "video":
            continue
        if stream.index == primary.index:
            return ordinal
        ordinal += 1
    return 0


def decodable_end_from_extent(
    extent: VideoPacketExtent,
    *,
    frame_period_s: float | None = None,
) -> float | None:
    """Exclusive-ish timeline end (seconds) from demuxed packet bounds."""
    if extent.last_end_s is not None and extent.last_end_s > 0:
        end = float(extent.last_end_s)
        if (
            extent.last_pts_s is not None
            and frame_period_s is not None
            and frame_period_s > 0
            and end <= float(extent.last_pts_s) + 1e-6
        ):
            return end + float(frame_period_s)
        return end
    if extent.last_pts_s is not None and extent.last_pts_s > 0:
        pad = frame_period_s if frame_period_s and frame_period_s > 0 else _FRAME_PERIOD_PAD_S
        return float(extent.last_pts_s) + pad
    return None


def enrich_media_decodable_extent(
    media: MediaInfo,
    source: Path,
    *,
    ffprobe: FFProbeAdapter | None = None,
) -> MediaInfo:
    """Attach ``decodable_end_s`` on the primary video stream when missing."""
    primary = media.primary_video
    if primary is None or primary.decodable_end_s is not None:
        return media

    adapter = ffprobe or FFProbeAdapter()
    duration_hint = media.fmt.duration_s if media.fmt is not None else None
    try:
        extent = adapter.probe_video_packet_extent(
            source,
            stream_index=_video_stream_ordinal(primary, media),
            duration_s=duration_hint,
        )
    except Exception as exc:
        log.warning("decodable extent probe failed (%s); using metadata duration only", exc)
        return media

    fps = parse_rational(primary.avg_frame_rate) or parse_rational(primary.r_frame_rate)
    frame_period = (1.0 / float(fps)) if fps is not None and fps > 0 else None
    decodable_end = decodable_end_from_extent(extent, frame_period_s=frame_period)
    if decodable_end is None or decodable_end <= 0:
        return media

    updated_streams: list[StreamInfo] = []
    for stream in media.streams:
        if stream.index == primary.index:
            updated_streams.append(
                stream.model_copy(update={"decodable_end_s": decodable_end}),
            )
        else:
            updated_streams.append(stream)

    fmt_dur = media.fmt.duration_s if media.fmt is not None else None
    if fmt_dur is not None and decodable_end < float(fmt_dur) - 0.05:
        log.info(
            "decodable video ends at %.3fs (container metadata reports %.3fs)",
            decodable_end,
            fmt_dur,
        )

    return media.model_copy(update={"streams": updated_streams})
