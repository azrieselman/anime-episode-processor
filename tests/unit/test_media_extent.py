"""Tests for demux-based decodable extent enrichment."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aep.adapters.ffprobe import VideoPacketExtent, parse_video_packets_compact
from aep.media.extent import decodable_end_from_extent, enrich_media_decodable_extent
from aep.media.models import FormatInfo, MediaInfo, StreamInfo
from aep.pipeline.batch_timing import resolve_planning_duration_s
from aep.util.proc import ProcResult

SAMPLE = """\
pts_time=0.000000|flags=K__|pkt_duration_time=0.042000
pts_time=1420.000000|flags=___|pkt_duration_time=0.042000
"""


def test_decodable_end_from_extent_uses_packet_duration() -> None:
    extent = parse_video_packets_compact(SAMPLE)
    end = decodable_end_from_extent(extent, frame_period_s=1.0 / 24.0)
    assert end == pytest.approx(1420.042)


def test_resolve_planning_duration_prefers_decodable_end_over_metadata() -> None:
    actual_s = 1420.0
    inflated_s = 1500.0
    primary = StreamInfo(
        index=0,
        kind="video",
        avg_frame_rate="24/1",
        r_frame_rate="24/1",
        duration_s=float(inflated_s),
        nb_frames=int(inflated_s * 24),
        decodable_end_s=float(actual_s),
    )
    media = MediaInfo(
        source_path="/tmp/x.mkv",
        fmt=FormatInfo(
            filename="/tmp/x.mkv",
            format_name="matroska",
            duration_s=float(inflated_s),
        ),
        streams=[primary],
    )
    assert resolve_planning_duration_s(media) == pytest.approx(actual_s)


def test_enrich_media_decodable_extent_sets_primary_stream(tmp_path: Path) -> None:
    src = tmp_path / "in.mkv"
    src.write_bytes(b"\x00")
    inflated_s = 1500.0
    primary = StreamInfo(
        index=0,
        kind="video",
        avg_frame_rate="24/1",
        r_frame_rate="24/1",
    )
    media = MediaInfo(
        source_path=str(src),
        fmt=FormatInfo(
            filename=str(src),
            format_name="matroska",
            duration_s=float(inflated_s),
        ),
        streams=[primary],
    )
    extent = VideoPacketExtent(
        last_pts_s=1420.0,
        last_end_s=1420.042,
        keyframe_times_s=[0.0],
    )

    with patch("aep.media.extent.FFProbeAdapter") as mock_cls:
        mock_cls.return_value.probe_video_packet_extent.return_value = extent
        enriched = enrich_media_decodable_extent(media, src)

    pv = enriched.primary_video
    assert pv is not None
    assert pv.decodable_end_s == pytest.approx(1420.042)
