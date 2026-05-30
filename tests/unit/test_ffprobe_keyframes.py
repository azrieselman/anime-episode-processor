"""Keyframe packet parser and probe timeout (no real ffprobe binary)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from aep.adapters.ffprobe import (
    FFProbeAdapter,
    keyframe_probe_timeout,
    parse_keyframe_packets_compact,
)
from aep.util.proc import ProcResult

SAMPLE_COMPACT = """\
pts_time=0.000000|flags=K__
pts_time=0.042000|flags=___
pts_time=2.000000|flags=K__
pkt_pts_time=4.000000|flags=K__
pts_time=6.000000|flags=K__
"""


def test_keyframe_probe_timeout_unknown() -> None:
    assert keyframe_probe_timeout(None) == 300.0
    assert keyframe_probe_timeout(0) == 300.0


def test_keyframe_probe_timeout_scales_with_duration() -> None:
    assert keyframe_probe_timeout(600.0) == 300.0
    assert keyframe_probe_timeout(8100.0) == pytest.approx(4050.0)


def test_parse_keyframe_packets_compact() -> None:
    kfs = parse_keyframe_packets_compact(SAMPLE_COMPACT)
    assert kfs == [0.0, 2.0, 4.0, 6.0]


def test_parse_keyframe_packets_compact_empty() -> None:
    assert parse_keyframe_packets_compact("") == []


def test_list_video_keyframes_uses_packet_probe(tmp_path: Path) -> None:
    src = tmp_path / "in.mkv"
    src.write_bytes(b"\x00")
    adapter = FFProbeAdapter()
    adapter._path = "ffprobe"  # noqa: SLF001

    with patch("aep.adapters.ffprobe.run_capture") as mock_run:
        mock_run.return_value = ProcResult(
            ["ffprobe"],
            0,
            SAMPLE_COMPACT,
            "",
        )
        kfs = adapter.list_video_keyframes(src, duration_s=120.0)

    assert kfs == [0.0, 2.0, 4.0, 6.0]
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-show_entries" in cmd
    idx = cmd.index("-show_entries")
    assert "packet=" in cmd[idx + 1]
    assert cmd[cmd.index("-of") + 1] == "compact=p=0"
    assert mock_run.call_args[1]["timeout"] == 300.0
