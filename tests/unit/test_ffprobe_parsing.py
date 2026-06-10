"""Parse a synthetic ffprobe JSON payload through FfprobeAnalyzer (without invoking ffprobe).

We monkeypatch FFProbeAdapter.probe_full_json so the test does not require the real
ffprobe binary. This locks the JSON-shape contract that the analyzer depends on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aep.media.ffprobe import FfprobeAnalyzer

SAMPLE = {
    "streams": [
        {
            "index": 0, "codec_type": "video", "codec_name": "h264",
            "width": 1280, "height": 720, "pix_fmt": "yuv420p",
            "avg_frame_rate": "24000/1001", "r_frame_rate": "24000/1001",
            "tags": {"language": "und", "title": "Main"},
            "disposition": {"default": 1},
        },
        {
            "index": 1, "codec_type": "audio", "codec_name": "aac",
            "channels": 2, "channel_layout": "stereo", "sample_rate": "48000",
            "tags": {"language": "jpn", "title": "Japanese 2.0"},
            "disposition": {"default": 1},
        },
        {
            "index": 2, "codec_type": "audio", "codec_name": "ac3",
            "channels": 6, "channel_layout": "5.1",
            "tags": {"language": "eng", "title": "English 5.1"},
            "disposition": {"default": 0},
        },
        {
            "index": 3, "codec_type": "subtitle", "codec_name": "ass",
            "tags": {"language": "eng", "title": "Full Subs"},
            "disposition": {"default": 1, "forced": 0},
        },
        {
            "index": 4, "codec_type": "subtitle", "codec_name": "ass",
            "tags": {"language": "eng", "title": "Signs & Songs"},
            "disposition": {"default": 0, "forced": 1},
        },
        {
            "index": 5, "codec_type": "attachment",
            "tags": {"filename": "OpenSans-Regular.ttf", "mimetype": "application/x-truetype-font"},
        },
    ],
    "chapters": [
        {"id": 0, "start_time": "0.000", "end_time": "90.000", "tags": {"title": "Prologue"}},
        {"id": 1, "start_time": "90.000", "end_time": "180.000", "tags": {"title": "OP"}},
    ],
    "format": {
        "filename": "fake.mkv",
        "format_name": "matroska,webm",
        "duration": "1440.000",
        "size": "1234567890",
        "bit_rate": "5000000",
        "tags": {"title": "Episode 01"},
    },
}


class _FakeAdapter:
    version = "n8.1.1"

    def probe_full_json(self, _: Path) -> str:
        return json.dumps(SAMPLE)


def test_parse_full(tmp_path: Path) -> None:
    fake_file = tmp_path / "fake.mkv"
    fake_file.write_bytes(b"\x00")  # exists; we don't actually probe
    analyzer = FfprobeAnalyzer(_FakeAdapter())  # type: ignore[arg-type]
    info = analyzer.analyze(fake_file)

    assert len(info.video_streams) == 1
    assert info.video_streams[0].width == 1280
    assert info.video_streams[0].height == 720

    assert len(info.audio_streams) == 2
    assert info.audio_streams[0].language == "jpn"
    assert info.audio_streams[1].channels == 6

    assert len(info.subtitle_streams) == 2
    forced = [s for s in info.subtitle_streams if s.disposition.forced]
    assert len(forced) == 1
    assert forced[0].title == "Signs & Songs"

    assert len(info.attachments) == 1
    assert info.attachments[0].filename == "OpenSans-Regular.ttf"

    assert len(info.chapters) == 2
    assert info.chapters[1].title == "OP"

    assert info.is_matroska is True
    assert info.fmt.duration_s == pytest.approx(1440.0)
