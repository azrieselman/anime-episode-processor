"""Tests for `aep.pipeline.stages.s10_validate._validate`.

We test the pure validator with synthetic MediaInfo + plan dicts.
"""

from __future__ import annotations

from copy import deepcopy

from aep.media.models import (
    Chapter,
    Disposition,
    FormatInfo,
    MediaInfo,
    StreamInfo,
)
from aep.pipeline.stages.s10_validate import _validate


def _v(width: int = 1920, height: int = 1080) -> StreamInfo:
    return StreamInfo(index=0, kind="video", codec_name="hevc",
                      width=width, height=height, pix_fmt="yuv420p10le")


def _a(idx: int, default: bool = False) -> StreamInfo:
    return StreamInfo(index=idx, kind="audio", codec_name="aac",
                      disposition=Disposition(default=default))


def _s(idx: int, default: bool = False) -> StreamInfo:
    return StreamInfo(index=idx, kind="subtitle", codec_name="ass",
                      disposition=Disposition(default=default))


def _media(
    *,
    duration: float = 1440.0,
    video: StreamInfo | None = None,
    audio: list[StreamInfo] | None = None,
    subs: list[StreamInfo] | None = None,
    chapters: list[Chapter] | None = None,
    attachments: list[StreamInfo] | None = None,
    filename: str = "/fake/out.mkv",
) -> MediaInfo:
    streams = [video or _v()] + (audio or []) + (subs or [])
    return MediaInfo(
        source_path=filename,
        fmt=FormatInfo(filename=filename, format_name="matroska,webm",
                       duration_s=duration),
        streams=streams,
        chapters=chapters or [],
        attachments=attachments or [],
        is_matroska=True,
    )


def _plan(
    *,
    container: str = "mkv",
    audio_count: int = 1,
    sub_count: int = 1,
    target_w: int | None = 1920,
    target_h: int | None = 1080,
    copy_chapters: bool = True,
    copy_attachments: bool = True,
) -> dict:
    return {
        "container": container,
        "stream_mapping": {
            "audio": [{"source_index": i} for i in range(audio_count)],
            "subtitles": [{"source_index": i} for i in range(sub_count)],
            "skipped": [],
            "copy_chapters": copy_chapters,
        },
        "target_geometry": {"width": target_w, "height": target_h,
                            "preserved": target_w is None and target_h is None},
        "preset": {"streams": {"copy_attachments": copy_attachments}},
    }


def test_passes_clean_match():
    src = _media(audio=[_a(1)], subs=[_s(2)],
                 chapters=[Chapter(id=1, start_time_s=0, end_time_s=10)])
    out = _media(audio=[_a(1)], subs=[_s(2)],
                 chapters=[Chapter(id=1, start_time_s=0, end_time_s=10)])
    report = _validate(src, out, _plan())
    assert report.passed
    assert report.failures == []


def test_duration_drift_fails():
    src = _media(duration=1440.0, audio=[_a(1)], subs=[_s(2)])
    out = _media(duration=1437.0, audio=[_a(1)], subs=[_s(2)])
    report = _validate(src, out, _plan())
    assert not report.passed
    codes = {f.code for f in report.failures}
    assert "duration_mismatch" in codes


def test_audio_count_mismatch_fails():
    src = _media(audio=[_a(1), _a(2)], subs=[_s(3)])
    out = _media(audio=[_a(1)], subs=[_s(3)])
    report = _validate(src, out, _plan(audio_count=2))
    codes = {f.code for f in report.failures}
    assert "audio_stream_count" in codes


def test_subtitle_count_mismatch_fails():
    src = _media(audio=[_a(1)], subs=[_s(2), _s(3)])
    out = _media(audio=[_a(1)], subs=[_s(2)])
    report = _validate(src, out, _plan(sub_count=2))
    codes = {f.code for f in report.failures}
    assert "subtitle_stream_count" in codes


def test_chapters_lost_fails():
    chaps = [Chapter(id=i, start_time_s=i * 60, end_time_s=(i + 1) * 60) for i in range(3)]
    src = _media(audio=[_a(1)], subs=[_s(2)], chapters=chaps)
    out = _media(audio=[_a(1)], subs=[_s(2)], chapters=[])
    report = _validate(src, out, _plan())
    codes = {f.code for f in report.failures}
    assert "chapters_lost" in codes


def test_attachments_lost_fails_for_mkv():
    attach = [StreamInfo(index=10, kind="attachment", filename="font.otf")]
    src = _media(audio=[_a(1)], subs=[_s(2)], attachments=attach)
    out = _media(audio=[_a(1)], subs=[_s(2)], attachments=[])
    report = _validate(src, out, _plan())
    codes = {f.code for f in report.failures}
    assert "attachments_lost" in codes


def test_attachments_skipped_for_mp4():
    attach = [StreamInfo(index=10, kind="attachment", filename="font.otf")]
    src = _media(audio=[_a(1)], subs=[_s(2)], attachments=attach)
    out = _media(audio=[_a(1)], subs=[_s(2)], attachments=[])
    report = _validate(src, out, _plan(container="mp4"))
    # MP4 can't carry attachments, so we don't fail.
    assert "attachments_lost" not in {f.code for f in report.failures}


def test_video_resolution_off_target_fails():
    src = _media(video=_v(1920, 1080), audio=[_a(1)], subs=[_s(2)])
    out = _media(video=_v(1280, 720), audio=[_a(1)], subs=[_s(2)])
    report = _validate(src, out, _plan(target_w=1920, target_h=1080))
    codes = {f.code for f in report.failures}
    assert "video_resolution" in codes


def test_video_resolution_within_tolerance_passes():
    src = _media(video=_v(1920, 1080), audio=[_a(1)], subs=[_s(2)])
    # Off by 1 px (rounding) → should pass.
    out = _media(video=_v(1921, 1079), audio=[_a(1)], subs=[_s(2)])
    report = _validate(src, out, _plan(target_w=1920, target_h=1080))
    assert "video_resolution" not in {f.code for f in report.failures}


def test_default_flag_full_loss_fails():
    src = _media(audio=[_a(1, default=True)], subs=[_s(2)])
    out = _media(audio=[_a(1, default=False)], subs=[_s(2)])
    report = _validate(src, out, _plan())
    codes = {f.code for f in report.failures}
    assert "audio_streams_default_lost" in codes


def test_default_flag_partial_loss_warning_only():
    src = _media(audio=[_a(1, default=True), _a(2, default=True)], subs=[_s(3)])
    out = _media(audio=[_a(1, default=True), _a(2, default=False)], subs=[_s(3)])
    plan = _plan(audio_count=2, sub_count=1)
    report = _validate(src, out, plan)
    # Partial drop = note, not failure.
    codes = {f.code for f in report.failures}
    assert "audio_streams_default_lost" not in codes
    assert any("default flags preserved" in n for n in report.notes)


def test_video_stream_count_must_be_one():
    extra_v = StreamInfo(index=4, kind="video", codec_name="hevc")
    src = _media(audio=[_a(1)], subs=[_s(2)])
    out = _media(audio=[_a(1)], subs=[_s(2)])
    out.streams.append(extra_v)
    report = _validate(src, out, _plan())
    codes = {f.code for f in report.failures}
    assert "video_stream_count" in codes


def test_plan_doc_immutability():
    """Validator must not mutate the plan dict."""
    src = _media(audio=[_a(1)], subs=[_s(2)])
    out = _media(audio=[_a(1)], subs=[_s(2)])
    plan = _plan()
    plan_before = deepcopy(plan)
    _validate(src, out, plan)
    assert plan == plan_before
