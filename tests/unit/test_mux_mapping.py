"""Tests for `aep.mux.mapping`.

These tests are pure (no subprocess); they build synthetic MediaInfo and assert on
the produced argv lists and decision logic.
"""

from __future__ import annotations

from aep.media.models import (
    Chapter,
    Disposition,
    FormatInfo,
    MediaInfo,
    StreamInfo,
)
from aep.mux.mapping import decide_mux_tool, plan_streams
from aep.persist.presets import StreamMappingCfg


def _media(
    *,
    audio_streams: list[StreamInfo] | None = None,
    subtitle_streams: list[StreamInfo] | None = None,
    chapters: list[Chapter] | None = None,
    attachments: list[StreamInfo] | None = None,
    is_mkv: bool = True,
    fmt_tags: dict[str, str] | None = None,
) -> MediaInfo:
    video = StreamInfo(index=0, kind="video", codec_name="hevc", width=1920, height=1080,
                       pix_fmt="yuv420p10le")
    streams = [video] + (audio_streams or []) + (subtitle_streams or [])
    return MediaInfo(
        source_path="/fake/in.mkv",
        fmt=FormatInfo(filename="/fake/in.mkv", format_name="matroska,webm",
                       duration_s=1440.0, tags=fmt_tags or {}),
        streams=streams,
        chapters=chapters or [],
        attachments=attachments or [],
        is_matroska=is_mkv,
    )


def _aud(idx: int, *, lang: str | None = "jpn", title: str | None = None,
         codec: str = "aac", default: bool = False, forced: bool = False) -> StreamInfo:
    return StreamInfo(
        index=idx, kind="audio", codec_name=codec,
        language=lang, title=title,
        disposition=Disposition(default=default, forced=forced),
    )


def _sub(idx: int, *, lang: str | None = "eng", title: str | None = None,
         codec: str = "ass", default: bool = False, forced: bool = False) -> StreamInfo:
    return StreamInfo(
        index=idx, kind="subtitle", codec_name=codec,
        language=lang, title=title,
        disposition=Disposition(default=default, forced=forced),
    )


def _attachment(idx: int, name: str = "font.otf") -> StreamInfo:
    return StreamInfo(index=idx, kind="attachment", filename=name, mimetype="font/otf")


# ----- plan_streams ---------------------------------------------------------


def test_video_always_mapped_first():
    media = _media(audio_streams=[_aud(1)])
    plan = plan_streams(media, StreamMappingCfg())
    # First two map_args entries: -map 0:v:0
    assert plan.map_args[0] == "-map"
    assert plan.map_args[1] == "0:v:0"
    # First copy_args is -c:v copy
    assert plan.copy_args[0] == "-c:v"
    assert plan.copy_args[1] == "copy"


def test_audio_streams_mapped_with_metadata():
    media = _media(audio_streams=[
        _aud(1, lang="jpn", title="Japanese stereo", default=True),
        _aud(2, lang="eng", title="English dub"),
    ])
    plan = plan_streams(media, StreamMappingCfg())
    assert len(plan.audio_streams) == 2
    assert plan.audio_streams[0].language == "jpn"
    # ffmpeg argv should include language metadata for each.
    args = " ".join(plan.metadata_args)
    assert "language=jpn" in args
    assert "language=eng" in args
    assert "title=Japanese stereo" in args


def test_subtitles_in_mp4_skip_ass_with_warning():
    media = _media(subtitle_streams=[_sub(2, codec="ass"), _sub(3, codec="mov_text")])
    plan = plan_streams(media, StreamMappingCfg(), container="mp4")
    # Only mov_text survives.
    assert len(plan.subtitle_streams) == 1
    assert plan.subtitle_streams[0].codec_name == "mov_text"
    # Warning text mentions container can't carry the codec; reason in skipped_streams notes "not supported" in spirit.
    assert any("cannot carry" in w for w in plan.warnings)
    # skipped_streams is list[tuple[source_index, reason]]; reason mentions "not supported".
    assert any("not supported" in reason for _, reason in plan.skipped_streams)


def test_dispositions_emitted():
    media = _media(audio_streams=[_aud(1, default=True, forced=True)])
    plan = plan_streams(media, StreamMappingCfg())
    args = " ".join(plan.disposition_args)
    assert "-disposition:a:0" in args
    assert "default" in args
    assert "forced" in args


def test_disposition_zero_when_no_flags_set():
    media = _media(audio_streams=[_aud(1, default=False, forced=False)])
    plan = plan_streams(media, StreamMappingCfg())
    # The value after `-disposition:a:0` should be "0".
    idx = plan.disposition_args.index("-disposition:a:0")
    assert plan.disposition_args[idx + 1] == "0"


def test_audio_off_yields_no_audio_streams():
    media = _media(audio_streams=[_aud(1), _aud(2)])
    plan = plan_streams(media, StreamMappingCfg(copy_audio=False))
    assert plan.audio_streams == []
    assert "-map" in plan.map_args
    # only the video map remains
    assert plan.map_args.count("0:v:0") == 1


def test_chapters_flag_only_when_present_and_enabled():
    media_with = _media(chapters=[Chapter(id=1, start_time_s=0, end_time_s=10)])
    plan = plan_streams(media_with, StreamMappingCfg())
    assert plan.copy_chapters is True

    media_none = _media(chapters=[])
    plan2 = plan_streams(media_none, StreamMappingCfg())
    assert plan2.copy_chapters is False


def test_mkvpropedit_segment_title_carried_forward():
    media = _media(fmt_tags={"title": "Episode 01 — Beginnings"})
    plan = plan_streams(media, StreamMappingCfg())
    assert plan.mkvpropedit_segment_title == "Episode 01 — Beginnings"


def test_mkvpropedit_track_edits_for_audio_and_sub():
    media = _media(
        audio_streams=[_aud(1, lang="jpn", title="JP", default=True)],
        subtitle_streams=[_sub(2, lang="eng", title="Signs", forced=True)],
    )
    plan = plan_streams(media, StreamMappingCfg())
    selectors = [edit.selector for edit in plan.mkvpropedit_track_edits]
    assert "track:a1" in selectors
    assert "track:s1" in selectors
    # Look up the subtitle edit and confirm forced flag set.
    sub_edit = next(e for e in plan.mkvpropedit_track_edits if e.selector == "track:s1")
    pairs = dict(sub_edit.set_pairs)
    assert pairs.get("flag-forced") == "1"


def test_mp4_container_skips_propedit_block():
    media = _media(audio_streams=[_aud(1)])
    plan = plan_streams(media, StreamMappingCfg(), container="mp4")
    assert plan.mkvpropedit_segment_title is None
    assert plan.mkvpropedit_track_edits == []


# ----- decide_mux_tool ------------------------------------------------------


def test_mp4_always_uses_ffmpeg():
    media = _media(attachments=[_attachment(10)])
    decision = decide_mux_tool(media, StreamMappingCfg(), container="mp4")
    assert decision.tool == "ffmpeg"
    assert decision.needs_propedit_pass is False


def test_mkv_with_attachments_uses_mkvmerge():
    media = _media(attachments=[_attachment(10), _attachment(11)])
    decision = decide_mux_tool(media, StreamMappingCfg(), container="mkv")
    assert decision.tool == "mkvmerge"


def test_mkv_attachments_disabled_falls_back_to_ffmpeg():
    media = _media(attachments=[_attachment(10)])
    cfg = StreamMappingCfg(copy_attachments=False)
    decision = decide_mux_tool(media, cfg, container="mkv")
    assert decision.tool == "ffmpeg"


def test_mkv_with_chapters_uses_mkvmerge_when_multi():
    chaps = [Chapter(id=i, start_time_s=i * 60, end_time_s=(i + 1) * 60) for i in range(5)]
    media = _media(chapters=chaps)
    decision = decide_mux_tool(media, StreamMappingCfg(), container="mkv")
    assert decision.tool == "mkvmerge"


def test_default_mkv_path_uses_ffmpeg_with_propedit():
    media = _media(audio_streams=[_aud(1)])
    decision = decide_mux_tool(media, StreamMappingCfg(), container="mkv")
    assert decision.tool == "ffmpeg"
    assert decision.needs_propedit_pass is True
