"""Normalized media-info dataclasses.

ffprobe's JSON is ergonomically awful (string types, optional keys, mixed conventions).
We parse once into these typed models and pass them around the rest of the app. This is
also the model the GUI's stream inspector binds to.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

StreamKind = Literal["video", "audio", "subtitle", "attachment", "data", "unknown"]


class Disposition(BaseModel):
    default: bool = False
    forced: bool = False
    attached_pic: bool = False
    hearing_impaired: bool = False
    visual_impaired: bool = False
    captions: bool = False
    descriptions: bool = False
    original: bool = False
    comment: bool = False
    dub: bool = False


class StreamInfo(BaseModel):
    index: int
    kind: StreamKind
    codec_name: str | None = None
    codec_long_name: str | None = None
    profile: str | None = None
    language: str | None = None
    title: str | None = None
    disposition: Disposition = Field(default_factory=Disposition)
    bit_rate: int | None = None
    duration_s: float | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    # video-specific
    width: int | None = None
    height: int | None = None
    pix_fmt: str | None = None
    color_space: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_range: str | None = None
    field_order: str | None = None
    avg_frame_rate: str | None = None       # "24000/1001"
    r_frame_rate: str | None = None
    nb_frames: int | None = None
    sar: str | None = None                  # "1:1"
    dar: str | None = None                  # "16:9"
    bits_per_raw_sample: int | None = None

    # audio-specific
    sample_rate: int | None = None
    channels: int | None = None
    channel_layout: str | None = None

    # attachment-specific (fonts etc.)
    filename: str | None = None
    mimetype: str | None = None


class Chapter(BaseModel):
    id: int
    start_time_s: float
    end_time_s: float
    title: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class FormatInfo(BaseModel):
    filename: str
    format_name: str
    format_long_name: str | None = None
    duration_s: float | None = None
    bit_rate: int | None = None
    size_bytes: int | None = None
    tags: dict[str, str] = Field(default_factory=dict)


class MediaInfo(BaseModel):
    """Top-level analysis result for a single file."""
    source_path: str
    fmt: FormatInfo
    streams: list[StreamInfo]
    chapters: list[Chapter] = Field(default_factory=list)
    attachments: list[StreamInfo] = Field(default_factory=list)  # attachment streams
    is_matroska: bool = False
    likely_anime: bool | None = None
    notes: list[str] = Field(default_factory=list)

    # Convenience views
    @property
    def video_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.kind == "video"]

    @property
    def audio_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.kind == "audio"]

    @property
    def subtitle_streams(self) -> list[StreamInfo]:
        return [s for s in self.streams if s.kind == "subtitle"]

    @property
    def primary_video(self) -> StreamInfo | None:
        return self.video_streams[0] if self.video_streams else None
