"""Stream mapping engine.

Given a source `MediaInfo` and a `StreamMappingCfg` from the active preset, produce a
`StreamMappingPlan` that the ffmpeg-mux or mkvmerge-mux backends can execute.

The plan is intentionally explicit — for every audio/subtitle stream we emit:
  * a `-map 1:<absolute-index>` (pulling from the original-source input, which is the
    SECOND input to ffmpeg; the first input is our re-encoded video)
  * a `-c:<output-stream-spec>` (always `copy` for non-video — we never re-encode audio
    or subtitles implicitly)
  * `-disposition:<output-stream-spec>` to mirror the source disposition flags
  * `-metadata:s:<output-stream-spec>` for language and title tags

ffmpeg's stream specifier model:
  Inputs are numbered; within the *output*, streams are numbered separately. We use
  output-side specifiers (`-c:a:0`, `-disposition:s:1`) so they're stable regardless
  of how many streams we're mapping.

Why we still produce mkvpropedit fixups in the plan even when we use ffmpeg-mux:
  ffmpeg's mkv writer occasionally drops `track_name` or normalizes language codes from
  the source MediaInfo. We capture the canonical values here and let stage 09 run
  mkvpropedit as a fast post-mux corrective pass on Matroska outputs.

This module has NO side effects — no subprocess calls, no file I/O. That keeps it
trivially unit-testable from synthetic MediaInfo fixtures.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from aep.adapters.mkvtoolnix import TrackEdit
from aep.media.models import Disposition, MediaInfo, StreamInfo
from aep.persist.presets import StreamMappingCfg

log = logging.getLogger(__name__)


ContainerTarget = Literal["mkv", "mp4"]


# ---- Plan model ----------------------------------------------------------


@dataclass(frozen=True)
class MappedStream:
    """One stream's worth of ffmpeg arguments + accounting info."""

    source_index: int            # absolute index in the source file (input 1)
    out_kind: Literal["audio", "subtitle"]
    out_idx: int                 # 0-based index within its kind in the output
    codec_name: str | None       # source codec (informational; we always copy)
    language: str | None
    title: str | None
    disposition: Disposition

    def specifier(self) -> str:
        """ffmpeg output-side specifier, e.g. `a:0`, `s:1`."""
        return f"{'a' if self.out_kind == 'audio' else 's'}:{self.out_idx}"


@dataclass
class StreamMappingPlan:
    """The full output plan."""

    container: ContainerTarget

    # ffmpeg argv chunks ------------------------------------------------
    map_args: list[str] = field(default_factory=list)
    copy_args: list[str] = field(default_factory=list)
    metadata_args: list[str] = field(default_factory=list)
    disposition_args: list[str] = field(default_factory=list)

    # global flags -----------------------------------------------------
    copy_global_metadata: bool = True
    copy_chapters: bool = True

    # accounting -------------------------------------------------------
    audio_streams: list[MappedStream] = field(default_factory=list)
    subtitle_streams: list[MappedStream] = field(default_factory=list)
    skipped_streams: list[tuple[int, str]] = field(default_factory=list)  # (idx, reason)

    # post-ffmpeg corrections (only meaningful for mkv) -----------------
    mkvpropedit_segment_title: str | None = None
    mkvpropedit_track_edits: list[TrackEdit] = field(default_factory=list)

    # informational ----------------------------------------------------
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def all_ffmpeg_args(self) -> list[str]:
        """Convenience: flat ffmpeg argv chunk in the canonical order."""
        return [
            *self.map_args,
            *self.copy_args,
            *self.metadata_args,
            *self.disposition_args,
        ]


# ---- Helpers -------------------------------------------------------------


# ffmpeg `-disposition:<spec>` accepts a `+` / `-` flag list. We translate the
# Disposition model into the literal flag list the source had so we can roundtrip.
_DISPOSITION_FLAGS: list[tuple[str, str]] = [
    ("default", "default"),
    ("forced", "forced"),
    ("hearing_impaired", "hearing_impaired"),
    ("visual_impaired", "visual_impaired"),
    ("captions", "captions"),
    ("descriptions", "descriptions"),
    ("original", "original"),
    ("comment", "comment"),
    ("dub", "dub"),
]


def _disposition_to_ffmpeg(disp: Disposition) -> str:
    """Convert a Disposition into an ffmpeg `-disposition` value string.

    Returns a comma-less, plus-prefixed flag set (ffmpeg accepts `default+forced`).
    If no flags are set, returns `0` which clears any inherited dispositions on output.
    """
    on: list[str] = []
    for attr, flag in _DISPOSITION_FLAGS:
        if getattr(disp, attr, False):
            on.append(flag)
    return "+".join(on) if on else "0"


def _is_text_subtitle(stream: StreamInfo) -> bool:
    codec = (stream.codec_name or "").lower()
    return codec in {"ass", "ssa", "subrip", "srt", "webvtt", "mov_text"}


def _is_image_subtitle(stream: StreamInfo) -> bool:
    codec = (stream.codec_name or "").lower()
    return codec in {"hdmv_pgs_subtitle", "pgs", "dvd_subtitle", "dvb_subtitle"}


def _subtitle_compatible_with_container(stream: StreamInfo, container: ContainerTarget) -> bool:
    """MP4 doesn't carry ASS/SRT/PGS; only mov_text. MKV carries them all.

    Returning False causes the planner to skip the stream and surface a warning instead
    of producing an output that ffmpeg will reject mid-encode.
    """
    if container == "mkv":
        return True
    # mp4
    codec = (stream.codec_name or "").lower()
    return codec == "mov_text"


# ---- Public API ----------------------------------------------------------


def plan_streams(
    media: MediaInfo,
    cfg: StreamMappingCfg,
    *,
    container: ContainerTarget = "mkv",
    source_input_index: int = 1,
) -> StreamMappingPlan:
    """Build a complete StreamMappingPlan.

    The output's stream-0 is the encoded video (input 0). Audio and subtitle indices
    are renumbered starting at 0 within their kind, in the same order as the source.

    `source_input_index` is 1 in the standard pipeline (input 0 = encoded video,
    input 1 = original source) but is parameterized for tests / variant pipelines.
    """
    plan = StreamMappingPlan(container=container)

    # Always start by mapping the encoded video — input 0, video stream 0.
    plan.map_args += ["-map", "0:v:0"]
    plan.copy_args += ["-c:v", "copy"]
    plan.rationale.append("video: stream-copy from encoded artifact (input 0).")

    # ----- audio --------------------------------------------------------
    audio_idx_out = 0
    if cfg.copy_audio:
        for stream in media.audio_streams:
            mapped = MappedStream(
                source_index=stream.index,
                out_kind="audio",
                out_idx=audio_idx_out,
                codec_name=stream.codec_name,
                language=stream.language,
                title=stream.title,
                disposition=stream.disposition,
            )
            plan.map_args += ["-map", f"{source_input_index}:{stream.index}"]
            plan.copy_args += [f"-c:{mapped.specifier()}", "copy"]
            if cfg.copy_stream_metadata:
                if stream.language:
                    plan.metadata_args += [
                        f"-metadata:s:{mapped.specifier()}", f"language={stream.language}",
                    ]
                if stream.title:
                    plan.metadata_args += [
                        f"-metadata:s:{mapped.specifier()}", f"title={stream.title}",
                    ]
            if cfg.copy_dispositions:
                plan.disposition_args += [
                    f"-disposition:{mapped.specifier()}",
                    _disposition_to_ffmpeg(stream.disposition),
                ]
            plan.audio_streams.append(mapped)
            audio_idx_out += 1
        plan.rationale.append(f"audio: copied {audio_idx_out} streams from source.")
    else:
        plan.rationale.append("audio: skipped per StreamMappingCfg.copy_audio=False.")

    # ----- subtitles ----------------------------------------------------
    sub_idx_out = 0
    if cfg.copy_subtitles:
        for stream in media.subtitle_streams:
            if not _subtitle_compatible_with_container(stream, container):
                plan.skipped_streams.append((
                    stream.index,
                    f"codec {stream.codec_name!r} not supported in {container}",
                ))
                plan.warnings.append(
                    f"subtitle stream {stream.index} ({stream.codec_name}) skipped: "
                    f"container '{container}' cannot carry it. Use MKV to preserve."
                )
                continue
            mapped = MappedStream(
                source_index=stream.index,
                out_kind="subtitle",
                out_idx=sub_idx_out,
                codec_name=stream.codec_name,
                language=stream.language,
                title=stream.title,
                disposition=stream.disposition,
            )
            plan.map_args += ["-map", f"{source_input_index}:{stream.index}"]
            plan.copy_args += [f"-c:{mapped.specifier()}", "copy"]
            if cfg.copy_stream_metadata:
                if stream.language:
                    plan.metadata_args += [
                        f"-metadata:s:{mapped.specifier()}", f"language={stream.language}",
                    ]
                if stream.title:
                    plan.metadata_args += [
                        f"-metadata:s:{mapped.specifier()}", f"title={stream.title}",
                    ]
            if cfg.copy_dispositions:
                plan.disposition_args += [
                    f"-disposition:{mapped.specifier()}",
                    _disposition_to_ffmpeg(stream.disposition),
                ]
            plan.subtitle_streams.append(mapped)
            sub_idx_out += 1
        plan.rationale.append(f"subtitles: copied {sub_idx_out} streams from source.")
        if plan.skipped_streams:
            plan.rationale.append(
                f"subtitles: skipped {len(plan.skipped_streams)} due to container limits."
            )
    else:
        plan.rationale.append("subtitles: skipped per StreamMappingCfg.copy_subtitles=False.")

    # ----- chapters & global metadata ----------------------------------
    plan.copy_chapters = cfg.copy_chapters and bool(media.chapters)
    plan.copy_global_metadata = cfg.copy_global_metadata
    if cfg.copy_chapters and not media.chapters:
        plan.rationale.append("chapters: source has none; nothing to copy.")
    elif cfg.copy_chapters:
        plan.rationale.append(f"chapters: {len(media.chapters)} entries will be copied.")
    else:
        plan.rationale.append("chapters: skipped per StreamMappingCfg.copy_chapters=False.")

    if cfg.copy_global_metadata:
        plan.rationale.append("global metadata: -map_metadata from source enabled.")

    # Attachments (fonts etc.) are not handled by ffmpeg's `-map` reliably. The mux
    # backend (ffmpeg or mkvmerge) handles attachments separately. We just record the
    # intent here so the backend can act on it.
    if cfg.copy_attachments and media.attachments:
        plan.rationale.append(
            f"attachments: {len(media.attachments)} present; "
            "preserved by mux backend (mkvmerge or ffmpeg -map_attachments)."
        )

    # ----- mkvpropedit corrections ------------------------------------
    if container == "mkv":
        # Carry the segment title forward if present.
        seg_title = (media.fmt.tags or {}).get("title")
        if seg_title:
            plan.mkvpropedit_segment_title = seg_title

        # Per-track corrections — set track_name/language from canonical MediaInfo
        # so any stripping by ffmpeg's matroska muxer is corrected post-hoc.
        # Selectors use 1-based per-kind indices: track:v1, track:a1..., track:s1...
        for i, mapped in enumerate(plan.audio_streams, start=1):
            edits: list[tuple[str, str | None]] = []
            if cfg.copy_stream_metadata and mapped.language:
                edits.append(("language", mapped.language))
            if cfg.copy_stream_metadata and mapped.title:
                edits.append(("name", mapped.title))
            if cfg.copy_dispositions:
                edits.append(("flag-default", "1" if mapped.disposition.default else "0"))
                edits.append(("flag-forced", "1" if mapped.disposition.forced else "0"))
            if edits:
                plan.mkvpropedit_track_edits.append(
                    TrackEdit(selector=f"track:a{i}", set_pairs=edits)
                )
        for i, mapped in enumerate(plan.subtitle_streams, start=1):
            edits = []
            if cfg.copy_stream_metadata and mapped.language:
                edits.append(("language", mapped.language))
            if cfg.copy_stream_metadata and mapped.title:
                edits.append(("name", mapped.title))
            if cfg.copy_dispositions:
                edits.append(("flag-default", "1" if mapped.disposition.default else "0"))
                edits.append(("flag-forced", "1" if mapped.disposition.forced else "0"))
            if edits:
                plan.mkvpropedit_track_edits.append(
                    TrackEdit(selector=f"track:s{i}", set_pairs=edits)
                )

    return plan


# ---- Mux tool decision ---------------------------------------------------


@dataclass(frozen=True)
class MuxToolDecision:
    tool: Literal["ffmpeg", "mkvmerge"]
    reason: str
    needs_propedit_pass: bool


def decide_mux_tool(
    media: MediaInfo,
    cfg: StreamMappingCfg,
    *,
    container: ContainerTarget,
) -> MuxToolDecision:
    """Pick the mux tool.

    Heuristic (deterministic, conservative):
      1. If container is MP4 → ffmpeg always. mkvmerge only writes Matroska.
      2. If container is MKV and we're preserving attachments AND the source has at
         least one attachment → mkvmerge. ffmpeg does carry attachments via
         `-map 1:t?` but historically loses some metadata fields, so mkvmerge is
         safer for the spec promise of "preserve attachments wherever possible."
      3. If container is MKV and the source has chapter editions (multiple `<EditionEntry>`)
         → mkvmerge, which is the only tool that round-trips edition UIDs cleanly.
         (We approximate this with chapter_count > 0 + is_matroska, since our MediaInfo
         doesn't yet model edition entries — fidelity may be improved in a future release.)
      4. Otherwise ffmpeg + a mkvpropedit correction pass for metadata.
    """
    if container == "mp4":
        return MuxToolDecision(
            tool="ffmpeg",
            reason="container=mp4 → ffmpeg (mkvmerge writes only Matroska).",
            needs_propedit_pass=False,
        )

    if cfg.copy_attachments and media.attachments:
        return MuxToolDecision(
            tool="mkvmerge",
            reason=(
                f"source has {len(media.attachments)} attachment(s) and "
                "copy_attachments=True; mkvmerge preserves them more reliably."
            ),
            needs_propedit_pass=False,
        )

    if media.is_matroska and media.chapters and len(media.chapters) > 1:
        return MuxToolDecision(
            tool="mkvmerge",
            reason=(
                f"source is matroska with {len(media.chapters)} chapter entries; "
                "mkvmerge is the canonical chapter-edition writer."
            ),
            needs_propedit_pass=False,
        )

    # Default: ffmpeg + post-pass propedit for stream metadata.
    return MuxToolDecision(
        tool="ffmpeg",
        reason="standard pipeline: ffmpeg mux + mkvpropedit metadata fixup.",
        needs_propedit_pass=container == "mkv",
    )
