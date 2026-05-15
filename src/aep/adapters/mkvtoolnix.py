"""MKVToolNix adapter — mkvmerge / mkvpropedit / mkvinfo wrappers.

Why we ship MKVToolNix in addition to FFmpeg:

* `mkvmerge -J <file>` is the authoritative source for Matroska track UIDs, attachment
  metadata (filename, mime, size), chapter editions, and dispositions. ffprobe exposes
  some of this but with version-dependent gaps.

* `mkvmerge` (the mux tool, not the introspector) preserves attachment streams more
  reliably than ffmpeg in some FFmpeg versions, especially TrueType fonts with unusual
  metadata. It also lets us cherry-pick attachments and chapters without touching A/V.

* `mkvpropedit` lets us fix track titles, language tags, default/forced flags, and
  set the segment title without re-muxing. We use it as a last-mile cleanup tool in
  stage 09 to ensure metadata matches exactly what the source had (or what the user
  configured to override).

The adapter only wraps subprocess invocations; semantic decisions (which attachments to
preserve, which mux tool to use) live in `aep.mux`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aep.adapters.base import ToolAdapter
from aep.constants import BIN_MKVINFO, BIN_MKVMERGE, BIN_MKVPROPEDIT
from aep.errors import MuxError
from aep.util.proc import ProcError, run_capture

log = logging.getLogger(__name__)


_VERSION_RE = re.compile(r"v(\d+(?:\.\d+){1,2})")


# ============================================================================
# mkvmerge -J normalized models
# ============================================================================


@dataclass(frozen=True)
class MkvTrack:
    id: int
    type: str            # "video"|"audio"|"subtitles"
    codec: str           # mkvmerge codec ID (e.g. "AVC/H.264/MPEG-4p10")
    codec_id: str        # raw codec id (e.g. "V_MPEG4/ISO/AVC")
    language: str | None
    track_name: str | None
    default: bool
    forced: bool
    enabled: bool
    track_uid: int | None
    properties: dict[str, Any]


@dataclass(frozen=True)
class MkvAttachment:
    id: int
    file_name: str
    content_type: str | None
    description: str | None
    size: int
    uid: int | None


@dataclass(frozen=True)
class MkvChapterEntry:
    start_ns: int
    end_ns: int | None
    name: str | None


@dataclass(frozen=True)
class MkvIdentify:
    file_name: str
    container_type: str
    is_matroska: bool
    title: str | None
    duration_ns: int | None
    tracks: list[MkvTrack]
    attachments: list[MkvAttachment]
    chapter_count: int


# ============================================================================
# mkvmerge adapter
# ============================================================================


class MkvmergeAdapter(ToolAdapter):
    tool_id = "mkvmerge"
    bin_name = BIN_MKVMERGE
    tools_subdir = "mkvtoolnix"

    def _detect_version(self) -> str:
        result = run_capture([self.path, "--version"], timeout=15.0)
        m = _VERSION_RE.search(result.stdout + result.stderr)
        return m.group(1) if m else "unknown"

    # ----- introspection ----------------------------------------------

    def identify(self, source: Path) -> MkvIdentify:
        """Run `mkvmerge -J <file>` and parse into MkvIdentify."""
        result = run_capture(
            [self.path, "-J", str(source)],
            timeout=120.0,
            check=False,
        )
        if result.returncode not in (0, 1):
            # mkvmerge uses 0=ok, 1=warnings, 2=errors. We accept warnings.
            raise MuxError(
                f"mkvmerge -J failed (exit {result.returncode})",
                context={"stderr": result.stderr[:1000]},
            )
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MuxError("mkvmerge -J returned invalid JSON",
                           context={"reason": str(exc)}) from exc

        return _parse_mkvmerge_identify(data, source)

    # ----- mux ---------------------------------------------------------

    def build_mkv_mux(
        self,
        *,
        encoded_video: Path,
        source: Path,
        output: Path,
        copy_audio: bool = True,
        copy_subtitles: bool = True,
        copy_chapters: bool = True,
        copy_attachments: bool = True,
        copy_global_tags: bool = True,
        excluded_track_ids: list[int] | None = None,
    ) -> list[str | Path]:
        """Build an mkvmerge command that combines our encoded video with selected
        streams/attachments/chapters from the source.

        mkvmerge's track-selection model is the inverse of ffmpeg's: by default it
        includes everything; you exclude with `--no-audio`, `--no-subtitles`, etc., or
        use `--audio-tracks !1,2` syntax. We always pass explicit flags rather than
        relying on defaults so the resulting command is reviewable.

        Output structure:
            mkvmerge -o OUTPUT \
                --no-audio --no-subtitles --no-chapters --no-attachments --no-global-tags ENCODED \
                [--no-video] \
                [--no-audio | --audio-tracks ...] \
                [--no-subtitles | --subtitle-tracks ...] \
                [--no-chapters] \
                [--no-attachments] \
                [--no-global-tags] \
                SOURCE
        """
        excluded = excluded_track_ids or []
        cmd: list[str | Path] = [self.path, "-o", str(output)]

        # First file: our encoded video. Strip everything except video.
        cmd += [
            "--no-audio", "--no-subtitles", "--no-chapters",
            "--no-attachments", "--no-global-tags",
            str(encoded_video),
        ]

        # Second file: original source. Cherry-pick streams.
        cmd += ["--no-video"]
        if not copy_audio:
            cmd += ["--no-audio"]
        elif excluded:
            # mkvmerge accepts comma-separated track IDs in the source-file numbering;
            # !id syntax means "exclude these IDs."
            ex_aud = ",".join(f"!{i}" for i in excluded)
            if ex_aud:
                cmd += ["--audio-tracks", ex_aud]
        if not copy_subtitles:
            cmd += ["--no-subtitles"]
        if not copy_chapters:
            cmd += ["--no-chapters"]
        if not copy_attachments:
            cmd += ["--no-attachments"]
        if not copy_global_tags:
            cmd += ["--no-global-tags"]
        cmd += [str(source)]

        return cmd


# ============================================================================
# mkvpropedit adapter
# ============================================================================


class MkvpropeditAdapter(ToolAdapter):
    tool_id = "mkvpropedit"
    bin_name = BIN_MKVPROPEDIT
    tools_subdir = "mkvtoolnix"

    def _detect_version(self) -> str:
        result = run_capture([self.path, "--version"], timeout=15.0)
        m = _VERSION_RE.search(result.stdout + result.stderr)
        return m.group(1) if m else "unknown"

    def build_track_metadata_fix(
        self,
        *,
        target: Path,
        track_edits: list[TrackEdit],
        segment_title: str | None = None,
    ) -> list[str | Path]:
        """Build a mkvpropedit command that updates track metadata in place.

        Used in stage 09 to copy original track names/languages/dispositions onto the
        muxed output when ffmpeg's mux dropped or remapped them. mkvpropedit uses 1-based
        track selectors (track:1 = first video track, track:a1 = first audio, etc.).
        """
        cmd: list[str | Path] = [self.path, str(target)]
        if segment_title is not None:
            cmd += ["--edit", "info", "--set", f"title={segment_title}"]
        for edit in track_edits:
            cmd += ["--edit", edit.selector]
            for key, value in edit.set_pairs:
                if value is None:
                    cmd += ["--delete", key]
                else:
                    cmd += ["--set", f"{key}={value}"]
        return cmd

    def apply_edits(
        self,
        target: Path,
        track_edits: list[TrackEdit],
        *,
        segment_title: str | None = None,
    ) -> None:
        cmd = self.build_track_metadata_fix(
            target=target, track_edits=track_edits, segment_title=segment_title,
        )
        try:
            run_capture(cmd, timeout=120.0)
        except ProcError as exc:
            raise MuxError("mkvpropedit failed",
                           context={"stderr": exc.result.stderr[:1000]}) from exc


@dataclass(frozen=True)
class TrackEdit:
    """Selector + key/value list for a mkvpropedit edit block.

    selector examples:
        "track:v1"  — first video track
        "track:a2"  — second audio track
        "track:s1"  — first subtitle track
    """
    selector: str
    set_pairs: list[tuple[str, str | None]]   # value=None → delete


# ============================================================================
# mkvinfo (rarely needed, included for completeness / debugging)
# ============================================================================


class MkvinfoAdapter(ToolAdapter):
    tool_id = "mkvinfo"
    bin_name = BIN_MKVINFO
    tools_subdir = "mkvtoolnix"

    def _detect_version(self) -> str:
        result = run_capture([self.path, "--version"], timeout=15.0)
        m = _VERSION_RE.search(result.stdout + result.stderr)
        return m.group(1) if m else "unknown"


# ============================================================================
# Parser for mkvmerge -J JSON
# ============================================================================


def _parse_mkvmerge_identify(data: dict, source: Path) -> MkvIdentify:
    container = data.get("container", {}) or {}
    container_type = container.get("type", "")
    is_mkv = container_type.lower() in {"matroska", "webm"}
    properties = container.get("properties", {}) or {}
    duration_ns = properties.get("duration")
    title = properties.get("title")

    tracks: list[MkvTrack] = []
    for raw in data.get("tracks", []):
        props = raw.get("properties", {}) or {}
        tracks.append(MkvTrack(
            id=int(raw.get("id", -1)),
            type=str(raw.get("type", "")),
            codec=str(raw.get("codec", "")),
            codec_id=str(props.get("codec_id", "")),
            language=props.get("language"),
            track_name=props.get("track_name"),
            default=bool(props.get("default_track", False)),
            forced=bool(props.get("forced_track", False)),
            enabled=bool(props.get("enabled_track", True)),
            track_uid=int(props["uid"]) if "uid" in props else None,
            properties=props,
        ))

    attachments: list[MkvAttachment] = []
    for raw in data.get("attachments", []):
        attachments.append(MkvAttachment(
            id=int(raw.get("id", -1)),
            file_name=str(raw.get("file_name", "")),
            content_type=raw.get("content_type"),
            description=raw.get("description"),
            size=int(raw.get("size", 0)),
            uid=int(raw["properties"]["uid"]) if "properties" in raw and "uid" in raw["properties"] else None,
        ))

    chapter_count = 0
    for chap in data.get("chapters", []):
        chapter_count += int(chap.get("num_entries", 0))

    return MkvIdentify(
        file_name=str(data.get("file_name") or source),
        container_type=container_type,
        is_matroska=is_mkv,
        title=title,
        duration_ns=int(duration_ns) if duration_ns is not None else None,
        tracks=tracks,
        attachments=attachments,
        chapter_count=chapter_count,
    )
