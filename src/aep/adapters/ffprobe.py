"""ffprobe adapter — focused on probing/inspection.

Encoding, decoding, and frame streaming are owned by the ffmpeg adapter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.base import ToolAdapter, env_with_tool_dirs
from aep.constants import BIN_FFPROBE
from aep.util.proc import run_capture

_VERSION_RE = re.compile(r"ffprobe version (\S+)")


def keyframe_probe_timeout(duration_s: float | None) -> float:
    """Wall-clock budget for demux-only keyframe enumeration."""
    if duration_s is None or duration_s <= 0:
        return 300.0
    return max(300.0, duration_s * 0.5)


def _packet_pts_seconds(fields: dict[str, str]) -> float | None:
    for key in ("pkt_pts_time", "pts_time", "best_effort_timestamp_time"):
        val = fields.get(key)
        if val and val not in ("", "N/A"):
            try:
                return float(val)
            except ValueError:
                continue
    return None


def _packet_duration_seconds(fields: dict[str, str]) -> float | None:
    for key in ("pkt_duration_time", "duration_time"):
        val = fields.get(key)
        if val and val not in ("", "N/A"):
            try:
                d = float(val)
            except ValueError:
                continue
            if d > 0:
                return d
    return None


@dataclass(frozen=True)
class VideoPacketExtent:
    """Timeline bounds inferred from demuxed video packets."""

    last_pts_s: float | None
    last_end_s: float | None
    keyframe_times_s: list[float]


def parse_video_packets_compact(stdout: str) -> VideoPacketExtent:
    """Parse ``-of compact=p=0`` packet lines into keyframes and timeline bounds."""
    keyframes: list[float] = []
    last_pts: float | None = None
    last_end: float | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields: dict[str, str] = {}
        for part in line.split("|"):
            if "=" not in part:
                continue
            k, _, v = part.partition("=")
            fields[k.strip()] = v.strip()
        pts = _packet_pts_seconds(fields)
        if pts is None:
            continue
        dur = _packet_duration_seconds(fields)
        end = pts + dur if dur is not None else pts
        last_pts = pts if last_pts is None else max(last_pts, pts)
        last_end = end if last_end is None else max(last_end, end)
        if "K" in fields.get("flags", ""):
            keyframes.append(pts)
    keyframes.sort()
    return VideoPacketExtent(
        last_pts_s=last_pts,
        last_end_s=last_end,
        keyframe_times_s=keyframes,
    )


def parse_keyframe_packets_compact(stdout: str) -> list[float]:
    """Parse ``-of compact=p=0`` packet lines; return sorted keyframe PTS (seconds)."""
    return parse_video_packets_compact(stdout).keyframe_times_s


class FFProbeAdapter(ToolAdapter):
    tool_id = "ffprobe"
    bin_name = BIN_FFPROBE
    tools_subdir = "ffmpeg"

    def _detect_version(self) -> str:
        result = run_capture([self.path, "-version"], timeout=15.0)
        m = _VERSION_RE.search(result.stdout)
        if not m:
            return "unknown"
        return m.group(1)

    def _video_packet_probe_cmd(
        self,
        source: Path,
        *,
        stream_index: int = 0,
    ) -> list[str | Path]:
        return [
            self.path,
            "-v", "error",
            "-hide_banner",
            "-select_streams", f"v:{int(stream_index)}",
            "-show_entries",
            "packet=pts_time,pkt_pts_time,best_effort_timestamp_time,"
            "pkt_duration_time,duration_time,flags",
            "-of", "compact=p=0",
            str(source),
        ]

    def probe_video_packet_extent(
        self,
        source: Path,
        *,
        stream_index: int = 0,
        duration_s: float | None = None,
        timeout: float | None = None,
    ) -> VideoPacketExtent:
        """Demux the primary video stream and infer decodable timeline bounds.

        Container and stream metadata can claim a longer duration than the last
        real video packet (common in bad anime encodes). This scan stops at EOF
        for packets and therefore reflects what decode can actually read.
        """
        effective_timeout = (
            timeout if timeout is not None else keyframe_probe_timeout(duration_s)
        )
        result = run_capture(
            self._video_packet_probe_cmd(source, stream_index=stream_index),
            env=env_with_tool_dirs(),
            timeout=effective_timeout,
        )
        return parse_video_packets_compact(result.stdout)

    def list_video_keyframes(
        self,
        source: Path,
        *,
        stream_index: int = 0,
        duration_s: float | None = None,
        timeout: float | None = None,
    ) -> list[float]:
        """Return sorted ascending list of keyframe presentation times (seconds).

        Used by the batch planner to snap batch boundaries to source
        keyframes — decoder seek to a keyframe is free, anything else
        triggers a re-decode of the preceding GOP.

        Implementation notes:
          * Demux-only ``-show_entries packet=`` with ``flags`` containing ``K``
            avoids decoding every frame (much faster than frame-level probes on
            long sources).
          * ``pkt_pts_time`` is preferred when present; we fall back to
            ``pts_time`` and then ``best_effort_timestamp_time``.
          * Compact output keeps memory bounded vs JSON ``-show_packets``.
          * Timeout defaults to ``max(300, duration_s * 0.5)`` when
            ``duration_s`` is known.
        """
        extent = self.probe_video_packet_extent(
            source,
            stream_index=stream_index,
            duration_s=duration_s,
            timeout=timeout,
        )
        return extent.keyframe_times_s

    def probe_full_json(self, source: Path) -> str:
        """Return ffprobe JSON for format + streams + chapters.

        We deliberately do NOT use ``-show_data`` (way too verbose). Packet dumps
        are only used by :meth:`probe_video_packet_extent` with compact output.
        For attachments, we collect what ffprobe emits; mkvmerge -J is run
        separately when authoritative attachment metadata is needed.
        """
        cmd: list[str | Path] = [
            self.path,
            "-v", "error",
            "-hide_banner",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(source),
        ]
        result = run_capture(cmd, env=env_with_tool_dirs(), timeout=120.0)
        return result.stdout
