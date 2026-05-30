"""ffprobe adapter — focused on probing/inspection.

Encoding, decoding, and frame streaming are owned by the ffmpeg adapter.
"""

from __future__ import annotations

import re
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


def parse_keyframe_packets_compact(stdout: str) -> list[float]:
    """Parse ``-of compact=p=0`` packet lines; return sorted keyframe PTS (seconds)."""
    kfs: list[float] = []
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
        flags = fields.get("flags", "")
        if "K" not in flags:
            continue
        for key in ("pkt_pts_time", "pts_time", "best_effort_timestamp_time"):
            val = fields.get(key)
            if val and val not in ("", "N/A"):
                try:
                    kfs.append(float(val))
                    break
                except ValueError:
                    continue
    kfs.sort()
    return kfs


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
        effective_timeout = (
            timeout if timeout is not None else keyframe_probe_timeout(duration_s)
        )
        cmd: list[str | Path] = [
            self.path,
            "-v", "error",
            "-hide_banner",
            "-select_streams", f"v:{int(stream_index)}",
            "-show_entries", "packet=pts_time,pkt_pts_time,best_effort_timestamp_time,flags",
            "-of", "compact=p=0",
            str(source),
        ]
        result = run_capture(cmd, env=env_with_tool_dirs(), timeout=effective_timeout)
        return parse_keyframe_packets_compact(result.stdout)

    def probe_full_json(self, source: Path) -> str:
        """Return ffprobe JSON for format + streams + chapters.

        We deliberately do NOT use ``-show_data`` (way too verbose). Packet dumps
        are only used by :meth:`list_video_keyframes` with compact output. For
        attachments, we collect what ffprobe emits; mkvmerge -J is run separately
        when authoritative attachment metadata is needed.
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
