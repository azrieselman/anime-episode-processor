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
        timeout: float = 300.0,
    ) -> list[float]:
        """Return sorted ascending list of keyframe presentation times (seconds).

        Used by the batch planner to snap batch boundaries to source
        keyframes — decoder seek to a keyframe is free, anything else
        triggers a re-decode of the preceding GOP.

        Implementation notes:
          * `-skip_frame nokey` filters non-keyframes at the demuxer layer,
            so this is much cheaper than `-show_frames` on the full stream.
          * `pkt_pts_time` is preferred when present; we fall back to
            `pts_time` (older ffprobe builds) and then `best_effort_timestamp_time`.
          * Streams with no decodable timestamps (rare; some MPEG-TS
            captures) come back as an empty list — callers fall back to
            time-based boundaries.
          * Output is parsed as plain key=value lines (`-of default`)
            because JSON output for `-show_packets` balloons memory on
            long sources where every keyframe is a separate object.
        """
        cmd: list[str | Path] = [
            self.path,
            "-v", "error",
            "-hide_banner",
            "-skip_frame", "nokey",
            "-select_streams", f"v:{int(stream_index)}",
            "-show_entries", "frame=pkt_pts_time,pts_time,best_effort_timestamp_time,key_frame",
            "-of", "default=nw=1:nk=0",
            str(source),
        ]
        result = run_capture(cmd, env=env_with_tool_dirs(), timeout=timeout)
        kfs: list[float] = []
        # Parse "key=value" lines. We accumulate fields per frame and emit a
        # timestamp when we hit a frame boundary (next key_frame= line, or EOF).
        cur: dict[str, str] = {}

        def _emit(d: dict[str, str]) -> None:
            if d.get("key_frame") not in {"1", "true", "True"}:
                # Belt-and-suspenders: -skip_frame nokey already filtered, but
                # some builds still emit non-keyframes if metadata claims kf=0.
                return
            for k in ("pkt_pts_time", "pts_time", "best_effort_timestamp_time"):
                if k in d and d[k] not in ("", "N/A"):
                    try:
                        kfs.append(float(d[k]))
                        return
                    except ValueError:
                        continue

        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            # ffprobe emits one [FRAME] block at a time; key_frame is the
            # first field per frame in this output mode. When we see it
            # again, the prior frame is complete.
            if k == "key_frame" and cur:
                _emit(cur)
                cur = {}
            cur[k] = v
        if cur:
            _emit(cur)

        kfs.sort()
        return kfs

    def probe_full_json(self, source: Path) -> str:
        """Return ffprobe JSON for format + streams + chapters.

        We deliberately do NOT use `-show_data` (way too verbose) or `-show_packets`
        (wrong tool for the job). For attachments, we collect what ffprobe emits;
        mkvmerge -J is run separately when authoritative attachment metadata is needed.
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
