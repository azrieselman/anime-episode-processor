"""Scene-cut detection helpers (PySceneDetect or FFmpeg scdet)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

_SCDET_LINE_RE = re.compile(
    r"lavfi\.scd\.score:\s*([\d.]+),\s*lavfi\.scd\.time:\s*(.+?)\s*$",
)


@dataclass(frozen=True)
class SceneCut:
    """A single detected scene cut."""
    frame_index: int
    pts_time_s: float


def scdet_timestr_to_seconds(raw: str) -> float:
    """Parse ``lavfi.scd.time`` strings emitted by FFmpeg's scdet filter."""
    s = raw.strip()
    if not s or s.upper() == "N/A":
        raise ValueError(f"invalid scdet time: {raw!r}")
    if ":" not in s:
        return float(s)
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + sec
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    raise ValueError(f"unrecognized scdet time format: {raw!r}")


def parse_scdet_log(text: str) -> list[tuple[float, float]]:
    """Extract (score, time_seconds) for each scene cut line in FFmpeg log output."""
    out: list[tuple[float, float]] = []
    for line in text.splitlines():
        m = _SCDET_LINE_RE.search(line)
        if not m:
            continue
        score = float(m.group(1))
        t_s = scdet_timestr_to_seconds(m.group(2))
        out.append((score, t_s))
    return out


def detect_scene_cuts_ffmpeg_scdet(
    source_path: str | Path,
    *,
    ffmpeg_executable: str | Path,
    video_stream_index: int,
    threshold_percent: float,
    fps: Fraction,
    scale_width: int = 320,
    timeout: float = 3600.0,
) -> list[SceneCut]:
    """Run FFmpeg ``scdet`` on the given stream; return raw cut records.

    When ``scale_width`` > 0, frames are downscaled with ``scale=W:-1`` before
    ``scdet`` for faster analysis; timestamps map back via source frame rate.
    """
    from aep.adapters.base import env_with_tool_dirs
    from aep.util.proc import run_capture

    pct = max(0.0, min(100.0, float(threshold_percent)))
    w = int(scale_width)
    vf = f"scale={w}:-1,scdet=t={pct}" if w > 0 else f"scdet=t={pct}"
    cmd: list[str | Path] = [
        ffmpeg_executable,
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        Path(source_path),
        "-map",
        f"0:{int(video_stream_index)}",
        "-vf",
        vf,
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]
    result = run_capture(
        cmd,
        env=env_with_tool_dirs(),
        timeout=timeout,
        check=True,
    )
    combined = result.stderr + "\n" + result.stdout
    scored_times = parse_scdet_log(combined)
    fps_f = float(fps)
    cuts: list[SceneCut] = []
    for _score, t_s in scored_times:
        frame_idx = int(round(t_s * fps_f))
        if frame_idx > 0:
            cuts.append(SceneCut(frame_index=frame_idx, pts_time_s=t_s))
    cuts.sort(key=lambda c: c.frame_index)
    return cuts


def detect_scene_cuts(source_path: str, *, threshold: float, min_scene_len: int = 15) -> list[SceneCut]:
    """Run PySceneDetect `ContentDetector` and return raw cut records."""
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(source_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))
    manager.detect_scenes(video=video)
    scenes = manager.get_scene_list()
    # Scene list contains (start, end) pairs, where each start after the first is a cut.
    cuts: list[SceneCut] = []
    for start, _ in scenes[1:]:
        frame_idx = int(start.get_frames())
        if frame_idx <= 0:
            continue
        cuts.append(SceneCut(frame_index=frame_idx, pts_time_s=float(start.get_seconds())))
    cuts.sort(key=lambda c: c.frame_index)
    return cuts


def cuts_to_frame_indices(
    cuts: list[SceneCut],
    *,
    total_frames: int | None = None,
) -> list[int]:
    """Validate/normalize cut frame indices (sorted, deduped, in-range)."""
    seen: set[int] = set()
    out: list[int] = []
    for c in cuts:
        idx = c.frame_index
        if idx <= 0:
            continue
        if total_frames is not None and idx >= total_frames:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    out.sort()
    return out
