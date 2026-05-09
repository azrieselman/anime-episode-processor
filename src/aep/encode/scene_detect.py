"""Scene-cut detection helpers backed by PySceneDetect."""

from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SceneCut:
    """A single detected scene cut."""
    frame_index: int
    pts_time_s: float


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
