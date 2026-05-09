"""Batch planner — splits a source into time-aligned chunks for the pipeline.

A `BatchSpec` is the contract between the planner (s01_plan) and the per-batch
stages (04_decode_serve … 08_encode). Each batch declares:

  * `index`: stable ordinal, used for `batch_<NN>` directory naming.
  * `start_pts` / `end_pts`: presentation timestamps (seconds) covering this
    batch. `start_pts` is inclusive, `end_pts` is exclusive — the next batch
    picks up at the previous batch's `end_pts`. The final batch's `end_pts`
    equals the source duration (i.e. covers the tail completely).
  * `frame_count_estimate`: how many *output* frames the encode stage will see
    after interpolation. Used for tqdm-style progress and ramdisk sizing.
  * `est_bytes`: peak intermediate frame storage in bytes. Drives the
    pre-batch RAM-disk free-space gate in stage 04.

The planner itself does not seek or decode — it only computes timestamps. The
decode-serve stage applies the boundaries via `-ss` / `-to`.

Boundary policy:

    keyframe    snap each boundary to the nearest source keyframe ≤ target.
                The first batch always starts at 0.0. Last batch's end is the
                source duration regardless of policy.
    exact       use exact `chunk_seconds` multiples; the decode stage pays a
                re-decode-from-prior-keyframe penalty per chunk but boundaries
                are deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)


BoundaryPolicy = Literal["keyframe", "exact"]


@dataclass(frozen=True)
class BatchSpec:
    """One pipeline chunk. See module docstring for field semantics."""
    index: int
    start_pts: float
    end_pts: float
    frame_count_estimate: int
    est_bytes: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_pts - self.start_pts)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form for plan.json / cache key inputs."""
        return {
            "index": self.index,
            "start_pts": self.start_pts,
            "end_pts": self.end_pts,
            "frame_count_estimate": self.frame_count_estimate,
            "est_bytes": self.est_bytes,
        }


def plan_batches(
    *,
    duration_s: float,
    chunk_seconds: int,
    boundary_policy: BoundaryPolicy = "keyframe",
    keyframes: list[float] | None = None,
    output_fps: float | None = None,
    bytes_per_output_frame: int = 0,
) -> list[BatchSpec]:
    """Compute the batch layout for a source.

    Parameters
    ----------
    duration_s
        Total source duration in seconds. Must be > 0.
    chunk_seconds
        Target chunk length. Real chunks may be slightly longer or shorter
        once boundaries snap to keyframes.
    boundary_policy
        How to round target boundaries. See module docstring.
    keyframes
        Sorted ascending list of source keyframe times. Required when
        `boundary_policy="keyframe"`. Ignored otherwise. The first keyframe
        is assumed to be at 0.0; if not present we behave as if it were.
    output_fps
        Output frame rate after interpolation, used for `frame_count_estimate`.
        Pass None when unknown — frame_count_estimate falls back to 0.
    bytes_per_output_frame
        Per-frame byte budget (output geometry × bytes-per-pixel). Used for
        `est_bytes`. Pass 0 when unknown.

    Returns
    -------
    list[BatchSpec]
        At least one batch covering the whole source. The list is sorted
        ascending by `index` and contiguous: batch[i].end_pts == batch[i+1].start_pts.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
    if chunk_seconds <= 0:
        raise ValueError(f"chunk_seconds must be > 0, got {chunk_seconds!r}")

    if boundary_policy == "keyframe":
        boundaries = _keyframe_boundaries(
            duration_s=duration_s,
            chunk_seconds=float(chunk_seconds),
            keyframes=keyframes or [],
        )
    elif boundary_policy == "exact":
        boundaries = _exact_boundaries(
            duration_s=duration_s,
            chunk_seconds=float(chunk_seconds),
        )
    else:  # pragma: no cover — Literal type guards this at the public surface
        raise ValueError(f"unknown boundary_policy: {boundary_policy!r}")

    # Convert sorted boundaries into BatchSpec list. Boundaries are the
    # *interior* split points — start (0.0) and end (duration_s) are added
    # implicitly here so the planner returns N+1 batches for N split points.
    edges = [0.0, *boundaries, duration_s]
    # Defensive de-dup: keyframe snap may produce 0.0 duplicates if the first
    # interior boundary is itself ≈ 0.0, and exact mode produces duration_s
    # as both the last interior and the implicit tail. dict.fromkeys preserves
    # insertion order while removing dupes.
    deduped = list(dict.fromkeys(round(e, 6) for e in edges))
    deduped.sort()
    if deduped[0] != 0.0:
        deduped.insert(0, 0.0)
    if deduped[-1] < duration_s:
        deduped.append(duration_s)

    batches: list[BatchSpec] = []
    for i in range(len(deduped) - 1):
        start = deduped[i]
        end = deduped[i + 1]
        if end <= start:
            # Skip degenerate zero-length intervals (can happen if a keyframe
            # lands exactly on a target boundary).
            continue
        dur = end - start
        if output_fps is not None and output_fps > 0:
            frame_count = int(round(dur * float(output_fps)))
        else:
            frame_count = 0
        est_bytes = (
            frame_count * int(bytes_per_output_frame)
            if bytes_per_output_frame > 0 else 0
        )
        batches.append(BatchSpec(
            index=len(batches),
            start_pts=start,
            end_pts=end,
            frame_count_estimate=frame_count,
            est_bytes=est_bytes,
        ))
    if not batches:
        # Source shorter than one chunk — emit a single batch covering it all.
        frame_count = (
            int(round(duration_s * float(output_fps)))
            if output_fps and output_fps > 0 else 0
        )
        est_bytes = (
            frame_count * int(bytes_per_output_frame)
            if bytes_per_output_frame > 0 else 0
        )
        batches.append(BatchSpec(
            index=0,
            start_pts=0.0,
            end_pts=duration_s,
            frame_count_estimate=frame_count,
            est_bytes=est_bytes,
        ))
    return batches


# ---------- internal boundary computers ------------------------------------


def _exact_boundaries(*, duration_s: float, chunk_seconds: float) -> list[float]:
    """Return interior split points at exact chunk_seconds multiples.

    Excludes 0.0 and duration_s (those are implicit edges).
    """
    boundaries: list[float] = []
    t = chunk_seconds
    while t < duration_s:
        boundaries.append(t)
        t += chunk_seconds
    return boundaries


def _keyframe_boundaries(
    *,
    duration_s: float,
    chunk_seconds: float,
    keyframes: list[float],
) -> list[float]:
    """For each target time, pick the latest keyframe ≤ target.

    Why "≤ target" rather than "nearest"?
        Decode-serve uses `-ss <keyframe>` to seek. If we picked a keyframe
        *after* the target we'd skip frames; if we picked the nearest one
        we might extend a chunk past 2× chunk_seconds when keyframes are
        sparse. "Latest ≤ target" is the only choice that preserves both
        coverage and bounded chunk size.

    If keyframes is empty (or no keyframe ≤ target exists), the boundary
    falls back to the exact target time. The decode stage will still seek
    correctly — it just pays the re-decode penalty for that chunk.

    The first batch always starts at 0.0 even if there's no keyframe there;
    this is virtually always true in practice (every clean container has a
    keyframe at the first packet) but we don't rely on it.
    """
    if not keyframes:
        return _exact_boundaries(
            duration_s=duration_s, chunk_seconds=chunk_seconds,
        )

    boundaries: list[float] = []
    target = chunk_seconds
    last_picked = 0.0
    while target < duration_s:
        # Find the latest keyframe ≤ target. Linear scan is fine because
        # `keyframes` is small (a 24-min episode at GOP=120 has ~720 entries)
        # and this function runs once per job.
        best = None
        for kf in keyframes:
            if kf > target:
                break
            best = kf
        if best is None or best <= last_picked:
            # No keyframe ≤ target, OR the only candidate is one we've already
            # used (would create a zero-length batch). Fall back to the exact
            # target time — decode will deal with it.
            picked = target
        else:
            picked = best
        boundaries.append(picked)
        last_picked = picked
        target = picked + chunk_seconds  # advance from the *picked* boundary
        # so chunks stay ≥ chunk_seconds long even if a keyframe landed
        # well before target.
    return boundaries
