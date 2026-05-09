"""Tests for the M6.5 batch planner (aep.pipeline.batches.plan_batches).

The planner has two boundary policies: keyframe (default) and exact. Both
must produce a contiguous list covering the full source. The exhaustive
edge cases live here so the runner-side tests can stay focused on iteration.
"""

from __future__ import annotations

import pytest

from aep.pipeline.batches import BatchSpec, plan_batches

# ---------- exact-boundary policy ------------------------------------------


def test_exact_simple_split() -> None:
    batches = plan_batches(
        duration_s=90.0, chunk_seconds=30, boundary_policy="exact",
    )
    assert [(b.start_pts, b.end_pts) for b in batches] == [
        (0.0, 30.0), (30.0, 60.0), (60.0, 90.0),
    ]
    assert [b.index for b in batches] == [0, 1, 2]


def test_exact_last_chunk_absorbs_remainder() -> None:
    # 95 s with 30 s chunks → [0..30, 30..60, 60..90, 90..95]
    batches = plan_batches(
        duration_s=95.0, chunk_seconds=30, boundary_policy="exact",
    )
    assert len(batches) == 4
    assert batches[-1].start_pts == 90.0
    assert batches[-1].end_pts == 95.0
    assert batches[-1].duration_s == pytest.approx(5.0)


def test_exact_source_shorter_than_one_chunk_emits_single_batch() -> None:
    batches = plan_batches(
        duration_s=12.5, chunk_seconds=30, boundary_policy="exact",
    )
    assert len(batches) == 1
    assert batches[0].start_pts == 0.0
    assert batches[0].end_pts == 12.5


def test_exact_contiguous_no_gaps_or_overlaps() -> None:
    batches = plan_batches(
        duration_s=200.0, chunk_seconds=45, boundary_policy="exact",
    )
    # Every batch's end equals the next batch's start.
    for a, b in zip(batches, batches[1:], strict=False):
        assert a.end_pts == b.start_pts
    assert batches[0].start_pts == 0.0
    assert batches[-1].end_pts == 200.0


# ---------- keyframe-snap policy -------------------------------------------


def test_keyframe_snap_picks_latest_kf_le_target() -> None:
    # Target=30  → latest kf ≤ 30 is 28. Advance from 28: target=58 → latest ≤ 58 is 36.
    # 36+30=66 → 60.  60+30=90 → 84.  84+30=114 → 100.  100+30=130 > 120, stop.
    batches = plan_batches(
        duration_s=120.0,
        chunk_seconds=30,
        boundary_policy="keyframe",
        keyframes=[0.0, 12.0, 24.0, 28.0, 36.0, 60.0, 84.0, 100.0],
    )
    starts_ends = [(b.start_pts, b.end_pts) for b in batches]
    assert starts_ends == [
        (0.0, 28.0),
        (28.0, 36.0),
        (36.0, 60.0),
        (60.0, 84.0),
        (84.0, 100.0),
        (100.0, 120.0),
    ]


def test_keyframe_empty_list_falls_back_to_exact() -> None:
    batches = plan_batches(
        duration_s=90.0,
        chunk_seconds=30,
        boundary_policy="keyframe",
        keyframes=[],
    )
    assert [(b.start_pts, b.end_pts) for b in batches] == [
        (0.0, 30.0), (30.0, 60.0), (60.0, 90.0),
    ]


def test_keyframe_sparse_keyframes_dont_advance_past_themselves() -> None:
    # Only one keyframe at 0.0 — every target should fall back to the exact
    # value (no zero-length batches, no infinite loops).
    batches = plan_batches(
        duration_s=90.0,
        chunk_seconds=30,
        boundary_policy="keyframe",
        keyframes=[0.0],
    )
    # Falls through to exact-style boundaries.
    assert [(b.start_pts, b.end_pts) for b in batches] == [
        (0.0, 30.0), (30.0, 60.0), (60.0, 90.0),
    ]


def test_keyframe_clusters_at_start_dont_collapse_chunks() -> None:
    # Several keyframes near the start; planner must still advance by
    # chunk_seconds from the picked boundary, not from the original target.
    batches = plan_batches(
        duration_s=120.0,
        chunk_seconds=30,
        boundary_policy="keyframe",
        keyframes=[0.0, 1.0, 2.0, 5.0, 10.0, 50.0, 80.0, 110.0],
    )
    # First target=30 → latest≤30 is 10. Next = 10+30=40 → latest≤40 is 10
    # again, but that would be a duplicate / zero-length, so we fall back
    # to the exact target (40). Next = 40+30=70 → latest≤70 is 50.
    # Then 50+30=80 → latest≤80 is 80. Then 80+30=110 → latest≤110 is 110.
    starts_ends = [(b.start_pts, b.end_pts) for b in batches]
    assert starts_ends[0] == (0.0, 10.0)
    # Every batch is non-empty:
    for b in batches:
        assert b.duration_s > 0
    # Coverage:
    assert batches[0].start_pts == 0.0
    assert batches[-1].end_pts == 120.0


# ---------- frame_count + est_bytes derivation -----------------------------


def test_frame_count_uses_output_fps_and_chunk_duration() -> None:
    batches = plan_batches(
        duration_s=60.0,
        chunk_seconds=30,
        boundary_policy="exact",
        output_fps=60.0,
    )
    # 30 s × 60 fps = 1800 frames per batch.
    assert all(b.frame_count_estimate == 1800 for b in batches)


def test_est_bytes_scales_with_frame_count() -> None:
    bytes_per_frame = 1920 * 1080 * 4  # 8.3 MB
    batches = plan_batches(
        duration_s=60.0,
        chunk_seconds=30,
        boundary_policy="exact",
        output_fps=60.0,
        bytes_per_output_frame=bytes_per_frame,
    )
    assert all(b.est_bytes == 1800 * bytes_per_frame for b in batches)


def test_unknown_fps_yields_zero_estimates() -> None:
    batches = plan_batches(
        duration_s=60.0, chunk_seconds=30, boundary_policy="exact",
    )
    assert all(b.frame_count_estimate == 0 for b in batches)
    assert all(b.est_bytes == 0 for b in batches)


# ---------- error handling -------------------------------------------------


def test_zero_duration_raises() -> None:
    with pytest.raises(ValueError, match="duration_s must be > 0"):
        plan_batches(duration_s=0.0, chunk_seconds=30)


def test_zero_chunk_raises() -> None:
    with pytest.raises(ValueError, match="chunk_seconds must be > 0"):
        plan_batches(duration_s=60.0, chunk_seconds=0)


# ---------- to_dict round-trip --------------------------------------------


def test_batch_spec_to_dict_is_json_safe() -> None:
    import json
    b = BatchSpec(index=2, start_pts=12.5, end_pts=42.5, frame_count_estimate=1800, est_bytes=99)
    payload = json.dumps(b.to_dict())
    rt = json.loads(payload)
    assert rt == {
        "index": 2,
        "start_pts": 12.5,
        "end_pts": 42.5,
        "frame_count_estimate": 1800,
        "est_bytes": 99,
    }
