"""Tests for batch frame accounting (exclusive PTS windows, overlap, trim)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest

from aep.adapters.rife import local_cuts_from_global
from aep.pipeline.batch_timing import (
    BatchFramePlan,
    batch_frame_plan_for_actual_decode,
    batch_frame_plan_from_stored_decode,
    content_frame_offset_for_pts,
    count_numeric_frames_in_dir,
    drop_rife_output_prefix,
    expected_content_frames,
    expected_segment_duration_s,
    merge_batch_frame_plan_into_decode,
    reconcile_batch_decode_outputs,
    resolve_batch_frame_plan,
    trim_frames_dir,
)
from aep.pipeline.context import PipelineContext
from aep.util.fps import total_frames


def test_expected_content_frames_ntsc_twenty_seconds() -> None:
    fps = Fraction(24000, 1001)
    # 20 s × 24000/1001 ≈ 479.52 → 480 frames (half-up via total_frames).
    assert expected_content_frames(
        start_pts=0.0,
        end_pts=20.0,
        source_fps=fps,
    ) == total_frames(fps, 20.0)


def test_expected_content_frames_uses_cumulative_indices() -> None:
    fps = Fraction(24000, 1001)
    start, end = 715.784609916052, 872.9688029041514
    by_duration = total_frames(fps, end - start)
    by_indices = expected_content_frames(
        start_pts=start, end_pts=end, source_fps=fps,
    )
    assert by_indices == total_frames(fps, end) - total_frames(fps, start)
    assert by_duration - by_indices == 1


def test_expected_content_frames_capped_by_source_total() -> None:
    fps = Fraction(24000, 1001)
    start_idx = total_frames(fps, 1400.0)
    assert expected_content_frames(
        start_pts=1400.0,
        end_pts=1500.0,
        source_fps=fps,
        source_total_frames=start_idx + 10,
    ) == 10


def test_content_frame_offset_at_zero() -> None:
    fps = Fraction(24000, 1001)
    assert content_frame_offset_for_pts(0.0, fps) == 0
    assert content_frame_offset_for_pts(1.0, fps) == total_frames(fps, 1.0)


def test_batch_frame_plan_overlap_decode_start() -> None:
    fps = Fraction(24000, 1001)
    plan = BatchFramePlan(
        start_pts=20.0,
        end_pts=40.0,
        source_fps=fps,
        overlap_source_frames=1,
        content_frame_offset=total_frames(fps, 20.0),
        expected_content_frames=expected_content_frames(
            start_pts=20.0, end_pts=40.0, source_fps=fps,
        ),
        rife_input_base=total_frames(fps, 20.0) - 1,
        rife_input_count=1 + expected_content_frames(
            start_pts=20.0, end_pts=40.0, source_fps=fps,
        ),
        rife_output_skip=2,
        expected_output_frames=expected_content_frames(
            start_pts=20.0, end_pts=40.0, source_fps=fps,
        ) * 2,
    )
    assert plan.decode_start_pts < 20.0
    assert plan.expected_decode_frames == 1 + plan.expected_content_frames


def test_local_cuts_with_overlap_maps_boundary_cut_to_local_two() -> None:
    # Content starts at global frame 480; overlap frame is 479. Scene cut at 480
    # is the first content frame → local index 2 when rife_input_base=479.
    out = local_cuts_from_global(
        [480],
        rife_input_base=479,
        in_count=481,
    )
    assert out == [2]


def test_trim_frames_dir_removes_tail(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 6):
        (frames / f"{i:08d}.png").write_bytes(b"x")
    removed = trim_frames_dir(frames, frame_format="png", keep_count=3)
    assert removed == 2
    assert sorted(p.name for p in frames.iterdir()) == [
        "00000001.png", "00000002.png", "00000003.png",
    ]


def test_drop_rife_output_prefix_renumbers(tmp_path: Path) -> None:
    out = tmp_path / "rife"
    out.mkdir()
    for i in range(1, 7):
        (out / f"{i:08d}.png").write_bytes(b"x")
    drop_rife_output_prefix(out, frame_format="png", drop_count=2)
    names = sorted(p.name for p in out.iterdir())
    assert names == [f"{i:08d}.png" for i in range(1, 5)]


def test_resolve_batch_frame_plan_uses_stored_decode_counts() -> None:
    fps = Fraction(24000, 1001)
    theoretical = BatchFramePlan(
        start_pts=1200.0,
        end_pts=1320.0,
        source_fps=fps,
        overlap_source_frames=1,
        content_frame_offset=28771,
        expected_content_frames=2877,
        rife_input_base=28770,
        rife_input_count=2878,
        rife_output_skip=2,
        expected_output_frames=5754,
    )
    reconciled = batch_frame_plan_for_actual_decode(theoretical, 225)
    ctx = PipelineContext(
        job_id="j",
        source_path=Path("src.mkv"),
        workdir=Path("work"),
        output_path=Path("out.mkv"),
        preset_id="p",
        preset_data={},
    )
    ctx._active_batch_idx = 10
    from types import SimpleNamespace

    ctx.media_info = SimpleNamespace(
        primary_video=SimpleNamespace(
            avg_frame_rate="24000/1001",
            r_frame_rate="24000/1001",
            nb_frames=None,
        ),
        fmt=SimpleNamespace(duration_s=1425.0),
    )
    ctx.plan = {
        "decode": {
            "pts_window": (1200.0, 1320.0),
        },
        "interpolate": {"multiplier": 2},
    }
    merge_batch_frame_plan_into_decode(ctx, reconciled)
    resolved = resolve_batch_frame_plan(ctx)
    assert resolved is not None
    assert resolved.rife_input_count == 225
    assert resolved.expected_content_frames == 224
    assert resolved.expected_output_frames == 448
    # PTS-only math would still claim the full batch.
    assert resolved.rife_input_count != theoretical.rife_input_count


def test_reconcile_batch_decode_outputs_shortfall(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(1, 4):
        (frames / f"{i:08d}.png").write_bytes(b"x")
    fps = Fraction(24000, 1001)
    plan = BatchFramePlan(
        start_pts=10.0,
        end_pts=20.0,
        source_fps=fps,
        overlap_source_frames=1,
        content_frame_offset=total_frames(fps, 10.0),
        expected_content_frames=100,
        rife_input_base=0,
        rife_input_count=101,
        rife_output_skip=2,
        expected_output_frames=200,
    )
    ctx = PipelineContext(
        job_id="j",
        source_path=tmp_path / "src.mkv",
        workdir=tmp_path / "work",
        output_path=tmp_path / "out.mkv",
        preset_id="p",
        preset_data={},
    )
    ctx._active_batch_idx = 2
    ctx.plan = {"decode": {"pts_window": (10.0, 20.0)}}
    n, trimmed, updated, reconciled = reconcile_batch_decode_outputs(
        ctx,
        out_dir=frames,
        frame_format="png",
        plan=plan,
    )
    assert n == 3
    assert trimmed == 0
    assert reconciled is True
    assert updated.expected_decode_frames == 3
    assert updated.expected_content_frames == 2
    assert ctx.plan["decode"]["batch_expected_content_frames"] == 2


def test_batch_frame_plan_for_actual_decode_preserves_multiplier() -> None:
    fps = Fraction(24000, 1001)
    plan = BatchFramePlan(
        start_pts=0.0,
        end_pts=1.0,
        source_fps=fps,
        overlap_source_frames=0,
        content_frame_offset=0,
        expected_content_frames=24,
        rife_input_base=0,
        rife_input_count=24,
        rife_output_skip=0,
        expected_output_frames=48,
    )
    adjusted = batch_frame_plan_for_actual_decode(plan, 20)
    assert adjusted.expected_content_frames == 20
    assert adjusted.expected_output_frames == 40


def test_count_numeric_frames_ignores_non_numeric(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "00000001.png").write_bytes(b"x")
    (frames / "sidecar.png").write_bytes(b"x")
    assert count_numeric_frames_in_dir(frames, frame_format="png") == 1


def test_expected_segment_duration_s() -> None:
    fps = Fraction(48000, 1001)
    plan = BatchFramePlan(
        start_pts=0.0,
        end_pts=1.0,
        source_fps=Fraction(24000, 1001),
        overlap_source_frames=0,
        content_frame_offset=0,
        expected_content_frames=24,
        rife_input_base=0,
        rife_input_count=24,
        rife_output_skip=0,
        expected_output_frames=48,
    )
    dur = expected_segment_duration_s(plan, output_fps=fps)
    assert dur == pytest.approx(48 / float(fps), rel=1e-4)
