from __future__ import annotations

from aep.bench.segment import resolve_benchmark_segment, segment_batch_spec


def test_resolve_benchmark_segment_clamps_to_source_duration() -> None:
    seg = resolve_benchmark_segment(
        source_duration_s=120.0,
        start_s=110.0,
        duration_s=30.0,
    )
    assert seg.start_s == 110.0
    assert seg.end_s == 120.0
    assert seg.duration_s == 10.0


def test_resolve_benchmark_segment_normalizes_negative_inputs() -> None:
    seg = resolve_benchmark_segment(
        source_duration_s=60.0,
        start_s=-5.0,
        duration_s=-1.0,
    )
    assert seg.start_s == 0.0
    assert seg.end_s > seg.start_s


def test_segment_batch_spec_uses_window_fps_and_bytes_per_frame() -> None:
    seg = resolve_benchmark_segment(
        source_duration_s=100.0,
        start_s=10.0,
        duration_s=20.0,
    )
    batch = segment_batch_spec(
        segment=seg,
        output_fps=30.0,
        bytes_per_output_frame=1000,
    )
    assert batch.start_pts == 10.0
    assert batch.end_pts == 30.0
    assert batch.frame_count_estimate == 600
    assert batch.est_bytes == 600_000
