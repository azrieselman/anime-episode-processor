"""Helpers for benchmark segment planning."""

from __future__ import annotations

from dataclasses import dataclass

from aep.pipeline.batches import BatchSpec


@dataclass(frozen=True)
class BenchmarkSegment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, float]:
        return {
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
        }


def resolve_benchmark_segment(
    *,
    source_duration_s: float,
    start_s: float,
    duration_s: float,
) -> BenchmarkSegment:
    if source_duration_s <= 0:
        raise ValueError(f"source_duration_s must be > 0, got {source_duration_s!r}")
    safe_start = max(0.0, min(float(start_s), float(source_duration_s)))
    safe_duration = max(0.1, float(duration_s))
    safe_end = min(float(source_duration_s), safe_start + safe_duration)
    if safe_end <= safe_start:
        safe_end = min(float(source_duration_s), safe_start + 0.1)
    return BenchmarkSegment(start_s=safe_start, end_s=safe_end)


def segment_batch_spec(
    *,
    segment: BenchmarkSegment,
    output_fps: float | None,
    bytes_per_output_frame: int,
) -> BatchSpec:
    frame_count = (
        int(round(segment.duration_s * float(output_fps)))
        if output_fps is not None and output_fps > 0
        else 0
    )
    est_bytes = frame_count * int(bytes_per_output_frame) if bytes_per_output_frame > 0 else 0
    return BatchSpec(
        index=0,
        start_pts=float(segment.start_s),
        end_pts=float(segment.end_s),
        frame_count_estimate=frame_count,
        est_bytes=est_bytes,
    )
