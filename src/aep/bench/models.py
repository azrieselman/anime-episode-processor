"""Benchmark run request/result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

BenchmarkScope = Literal["full", "encode_only"]


@dataclass(frozen=True)
class BenchmarkRequest:
    source_path: Path
    preset_id: str
    scope: BenchmarkScope = "full"
    start_s: float = 0.0
    duration_s: float = 30.0
    preset_overrides: dict[str, Any] | None = None
    verbose_ffmpeg: bool = True
    compute_vmaf: bool = True

    def benchmark_extra(self) -> dict[str, Any]:
        return {
            "start_s": float(self.start_s),
            "duration_s": float(self.duration_s),
            "scope": self.scope,
            "verbose_ffmpeg": bool(self.verbose_ffmpeg),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "preset_id": self.preset_id,
            "scope": self.scope,
            "start_s": float(self.start_s),
            "duration_s": float(self.duration_s),
            "preset_overrides": dict(self.preset_overrides or {}),
            "verbose_ffmpeg": bool(self.verbose_ffmpeg),
            "compute_vmaf": bool(self.compute_vmaf),
        }


@dataclass(frozen=True)
class VmafScores:
    mean: float
    harmonic_mean: float | None = None
    model: str = "vmaf_v0.6.1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": float(self.mean),
            "harmonic_mean": (
                float(self.harmonic_mean) if self.harmonic_mean is not None else None
            ),
            "model": self.model,
        }


@dataclass
class BenchmarkResult:
    run_id: str
    request: BenchmarkRequest
    workdir: Path
    ffmpeg_log_path: Path
    encoded_video_path: Path | None
    perf_profile: dict[str, Any]
    encode_samples: list[dict[str, Any]] = field(default_factory=list)
    vmaf: VmafScores | None = None
    hardware_fingerprint: str | None = None
    completed_at: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "request": self.request.to_dict(),
            "hardware_fingerprint": self.hardware_fingerprint,
            "perf_profile": self.perf_profile,
            "vmaf": self.vmaf.to_dict() if self.vmaf is not None else None,
            "encode_samples": list(self.encode_samples),
            "artifacts": {
                "encoded_video": (
                    str(self.encoded_video_path) if self.encoded_video_path is not None else None
                ),
                "ffmpeg_log": str(self.ffmpeg_log_path),
                "workdir": str(self.workdir),
            },
            "completed_at": self.completed_at,
            "warnings": list(self.warnings),
        }
