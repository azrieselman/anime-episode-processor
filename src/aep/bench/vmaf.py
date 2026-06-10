"""VMAF helpers for benchmark runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aep.adapters.base import env_with_tool_dirs
from aep.adapters.ffmpeg import FFmpegAdapter
from aep.bench.models import VmafScores
from aep.util.proc import ProcError, run_capture

_DEFAULT_MODEL = "vmaf_v0.6.1"


def _lavfi_log_path(log_path: Path) -> str:
    """Return a libvmaf ``log_path`` value safe for ``-lavfi`` filter graphs.

    Absolute Windows paths (``C:\\...``) cannot be embedded reliably in filter
    option strings — ffmpeg treats ``:`` as an option separator even when
    backslash-escaped.  Callers must run ffmpeg with ``cwd=log_path.parent`` and
    pass only the basename here.
    """
    return log_path.name


def is_libvmaf_available(ffmpeg_adapter: FFmpegAdapter | None = None) -> bool:
    ffmpeg = ffmpeg_adapter or FFmpegAdapter()
    cmd = [
        ffmpeg.command_executable(),
        "-hide_banner",
        "-filters",
    ]
    try:
        result = run_capture(cmd, env=env_with_tool_dirs(), timeout=20.0, check=False)
    except ProcError:
        return False
    haystack = f"{result.stdout}\n{result.stderr}".lower()
    return "libvmaf" in haystack


def build_vmaf_command(
    *,
    source_path: Path,
    encoded_path: Path,
    start_s: float,
    duration_s: float,
    log_path: Path,
    model: str = _DEFAULT_MODEL,
    ffmpeg_adapter: FFmpegAdapter | None = None,
) -> list[str | Path]:
    ffmpeg = ffmpeg_adapter or FFmpegAdapter()
    safe_start = max(0.0, float(start_s))
    safe_duration = max(0.1, float(duration_s))
    filter_graph = (
        "[0:v][1:v]scale2ref=flags=bicubic[ref][dist];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={_lavfi_log_path(log_path)}:model=version={model}"
    )
    return [
        ffmpeg.command_executable(),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "verbose",
        "-ss",
        f"{safe_start:.6f}",
        "-t",
        f"{safe_duration:.6f}",
        "-i",
        str(source_path),
        "-i",
        str(encoded_path),
        "-lavfi",
        filter_graph,
        "-an",
        "-sn",
        "-dn",
        "-f",
        "null",
        "-",
    ]


def parse_vmaf_json(payload: dict[str, Any], *, model: str = _DEFAULT_MODEL) -> VmafScores:
    pooled = payload.get("pooled_metrics") or {}
    vmaf = pooled.get("vmaf") if isinstance(pooled, dict) else None
    if isinstance(vmaf, dict):
        mean = vmaf.get("mean")
        harmonic = vmaf.get("harmonic_mean")
        if isinstance(mean, (int, float)):
            return VmafScores(
                mean=float(mean),
                harmonic_mean=(float(harmonic) if isinstance(harmonic, (int, float)) else None),
                model=model,
            )
    frames = payload.get("frames")
    if isinstance(frames, list):
        values: list[float] = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            metrics = frame.get("metrics")
            if not isinstance(metrics, dict):
                continue
            value = metrics.get("vmaf")
            if isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            return VmafScores(mean=sum(values) / len(values), harmonic_mean=None, model=model)
    raise ValueError("No VMAF score found in JSON log payload")


def compute_vmaf_for_segment(
    *,
    source_path: Path,
    encoded_path: Path,
    start_s: float,
    duration_s: float,
    log_path: Path,
    model: str = _DEFAULT_MODEL,
    ffmpeg_adapter: FFmpegAdapter | None = None,
) -> VmafScores:
    resolved_log = log_path.resolve()
    resolved_log.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_vmaf_command(
        source_path=source_path,
        encoded_path=encoded_path,
        start_s=start_s,
        duration_s=duration_s,
        log_path=resolved_log,
        model=model,
        ffmpeg_adapter=ffmpeg_adapter,
    )
    result = run_capture(
        cmd,
        env=env_with_tool_dirs(),
        cwd=resolved_log.parent,
        timeout=24 * 3600.0,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"VMAF ffmpeg invocation failed: {result.stderr[:800]}")
    payload = json.loads(resolved_log.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid VMAF payload: expected object")
    return parse_vmaf_json(payload, model=model)
