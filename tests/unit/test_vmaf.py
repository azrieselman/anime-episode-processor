from __future__ import annotations

import json
from pathlib import Path

from aep.bench.vmaf import (
    build_vmaf_command,
    compute_vmaf_for_segment,
    is_libvmaf_available,
    parse_vmaf_json,
)
from aep.util.proc import ProcResult


class _FakeFFmpeg:
    def command_executable(self) -> str:
        return "ffmpeg"


def test_build_vmaf_command_uses_basename_for_log_path(tmp_path: Path) -> None:
    log_path = tmp_path / "bench run" / "vmaf.json"
    cmd = build_vmaf_command(
        source_path=Path("source.mkv"),
        encoded_path=Path("encoded.mkv"),
        start_s=12.5,
        duration_s=20.0,
        log_path=log_path,
        ffmpeg_adapter=_FakeFFmpeg(),  # type: ignore[arg-type]
    )
    argv = [str(x) for x in cmd]
    filter_graph = argv[argv.index("-lavfi") + 1]
    assert "log_path=vmaf.json" in filter_graph
    assert "AppData" not in filter_graph
    assert "bench run" not in filter_graph
    assert "libvmaf=log_fmt=json" in " ".join(argv)
    assert "-ss" in argv and "12.500000" in argv
    assert "-t" in argv and "20.000000" in argv


def test_parse_vmaf_json_uses_pooled_metrics() -> None:
    payload = {
        "pooled_metrics": {
            "vmaf": {
                "mean": 93.25,
                "harmonic_mean": 92.81,
            },
        },
    }
    out = parse_vmaf_json(payload)
    assert out.mean == 93.25
    assert out.harmonic_mean == 92.81


def test_parse_vmaf_json_falls_back_to_frame_average() -> None:
    payload = {
        "frames": [
            {"metrics": {"vmaf": 90.0}},
            {"metrics": {"vmaf": 95.0}},
        ],
    }
    out = parse_vmaf_json(payload)
    assert out.mean == 92.5
    assert out.harmonic_mean is None


def test_is_libvmaf_available_reads_filters_output(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def _fake_run_capture(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return ProcResult(["ffmpeg", "-filters"], 0, " ... libvmaf ... ", "")

    monkeypatch.setattr("aep.bench.vmaf.run_capture", _fake_run_capture)
    assert is_libvmaf_available(ffmpeg_adapter=_FakeFFmpeg())  # type: ignore[arg-type]


def test_compute_vmaf_for_segment_parses_log_file(
    monkeypatch, tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    log_path = tmp_path / "vmaf.json"
    seen: dict[str, object] = {}

    def _fake_run_capture(*_args, **kwargs):  # type: ignore[no-untyped-def]
        seen["cwd"] = kwargs.get("cwd")
        payload = {
            "pooled_metrics": {
                "vmaf": {
                    "mean": 88.2,
                    "harmonic_mean": 87.9,
                },
            },
        }
        log_path.write_text(json.dumps(payload), encoding="utf-8")
        return ProcResult(["ffmpeg"], 0, "", "")

    monkeypatch.setattr("aep.bench.vmaf.run_capture", _fake_run_capture)
    out = compute_vmaf_for_segment(
        source_path=Path("source.mkv"),
        encoded_path=Path("encoded.mkv"),
        start_s=0.0,
        duration_s=30.0,
        log_path=log_path,
        ffmpeg_adapter=_FakeFFmpeg(),  # type: ignore[arg-type]
    )
    assert seen["cwd"] == log_path.parent.resolve()
    assert out.mean == 88.2
    assert out.harmonic_mean == 87.9
