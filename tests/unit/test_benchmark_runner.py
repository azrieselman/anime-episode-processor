from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from aep.bench.models import BenchmarkRequest, BenchmarkResult, VmafScores
from aep.bench.runner import BenchmarkRunner, _resolve_benchmark_encoded_video, delete_benchmark_run


class _Preset:
    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {"meta": {"id": "anime_balanced"}}


class _HardwareProfile:
    def fingerprint(self) -> str:
        return "hw-fingerprint"


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        hardware=SimpleNamespace(
            decode_hwaccel="off",
            prefer_hardware_encoder=True,
        ),
        paths=SimpleNamespace(ramdisk_path=""),
        pipeline=SimpleNamespace(order="interpolate_first"),
    )


def _patch_common(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("aep.bench.runner.bench_dir", lambda: tmp_path)
    monkeypatch.setattr("aep.bench.runner.load_settings", _settings)
    monkeypatch.setattr("aep.bench.runner.load_preset", lambda _preset_id: _Preset())
    monkeypatch.setattr(
        "aep.bench.runner.merge_preset_data_for_job",
        lambda base, _overrides, settings_decode_hwaccel: {
            **base,
            "decode": {"hwaccel": settings_decode_hwaccel},
            "upscaler": {"enabled": False, "engine": "none"},
            "interpolation": {"enabled": False, "engine": "none"},
            "postprocess": {"enabled": False, "deband": False, "deblock": False, "grain_addback": 0},
            "batching": {"mode": "manual", "enabled": False},
            "frame_dedupe": {"enabled": False},
        },
    )
    monkeypatch.setattr("aep.bench.runner.Preset.model_validate", lambda _payload: None)


def test_benchmark_runner_encode_only_filters_stages(
    monkeypatch, tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    _patch_common(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    def _build_default_stages(*, order):  # type: ignore[no-untyped-def]
        assert order == "interpolate_first"
        names = [
            "00_probe",
            "01_plan",
            "02_sample_bench",
            "03_scene_detect",
            "08_encode",
            "09_mux",
            "10_validate",
        ]
        return [SimpleNamespace(name=n) for n in names]

    class _FakePipelineRunner:
        def __init__(self, stages) -> None:  # type: ignore[no-untyped-def]
            seen["stages"] = [s.name for s in stages]

        def run(self, ctx, _events) -> None:  # type: ignore[no-untyped-def]
            ctx.extras["hardware_profile"] = _HardwareProfile()
            out_dir = ctx.workdir / "08_encode"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "video.mkv").write_bytes(b"video")
            (ctx.workdir / "perf_profile.json").write_text(
                json.dumps({"stages": {"08_encode": {"duration_s": 1.0}}}),
                encoding="utf-8",
            )

    monkeypatch.setattr("aep.bench.runner.build_default_stages", _build_default_stages)
    monkeypatch.setattr("aep.bench.runner.PipelineRunner", _FakePipelineRunner)

    runner = BenchmarkRunner()
    result = runner.run(
        BenchmarkRequest(
            source_path=Path("input.mkv"),
            preset_id="anime_balanced",
            scope="encode_only",
            compute_vmaf=False,
        ),
    )

    assert seen["stages"] == ["00_probe", "01_plan", "08_encode"]
    assert result.perf_profile["stages"]["08_encode"]["duration_s"] == 1.0
    assert result.encoded_video_path is not None and result.encoded_video_path.is_file()
    assert result.hardware_fingerprint == "hw-fingerprint"


def test_benchmark_runner_full_scope_drops_bench_mux_validate(
    monkeypatch, tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    _patch_common(monkeypatch, tmp_path)
    seen: dict[str, object] = {}

    def _build_default_stages(*, order):  # type: ignore[no-untyped-def]
        assert order == "interpolate_first"
        names = [
            "00_probe",
            "01_plan",
            "02_sample_bench",
            "03_scene_detect",
            "04_decode_serve",
            "05_upscale",
            "06_interpolate",
            "07_postprocess",
            "08_encode",
            "09_mux",
            "10_validate",
        ]
        return [SimpleNamespace(name=n) for n in names]

    class _FakePipelineRunner:
        def __init__(self, stages) -> None:  # type: ignore[no-untyped-def]
            seen["stages"] = [s.name for s in stages]

        def run(self, ctx, _events) -> None:  # type: ignore[no-untyped-def]
            (ctx.workdir / "perf_profile.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("aep.bench.runner.build_default_stages", _build_default_stages)
    monkeypatch.setattr("aep.bench.runner.PipelineRunner", _FakePipelineRunner)

    runner = BenchmarkRunner()
    runner.run(
        BenchmarkRequest(
            source_path=Path("input.mkv"),
            preset_id="anime_balanced",
            scope="full",
            compute_vmaf=False,
        ),
    )

    assert seen["stages"] == [
        "00_probe",
        "01_plan",
        "03_scene_detect",
        "04_decode_serve",
        "05_upscale",
        "06_interpolate",
        "07_postprocess",
        "08_encode",
    ]


def test_resolve_benchmark_encoded_video_prefers_batch_segment(
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "bench_run"
    seg_dir = workdir / "batch_segments"
    seg_dir.mkdir(parents=True)
    segment = seg_dir / "segment_00.mkv"
    segment.write_bytes(b"segment")

    ctx = SimpleNamespace(
        encoded_segments=[segment],
        stage_results={},
        workdir=workdir,
    )

    resolved = _resolve_benchmark_encoded_video(ctx, workdir)  # type: ignore[arg-type]
    assert resolved == segment


def test_benchmark_runner_runs_vmaf_on_batch_segment(
    monkeypatch, tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    _patch_common(monkeypatch, tmp_path)

    def _build_default_stages(*, order):  # type: ignore[no-untyped-def]
        return [SimpleNamespace(name="08_encode")]

    class _FakePipelineRunner:
        def __init__(self, _stages) -> None:
            pass

        def run(self, ctx, _events) -> None:  # type: ignore[no-untyped-def]
            ctx.extras["hardware_profile"] = _HardwareProfile()
            seg_dir = ctx.workdir / "batch_segments"
            seg_dir.mkdir(parents=True, exist_ok=True)
            segment = seg_dir / "segment_00.mkv"
            segment.write_bytes(b"segment")
            ctx.encoded_segments = [segment]
            (ctx.workdir / "perf_profile.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr("aep.bench.runner.build_default_stages", _build_default_stages)
    monkeypatch.setattr("aep.bench.runner.PipelineRunner", _FakePipelineRunner)
    monkeypatch.setattr("aep.bench.runner.is_libvmaf_available", lambda: True)
    monkeypatch.setattr(
        "aep.bench.runner.compute_vmaf_for_segment",
        lambda **_kwargs: VmafScores(mean=91.5, harmonic_mean=90.0),
    )

    runner = BenchmarkRunner()
    result = runner.run(
        BenchmarkRequest(
            source_path=Path("input.mkv"),
            preset_id="anime_balanced",
            scope="full",
            compute_vmaf=True,
        ),
    )

    assert result.encoded_video_path == tmp_path / result.run_id / "batch_segments" / "segment_00.mkv"
    assert result.vmaf is not None and result.vmaf.mean == 91.5
    assert not any("encoded segment missing" in warning for warning in result.warnings)


def _bench_result(*, run_id: str, workdir: Path) -> BenchmarkResult:
    return BenchmarkResult(
        run_id=run_id,
        request=BenchmarkRequest(
            source_path=Path("input.mkv"),
            preset_id="anime_balanced",
        ),
        workdir=workdir,
        ffmpeg_log_path=workdir / "ffmpeg.log",
        encoded_video_path=None,
        perf_profile={},
    )


def test_delete_benchmark_run_removes_workdir(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("aep.bench.runner.bench_dir", lambda: tmp_path)
    run_id = "bench_20260101T000000_abcd1234"
    workdir = tmp_path / run_id
    workdir.mkdir(parents=True)
    (workdir / "ffmpeg.log").write_text("log", encoding="utf-8")

    delete_benchmark_run(_bench_result(run_id=run_id, workdir=workdir))

    assert not workdir.exists()


def test_delete_benchmark_run_rejects_path_outside_bench_dir(
    monkeypatch, tmp_path: Path,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr("aep.bench.runner.bench_dir", lambda: tmp_path / "bench")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    run_id = "bench_20260101T000000_abcd1234"

    try:
        delete_benchmark_run(_bench_result(run_id=run_id, workdir=outside))
    except ValueError as exc:
        assert "outside bench dir" in str(exc)
    else:
        raise AssertionError("expected ValueError")
