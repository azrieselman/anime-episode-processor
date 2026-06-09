from __future__ import annotations

from pathlib import Path

from aep.adapters.ncnn_base import NcnnRunResult
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink
from aep.pipeline.stages.s05_upscale import UpscaleStage


class _FakeAnime4kAdapter:
    version = "2.0"

    def run_frame_sequence(self, **kwargs) -> NcnnRunResult:
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "00000001.png").write_bytes(b"png")
        return NcnnRunResult(
            output_dir=out_dir,
            frames_in=1,
            frames_out=1,
            tile_size_used=128,
            duration_s=0.01,
            attempts=1,
        )


class _FakeAnime4kVsAdapter:
    version = "R74"

    def run_frame_sequence(self, **kwargs) -> NcnnRunResult:
        out_dir = Path(kwargs["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "00000001.png").write_bytes(b"png")
        return NcnnRunResult(
            output_dir=out_dir,
            frames_in=1,
            frames_out=1,
            tile_size_used=0,
            duration_s=0.01,
            attempts=1,
        )


def test_stage05_dispatches_anime4k_engine(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    decode_dir = workdir / "04_decode_serve" / "frames"
    decode_dir.mkdir(parents=True, exist_ok=True)
    (decode_dir / "00000001.png").write_bytes(b"png")

    ctx = PipelineContext(
        job_id="job1",
        source_path=workdir / "input.mkv",
        workdir=workdir,
        output_path=workdir / "out.mkv",
        preset_id="anime_balanced",
        preset_data={},
        plan={
            "decode": {"dir": str(decode_dir)},
            "upscale": {
                "active": True,
                "engine": "anime4kcpp",
                "model": "acnet-f8b8-hdn",
                "scale": 2,
                "denoise": 1,
                "tile_size": 256,
                "tta": False,
                "frame_format": "png",
            },
        },
    )

    stage = UpscaleStage(anime4kcpp=_FakeAnime4kAdapter())  # type: ignore[arg-type]
    plan = stage.plan(ctx)
    result = stage.run(ctx, plan, EventSink())

    assert result.success is True
    assert result.metrics["engine"] == "anime4kcpp"
    assert ctx.plan["upscale"]["count"] == 1


def test_stage05_dispatches_anime4k_vs_engine(tmp_path: Path) -> None:
    workdir = tmp_path / "work"
    decode_dir = workdir / "04_decode_serve" / "frames"
    decode_dir.mkdir(parents=True, exist_ok=True)
    (decode_dir / "00000001.png").write_bytes(b"png")

    ctx = PipelineContext(
        job_id="job1",
        source_path=workdir / "input.mkv",
        workdir=workdir,
        output_path=workdir / "out.mkv",
        preset_id="anime_balanced",
        preset_data={},
        plan={
            "decode": {"dir": str(decode_dir)},
            "upscale": {
                "active": True,
                "engine": "anime4kcpp-vs",
                "model": "acnet-f8b8-hdn",
                "scale": 2,
                "denoise": 1,
                "tile_size": 256,
                "tta": False,
                "frame_format": "png",
            },
        },
    )

    stage = UpscaleStage(anime4kcpp_vs=_FakeAnime4kVsAdapter())  # type: ignore[arg-type]
    plan = stage.plan(ctx)
    result = stage.run(ctx, plan, EventSink())

    assert result.success is True
    assert result.metrics["engine"] == "anime4kcpp-vs"
    assert ctx.plan["upscale"]["count"] == 1
