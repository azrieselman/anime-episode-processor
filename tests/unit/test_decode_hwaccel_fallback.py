from __future__ import annotations

import threading
from pathlib import Path

from aep.persist.presets import EncoderCfg
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink
from aep.pipeline.stage import StagePlan
from aep.adapters.ffmpeg import decode_hwaccel_uses_hardware_decode
from aep.pipeline.stages.s04_decode_serve import DecodeServeStage
from aep.pipeline.stages.s08_encode import EncodeStage
from aep.util.proc import ProcError, ProcResult


class _FakeFFmpeg:
    path = "ffmpeg"
    version = "test"

    def build_decode_to_frames(self, **kwargs):
        return ["ffmpeg", kwargs.get("decode_hwaccel", "off")]

    def build_decode_to_frames_with_scene_metadata_fused(self, **kwargs):
        return ["ffmpeg", "fused", kwargs.get("decode_hwaccel", "off")]

    def build_scene_score_scan(self, **kwargs):
        return ["ffmpeg", "scan", kwargs.get("decode_hwaccel", "off")]

    def build_passthrough_video_encode(self, **kwargs):
        return ["ffmpeg", kwargs.get("decode_hwaccel", "off")]


def _make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        job_id="job-1",
        source_path=tmp_path / "in.mkv",
        workdir=tmp_path / "work",
        output_path=tmp_path / "out.mkv",
        preset_id="anime_balanced",
        preset_data={},
        cancel_event=threading.Event(),
        pause_event=threading.Event(),
    )


def _hwaccel_token_in_cmd(cmd_strs: list[str]) -> bool:
    return any(t in cmd_strs for t in ("d3d11va", "cuda"))


def _make_streaming_fake(calls: list[list[str]]):
    """Return a ``run_streaming`` replacement that records calls and raises on hw decode."""
    def _fake_run_streaming(cmd, **kwargs):
        cmd_strs = [str(x) for x in cmd]
        calls.append(cmd_strs)
        if _hwaccel_token_in_cmd(cmd_strs):
            raise ProcError(ProcResult(cmd_strs, 1, "", "hwaccel decode failed"))
        if False:  # pragma: no cover — make this a generator
            yield ("stdout", "")
    return _fake_run_streaming


def _make_capture_fake(calls: list[list[str]]):
    def _fake_run_capture(cmd, **kwargs):
        cmd_strs = [str(x) for x in cmd]
        calls.append(cmd_strs)
        rc = 1 if _hwaccel_token_in_cmd(cmd_strs) else 0
        return ProcResult(cmd_strs, rc, "", "err" if rc else "")
    return _fake_run_capture


def test_decode_hwaccel_uses_hardware_decode() -> None:
    assert decode_hwaccel_uses_hardware_decode("cuda") is True
    assert decode_hwaccel_uses_hardware_decode("d3d11va") is True
    assert decode_hwaccel_uses_hardware_decode("off") is False
    assert decode_hwaccel_uses_hardware_decode("") is False


def _successful_streaming(calls: list[list[str]]):
    def _fake_run_streaming(cmd, **kwargs):
        calls.append([str(x) for x in cmd])
        if False:  # pragma: no cover
            yield ("stdout", "")
    return _fake_run_streaming


def test_frame_dedupe_with_cuda_skips_fused_runs_decode_then_scan(monkeypatch, tmp_path: Path) -> None:
    """Hardware decode + dedupe must not use fused decode+scene graph (stage 04 policy)."""
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "aep.pipeline.stages.s04_decode_serve.run_streaming",
        _successful_streaming(calls),
    )
    monkeypatch.setattr(
        "aep.pipeline.stages.s04_decode_serve.run_capture",
        _make_capture_fake(calls),
    )
    monkeypatch.setattr(
        PipelineContext,
        "get_frame_manifest",
        lambda *a, **k: {"count": 2, "bytes": 0},
    )

    stage = DecodeServeStage(ffmpeg=_FakeFFmpeg())
    ctx = _make_ctx(tmp_path)
    ctx.plan = {
        "decode": {"target_w": None, "target_h": None},
        "hdr": {},
        "frame_dedupe": {"active": True, "threshold": 0.02, "protect_scene_cuts": True},
    }
    plan = StagePlan(
        stage_name="04_decode_serve",
        cache_key="k",
        params={"active": True, "frame_format": "png", "bt709_normalize": True, "decode_hwaccel": "cuda"},
        outputs=[tmp_path / "frames"],
    )
    stage.run(ctx, plan, EventSink())
    flat = " | ".join(" ".join(c) for c in calls)
    assert "fused" not in flat
    assert any("cuda" in c and "fused" not in c for c in calls)
    assert any("scan" in c for c in calls)


def test_decode_stage_retries_without_hwaccel(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "aep.pipeline.stages.s04_decode_serve.run_streaming",
        _make_streaming_fake(calls),
    )
    monkeypatch.setattr(
        "aep.pipeline.stages.s04_decode_serve.run_capture",
        _make_capture_fake(calls),
    )
    monkeypatch.setattr(
        PipelineContext,
        "get_frame_manifest",
        lambda *a, **k: {"count": 2, "bytes": 0},
    )

    stage = DecodeServeStage(ffmpeg=_FakeFFmpeg())
    ctx = _make_ctx(tmp_path)
    ctx.plan = {"decode": {"target_w": None, "target_h": None}, "hdr": {}}
    plan = StagePlan(
        stage_name="04_decode_serve",
        cache_key="k",
        params={"active": True, "frame_format": "png", "bt709_normalize": True, "decode_hwaccel": "d3d11va"},
        outputs=[tmp_path / "frames"],
    )
    stage.run(ctx, plan, EventSink())
    assert len(calls) == 2
    assert "d3d11va" in calls[0]
    assert "off" in calls[1]


def test_decode_stage_retries_without_hwaccel_cuda(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "aep.pipeline.stages.s04_decode_serve.run_streaming",
        _make_streaming_fake(calls),
    )
    monkeypatch.setattr(
        "aep.pipeline.stages.s04_decode_serve.run_capture",
        _make_capture_fake(calls),
    )
    monkeypatch.setattr(
        PipelineContext,
        "get_frame_manifest",
        lambda *a, **k: {"count": 2, "bytes": 0},
    )

    stage = DecodeServeStage(ffmpeg=_FakeFFmpeg())
    ctx = _make_ctx(tmp_path)
    ctx.plan = {"decode": {"target_w": None, "target_h": None}, "hdr": {}}
    plan = StagePlan(
        stage_name="04_decode_serve",
        cache_key="k",
        params={"active": True, "frame_format": "png", "bt709_normalize": True, "decode_hwaccel": "cuda"},
        outputs=[tmp_path / "frames"],
    )
    stage.run(ctx, plan, EventSink())
    assert len(calls) == 2
    assert "cuda" in calls[0]
    assert "off" in calls[1]


def test_encode_source_retries_without_hwaccel(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "aep.pipeline.stages.s08_encode.run_streaming",
        _make_streaming_fake(calls),
    )
    monkeypatch.setattr(
        "aep.pipeline.stages.s08_encode.run_capture",
        _make_capture_fake(calls),
    )

    class _Primary:
        pix_fmt = "yuv420p"

    class _Media:
        primary_video = _Primary()

    stage = EncodeStage(ffmpeg=_FakeFFmpeg())
    ctx = _make_ctx(tmp_path)
    out_path = tmp_path / "video.mkv"
    out_path.write_bytes(b"ok")
    ctx.media_info = _Media()
    ctx.plan = {
        "encoder": {"cfg": EncoderCfg().model_dump(mode="json")},
        "decode": {"pts_window": None},
        "output_fps": "24000/1001",
    }
    plan = StagePlan(
        stage_name="08_encode",
        cache_key="k",
        params={"mode": "source", "target_w": None, "target_h": None, "decode_hwaccel": "d3d11va"},
        outputs=[out_path],
    )
    stage.run(ctx, plan, EventSink())
    assert len(calls) == 2
    assert "d3d11va" in calls[0]
    assert "off" in calls[1]


def test_encode_source_retries_without_hwaccel_cuda(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "aep.pipeline.stages.s08_encode.run_streaming",
        _make_streaming_fake(calls),
    )
    monkeypatch.setattr(
        "aep.pipeline.stages.s08_encode.run_capture",
        _make_capture_fake(calls),
    )

    class _Primary:
        pix_fmt = "yuv420p"

    class _Media:
        primary_video = _Primary()

    stage = EncodeStage(ffmpeg=_FakeFFmpeg())
    ctx = _make_ctx(tmp_path)
    out_path = tmp_path / "video.mkv"
    out_path.write_bytes(b"ok")
    ctx.media_info = _Media()
    ctx.plan = {
        "encoder": {"cfg": EncoderCfg().model_dump(mode="json")},
        "decode": {"pts_window": None},
        "output_fps": "24000/1001",
    }
    plan = StagePlan(
        stage_name="08_encode",
        cache_key="k",
        params={"mode": "source", "target_w": None, "target_h": None, "decode_hwaccel": "cuda"},
        outputs=[out_path],
    )
    stage.run(ctx, plan, EventSink())
    assert len(calls) == 2
    assert "cuda" in calls[0]
    assert "off" in calls[1]
