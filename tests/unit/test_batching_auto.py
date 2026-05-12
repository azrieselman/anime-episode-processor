"""Unit tests for auto batching (disk-sized chunk selection)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aep.errors import PipelineError
from aep.media.models import FormatInfo, MediaInfo, StreamInfo
from aep.persist.presets import (
    BatchingCfg,
    DecodeCfg,
    EncoderCfg,
    InterpolationCfg,
    PostprocessCfg,
    Preset,
    PresetMeta,
    StreamMappingCfg,
    TargetResolution,
    UpscalerCfg,
)
from aep.pipeline.context import PipelineContext
from aep.pipeline.stages.s01_plan import (
    _estimate_frame_bytes,
    _plan_m3_video_path,
    _plan_video_batches,
    _resolve_decode_hwaccel,
    _resolve_target_geometry,
)


def _small_preset(*, batching: BatchingCfg) -> Preset:
    return Preset(
        meta=PresetMeta(id="t", name="t"),
        batching=batching,
        decode=DecodeCfg(hwaccel="off"),
        encoder=EncoderCfg(name="libx264"),
        streams=StreamMappingCfg(),
        target_resolution=TargetResolution(mode="scale_only"),
        upscaler=UpscalerCfg(enabled=False),
        interpolation=InterpolationCfg(enabled=False, target_fps=None),
        postprocess=PostprocessCfg(enabled=False),
    )


def _make_media(*, duration_s: float, width: int, height: int) -> tuple[MediaInfo, object]:
    primary = StreamInfo(
        index=0,
        kind="video",
        codec_name="h264",
        pix_fmt="yuv420p",
        avg_frame_rate="24/1",
        r_frame_rate="24/1",
        width=width,
        height=height,
        nb_frames=int(duration_s * 24),
    )
    media = MediaInfo(
        source_path="/tmp/x.mkv",
        fmt=FormatInfo(
            filename="/tmp/x.mkv", format_name="matroska", duration_s=float(duration_s),
        ),
        streams=[primary],
        is_matroska=True,
    )
    return media, media.primary_video


def _ctx(tmp_path, *, ramdisk: bool = True) -> PipelineContext:
    root = tmp_path / "job"
    root.mkdir()
    ram_path = tmp_path / "ram"
    if ramdisk:
        ram_path.mkdir()
    return PipelineContext(
        job_id="j1",
        source_path=tmp_path / "in.mkv",
        workdir=root,
        output_path=tmp_path / "out.mkv",
        preset_id="p",
        preset_data={},
        ramdisk_path=ram_path if ramdisk else None,
    )


def test_auto_unbatched_when_full_estimate_fits_free_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _small_preset(batching=BatchingCfg(mode="auto", chunk_seconds=30))
    media, primary = _make_media(duration_s=60.0, width=320, height=240)
    ctx = _ctx(tmp_path)
    m3, _w, _r = _plan_m3_video_path(
        preset, media, primary, decode_hwaccel="off",
    )
    tw, th = _resolve_target_geometry(preset, media)
    est = _estimate_frame_bytes(media=media, target_w=tw, target_h=th, m3_plan=m3)

    mock_usage = MagicMock()
    mock_usage.free = est + 100 * 1024 * 1024
    monkeypatch.setattr(
        "aep.pipeline.stages.s01_plan.shutil.disk_usage", lambda _p: mock_usage,
    )

    batches, meta = _plan_video_batches(
        ctx=ctx,
        preset=preset,
        media=media,
        primary=primary,
        target_w=tw,
        target_h=th,
        m3_plan=m3,
        ramdisk_estimate=est,
    )
    assert batches == []
    assert meta.get("auto_unbatched_reason") == "estimate_fits_free_space"


def test_auto_resolves_batches_when_tight_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _small_preset(
        batching=BatchingCfg(
            mode="auto",
            chunk_seconds=600,
            boundary_policy="exact",
        ),
    )
    media, primary = _make_media(duration_s=300.0, width=320, height=240)
    ctx = _ctx(tmp_path, ramdisk=True)
    m3, _w, _r = _plan_m3_video_path(
        preset, media, primary, decode_hwaccel=_resolve_decode_hwaccel("off"),
    )
    tw, th = _resolve_target_geometry(preset, media)
    est = _estimate_frame_bytes(media=media, target_w=tw, target_h=th, m3_plan=m3)
    assert est > 50 * 1024 * 1024

    mock_usage = MagicMock()
    mock_usage.free = 80 * 1024 * 1024
    monkeypatch.setattr(
        "aep.pipeline.stages.s01_plan.shutil.disk_usage", lambda _p: mock_usage,
    )

    batches, meta = _plan_video_batches(
        ctx=ctx,
        preset=preset,
        media=media,
        primary=primary,
        target_w=tw,
        target_h=th,
        m3_plan=m3,
        ramdisk_estimate=est,
    )
    assert len(batches) >= 2
    assert meta.get("resolved_chunk_seconds") is not None
    assert meta.get("resolved_chunk_seconds", 999) <= 600


def test_auto_resolves_batches_when_m3_output_fps_unset_but_nb_frames_known(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto sizing must use byte estimates even when ffprobe yields no usable fps.

    If ``plan_batches`` sees ``output_fps=None`` it assigns zero ``est_bytes`` per
    batch, so the binary search always picks ``chunk_seconds`` as the cap — the
    bug this regression guards against.
    """
    preset = _small_preset(
        batching=BatchingCfg(
            mode="auto",
            chunk_seconds=600,
            boundary_policy="exact",
        ),
    )
    primary = StreamInfo(
        index=0,
        kind="video",
        codec_name="h264",
        pix_fmt="yuv420p",
        avg_frame_rate="",
        r_frame_rate="",
        width=320,
        height=240,
        nb_frames=int(300.0 * 24),
    )
    media = MediaInfo(
        source_path="/tmp/x.mkv",
        fmt=FormatInfo(
            filename="/tmp/x.mkv", format_name="matroska", duration_s=300.0,
        ),
        streams=[primary],
        is_matroska=True,
    )
    ctx = _ctx(tmp_path, ramdisk=True)
    m3, _w, _r = _plan_m3_video_path(
        preset, media, primary, decode_hwaccel=_resolve_decode_hwaccel("off"),
    )
    assert not (m3.get("output_fps") or "").strip()
    tw, th = _resolve_target_geometry(preset, media)
    est = _estimate_frame_bytes(media=media, target_w=tw, target_h=th, m3_plan=m3)
    assert est > 50 * 1024 * 1024

    mock_usage = MagicMock()
    mock_usage.free = 80 * 1024 * 1024
    monkeypatch.setattr(
        "aep.pipeline.stages.s01_plan.shutil.disk_usage", lambda _p: mock_usage,
    )

    batches, meta = _plan_video_batches(
        ctx=ctx,
        preset=preset,
        media=media,
        primary=primary,
        target_w=tw,
        target_h=th,
        m3_plan=m3,
        ramdisk_estimate=est,
    )
    assert len(batches) >= 2
    assert meta.get("resolved_chunk_seconds") is not None
    assert meta.get("resolved_chunk_seconds", 999) < 600


def test_auto_raises_when_batching_needed_but_no_ramdisk_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = _small_preset(batching=BatchingCfg(mode="auto"))
    media, primary = _make_media(duration_s=300.0, width=1920, height=1080)
    ctx = _ctx(tmp_path, ramdisk=False)
    m3, _w, _r = _plan_m3_video_path(preset, media, primary, decode_hwaccel="off")
    tw, th = _resolve_target_geometry(preset, media)
    est = _estimate_frame_bytes(media=media, target_w=tw, target_h=th, m3_plan=m3)

    mock_usage = MagicMock()
    mock_usage.free = 50 * 1024 * 1024
    monkeypatch.setattr(
        "aep.pipeline.stages.s01_plan.shutil.disk_usage", lambda _p: mock_usage,
    )

    with pytest.raises(PipelineError, match="ramdisk_path"):
        _plan_video_batches(
            ctx=ctx,
            preset=preset,
            media=media,
            primary=primary,
            target_w=tw,
            target_h=th,
            m3_plan=m3,
            ramdisk_estimate=est,
        )
