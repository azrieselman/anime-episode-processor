"""Smoke tests for ffmpeg encoder argv builders."""

from __future__ import annotations

import pytest

from aep.encode.encoders import QSV_GLOBAL_PREFIX, build_encoder_args
from aep.persist.presets import EncoderCfg


@pytest.mark.parametrize(
    "name,expect_codec",
    [
        ("hevc_qsv", "hevc_qsv"),
        ("h264_qsv", "h264_qsv"),
        ("av1_qsv", "av1_qsv"),
        ("hevc_amf", "hevc_amf"),
        ("h264_amf", "h264_amf"),
        ("av1_amf", "av1_amf"),
    ],
)
def test_hw_encoder_argv_contains_codec(name: str, expect_codec: str) -> None:
    cfg = EncoderCfg(name=name)  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    vi = r.args.index("-c:v")
    assert r.args[vi + 1] == expect_codec


def test_qsv_sets_global_prefix() -> None:
    cfg = EncoderCfg(name="hevc_qsv")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    assert r.global_prefix == QSV_GLOBAL_PREFIX
    assert "format=qsv" in "".join(r.args)


def test_amf_has_no_global_prefix() -> None:
    cfg = EncoderCfg(name="hevc_amf")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    assert r.global_prefix == ()


def test_nvenc_relaxed_strategies_fullres_then_simpler() -> None:
    from aep.pipeline.stages.s08_encode import nvenc_relaxed_strategies

    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_nvenc",
        nvenc_multipass="fullres",
        nvenc_temporal_aq=True,
    )
    flat = nvenc_relaxed_strategies(cfg, source_is_10bit=False)
    assert flat[0][0].nvenc_multipass == "fullres"
    assert flat[1][0].nvenc_multipass == "qres"
    assert any(c.nvenc_multipass == "disabled" and c.nvenc_temporal_aq is False for c, _ in flat)


def test_nvenc_relaxed_strategies_adds_8bit_for_hdr_sources() -> None:
    from aep.pipeline.stages.s08_encode import nvenc_relaxed_strategies

    cfg = EncoderCfg(name="hevc_nvenc", nvenc_multipass="disabled", nvenc_temporal_aq=False)  # type: ignore[arg-type]
    flat = nvenc_relaxed_strategies(cfg, source_is_10bit=True)
    assert flat[-1][1] == "yuv420p"


def test_nvenc_relaxed_strategies_non_nvenc_is_identity() -> None:
    from aep.pipeline.stages.s08_encode import nvenc_relaxed_strategies

    cfg = EncoderCfg(name="libx264")  # type: ignore[arg-type]
    assert nvenc_relaxed_strategies(cfg, source_is_10bit=True) == [(cfg, None)]
