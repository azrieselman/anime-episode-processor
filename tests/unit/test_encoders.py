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
