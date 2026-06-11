"""Smoke tests for ffmpeg encoder argv builders."""

from __future__ import annotations

import pytest

from aep.encode.encoders import QSV_GLOBAL_PREFIX, VULKAN_GLOBAL_PREFIX, build_encoder_args
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
        ("h264_d3d12", "h264_d3d12"),
        ("av1_d3d12", "av1_d3d12"),
        ("h264_vulkan", "h264_vulkan"),
        ("hevc_vulkan", "hevc_vulkan"),
        ("av1_vulkan", "av1_vulkan"),
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


def test_qsv_emits_extbrc_lookahead_and_bframes() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_qsv",
        qsv_extbrc=True,
        qsv_look_ahead_depth=40,
        qsv_bf=3,
        qsv_low_power=False,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-extbrc 1" in joined
    assert "-look_ahead_depth 40" in joined
    assert "-bf 3" in joined
    assert "-low_power 0" in joined


def test_amf_has_no_global_prefix() -> None:
    cfg = EncoderCfg(name="hevc_amf")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    assert r.global_prefix == ()


def test_d3d12_has_no_global_prefix() -> None:
    cfg = EncoderCfg(name="h264_d3d12")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    assert r.global_prefix == ()
    assert "-c:v" in r.args and "h264_d3d12" in r.args


def test_vulkan_sets_global_prefix_and_hwupload() -> None:
    cfg = EncoderCfg(name="hevc_vulkan")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    assert r.global_prefix == VULKAN_GLOBAL_PREFIX
    joined = " ".join(r.args)
    assert "format=nv12,hwupload" in joined
    assert "-c:v hevc_vulkan" in joined


def test_amf_emits_quality_and_hevc_skips_qp_b() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_amf",
        amf_rc="cqp",
        amf_preanalysis=True,
        amf_vbaq=True,
        amf_g=250,
        amf_qp_i=19,
        amf_qp_p=19,
        amf_qp_b=20,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-quality quality" in joined
    assert "-preset" not in joined
    assert "-preanalysis true" in joined
    assert "-vbaq false" in joined
    assert "-g 250" in joined
    assert "-header_insertion_mode" not in joined
    assert "-qp_b" not in joined


def test_amf_high_quality_maps_to_usage() -> None:
    cfg = EncoderCfg(name="hevc_amf", amf_quality="high_quality")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-usage high_quality" in joined
    assert "-quality quality" in joined


def test_h264_amf_emits_qp_b() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="h264_amf",
        amf_rc="cqp",
        amf_qp_i=19,
        amf_qp_p=19,
        amf_qp_b=20,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-qp_b 20" in joined


def test_amf_vbr_emits_maxrate_and_bufsize() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_amf",
        amf_rc="vbr_peak",
        amf_bitrate=6_000_000,
        amf_maxrate=8_000_000,
        amf_bufsize=8_000_000,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-b:v 6000000" in joined
    assert "-maxrate 8000000" in joined
    assert "-bufsize 8000000" in joined
    assert "-qp_i" not in joined


def test_amf_vbr_skips_b_v_when_zero() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_amf",
        amf_rc="vbr_peak",
        amf_bitrate=0,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-b:v" not in joined


def test_nvenc_vbr_emits_configurable_bitrate() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_nvenc",
        nvenc_rc="vbr",
        nvenc_bitrate=10_000_000,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-b:v 10000000" in joined


def test_nvenc_vbr_default_bitrate_zero() -> None:
    cfg = EncoderCfg(name="hevc_nvenc", nvenc_rc="vbr")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-b:v 0" in joined


def test_nvenc_non_vbr_uses_zero_bitrate() -> None:
    cfg = EncoderCfg(name="hevc_nvenc", nvenc_rc="constqp", nvenc_bitrate=8_000_000)  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-b:v 0" in joined
    assert "-b:v 8000000" not in joined


def test_amf_vbr_emits_preencode_when_enabled() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_amf",
        amf_rc="vbr_peak",
        amf_preencode=True,
        amf_vbaq=True,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-preencode true" in joined
    assert "-vbaq true" in joined


def test_amf_cqp_emits_preencode_false() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_amf",
        amf_rc="cqp",
        amf_preencode=True,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-preencode false" in joined


def test_amf_emits_bf_and_pa_options() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="hevc_amf",
        amf_bf=2,
        amf_pa_lookahead_buffer_depth=40,
        amf_pa_taq_mode=2,
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-bf 2" in joined
    assert "-pa_lookahead_buffer_depth 40" in joined
    assert "-pa_taq_mode 2" in joined


def test_amf_skips_auto_pa_options() -> None:
    cfg = EncoderCfg(name="hevc_amf")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-pa_lookahead_buffer_depth" not in joined
    assert "-pa_taq_mode" not in joined
    assert "-bf 3" in joined


def test_nvenc_uses_configurable_rc_lookahead() -> None:
    cfg = EncoderCfg(name="hevc_nvenc", nvenc_rc_lookahead=20)  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-rc-lookahead 20" in joined


def test_x265_prefers_structured_x265_params() -> None:
    cfg = EncoderCfg(  # type: ignore[arg-type]
        name="libx265",
        x265_params="aq-mode=3:psy-rd=2.2:psy-rdoq=1.1:rd=4",
        extra_args=[],
    )
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p10le")
    joined = " ".join(r.args)
    assert "-x265-params aq-mode=3:psy-rd=2.2:psy-rdoq=1.1:rd=4" in joined


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


def test_hevc_amf_auto_10bit_source_uses_p010le() -> None:
    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="auto", amf_preanalysis=True)  # type: ignore[arg-type]
    r = build_encoder_args(
        cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p10le",
    )
    joined = " ".join(r.args)
    assert "-pix_fmt p010le" in joined
    assert "-preanalysis false" in joined
    assert r.pix_fmt == "yuv420p10le"
    vf_idx = r.args.index("-vf")
    vf = r.args[vf_idx + 1]
    assert "out_color_matrix=bt709" in vf
    assert "out_range=tv" in vf
    assert "format=p010le" in vf
    assert any("Main10" in note for note in r.rationale)
    assert any("preanalysis disabled" in note for note in r.rationale)
    assert any("limited (tv)" in note for note in r.rationale)


def test_hevc_amf_force_10_on_8bit_source() -> None:
    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="10")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-pix_fmt p010le" in joined
    vf_idx = r.args.index("-vf")
    assert "out_range=tv" in r.args[vf_idx + 1]
    assert any("amf_bit_depth=10" in note for note in r.rationale)


def test_hevc_amf_force_8_on_10bit_source() -> None:
    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="8")  # type: ignore[arg-type]
    r = build_encoder_args(
        cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p10le",
    )
    joined = " ".join(r.args)
    assert "-pix_fmt yuv420p" in joined
    assert r.pix_fmt == "yuv420p"
    assert any("amf_bit_depth=8" in note for note in r.rationale)


def test_av1_amf_force_10_uses_p010le() -> None:
    cfg = EncoderCfg(name="av1_amf", amf_bit_depth="10")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-pix_fmt p010le" in joined


def test_h264_amf_ignores_bit_depth_10() -> None:
    cfg = EncoderCfg(name="h264_amf", amf_bit_depth="10")  # type: ignore[arg-type]
    r = build_encoder_args(
        cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p10le",
    )
    joined = " ".join(r.args)
    assert "-pix_fmt yuv420p" in joined
    assert any("ignored" in note for note in r.rationale)


def test_hevc_amf_10bit_scale_includes_format_p010le() -> None:
    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="auto")  # type: ignore[arg-type]
    r = build_encoder_args(
        cfg, target_width=1920, target_height=1080, source_pix_fmt="yuv420p10le",
    )
    vf_idx = r.args.index("-vf")
    vf = r.args[vf_idx + 1]
    assert "format=p010le" in vf
    assert "scale=1920:1080" in vf
    assert "out_color_matrix=bt709" in vf
    assert "out_range=tv" in vf


def test_hevc_amf_10bit_emits_bt709_color_metadata() -> None:
    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="10")  # type: ignore[arg-type]
    r = build_encoder_args(cfg, target_width=None, target_height=None, source_pix_fmt="yuv420p")
    joined = " ".join(r.args)
    assert "-color_primaries bt709" in joined
    assert "-color_trc bt709" in joined
    assert "-colorspace bt709" in joined
    assert "-color_range tv" in joined


def test_hevc_amf_10bit_amf_hwaccel_uses_hwdownload() -> None:
    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="auto")  # type: ignore[arg-type]
    r = build_encoder_args(
        cfg,
        target_width=None,
        target_height=None,
        source_pix_fmt="yuv420p10le",
        decode_hwaccel="amf",
    )
    vf_idx = r.args.index("-vf")
    assert "hwdownload,format=p010le" in r.args[vf_idx + 1]


def test_amf_relaxed_strategies_10bit_source_adds_8bit_fallback() -> None:
    from aep.pipeline.stages.s08_encode import amf_relaxed_strategies

    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="auto")  # type: ignore[arg-type]
    flat = amf_relaxed_strategies(cfg, source_is_10bit=True)
    assert len(flat) == 2
    assert flat[0][0].amf_bit_depth == "auto"
    assert flat[1][0].amf_bit_depth == "8"


def test_amf_relaxed_strategies_force_10_adds_8bit_fallback() -> None:
    from aep.pipeline.stages.s08_encode import amf_relaxed_strategies

    cfg = EncoderCfg(name="av1_amf", amf_bit_depth="10")  # type: ignore[arg-type]
    flat = amf_relaxed_strategies(cfg, source_is_10bit=False)
    assert len(flat) == 2
    assert flat[1][0].amf_bit_depth == "8"


def test_amf_relaxed_strategies_force_8_single_attempt() -> None:
    from aep.pipeline.stages.s08_encode import amf_relaxed_strategies

    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="8")  # type: ignore[arg-type]
    flat = amf_relaxed_strategies(cfg, source_is_10bit=True)
    assert flat == [(cfg, None)]


def test_encoder_relaxed_strategies_dispatches_amf() -> None:
    from aep.pipeline.stages.s08_encode import encoder_relaxed_strategies

    cfg = EncoderCfg(name="hevc_amf", amf_bit_depth="10")  # type: ignore[arg-type]
    flat = encoder_relaxed_strategies(cfg, source_is_10bit=False)
    assert len(flat) == 2
    assert flat[1][0].amf_bit_depth == "8"
