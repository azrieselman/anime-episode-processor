"""Tests for the M3 video-path planner helper.

The planner is the single point where preset config + probed media + HDR policy
collapse into a frozen plan that downstream stages obey. It also decides whether
the encode stage ingests frames (M3 path) or a single source file (M2 fallback).
These tests pin the decision matrix so future refactors can't silently change
behavior.
"""

from __future__ import annotations

from aep.media.models import FormatInfo, MediaInfo, StreamInfo
from aep.persist.presets import (
    BatchingCfg,
    InterpolationCfg,
    PostprocessCfg,
    Preset,
    PresetMeta,
    UpscalerCfg,
)
from aep.pipeline.stages.s01_plan import _plan_m3_video_path, _resolve_decode_hwaccel

# ---------- fixtures -------------------------------------------------------


def _make_media(
    *,
    pix_fmt: str = "yuv420p",
    color_transfer: str | None = None,
    avg_frame_rate: str = "24000/1001",
    width: int = 1920,
    height: int = 1080,
) -> tuple[MediaInfo, StreamInfo]:
    primary = StreamInfo(
        index=0,
        kind="video",
        codec_name="h264",
        pix_fmt=pix_fmt,
        color_transfer=color_transfer,
        avg_frame_rate=avg_frame_rate,
        r_frame_rate=avg_frame_rate,
        width=width,
        height=height,
    )
    media = MediaInfo(
        source_path="/tmp/x.mkv",
        fmt=FormatInfo(filename="/tmp/x.mkv", format_name="matroska", duration_s=1440.0),
        streams=[primary],
        is_matroska=True,
    )
    return media, primary


def _make_preset(
    *,
    upscaler_enabled: bool = True,
    engine: str = "realcugan-ncnn-vulkan",
    model: str = "models-pro",
    scale: int = 2,
    denoise: int = 3,
    hdr_policy: str = "skip",
    interp_enabled: bool = True,
    target_fps: float | None = 60.0,
    multiplier: int | None = None,
    pp_enabled: bool = False,
    pp_deband: bool = False,
    pp_grain: int = 0,
    batching_enabled: bool = False,
) -> Preset:
    return Preset(
        meta=PresetMeta(id="test", name="Test"),
        upscaler=UpscalerCfg(
            enabled=upscaler_enabled,
            engine=engine,        # type: ignore[arg-type]
            model=model,
            scale=scale,
            denoise=denoise,
            hdr_policy=hdr_policy,  # type: ignore[arg-type]
        ),
        interpolation=InterpolationCfg(
            enabled=interp_enabled,
            target_fps=target_fps,
            multiplier=multiplier,
        ),
        postprocess=PostprocessCfg(
            enabled=pp_enabled, deband=pp_deband, grain_addback=pp_grain,
        ),
        batching=BatchingCfg(enabled=batching_enabled),
    )


# ---------- mode selection -------------------------------------------------


def test_batching_enabled_forces_frames_mode() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=False,
        interp_enabled=False,
        pp_enabled=False,
        batching_enabled=True,
    )
    plan, _w, rationale = _plan_m3_video_path(preset, media, primary)
    assert plan["encode_input_mode"] == "frames"
    assert plan["decode"]["active"] is True
    assert any("batching" in r for r in rationale)


def test_all_disabled_picks_source_mode() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=False, interp_enabled=False, pp_enabled=False,
    )
    plan, _w, rationale = _plan_m3_video_path(preset, media, primary)
    assert plan["encode_input_mode"] == "source"
    assert plan["upscale"]["active"] is False
    assert plan["interpolate"]["active"] is False
    assert plan["postprocess"]["enabled"] is False
    assert plan["decode"]["active"] is False
    assert any("source" in r for r in rationale)


def test_only_upscaler_enabled_picks_frames_mode() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=True, interp_enabled=False, pp_enabled=False,
    )
    plan, _w, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["encode_input_mode"] == "frames"
    assert plan["upscale"]["active"] is True
    assert plan["decode"]["active"] is True


def test_only_postprocess_enabled_picks_frames_mode() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=False, interp_enabled=False,
        pp_enabled=True, pp_deband=True,
    )
    plan, _w, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["encode_input_mode"] == "frames"
    assert plan["postprocess"]["enabled"] is True


def test_postprocess_enabled_but_no_subfilters_does_not_force_frames() -> None:
    # `enabled=True` with all sub-filters off is effectively a no-op; the
    # planner treats it as inactive so encode-only fast path stays.
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=False, interp_enabled=False,
        pp_enabled=True, pp_deband=False, pp_grain=0,
    )
    plan, _w, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["postprocess"]["enabled"] is False
    assert plan["encode_input_mode"] == "source"


# ---------- HDR routing ----------------------------------------------------


def test_hdr_skip_disables_upscaler_with_warning() -> None:
    media, primary = _make_media(
        pix_fmt="yuv420p10le", color_transfer="smpte2084",
    )
    preset = _make_preset(
        hdr_policy="skip", interp_enabled=False, pp_enabled=False,
    )
    plan, warnings, rationale = _plan_m3_video_path(preset, media, primary)
    assert plan["upscale"]["active"] is False
    assert any("hdr_policy=skip" in w for w in warnings)
    assert any("forced False by hdr_policy=skip" in r for r in rationale)


def test_hdr_allow_8bit_roundtrip_keeps_upscaler_with_warning() -> None:
    media, primary = _make_media(
        pix_fmt="yuv420p10le", color_transfer="smpte2084",
    )
    preset = _make_preset(
        hdr_policy="allow_8bit_roundtrip",
        interp_enabled=False, pp_enabled=False,
    )
    plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["upscale"]["active"] is True
    assert any("8-bit" in w or "round-trip" in w for w in warnings)


def test_high_bit_depth_only_no_hdr_transfer_still_routes_through_hdr_policy() -> None:
    # 10-bit yuv420p10le SDR is still beyond NCNN's 8-bit input.
    media, primary = _make_media(pix_fmt="yuv420p10le", color_transfer="bt709")
    preset = _make_preset(
        hdr_policy="skip", interp_enabled=False, pp_enabled=False,
    )
    plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["upscale"]["active"] is False
    assert any("hdr_policy=skip" in w for w in warnings)


# ---------- waifu2x active (M4) --------------------------------------------


def test_waifu2x_engine_active_with_cunet_2x_denoise3() -> None:
    """Waifu2x on the cunet anime default should plan as active with no warnings."""
    media, primary = _make_media()
    preset = _make_preset(
        engine="waifu2x-ncnn-vulkan",
        model="models-cunet",
        scale=2,
        denoise=3,
        interp_enabled=False,
        pp_enabled=False,
    )
    plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["upscale"]["active"] is True
    assert plan["upscale"]["engine"] == "waifu2x-ncnn-vulkan"
    # No waifu2x-prefixed warnings on the supported anime default.
    assert not any(w.startswith("waifu2x:") for w in warnings)


def test_waifu2x_photo_model_emits_warning_but_stays_active() -> None:
    """Off-target model (photo) warns loudly but does not disable the stage."""
    media, primary = _make_media()
    preset = _make_preset(
        engine="waifu2x-ncnn-vulkan",
        model="models-upconv_7_photo",
        scale=2,
        denoise=3,
        interp_enabled=False,
        pp_enabled=False,
    )
    plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["upscale"]["active"] is True
    assert any("waifu2x:" in w and "photo model" in w for w in warnings)


# ---------- anime4k active --------------------------------------------------


def test_anime4k_balanced_active_with_known_model() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        engine="anime4kcpp",
        model="acnet-hdn-gan",
        scale=2,
        denoise=1,
        interp_enabled=False,
        pp_enabled=False,
    )
    plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["upscale"]["active"] is True
    assert plan["upscale"]["engine"] == "anime4kcpp"
    assert not any(w.startswith("anime4kcpp:") for w in warnings)


def test_anime4k_unknown_model_emits_warning_but_stays_active() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        engine="anime4kcpp",
        model="unknown-model",
        scale=2,
        denoise=1,
        interp_enabled=False,
        pp_enabled=False,
    )
    plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["upscale"]["active"] is True
    assert any("anime4kcpp:" in w and "not in our catalog" in w for w in warnings)


def test_anime4k_vs_active_with_known_model() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        engine="anime4kcpp-vs",
        model="acnet-hdn-gan",
        scale=2,
        denoise=1,
        interp_enabled=False,
        pp_enabled=False,
    )
    plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["upscale"]["active"] is True
    assert plan["upscale"]["engine"] == "anime4kcpp-vs"
    assert not any(w.startswith("anime4kcpp-vs:") for w in warnings)


# ---------- output_fps math ------------------------------------------------


def test_output_fps_24000_1001_x2_is_48000_1001() -> None:
    media, primary = _make_media(avg_frame_rate="24000/1001")
    preset = _make_preset(
        upscaler_enabled=False, interp_enabled=True,
        target_fps=None, multiplier=2, pp_enabled=False,
    )
    plan, _w, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["interpolate"]["active"] is True
    assert plan["interpolate"]["multiplier"] == 2
    assert plan["output_fps"] == "48000/1001"


def test_output_fps_no_interpolation_uses_source_rate() -> None:
    media, primary = _make_media(avg_frame_rate="24/1")
    preset = _make_preset(
        upscaler_enabled=True, interp_enabled=False, pp_enabled=False,
    )
    plan, _w, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["interpolate"]["active"] is False
    assert plan["output_fps"] == "24/1"


def test_interpolation_collapses_to_inactive_when_multiplier_is_one() -> None:
    # Source already at target → multiplier=1 → planner downgrades to inactive.
    media, primary = _make_media(avg_frame_rate="60/1")
    preset = _make_preset(
        upscaler_enabled=False, interp_enabled=True,
        target_fps=60.0, multiplier=None, pp_enabled=False,
    )
    plan, _w, rationale = _plan_m3_video_path(preset, media, primary)
    assert plan["interpolate"]["active"] is False
    assert any("multiplier=1" in r for r in rationale)


# ---------- decode plan ----------------------------------------------------


def test_decode_inherits_upscaler_frame_format() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=False, interp_enabled=False,
        pp_enabled=True, pp_deband=True,
    )
    # Force webp via direct construction since helper doesn't expose it.
    preset.upscaler = UpscalerCfg(
        enabled=False, engine="none", intermediate_format="webp",
    )
    plan, _w, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["decode"]["frame_format"] == "webp"
    assert plan["postprocess"]["frame_format"] == "webp"


def test_decode_target_geometry_set_when_resizing_without_upscaler() -> None:
    # No upscaler in chain but target is 1080p while source is 720p →
    # decode pre-resizes so the encoder sees final geometry directly.
    media, primary = _make_media(width=1280, height=720)
    preset = _make_preset(
        upscaler_enabled=False, interp_enabled=False,
        pp_enabled=True, pp_deband=True,
    )
    # Default target_resolution is named=1440p (per Preset default).
    plan, _w, _r = _plan_m3_video_path(preset, media, primary)
    assert plan["decode"]["target_w"] == 2560
    assert plan["decode"]["target_h"] == 1440


def test_decode_target_geometry_unset_when_upscaler_active() -> None:
    media, primary = _make_media(width=1280, height=720)
    preset = _make_preset(upscaler_enabled=True, interp_enabled=False, pp_enabled=False)
    plan, _w, _r = _plan_m3_video_path(preset, media, primary)
    # Upscaler will set the final pixel size; decode stays at source.
    assert plan["decode"]["target_w"] is None
    assert plan["decode"]["target_h"] is None


def test_decode_hwaccel_is_written_to_decode_plan() -> None:
    media, primary = _make_media()
    preset = _make_preset(upscaler_enabled=False, interp_enabled=False, pp_enabled=True, pp_deband=True)
    plan, _w, _r = _plan_m3_video_path(preset, media, primary, decode_hwaccel="d3d11va")
    assert plan["decode"]["hwaccel"] == "d3d11va"


def test_resolve_decode_hwaccel_auto_windows(monkeypatch) -> None:
    import aep.pipeline.stages.s01_plan as s01
    monkeypatch.setattr(s01.os, "name", "nt")
    assert _resolve_decode_hwaccel("auto") == "d3d11va"


def test_resolve_decode_hwaccel_auto_non_windows(monkeypatch) -> None:
    import aep.pipeline.stages.s01_plan as s01
    monkeypatch.setattr(s01.os, "name", "posix")
    assert _resolve_decode_hwaccel("auto") == "off"


# ---------- input_source chain --------------------------------------------


def test_input_sources_full_chain_interpolate_first() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=True, interp_enabled=True,
        target_fps=None, multiplier=2,
        pp_enabled=True, pp_deband=True,
    )
    plan, _w, _r = _plan_m3_video_path(
        preset, media, primary, pipeline_order="interpolate_first",
    )
    assert plan["pipeline_order"] == "interpolate_first"
    assert plan["interpolate"]["input_source"] == "decode"
    assert plan["upscale"]["input_source"] == "interpolate"
    assert plan["postprocess"]["input_source"] == "upscale"
    assert plan["encode_input_source"] == "postprocess"


def test_input_sources_full_chain_upscale_first() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=True, interp_enabled=True,
        target_fps=None, multiplier=2,
        pp_enabled=True, pp_deband=True,
    )
    plan, _w, _r = _plan_m3_video_path(
        preset, media, primary, pipeline_order="upscale_first",
    )
    assert plan["pipeline_order"] == "upscale_first"
    assert plan["interpolate"]["input_source"] == "upscale"
    assert plan["upscale"]["input_source"] == "decode"
    assert plan["postprocess"]["input_source"] == "interpolate"
    assert plan["encode_input_source"] == "postprocess"


def test_encode_input_source_without_postprocess() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=True, interp_enabled=True,
        target_fps=None, multiplier=2, pp_enabled=False,
    )
    plan_if, _, _ = _plan_m3_video_path(
        preset, media, primary, pipeline_order="interpolate_first",
    )
    assert plan_if["encode_input_source"] == "upscale"
    plan_uf, _, _ = _plan_m3_video_path(
        preset, media, primary, pipeline_order="upscale_first",
    )
    assert plan_uf["encode_input_source"] == "interpolate"


def test_postprocess_input_source_partial_chains() -> None:
    media, primary = _make_media()
    preset_up = _make_preset(
        upscaler_enabled=True, interp_enabled=False,
        pp_enabled=True, pp_deband=True,
    )
    for order in ("interpolate_first", "upscale_first"):
        plan, _, _ = _plan_m3_video_path(preset_up, media, primary, pipeline_order=order)  # type: ignore[arg-type]
        assert plan["postprocess"]["input_source"] == "upscale"

    preset_pp = _make_preset(
        upscaler_enabled=False, interp_enabled=False,
        pp_enabled=True, pp_deband=True,
    )
    for order in ("interpolate_first", "upscale_first"):
        plan, _, _ = _plan_m3_video_path(preset_pp, media, primary, pipeline_order=order)  # type: ignore[arg-type]
        assert plan["postprocess"]["input_source"] == "decode"


def test_plan_m3_default_pipeline_order_is_interpolate_first() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        upscaler_enabled=True, interp_enabled=True,
        target_fps=None, multiplier=2, pp_enabled=False,
    )
    plan, _, _ = _plan_m3_video_path(preset, media, primary)
    assert plan["pipeline_order"] == "interpolate_first"
    assert plan["interpolate"]["input_source"] == "decode"
    assert plan["upscale"]["input_source"] == "interpolate"


# ---------- engine validation surfacing -----------------------------------


def test_realcugan_off_grid_combination_emits_warning() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        engine="realcugan-ncnn-vulkan", model="models-pro",
        scale=4, denoise=3,   # not in _SUPPORTED_PRO
        interp_enabled=False, pp_enabled=False,
    )
    _plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert any("realcugan" in w for w in warnings)


def test_realesrgan_stills_model_emits_warning() -> None:
    media, primary = _make_media()
    preset = _make_preset(
        engine="realesrgan-ncnn-vulkan", model="realesrgan-x4plus-anime",
        scale=4, interp_enabled=False, pp_enabled=False,
    )
    _plan, warnings, _r = _plan_m3_video_path(preset, media, primary)
    assert any("realesrgan" in w for w in warnings)
