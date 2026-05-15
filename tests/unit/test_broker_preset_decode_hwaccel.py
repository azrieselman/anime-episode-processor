"""Broker preset merge: decode.hwaccel must not be clobbered by app settings."""

from __future__ import annotations

from aep.jobs.broker import merge_preset_data_for_job


def test_merge_keeps_preset_cuda_when_settings_auto() -> None:
    base = {"decode": {"hwaccel": "cuda"}, "encoder": {"name": "hevc_nvenc"}}
    out = merge_preset_data_for_job(
        base, None, settings_decode_hwaccel="auto",
    )
    assert out["decode"]["hwaccel"] == "cuda"


def test_merge_applies_settings_when_preset_auto() -> None:
    base = {"decode": {"hwaccel": "auto"}}
    out = merge_preset_data_for_job(
        base, None, settings_decode_hwaccel="cuda",
    )
    assert out["decode"]["hwaccel"] == "cuda"


def test_merge_job_override_wins_over_preset() -> None:
    base = {"decode": {"hwaccel": "d3d11va"}}
    out = merge_preset_data_for_job(
        base, {"decode": {"hwaccel": "cuda"}}, settings_decode_hwaccel="off",
    )
    assert out["decode"]["hwaccel"] == "cuda"


def test_merge_decode_png_intermediate_codec_override() -> None:
    base = {"decode": {"hwaccel": "off", "png_intermediate_codec": "mjpeg"}}
    out = merge_preset_data_for_job(
        base, {"decode": {"png_intermediate_codec": "libpng"}}, settings_decode_hwaccel="off",
    )
    assert out["decode"]["png_intermediate_codec"] == "libpng"
    assert out["decode"]["hwaccel"] == "off"


def test_merge_settings_fills_auto_after_override() -> None:
    base = {"decode": {"hwaccel": "auto"}}
    out = merge_preset_data_for_job(
        base, {"upscaler": {"scale": 2}}, settings_decode_hwaccel="cuda",
    )
    assert out["decode"]["hwaccel"] == "cuda"
