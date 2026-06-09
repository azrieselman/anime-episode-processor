"""Tests for `aep.encode.recommender`."""

from __future__ import annotations

from aep.bench.hardware import CpuInfo, GpuCapabilities, HardwareProfile
from aep.encode.recommender import recommend
from aep.persist.presets import (
    EncoderCfg,
    Preset,
    PresetMeta,
)


def _hw(
    *,
    has_nv: bool = True,
    h264: bool = True,
    hevc: bool = True,
    av1: bool = False,
    arch: str | None = "ampere",
    ffmpeg_encoders: list[str] | None = None,
    qsv_h264: bool = False,
    qsv_hevc: bool = False,
    qsv_av1: bool = False,
    amf_h264: bool = False,
    amf_hevc: bool = False,
    amf_av1: bool = False,
    primary_vendor: str = "unknown",
) -> HardwareProfile:
    return HardwareProfile(
        cpu=CpuInfo(logical_cores=8, ram_total_mib=32 * 1024),
        gpu=GpuCapabilities(
            has_nvidia=has_nv,
            nvenc_h264=h264 and has_nv,
            nvenc_hevc=hevc and has_nv,
            nvenc_av1=av1 and has_nv,
            arch=arch,
            vram_total_mib=10 * 1024 if has_nv else 0,
            vram_free_mib=8 * 1024 if has_nv else 0,
            driver_version="555.85" if has_nv else None,
            primary_vendor=primary_vendor,  # type: ignore[arg-type]
            qsv_h264=qsv_h264,
            qsv_hevc=qsv_hevc,
            qsv_av1=qsv_av1,
            amf_h264=amf_h264,
            amf_hevc=amf_hevc,
            amf_av1=amf_av1,
        ),
        ffmpeg_version="7.0.2",
        ffmpeg_encoders=ffmpeg_encoders or [
            "h264_nvenc", "hevc_nvenc", "libx264", "libx265",
        ],
    )


def _preset(encoder_name: str = "hevc_nvenc") -> Preset:
    return Preset(
        meta=PresetMeta(id="t", name="t"),
        encoder=EncoderCfg(name=encoder_name),  # type: ignore[arg-type]
    )


def test_hevc_nvenc_passes_through_on_compatible_gpu():
    rec = recommend(_preset("hevc_nvenc"), hardware=_hw())
    assert rec.encoder.name == "hevc_nvenc"
    assert not rec.warnings


def test_av1_nvenc_falls_back_to_hevc_on_ampere():
    rec = recommend(_preset("av1_nvenc"), hardware=_hw(av1=False, arch="ampere"))
    assert rec.encoder.name == "hevc_nvenc"
    assert any("av1_nvenc" in w for w in rec.warnings)


def test_no_nvidia_falls_back_to_software():
    rec = recommend(_preset("hevc_nvenc"), hardware=_hw(has_nv=False, arch=None))
    assert rec.encoder.name == "libx265"
    assert any("NVIDIA" in w for w in rec.warnings)


def test_h264_nvenc_warns_for_10bit_source():
    rec = recommend(_preset("h264_nvenc"), hardware=_hw(),
                    source_pix_fmt="yuv420p10le")
    assert rec.encoder.name == "h264_nvenc"
    assert any("10-bit" in w for w in rec.warnings)


def test_h264_target_warns_for_hevc_source():
    rec = recommend(_preset("h264_nvenc"), hardware=_hw(), source_codec="hevc")
    assert any("HEVC" in w for w in rec.warnings)


def test_goal_quality_bumps_nvenc_to_p7():
    rec = recommend(_preset("hevc_nvenc"), hardware=_hw(), goal="quality")
    assert rec.encoder.nvenc_preset == "p7"
    assert rec.encoder.nvenc_cq <= 19
    assert rec.encoder.nvenc_rc_lookahead >= 24


def test_goal_speed_relaxes_nvenc():
    rec = recommend(_preset("hevc_nvenc"), hardware=_hw(), goal="speed")
    assert rec.encoder.nvenc_preset == "p4"
    assert rec.encoder.nvenc_temporal_aq is False


def test_goal_compat_forces_h264():
    rec = recommend(_preset("hevc_nvenc"), hardware=_hw(), goal="compat")
    assert rec.encoder.name == "h264_nvenc"


def test_software_fallback_for_missing_libx265():
    hw = _hw(ffmpeg_encoders=["h264_nvenc", "hevc_nvenc", "libx264"])  # no libx265
    rec = recommend(_preset("libx265"), hardware=hw)
    # Should fall to hevc_nvenc since GPU has it.
    assert rec.encoder.name == "hevc_nvenc"


def test_goal_quality_tunes_qsv_new_fields() -> None:
    hw = _hw(
        has_nv=False,
        arch=None,
        ffmpeg_encoders=["hevc_qsv", "libx264", "libx265"],
        qsv_hevc=True,
        primary_vendor="intel",
    )
    rec = recommend(_preset("hevc_qsv"), hardware=hw, goal="quality")
    assert rec.encoder.qsv_extbrc is True
    assert rec.encoder.qsv_look_ahead_depth >= 40
    assert rec.encoder.qsv_low_power is False


def test_amf_cqp_disables_vbaq_on_quality_goal() -> None:
    hw = _hw(
        has_nv=False,
        arch=None,
        ffmpeg_encoders=["hevc_amf", "libx264", "libx265"],
        amf_hevc=True,
        primary_vendor="amd",
    )
    p = _preset("hevc_amf")
    p.encoder = p.encoder.model_copy(update={"amf_rc": "cqp", "amf_vbaq": True})
    rec = recommend(p, hardware=hw, goal="quality")
    assert rec.encoder.amf_vbaq is False
    assert any("VBAQ disabled" in r for r in rec.rationale)


def test_goal_speed_tunes_amf_new_fields() -> None:
    hw = _hw(
        has_nv=False,
        arch=None,
        ffmpeg_encoders=["hevc_amf", "libx264", "libx265"],
        amf_hevc=True,
        primary_vendor="amd",
    )
    rec = recommend(_preset("hevc_amf"), hardware=hw, goal="speed")
    assert rec.encoder.amf_quality == "speed"
    assert rec.encoder.amf_preanalysis is False
    assert rec.encoder.amf_vbaq is False


def test_prefer_hardware_encoder_false_skips_hardware_fallback() -> None:
    hw = _hw(ffmpeg_encoders=["h264_nvenc", "hevc_nvenc", "libx264"])  # no libx265
    rec = recommend(_preset("libx265"), hardware=hw, prefer_hardware_encoder=False)
    assert rec.encoder.name == "libx265"
    assert any("disabled by settings" in w for w in rec.warnings)


def test_goal_auto_uses_encoder_cfg_goal() -> None:
    p = _preset("hevc_nvenc")
    p.encoder = p.encoder.model_copy(update={"goal": "speed"})
    rec = recommend(p, hardware=_hw(), goal="auto")
    assert rec.encoder.nvenc_preset == "p4"


def test_rationale_always_records_final_encoder():
    rec = recommend(_preset("hevc_nvenc"), hardware=_hw())
    assert any(r.startswith("Final encoder:") for r in rec.rationale)


def test_hevc_qsv_passes_when_capable():
    hw = _hw(
        has_nv=False,
        arch=None,
        ffmpeg_encoders=["hevc_qsv", "libx264", "libx265"],
        qsv_hevc=True,
        primary_vendor="intel",
    )
    rec = recommend(_preset("hevc_qsv"), hardware=hw)
    assert rec.encoder.name == "hevc_qsv"


def test_hevc_qsv_falls_back_without_intel_caps():
    hw = _hw(
        has_nv=False,
        arch=None,
        ffmpeg_encoders=["hevc_qsv", "libx264", "libx265"],
        qsv_hevc=False,
    )
    rec = recommend(_preset("hevc_qsv"), hardware=hw)
    assert rec.encoder.name == "libx265"


def test_hevc_amf_passes_when_capable():
    hw = _hw(
        has_nv=False,
        arch=None,
        ffmpeg_encoders=["hevc_amf", "libx264", "libx265"],
        amf_hevc=True,
        primary_vendor="amd",
    )
    rec = recommend(_preset("hevc_amf"), hardware=hw)
    assert rec.encoder.name == "hevc_amf"


def test_goal_compat_maps_hevc_qsv_to_h264_qsv():
    hw = _hw(
        has_nv=False,
        arch=None,
        ffmpeg_encoders=["hevc_qsv", "h264_qsv", "libx264", "libx265"],
        qsv_hevc=True,
        qsv_h264=True,
        primary_vendor="intel",
    )
    rec = recommend(_preset("hevc_qsv"), hardware=hw, goal="compat")
    assert rec.encoder.name == "h264_qsv"
