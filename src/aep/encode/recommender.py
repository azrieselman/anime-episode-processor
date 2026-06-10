"""Encoder recommendation engine.

Given:
  * the user's preset (their stated preference)
  * the source MediaInfo (resolution, codec, bit depth, etc.)
  * the HardwareProfile (what's actually available)
  * an optional "goal" override ("quality" | "speed" | "archival" | "compat")

Returns an `EncoderRecommendation` with:
  * the encoder NAME we'll actually use (may differ from the preset if unavailable)
  * the (possibly adjusted) EncoderCfg
  * a list of rationale lines suitable for showing in the UI / writing to the manifest
  * a list of warnings (e.g. "preset asked for av1_nvenc but your GPU isn't Ada-class;
    falling back to hevc_nvenc")

Goal of the design: never silently override the user. If the preset asks for hardware
encoding that isn't available, we DO fall back, but every fallback is accompanied by an
explicit rationale string the GUI surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from aep.bench.hardware import HardwareProfile
from aep.persist.presets import EncoderCfg, Preset, coerce_amf_encoder

Goal = Literal["quality", "speed", "archival", "compat", "auto"]


@dataclass
class EncoderRecommendation:
    encoder: EncoderCfg
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.encoder.name


# Software fallback when a hardware encoder is unavailable.
_HW_TO_SOFTWARE: dict[str, str] = {
    "h264_nvenc": "libx264",
    "hevc_nvenc": "libx265",
    "av1_nvenc": "libx265",
    "h264_qsv": "libx264",
    "hevc_qsv": "libx265",
    "av1_qsv": "libx265",
    "h264_amf": "libx264",
    "hevc_amf": "libx265",
    "av1_amf": "libx265",
    "h264_d3d12": "libx264",
    "av1_d3d12": "libx265",
    "h264_vulkan": "libx264",
    "hevc_vulkan": "libx265",
    "av1_vulkan": "libx265",
}

_SOFTWARE_TO_HW_PREFERENCE: list[tuple[str, str]] = [
    ("libx264", "h264_nvenc"),
    ("libx264", "h264_qsv"),
    ("libx264", "h264_amf"),
    ("libx264", "h264_d3d12"),
    ("libx264", "h264_vulkan"),
    ("libx265", "hevc_nvenc"),
    ("libx265", "hevc_qsv"),
    ("libx265", "hevc_amf"),
    ("libx265", "hevc_vulkan"),
    ("libx265", "av1_vulkan"),
    ("libx265", "av1_d3d12"),
]


def _encoder_capable(name: str, hardware: HardwareProfile) -> bool:
    if not hardware.has_encoder(name):
        return False
    return bool(
        (name == "h264_d3d12" and hardware.gpu.d3d12_h264)
        or (name == "av1_d3d12" and hardware.gpu.d3d12_av1)
        or (name == "h264_vulkan" and hardware.gpu.vulkan_h264)
        or (name == "hevc_vulkan" and hardware.gpu.vulkan_hevc)
        or (name == "av1_vulkan" and hardware.gpu.vulkan_av1)
    )


def _universal_hw_candidates(name: str) -> list[str]:
    codec = name.split("_", 1)[0]
    if codec == "h264":
        return ["h264_d3d12", "h264_vulkan"]
    if codec == "av1":
        return ["av1_d3d12", "av1_vulkan"]
    if codec == "hevc":
        return ["hevc_vulkan"]
    return []


def _maybe_fallback_to_universal_hw(
    cfg: EncoderCfg,
    hardware: HardwareProfile,
    *,
    prefer_hardware_encoder: bool,
) -> tuple[EncoderCfg | None, list[str], list[str]]:
    warnings: list[str] = []
    rationale: list[str] = []
    for alt in _universal_hw_candidates(cfg.name):
        if not _encoder_capable(alt, hardware):
            continue
        warnings.append(f"Falling back to {alt}.")
        cfg2 = cfg.model_copy(update={"name": alt})  # type: ignore[arg-type]
        cfg3, w2, r2 = _enforce_hardware_availability(
            cfg2,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg3, warnings + w2, rationale + r2
    return None, warnings, rationale


def recommend(
    preset: Preset,
    *,
    hardware: HardwareProfile,
    source_codec: str | None = None,
    source_pix_fmt: str | None = None,
    goal: Goal = "auto",
    prefer_hardware_encoder: bool = True,
) -> EncoderRecommendation:
    cfg = preset.encoder.model_copy(deep=True)
    rationale: list[str] = []
    warnings: list[str] = []

    requested = cfg.name
    rationale.append(f"Preset '{preset.meta.id}' requests encoder: {requested}")

    cfg, fb_warnings, fb_rationale = _enforce_hardware_availability(
        cfg,
        hardware,
        prefer_hardware_encoder=prefer_hardware_encoder,
    )
    rationale.extend(fb_rationale)
    warnings.extend(fb_warnings)

    effective_goal: Goal = cfg.goal if goal == "auto" else goal
    cfg, goal_rationale = _apply_goal(cfg, effective_goal)
    rationale.extend(goal_rationale)

    # compat changes encoder family (e.g. HEVC→H.264); re-validate against hardware.
    if effective_goal == "compat":
        cfg, fb2, _ = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        warnings.extend(fb2)

    if source_pix_fmt and "10" in source_pix_fmt and cfg.name in {
        "h264_nvenc", "h264_qsv", "h264_amf", "h264_d3d12", "h264_vulkan",
    }:
        warnings.append(
            "Source is 10-bit but H.264 output will downconvert to 8-bit for compatibility. "
            "Use HEVC or libx265 to retain bit depth."
        )

    if source_codec and source_codec.lower() in {"hevc", "h265"} and cfg.name in {
        "h264_nvenc", "h264_qsv", "h264_amf", "h264_d3d12", "h264_vulkan", "libx264",
    }:
        warnings.append(
            "Source is HEVC but you selected H.264 output; this typically increases "
            "filesize for the same quality."
        )

    cfg, amf_rationale = _coerce_amf(cfg)
    rationale.extend(amf_rationale)

    rationale.append(f"Final encoder: {cfg.name}")
    return EncoderRecommendation(encoder=cfg, rationale=rationale, warnings=warnings)


def _coerce_amf(cfg: EncoderCfg) -> tuple[EncoderCfg, list[str]]:
    if not cfg.name.endswith("_amf"):
        return cfg, []
    before_vbaq = cfg.amf_vbaq
    before_preencode = cfg.amf_preencode
    cfg = coerce_amf_encoder(cfg)
    rationale: list[str] = []
    if before_vbaq and not cfg.amf_vbaq:
        rationale.append("AMF VBAQ disabled: incompatible with CQP rate control.")
    if before_preencode and not cfg.amf_preencode:
        rationale.append("AMF preencode disabled: incompatible with CQP rate control.")
    return cfg, rationale


def _nvenc_chain(
    cfg: EncoderCfg,
    hardware: HardwareProfile,
    *,
    prefer_hardware_encoder: bool,
) -> tuple[EncoderCfg, list[str], list[str]]:
    warnings: list[str] = []
    rationale: list[str] = []
    name = cfg.name

    if not hardware.gpu.has_nvidia:
        warnings.append(f"{name} requires an NVIDIA GPU; none detected.")
        cfg2, w2, r2 = _maybe_fallback_to_universal_hw(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        if cfg2 is not None:
            return cfg2, warnings + w2, rationale + r2
        sw = _HW_TO_SOFTWARE.get(name, "libx264")
        warnings.append(f"Falling back to {sw}.")
        cfg = cfg.model_copy(update={"name": sw})  # type: ignore[arg-type]
        return cfg, warnings, rationale

    if name == "av1_nvenc" and not hardware.gpu.nvenc_av1:
        warnings.append(
            f"av1_nvenc requires Ada (RTX 40xx) or newer; your arch is "
            f"{hardware.gpu.arch}. Falling back to hevc_nvenc."
        )
        cfg = cfg.model_copy(update={"name": "hevc_nvenc"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2

    if name == "hevc_nvenc" and not hardware.gpu.nvenc_hevc:
        warnings.append("hevc_nvenc unavailable; falling back to h264_nvenc.")
        cfg = cfg.model_copy(update={"name": "h264_nvenc"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2

    if name == "h264_nvenc" and not hardware.gpu.nvenc_h264:
        warnings.append("h264_nvenc unavailable.")
        cfg2, w2, r2 = _maybe_fallback_to_universal_hw(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        if cfg2 is not None:
            return cfg2, warnings + w2, rationale + r2
        cfg = cfg.model_copy(update={"name": "libx264"})  # type: ignore[arg-type]
        warnings.append("Falling back to libx264.")
        return cfg, warnings, rationale

    rationale.append(
        f"{name} confirmed available "
        f"(arch={hardware.gpu.arch}, vram={hardware.gpu.vram_total_mib} MiB)."
    )
    return cfg, warnings, rationale


def _qsv_chain(
    cfg: EncoderCfg,
    hardware: HardwareProfile,
    *,
    prefer_hardware_encoder: bool,
) -> tuple[EncoderCfg, list[str], list[str]]:
    warnings: list[str] = []
    rationale: list[str] = []
    name = cfg.name
    caps = {
        "h264_qsv": hardware.gpu.qsv_h264,
        "hevc_qsv": hardware.gpu.qsv_hevc,
        "av1_qsv": hardware.gpu.qsv_av1,
    }
    if caps.get(name, False):
        rationale.append(f"{name} confirmed available (Intel QSV path).")
        return cfg, warnings, rationale

    warnings.append(f"{name} is not available (driver/ffmpeg or no Intel adapter).")
    if name == "av1_qsv" and hardware.gpu.qsv_hevc and hardware.has_encoder("hevc_qsv"):
        warnings.append("Falling back to hevc_qsv.")
        cfg = cfg.model_copy(update={"name": "hevc_qsv"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2
    if name in {"av1_qsv", "hevc_qsv"} and hardware.gpu.qsv_h264 and hardware.has_encoder("h264_qsv"):
        warnings.append("Falling back to h264_qsv.")
        cfg = cfg.model_copy(update={"name": "h264_qsv"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2

    cfg2, w2, r2 = _maybe_fallback_to_universal_hw(
        cfg,
        hardware,
        prefer_hardware_encoder=prefer_hardware_encoder,
    )
    if cfg2 is not None:
        return cfg2, warnings + w2, rationale + r2

    sw = _HW_TO_SOFTWARE[name]
    warnings.append(f"Falling back to {sw}.")
    cfg = cfg.model_copy(update={"name": sw})  # type: ignore[arg-type]
    return cfg, warnings, rationale


def _amf_chain(
    cfg: EncoderCfg,
    hardware: HardwareProfile,
    *,
    prefer_hardware_encoder: bool,
) -> tuple[EncoderCfg, list[str], list[str]]:
    warnings: list[str] = []
    rationale: list[str] = []
    name = cfg.name
    caps = {
        "h264_amf": hardware.gpu.amf_h264,
        "hevc_amf": hardware.gpu.amf_hevc,
        "av1_amf": hardware.gpu.amf_av1,
    }
    if caps.get(name, False):
        rationale.append(f"{name} confirmed available (AMD AMF path).")
        return cfg, warnings, rationale

    warnings.append(f"{name} is not available (driver/ffmpeg or no AMD adapter).")
    if name == "av1_amf" and hardware.gpu.amf_hevc and hardware.has_encoder("hevc_amf"):
        warnings.append("Falling back to hevc_amf.")
        cfg = cfg.model_copy(update={"name": "hevc_amf"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2
    if name in {"av1_amf", "hevc_amf"} and hardware.gpu.amf_h264 and hardware.has_encoder("h264_amf"):
        warnings.append("Falling back to h264_amf.")
        cfg = cfg.model_copy(update={"name": "h264_amf"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2

    cfg2, w2, r2 = _maybe_fallback_to_universal_hw(
        cfg,
        hardware,
        prefer_hardware_encoder=prefer_hardware_encoder,
    )
    if cfg2 is not None:
        return cfg2, warnings + w2, rationale + r2

    sw = _HW_TO_SOFTWARE[name]
    warnings.append(f"Falling back to {sw}.")
    cfg = cfg.model_copy(update={"name": sw})  # type: ignore[arg-type]
    return cfg, warnings, rationale


def _d3d12_chain(
    cfg: EncoderCfg,
    hardware: HardwareProfile,
    *,
    prefer_hardware_encoder: bool,
) -> tuple[EncoderCfg, list[str], list[str]]:
    warnings: list[str] = []
    rationale: list[str] = []
    name = cfg.name
    caps = {
        "h264_d3d12": hardware.gpu.d3d12_h264,
        "av1_d3d12": hardware.gpu.d3d12_av1,
    }
    if caps.get(name, False):
        rationale.append(f"{name} confirmed available (DirectX D3D12 path).")
        return cfg, warnings, rationale

    warnings.append(f"{name} is not available in this ffmpeg/driver D3D12 path.")
    if name == "av1_d3d12" and hardware.gpu.d3d12_h264 and hardware.has_encoder("h264_d3d12"):
        warnings.append("Falling back to h264_d3d12.")
        cfg = cfg.model_copy(update={"name": "h264_d3d12"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2

    if name == "av1_d3d12" and _encoder_capable("av1_vulkan", hardware):
        warnings.append("Falling back to av1_vulkan.")
        cfg = cfg.model_copy(update={"name": "av1_vulkan"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2
    if name == "h264_d3d12" and _encoder_capable("h264_vulkan", hardware):
        warnings.append("Falling back to h264_vulkan.")
        cfg = cfg.model_copy(update={"name": "h264_vulkan"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2

    sw = _HW_TO_SOFTWARE[name]
    warnings.append(f"Falling back to {sw}.")
    cfg = cfg.model_copy(update={"name": sw})  # type: ignore[arg-type]
    return cfg, warnings, rationale


def _vulkan_chain(
    cfg: EncoderCfg,
    hardware: HardwareProfile,
    *,
    prefer_hardware_encoder: bool,
) -> tuple[EncoderCfg, list[str], list[str]]:
    warnings: list[str] = []
    rationale: list[str] = []
    name = cfg.name
    caps = {
        "h264_vulkan": hardware.gpu.vulkan_h264,
        "hevc_vulkan": hardware.gpu.vulkan_hevc,
        "av1_vulkan": hardware.gpu.vulkan_av1,
    }
    if caps.get(name, False):
        rationale.append(f"{name} confirmed available (Vulkan path).")
        return cfg, warnings, rationale

    warnings.append(f"{name} is not available in this ffmpeg/driver Vulkan path.")
    if name == "av1_vulkan" and hardware.gpu.vulkan_hevc and hardware.has_encoder("hevc_vulkan"):
        warnings.append("Falling back to hevc_vulkan.")
        cfg = cfg.model_copy(update={"name": "hevc_vulkan"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2
    if name in {"av1_vulkan", "hevc_vulkan"} and hardware.gpu.vulkan_h264 and hardware.has_encoder("h264_vulkan"):
        warnings.append("Falling back to h264_vulkan.")
        cfg = cfg.model_copy(update={"name": "h264_vulkan"})  # type: ignore[arg-type]
        cfg2, w2, r2 = _enforce_hardware_availability(
            cfg,
            hardware,
            prefer_hardware_encoder=prefer_hardware_encoder,
        )
        return cfg2, warnings + w2, rationale + r2

    if name in {"h264_vulkan", "av1_vulkan"}:
        d3d12_alt = f"{name.split('_', 1)[0]}_d3d12"
        if _encoder_capable(d3d12_alt, hardware):
            warnings.append(f"Falling back to {d3d12_alt}.")
            cfg = cfg.model_copy(update={"name": d3d12_alt})  # type: ignore[arg-type]
            cfg2, w2, r2 = _enforce_hardware_availability(
                cfg,
                hardware,
                prefer_hardware_encoder=prefer_hardware_encoder,
            )
            return cfg2, warnings + w2, rationale + r2

    sw = _HW_TO_SOFTWARE[name]
    warnings.append(f"Falling back to {sw}.")
    cfg = cfg.model_copy(update={"name": sw})  # type: ignore[arg-type]
    return cfg, warnings, rationale


def _enforce_hardware_availability(
    cfg: EncoderCfg,
    hardware: HardwareProfile,
    *,
    prefer_hardware_encoder: bool,
) -> tuple[EncoderCfg, list[str], list[str]]:
    warnings: list[str] = []
    rationale: list[str] = []
    name = cfg.name

    if name in {"h264_nvenc", "hevc_nvenc", "av1_nvenc"}:
        return _nvenc_chain(cfg, hardware, prefer_hardware_encoder=prefer_hardware_encoder)

    if name in {"h264_qsv", "hevc_qsv", "av1_qsv"}:
        return _qsv_chain(cfg, hardware, prefer_hardware_encoder=prefer_hardware_encoder)

    if name in {"h264_amf", "hevc_amf", "av1_amf"}:
        return _amf_chain(cfg, hardware, prefer_hardware_encoder=prefer_hardware_encoder)

    if name in {"h264_d3d12", "av1_d3d12"}:
        return _d3d12_chain(cfg, hardware, prefer_hardware_encoder=prefer_hardware_encoder)

    if name in {"h264_vulkan", "hevc_vulkan", "av1_vulkan"}:
        return _vulkan_chain(cfg, hardware, prefer_hardware_encoder=prefer_hardware_encoder)

    if name in {"libx264", "libx265"} and not hardware.has_encoder(name):
        warnings.append(f"{name} not present in this ffmpeg build.")
        if not prefer_hardware_encoder:
            warnings.append("Hardware encoder fallback disabled by settings.")
            return cfg, warnings, rationale
        for sw_name, hw_name in _SOFTWARE_TO_HW_PREFERENCE:
            if sw_name != name:
                continue
            if not hardware.has_encoder(hw_name):
                continue
            ok = False
            if hw_name.endswith("_nvenc"):
                ok = bool(
                    (hw_name == "h264_nvenc" and hardware.gpu.nvenc_h264) or
                    (hw_name == "hevc_nvenc" and hardware.gpu.nvenc_hevc)
                )
            elif hw_name.endswith("_qsv"):
                ok = bool(
                    (hw_name == "h264_qsv" and hardware.gpu.qsv_h264) or
                    (hw_name == "hevc_qsv" and hardware.gpu.qsv_hevc)
                )
            elif hw_name.endswith("_amf"):
                ok = bool(
                    (hw_name == "h264_amf" and hardware.gpu.amf_h264) or
                    (hw_name == "hevc_amf" and hardware.gpu.amf_hevc)
                )
            elif hw_name.endswith("_d3d12"):
                ok = bool(
                    (hw_name == "h264_d3d12" and hardware.gpu.d3d12_h264) or
                    (hw_name == "av1_d3d12" and hardware.gpu.d3d12_av1)
                )
            elif hw_name.endswith("_vulkan"):
                ok = bool(
                    (hw_name == "h264_vulkan" and hardware.gpu.vulkan_h264) or
                    (hw_name == "hevc_vulkan" and hardware.gpu.vulkan_hevc) or
                    (hw_name == "av1_vulkan" and hardware.gpu.vulkan_av1)
                )
            if ok:
                warnings.append(f"Falling back to {hw_name}.")
                cfg = cfg.model_copy(update={"name": hw_name})  # type: ignore[arg-type]
                cfg2, w2, r2 = _enforce_hardware_availability(
                    cfg,
                    hardware,
                    prefer_hardware_encoder=prefer_hardware_encoder,
                )
                return cfg2, warnings + w2, rationale + r2

    return cfg, warnings, rationale


def _apply_goal(cfg: EncoderCfg, goal: Goal) -> tuple[EncoderCfg, list[str]]:
    rationale: list[str] = []
    if goal == "auto":
        return cfg, rationale

    name = cfg.name
    is_nvenc = name.endswith("_nvenc")
    is_qsv = name.endswith("_qsv")
    is_amf = name.endswith("_amf")
    is_d3d12 = name.endswith("_d3d12")
    is_vulkan = name.endswith("_vulkan")
    is_hw = is_nvenc or is_qsv or is_amf or is_d3d12 or is_vulkan

    if goal == "quality":
        if is_nvenc:
            cfg = cfg.model_copy(update={
                "nvenc_preset": "p6", "nvenc_cq": min(cfg.nvenc_cq, 19),
                "nvenc_multipass": "qres",
                "nvenc_spatial_aq": True, "nvenc_temporal_aq": False,
                "nvenc_rc_lookahead": max(cfg.nvenc_rc_lookahead, 24),
            })
            rationale.append("Goal=quality: NVENC bumped to p6, CQ ≤ 19, qres multipass.")
        elif is_qsv:
            cfg = cfg.model_copy(update={
                "qsv_preset": "veryslow",
                "qsv_global_quality": max(1, min(cfg.qsv_global_quality, 22)),
                "qsv_extbrc": True,
                "qsv_look_ahead_depth": max(cfg.qsv_look_ahead_depth, 40),
                "qsv_low_power": False,
            })
            rationale.append("Goal=quality: QSV preset veryslow, global_quality tightened.")
        elif is_amf:
            cfg = cfg.model_copy(update={
                "amf_quality": "quality",
                "amf_qp_i": max(0, min(cfg.amf_qp_i, 20)),
                "amf_qp_p": max(0, min(cfg.amf_qp_p, 22)),
                "amf_qp_b": max(0, min(cfg.amf_qp_b, 24)),
                "amf_preanalysis": True,
                "amf_vbaq": False,
            })
            rationale.append("Goal=quality: AMF quality=quality, QP tightened.")
        elif is_d3d12:
            cfg = cfg.model_copy(update={
                "d3d12_qp": max(0, min(cfg.d3d12_qp, 20)),
                "d3d12_quality": max(cfg.d3d12_quality, 0),
            })
            rationale.append("Goal=quality: D3D12 QP tightened.")
        elif is_vulkan:
            cfg = cfg.model_copy(update={
                "vulkan_qp": max(0, min(cfg.vulkan_qp, 20)),
                "vulkan_quality": max(cfg.vulkan_quality, 0),
                "vulkan_async_depth": max(cfg.vulkan_async_depth, 2),
            })
            rationale.append("Goal=quality: Vulkan QP tightened.")
        else:
            cfg = cfg.model_copy(update={"x_preset": "slow", "x_crf": min(cfg.x_crf, 21)})
            rationale.append("Goal=quality: software encoder set to slow preset, CRF ≤ 21.")
    elif goal == "speed":
        if is_nvenc:
            cfg = cfg.model_copy(update={
                "nvenc_preset": "p4", "nvenc_multipass": "qres",
                "nvenc_temporal_aq": False,
            })
            rationale.append("Goal=speed: NVENC set to p4, qres multipass, temporal_aq off.")
        elif is_qsv:
            cfg = cfg.model_copy(update={
                "qsv_preset": "veryfast",
                "qsv_global_quality": min(51, cfg.qsv_global_quality + 4),
                "qsv_look_ahead_depth": min(cfg.qsv_look_ahead_depth, 12),
                "qsv_low_power": True,
            })
            rationale.append("Goal=speed: QSV preset veryfast, global_quality relaxed.")
        elif is_amf:
            cfg = cfg.model_copy(update={
                "amf_quality": "speed",
                "amf_preanalysis": False,
                "amf_vbaq": False,
            })
            rationale.append("Goal=speed: AMF quality=speed.")
        elif is_d3d12:
            cfg = cfg.model_copy(update={
                "d3d12_qp": min(51, max(0, cfg.d3d12_qp) + 4),
            })
            rationale.append("Goal=speed: D3D12 QP relaxed.")
        elif is_vulkan:
            cfg = cfg.model_copy(update={
                "vulkan_qp": min(255, max(0, cfg.vulkan_qp) + 4),
                "vulkan_async_depth": min(64, max(cfg.vulkan_async_depth, 4)),
            })
            rationale.append("Goal=speed: Vulkan QP relaxed and async depth increased.")
        else:
            cfg = cfg.model_copy(update={"x_preset": "medium"})
            rationale.append("Goal=speed: software preset relaxed to medium.")
    elif goal == "archival":
        if is_nvenc:
            cfg = cfg.model_copy(update={
                "nvenc_preset": "p7", "nvenc_cq": min(cfg.nvenc_cq, 18),
                "nvenc_multipass": "fullres",
                "nvenc_bframes": max(cfg.nvenc_bframes, 4),
            })
            rationale.append("Goal=archival: NVENC at p7/CQ18, 4 B-frames.")
        elif is_qsv:
            cfg = cfg.model_copy(update={
                "qsv_preset": "veryslow",
                "qsv_global_quality": max(1, min(cfg.qsv_global_quality, 20)),
                "qsv_extbrc": True,
                "qsv_look_ahead_depth": max(cfg.qsv_look_ahead_depth, 50),
                "qsv_bf": max(cfg.qsv_bf, 4),
                "qsv_low_power": False,
            })
            rationale.append("Goal=archival: QSV veryslow / tight global_quality.")
        elif is_amf:
            cfg = cfg.model_copy(update={
                "amf_quality": "high_quality",
                "amf_qp_i": max(0, min(cfg.amf_qp_i, 18)),
                "amf_qp_p": max(0, min(cfg.amf_qp_p, 18)),
                "amf_qp_b": max(0, min(cfg.amf_qp_b, 19)),
                "amf_preanalysis": True,
                "amf_vbaq": False,
            })
            rationale.append("Goal=archival: AMF high_quality / lower QP.")
        elif is_d3d12:
            cfg = cfg.model_copy(update={
                "d3d12_qp": max(0, min(cfg.d3d12_qp, 18)),
                "d3d12_quality": max(cfg.d3d12_quality, 0),
            })
            rationale.append("Goal=archival: D3D12 QP lowered.")
        elif is_vulkan:
            cfg = cfg.model_copy(update={
                "vulkan_qp": max(0, min(cfg.vulkan_qp, 18)),
                "vulkan_quality": max(cfg.vulkan_quality, 0),
                "vulkan_async_depth": min(64, max(cfg.vulkan_async_depth, 4)),
            })
            rationale.append("Goal=archival: Vulkan QP lowered.")
        else:
            cfg = cfg.model_copy(update={"x_preset": "slower", "x_crf": min(cfg.x_crf, 16)})
            rationale.append("Goal=archival: software encoder set to slower / CRF ≤ 16.")
    elif goal == "compat":
        hevc_like = {
            "hevc_nvenc", "av1_nvenc", "libx265",
            "hevc_qsv", "av1_qsv", "hevc_amf", "av1_amf",
            "av1_d3d12", "hevc_vulkan", "av1_vulkan",
        }
        if name in hevc_like:
            if is_nvenc:
                target = "h264_nvenc"
            elif is_qsv:
                target = "h264_qsv"
            elif is_amf:
                target = "h264_amf"
            elif is_d3d12:
                target = "h264_d3d12"
            elif is_vulkan:
                target = "h264_vulkan"
            else:
                target = "libx264"
            rationale.append(
                f"Goal=compat: forcing {target} for streaming/device compatibility."
            )
            cfg = cfg.model_copy(update={"name": target})  # type: ignore[arg-type]
    return cfg, rationale
