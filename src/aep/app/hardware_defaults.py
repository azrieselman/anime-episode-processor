"""One-time hardware-aware defaults (e.g. non-NVIDIA → QSV/AMF user preset overlay)."""

from __future__ import annotations

import logging

from aep.bench.hardware import probe_hardware
from aep.persist.presets import load_preset, save_user_preset
from aep.persist.settings import AppSettings

log = logging.getLogger(__name__)

# Bump when first-run logic changes so upgrades can re-evaluate once.
HARDWARE_ENCODER_DEFAULTS_VERSION = 1


def apply_hardware_encoder_defaults(settings: AppSettings) -> AppSettings:
    """If settings have not yet been migrated, probe hardware and optionally write a user
    ``anime_balanced`` preset that uses ``hevc_qsv`` or ``hevc_amf`` when NVIDIA NVENC is
    not the viable path.
    """
    if settings.hardware_encoder_defaults_version >= HARDWARE_ENCODER_DEFAULTS_VERSION:
        return settings

    try:
        hw = probe_hardware()
    except Exception as exc:
        log.warning("hardware defaults: probe failed: %s", exc)
        return settings.model_copy(update={
            "hardware_encoder_defaults_version": HARDWARE_ENCODER_DEFAULTS_VERSION,
        })

    new_ver = HARDWARE_ENCODER_DEFAULTS_VERSION

    if hw.gpu.has_nvidia:
        log.info("hardware defaults: NVIDIA GPU — keeping built-in anime_balanced encoder.")
        return settings.model_copy(update={"hardware_encoder_defaults_version": new_ver})

    target: str | None = None
    if hw.gpu.qsv_hevc and hw.has_encoder("hevc_qsv"):
        target = "hevc_qsv"
    elif hw.gpu.amf_hevc and hw.has_encoder("hevc_amf"):
        target = "hevc_amf"

    if target is None:
        log.info("hardware defaults: no Intel QSV / AMD AMF HEVC path — skipping preset overlay.")
        return settings.model_copy(update={"hardware_encoder_defaults_version": new_ver})

    try:
        preset = load_preset("anime_balanced")
    except Exception as exc:
        log.warning("hardware defaults: could not load anime_balanced: %s", exc)
        return settings.model_copy(update={"hardware_encoder_defaults_version": new_ver})

    updated_encoder = preset.encoder.model_copy(update={"name": target})  # type: ignore[arg-type]
    overlay = preset.model_copy(update={"encoder": updated_encoder})
    try:
        save_user_preset(overlay)
        log.info(
            "hardware defaults: wrote user preset anime_balanced with encoder=%s",
            target,
        )
    except Exception as exc:
        log.warning("hardware defaults: failed to save user preset: %s", exc)

    return settings.model_copy(update={"hardware_encoder_defaults_version": new_ver})
