"""Helpers for grouping encoder names by vendor/family."""

from __future__ import annotations

from typing import Literal

from aep.persist.presets import EncoderName

EncoderFamily = Literal["nvenc", "qsv", "amf", "d3d12", "vulkan", "x264", "x265"]
HardwareFamily = Literal["nvenc", "qsv", "amf", "d3d12", "vulkan"]
VideoCodec = Literal["h264", "hevc", "av1"]


def encoder_family(name: str) -> EncoderFamily:
    if name.endswith("_nvenc"):
        return "nvenc"
    if name.endswith("_qsv"):
        return "qsv"
    if name.endswith("_amf"):
        return "amf"
    if name.endswith("_d3d12"):
        return "d3d12"
    if name.endswith("_vulkan"):
        return "vulkan"
    if name == "libx264":
        return "x264"
    if name == "libx265":
        return "x265"
    raise ValueError(f"unsupported encoder: {name}")


def encode_name_for(hardware: HardwareFamily, codec: VideoCodec) -> EncoderName:
    if hardware == "d3d12" and codec == "hevc":
        raise ValueError("hevc_d3d12 is not supported by FFmpeg")
    return f"{codec}_{hardware}"  # type: ignore[return-value]


def software_name_for(codec: Literal["h264", "hevc"]) -> EncoderName:
    if codec == "h264":
        return "libx264"
    return "libx265"

