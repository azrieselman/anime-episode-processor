"""Hardware probe — combines NVIDIA + FFmpeg + CPU/RAM into a single HardwareProfile.

The encoder recommender (`aep.encode.recommender`) and the planner (`s01_plan`) consume
this profile to make decisions like:

* Is `hevc_nvenc` actually usable? (Both: GPU present AND ffmpeg lists it AND a quick
  encode probe doesn't fail at startup.)
* What tile size is safe? (VRAM total, current free.)
* Is hardware acceleration likely stable? (VRAM + compute capability help drive defaults.)

The profile is content-addressed via `fingerprint()` so the benchmark cache can
invalidate itself when hardware changes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from aep.adapters.nvidia import NvidiaProbe, probe_nvidia

if TYPE_CHECKING:
    from aep.adapters.ffmpeg import FFmpegAdapter

log = logging.getLogger(__name__)

GpuVendor = Literal["nvidia", "intel", "amd", "unknown"]

# Compute capability → architecture name. Used to enable/disable features that depend
# on Ampere (RTX 30xx), Ada (RTX 40xx), etc.
_ARCH_BY_CC: dict[str, str] = {
    "5.2": "maxwell",
    "6.1": "pascal",
    "7.0": "volta",
    "7.5": "turing",
    "8.0": "ampere",
    "8.6": "ampere",
    "8.9": "ada",
    "9.0": "hopper",
}


@dataclass(frozen=True)
class CpuInfo:
    logical_cores: int
    ram_total_mib: int | None = None


@dataclass(frozen=True)
class GpuCapabilities:
    """Higher-level capability flags derived from raw probe data."""
    has_nvidia: bool
    nvenc_h264: bool
    nvenc_hevc: bool
    nvenc_av1: bool
    arch: str | None              # "ampere" | "ada" | etc.
    vram_total_mib: int
    vram_free_mib: int
    driver_version: str | None
    primary_vendor: GpuVendor = "unknown"
    qsv_h264: bool = False
    qsv_hevc: bool = False
    qsv_av1: bool = False
    amf_h264: bool = False
    amf_hevc: bool = False
    amf_av1: bool = False
    d3d12_h264: bool = False
    d3d12_av1: bool = False
    vulkan_h264: bool = False
    vulkan_hevc: bool = False
    vulkan_av1: bool = False


@dataclass(frozen=True)
class HardwareProfile:
    cpu: CpuInfo
    gpu: GpuCapabilities
    ffmpeg_version: str | None
    ffmpeg_encoders: list[str] = field(default_factory=list)
    nvidia_raw: NvidiaProbe | None = None

    def fingerprint(self) -> str:
        """Stable fingerprint for benchmark cache keys."""
        h = hashlib.blake2b(digest_size=12)
        for piece in (
            str(self.cpu.logical_cores),
            str(self.cpu.ram_total_mib or 0),
            self.gpu.arch or "none",
            str(self.gpu.vram_total_mib),
            self.gpu.driver_version or "none",
            self.ffmpeg_version or "none",
            self.gpu.primary_vendor,
            f"{int(self.gpu.qsv_h264)}{int(self.gpu.qsv_hevc)}{int(self.gpu.qsv_av1)}",
            f"{int(self.gpu.amf_h264)}{int(self.gpu.amf_hevc)}{int(self.gpu.amf_av1)}",
            f"{int(self.gpu.d3d12_h264)}{int(self.gpu.d3d12_av1)}",
            f"{int(self.gpu.vulkan_h264)}{int(self.gpu.vulkan_hevc)}{int(self.gpu.vulkan_av1)}",
            ",".join(sorted(self.ffmpeg_encoders)),
        ):
            h.update(piece.encode("utf-8"))
            h.update(b"|")
        return h.hexdigest()

    def has_encoder(self, name: str) -> bool:
        return name in self.ffmpeg_encoders


def _ram_total_mib() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]
        return int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:
        # Best-effort fallback; on Windows os.sysconf is not available.
        return None


def _arch_from_cc(cc: str | None) -> str | None:
    if not cc:
        return None
    return _ARCH_BY_CC.get(cc)


def probe_hardware(*, ffmpeg_adapter: FFmpegAdapter | None = None) -> HardwareProfile:
    """Run all probes and assemble the profile.

    `ffmpeg_adapter` can be injected for tests; if None, we construct one. If FFmpeg is
    missing entirely we return a profile with empty encoder list — the recommender will
    surface that as a hard error rather than silently ignoring.
    """
    cpu = CpuInfo(
        logical_cores=os.cpu_count() or 1,
        ram_total_mib=_ram_total_mib(),
    )

    nv = probe_nvidia()
    primary = nv.primary
    if primary:
        gpu_caps_partial = {
            "has_nvidia": True,
            "arch": _arch_from_cc(primary.compute_capability),
            "vram_total_mib": primary.vram_total_mib,
            "vram_free_mib": primary.vram_free_mib,
            "driver_version": primary.driver_version,
        }
    else:
        gpu_caps_partial = {
            "has_nvidia": False,
            "arch": None,
            "vram_total_mib": 0,
            "vram_free_mib": 0,
            "driver_version": None,
        }

    ffmpeg_version: str | None = None
    encoders: list[str] = []
    if ffmpeg_adapter is None:
        try:
            from aep.adapters.ffmpeg import FFmpegAdapter
            ffmpeg_adapter = FFmpegAdapter()
        except Exception as exc:
            log.warning("could not construct FFmpegAdapter: %s", exc)
            ffmpeg_adapter = None

    if ffmpeg_adapter is not None:
        try:
            ffmpeg_version = ffmpeg_adapter.version
        except Exception as exc:
            log.warning("ffmpeg version probe failed: %s", exc)
        try:
            encoders = [e.name for e in ffmpeg_adapter.list_encoders()]
        except Exception as exc:
            log.warning("ffmpeg encoder enumeration failed: %s", exc)

    has_nv = bool(gpu_caps_partial["has_nvidia"]) and bool(encoders)
    arch = gpu_caps_partial["arch"]
    # NVENC AV1 requires Ada (RTX 40xx) and a sufficiently new driver. Even if the
    # encoder is in `ffmpeg -encoders` it may still fail at runtime on older arches.
    av1_eligible = has_nv and arch in {"ada", "hopper"} and "av1_nvenc" in encoders
    hevc_eligible = has_nv and arch in {"turing", "ampere", "ada", "hopper"} and "hevc_nvenc" in encoders
    h264_eligible = has_nv and "h264_nvenc" in encoders

    win_adapters: list[object] = []
    intel_ok = False
    amd_ok = False
    if sys.platform == "win32":
        from aep.adapters.windows_gpu import (
            has_amd_adapter,
            has_intel_adapter,
            infer_non_nvidia_vendor,
            probe_windows_adapters,
        )

        try:
            win_adapters = probe_windows_adapters()
        except Exception as exc:
            log.warning("windows_gpu probe failed: %s", exc)

        intel_ok = has_intel_adapter(win_adapters)
        amd_ok = has_amd_adapter(win_adapters)

    if primary:
        primary_vendor: GpuVendor = "nvidia"
    elif sys.platform == "win32" and win_adapters:
        primary_vendor = infer_non_nvidia_vendor(win_adapters)
    else:
        primary_vendor = "unknown"

    qsv_h264 = bool(encoders) and "h264_qsv" in encoders and intel_ok
    qsv_hevc = bool(encoders) and "hevc_qsv" in encoders and intel_ok
    qsv_av1 = bool(encoders) and "av1_qsv" in encoders and intel_ok
    amf_h264 = bool(encoders) and "h264_amf" in encoders and amd_ok
    amf_hevc = bool(encoders) and "hevc_amf" in encoders and amd_ok
    amf_av1 = bool(encoders) and "av1_amf" in encoders and amd_ok
    d3d12_h264 = bool(encoders) and "h264_d3d12" in encoders
    d3d12_av1 = bool(encoders) and "av1_d3d12" in encoders
    vulkan_h264 = bool(encoders) and "h264_vulkan" in encoders
    vulkan_hevc = bool(encoders) and "hevc_vulkan" in encoders
    vulkan_av1 = bool(encoders) and "av1_vulkan" in encoders

    gpu = GpuCapabilities(
        has_nvidia=has_nv,
        nvenc_h264=h264_eligible,
        nvenc_hevc=hevc_eligible,
        nvenc_av1=av1_eligible,
        arch=arch,
        vram_total_mib=int(gpu_caps_partial["vram_total_mib"]),
        vram_free_mib=int(gpu_caps_partial["vram_free_mib"]),
        driver_version=gpu_caps_partial["driver_version"],
        primary_vendor=primary_vendor,
        qsv_h264=qsv_h264,
        qsv_hevc=qsv_hevc,
        qsv_av1=qsv_av1,
        amf_h264=amf_h264,
        amf_hevc=amf_hevc,
        amf_av1=amf_av1,
        d3d12_h264=d3d12_h264,
        d3d12_av1=d3d12_av1,
        vulkan_h264=vulkan_h264,
        vulkan_hevc=vulkan_hevc,
        vulkan_av1=vulkan_av1,
    )

    profile = HardwareProfile(
        cpu=cpu,
        gpu=gpu,
        ffmpeg_version=ffmpeg_version,
        ffmpeg_encoders=encoders,
        nvidia_raw=nv,
    )
    log.info(
        "hardware: cpu_cores=%d ram=%s GPU=%s arch=%s vendor=%s vram=%dMiB "
        "nvenc(h264/hevc/av1)=%s/%s/%s qsv(h264/hevc/av1)=%s/%s/%s amf(h264/hevc/av1)=%s/%s/%s "
        "d3d12(h264/av1)=%s/%s vulkan(h264/hevc/av1)=%s/%s/%s ffmpeg=%s",
        cpu.logical_cores,
        f"{(cpu.ram_total_mib or 0) // 1024} GiB" if cpu.ram_total_mib else "?",
        primary.name if primary else "(none)",
        arch or "?",
        gpu.primary_vendor,
        gpu.vram_total_mib,
        gpu.nvenc_h264, gpu.nvenc_hevc, gpu.nvenc_av1,
        gpu.qsv_h264, gpu.qsv_hevc, gpu.qsv_av1,
        gpu.amf_h264, gpu.amf_hevc, gpu.amf_av1,
        gpu.d3d12_h264, gpu.d3d12_av1,
        gpu.vulkan_h264, gpu.vulkan_hevc, gpu.vulkan_av1,
        ffmpeg_version,
    )
    return profile
