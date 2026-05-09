"""NVIDIA hardware probe.

We shell out to `nvidia-smi` rather than linking against NVML — avoids a runtime DLL
dependency that ships with every Python release of pynvml, and `nvidia-smi` is part of
every NVIDIA driver install. If nvidia-smi is missing, the whole probe degrades to "no
NVIDIA GPU detected" and the rest of the app falls back to software encoders.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass

from aep.util.proc import run_capture

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_total_mib: int
    vram_free_mib: int
    driver_version: str
    cuda_version: str | None
    compute_capability: str | None  # e.g. "8.6" for Ampere
    pci_bus_id: str | None


@dataclass(frozen=True)
class NvidiaProbe:
    nvidia_smi_present: bool
    gpus: list[GpuInfo]
    error: str | None = None

    @property
    def primary(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None


def probe_nvidia() -> NvidiaProbe:
    nvsmi = shutil.which("nvidia-smi")
    if not nvsmi:
        return NvidiaProbe(nvidia_smi_present=False, gpus=[], error="nvidia-smi not on PATH")

    fields = [
        "name",
        "memory.total",
        "memory.free",
        "driver_version",
        "compute_cap",
        "pci.bus_id",
    ]
    try:
        result = run_capture(
            [
                nvsmi,
                f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits",
            ],
            timeout=10.0,
        )
    except Exception as exc:
        return NvidiaProbe(nvidia_smi_present=True, gpus=[], error=str(exc))

    cuda_version = _query_cuda_version(nvsmi)

    gpus: list[GpuInfo] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(fields):
            continue
        try:
            gpus.append(GpuInfo(
                name=parts[0],
                vram_total_mib=int(parts[1]),
                vram_free_mib=int(parts[2]),
                driver_version=parts[3],
                cuda_version=cuda_version,
                compute_capability=parts[4] or None,
                pci_bus_id=parts[5] or None,
            ))
        except (ValueError, IndexError):
            log.warning("could not parse nvidia-smi line: %r", line)

    return NvidiaProbe(nvidia_smi_present=True, gpus=gpus)


def _query_cuda_version(nvsmi: str) -> str | None:
    """`nvidia-smi -q -x` is heavy; the cheap path is to parse `nvidia-smi` text output
    once for the "CUDA Version: X.Y" header line."""
    try:
        result = subprocess.run(
            [nvsmi],
            capture_output=True,
            text=True,
            timeout=8.0,
            check=False,
        )
    except Exception:
        return None
    for line in result.stdout.splitlines():
        if "CUDA Version:" in line:
            tail = line.split("CUDA Version:")[1].strip()
            return tail.split()[0] if tail else None
    return None
