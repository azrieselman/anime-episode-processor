"""Windows display adapter enumeration (Intel / AMD / NVIDIA hints).

Uses PowerShell CIM over Win32_VideoController — no extra Python deps.

Hybrid laptops may list both Intel iGPU and AMD/NVIDIA dGPU. When **no** NVIDIA GPU is
handled by `probe_nvidia()`, we use this to decide whether QSV (Intel) or AMF (AMD) is
the plausible hardware encode path for first-run defaults.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)

VendorHint = Literal["intel", "amd", "nvidia", "unknown"]


@dataclass(frozen=True)
class WindowsAdapter:
    name: str
    adapter_ram_bytes: int | None


def probe_windows_adapters() -> list[WindowsAdapter]:
    """Return video adapters on Windows; empty list on failure or non-Windows."""
    if sys.platform != "win32":
        return []
    ps = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress -Depth 3"
    )
    try:
        proc = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                ps,
            ],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("windows_gpu: PowerShell probe failed: %s", exc)
        return []

    if proc.returncode != 0:
        log.warning("windows_gpu: PowerShell exit %s: %s", proc.returncode, proc.stderr[:500])
        return []

    raw = proc.stdout.strip()
    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("windows_gpu: could not parse JSON from PowerShell")
        return []

    if isinstance(data, dict):
        data = [data]

    out: list[WindowsAdapter] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        # Skip pure software renderers
        if "microsoft basic" in name.lower() or "remote display" in name.lower():
            continue
        ram = row.get("AdapterRAM")
        ram_int: int | None
        try:
            ram_int = int(ram) if ram is not None else None
        except (TypeError, ValueError):
            ram_int = None
        out.append(WindowsAdapter(name=name, adapter_ram_bytes=ram_int))
    return out


def _vendor_from_name(name: str) -> VendorHint:
    low = name.lower()
    if "nvidia" in low:
        return "nvidia"
    if "intel" in low:
        return "intel"
    if "amd" in low or "ati" in low or "radeon" in low:
        return "amd"
    return "unknown"


def infer_non_nvidia_vendor(adapters: list[WindowsAdapter]) -> VendorHint:
    """Pick Intel vs AMD when nvidia-smi did not report an NVIDIA GPU.

    Rule (deterministic):
    * Collect adapters whose name implies Intel vs AMD (ignore unknown-only entries).
    * If both AMD and Intel appear (common hybrid): prefer the vendor whose adapter has
      **larger** AdapterRAM when both have RAM reported (discrete usually wins).
      If RAM is missing or tied, prefer **AMD** over Intel so an AMD dGPU is chosen over
      Intel iGPU when names are ambiguous.
    * If only one vendor class appears, return it.
    """
    intel: list[WindowsAdapter] = []
    amd: list[WindowsAdapter] = []
    for a in adapters:
        v = _vendor_from_name(a.name)
        if v == "intel":
            intel.append(a)
        elif v == "amd":
            amd.append(a)

    def max_ram(lst: list[WindowsAdapter]) -> int:
        vals = [x.adapter_ram_bytes for x in lst if x.adapter_ram_bytes is not None]
        return max(vals) if vals else 0

    if amd and intel:
        ri, ra = max_ram(intel), max_ram(amd)
        if ra > ri:
            return "amd"
        if ri > ra:
            return "intel"
        return "amd"
    if amd:
        return "amd"
    if intel:
        return "intel"
    return "unknown"


def has_intel_adapter(adapters: list[WindowsAdapter]) -> bool:
    return any(_vendor_from_name(a.name) == "intel" for a in adapters)


def has_amd_adapter(adapters: list[WindowsAdapter]) -> bool:
    return any(_vendor_from_name(a.name) == "amd" for a in adapters)
