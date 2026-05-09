"""Platform / environment helpers."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformInfo:
    system: str
    release: str
    machine: str
    python_version: str
    is_windows: bool


def platform_info() -> PlatformInfo:
    return PlatformInfo(
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python_version=sys.version.split()[0],
        is_windows=os.name == "nt",
    )
