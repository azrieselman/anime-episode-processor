"""Windows-only helpers for reading PE version resources.

NcnnVulkan-style CLIs frequently omit ``version …`` banners; MSVC-built EXEs often
carry the upstream ``YYYYMMDD`` tag in ``ProductVersion`` / ``FileVersion`` anyway.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

_TAG_RE = re.compile(r"\b(20\d{6})\b")


def _looks_like_calendar_yyyymmdd(s: str) -> bool:
    if len(s) != 8 or not s.isdigit():
        return False
    try:
        d = datetime.strptime(s, "%Y%m%d")
    except ValueError:
        return False
    return 2018 <= d.year <= 2099


def first_yyyymmdd_tag(*blobs, prefer=None):
    """Pick the first plausible ``YYYYMMDD`` token from text blobs.

    When ``prefer`` is set and present in any blob, that value wins so we stay
    aligned with our manifest pin even if the binary also mentions other dates.
    """
    if prefer and prefer.isdigit() and _looks_like_calendar_yyyymmdd(prefer):
        for b in blobs:
            if isinstance(b, str) and prefer in b:
                return prefer
    found: list[str] = []
    for b in blobs:
        if not isinstance(b, str):
            continue
        for m in _TAG_RE.finditer(b):
            cand = m.group(1)
            if _looks_like_calendar_yyyymmdd(cand):
                found.append(cand)
    return found[0] if found else None


def pe_version_resource_strings(path: Path) -> list[str]:
    """Return human-readable strings from the PE ``VERSIONINFO`` resource.

    Empty on failure or on non-Windows hosts (Linux CI stays quiet).
    """
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return []

    dll = ctypes.WinDLL("version", use_last_error=True)
    wpath = str(path.resolve())
    dw_dummy = wintypes.DWORD(0)

    GetFileVersionInfoSizeW = dll.GetFileVersionInfoSizeW
    GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    GetFileVersionInfoSizeW.restype = wintypes.DWORD

    GetFileVersionInfoW = dll.GetFileVersionInfoW
    GetFileVersionInfoW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
    ]
    GetFileVersionInfoW.restype = wintypes.BOOL

    VerQueryValueW = dll.VerQueryValueW
    VerQueryValueW.argtypes = [
        ctypes.c_void_p, wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT),
    ]
    VerQueryValueW.restype = wintypes.BOOL

    info_len = GetFileVersionInfoSizeW(wpath, ctypes.byref(dw_dummy))
    if info_len == 0:
        return []

    buf = ctypes.create_string_buffer(info_len)
    if not GetFileVersionInfoW(wpath, 0, info_len, buf):
        return []

    trans_len = wintypes.UINT(0)
    trans_ptr = ctypes.c_void_p()
    if not VerQueryValueW(
        buf, r"\VarFileInfo\Translation",
        ctypes.byref(trans_ptr), ctypes.byref(trans_len),
    ):
        return []

    pairs: list[int] = []
    n_dw = trans_len.value // ctypes.sizeof(wintypes.DWORD)
    dwords = ctypes.cast(trans_ptr, ctypes.POINTER(wintypes.DWORD))
    for i in range(max(n_dw, 0)):
        pairs.append(int(dwords[i]))

    collected: list[str] = []
    keys = ("ProductVersion", "FileVersion", "ProductName", "CompanyName", "Comments")
    for lang_cp in pairs:
        lang = lang_cp & 0xFFFF
        cp = (lang_cp >> 16) & 0xFFFF
        sub = f"{lang:04x}{cp:04x}"
        for key in keys:
            subpath = f"\\StringFileInfo\\{sub}\\{key}"
            val_len = wintypes.UINT(0)
            val_ptr = ctypes.c_void_p()
            if not VerQueryValueW(buf, subpath, ctypes.byref(val_ptr), ctypes.byref(val_len)):
                continue
            if not val_ptr.value:
                continue
            s = ctypes.wstring_at(val_ptr)
            if s:
                collected.append(s)
    return collected
