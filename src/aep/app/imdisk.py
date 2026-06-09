"""ImDisk Toolkit integration for AEP ramdisk management.

The GUI uses this module to install ImDisk Toolkit and manage the dynamic
ramdisk that backs ``settings.paths.ramdisk_path``. Pure helpers are kept
separate from the subprocess/elevation calls so the risky command construction
is easy to unit-test without touching the host system.
"""

from __future__ import annotations

import ctypes
import hashlib
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aep.persist.settings import AppSettings
from aep.util.paths import runtime_dir

log = logging.getLogger(__name__)

IMDISK_INSTALLER_URL = (
    "https://github.com/azrieselman/anime-episode-processor/releases/download/other/ImDiskTk-Installer.zip"
)
IMDISK_INSTALLER_SHA256 = "87c4507b9337b935d80efb5988aca4f06a3e58bacf7b7706dc8db0fe66a259c4"
IMDISK_INSTALL_DIR = Path(r"C:\Program Files\ImDisk")
IMDISK_INSTALLER_NAME = "ImDiskTk-Installer.zip"
VOLUME_LABEL = "AEP"

DRIVE_NO_ROOT_DIR = 1
DRIVE_FIXED = 3
_DOWNLOAD_CHUNK = 256 * 1024


class ImDiskError(RuntimeError):
    """Install or ramdisk-management operation failed."""


@dataclass(frozen=True)
class DownloadProgress:
    bytes_read: int
    total_bytes: int | None


@dataclass(frozen=True)
class VolumeStatus:
    mounted: bool
    total_bytes: int = 0
    free_bytes: int = 0


ProgressCallback = Callable[[str, DownloadProgress | None], None]


def normalize_drive_letter(letter: str) -> str:
    cleaned = letter.strip().upper().rstrip("\\/")
    if cleaned.endswith(":"):
        cleaned = cleaned[:-1]
    if len(cleaned) != 1 or not cleaned.isalpha():
        raise ValueError("Drive letter must be a single A-Z letter.")
    return cleaned


def mountpoint(letter: str) -> str:
    return f"{normalize_drive_letter(letter)}:"


def root_path(letter: str) -> str:
    return f"{normalize_drive_letter(letter)}:\\"


def size_gb_to_kb(size_gb: int) -> int:
    if size_gb < 1:
        raise ValueError("RamDisk size must be at least 1 GB.")
    return size_gb * 1024 * 1024


def ramdyn_exe(install_dir: Path = IMDISK_INSTALL_DIR) -> Path:
    return install_dir / "RamDyn.exe"


def imdisk_exe(install_dir: Path = IMDISK_INSTALL_DIR) -> Path:
    return install_dir / "ImDisk-Dlg.exe"


def imdisk_cli_exe() -> Path:
    # Detach uses the driver control program (installed to System32), not ImDisk-Dlg.
    return Path(r"C:\Windows\System32\imdisk.exe")


def is_installed(install_dir: Path = IMDISK_INSTALL_DIR) -> bool:
    return ramdyn_exe(install_dir).is_file() and imdisk_exe(install_dir).is_file()


def build_ramdyn_create_cmd(
    letter: str,
    size_gb: int,
    *,
    install_dir: Path = IMDISK_INSTALL_DIR,
) -> list[str]:
    return [
        str(ramdyn_exe(install_dir)),
        mountpoint(letter),
        str(size_gb_to_kb(size_gb)),
        "-1",
        "0",
        "14",
        "",
    ]


def build_ramdyn_launch_cmd(
    letter: str,
    size_gb: int,
    *,
    install_dir: Path = IMDISK_INSTALL_DIR,
) -> list[str]:
    # RamDyn.exe does not exit after handling a create request; launch it in the
    # background so elevated cmd returns and the GUI worker can continue.
    return ["cmd", "/c", "start", "/b", ""] + build_ramdyn_create_cmd(
        letter, size_gb, install_dir=install_dir
    )


def build_format_cmd(letter: str) -> list[str]:
    mp = mountpoint(letter)
    # format.com works from an elevated PowerShell session on ImDisk volumes.
    return [
        "powershell",
        "-NoProfile",
        "-Command",
        f"format {mp} /fs:ntfs /v:\"{VOLUME_LABEL}\" /q /y",
    ]


def build_remove_cmd(letter: str) -> list[str]:
    return [str(imdisk_cli_exe()), "-D", "-m", mountpoint(letter)]


def build_install_cmd(install_bat: Path) -> list[str]:
    # ImDisk's install.bat re-spawns itself asynchronously unless %1 is "7".
    # Pass "7" so the elevated cmd waits for the real install to finish.
    return ["cmd", "/c", str(install_bat), "7"]


def apply_ramdisk_path(settings: AppSettings, letter: str | None) -> AppSettings:
    updated = settings.model_copy(deep=True)
    updated.paths.ramdisk_path = root_path(letter) if letter else None
    return updated


def installer_download_path() -> Path:
    return runtime_dir() / "_downloads" / IMDISK_INSTALLER_NAME


def download_installer(
    dest: Path | None = None,
    *,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    out = dest or installer_download_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    progress_cb and progress_cb("downloading", None)

    h = hashlib.sha256()
    read = 0
    try:
        with urllib.request.urlopen(IMDISK_INSTALLER_URL, timeout=60) as resp:
            total_s = resp.headers.get("Content-Length")
            total = int(total_s) if total_s and total_s.isdecimal() else None
            with out.open("wb") as fp:
                while True:
                    chunk = resp.read(_DOWNLOAD_CHUNK)
                    if not chunk:
                        break
                    fp.write(chunk)
                    h.update(chunk)
                    read += len(chunk)
                    progress_cb and progress_cb("downloading", DownloadProgress(read, total))
    except Exception as exc:
        raise ImDiskError(f"Failed to download ImDisk Toolkit: {exc}") from exc

    digest = h.hexdigest()
    if digest != IMDISK_INSTALLER_SHA256:
        try:
            out.unlink()
        except OSError:
            pass
        raise ImDiskError(
            "Downloaded ImDisk Toolkit archive failed SHA256 verification "
            f"(expected {IMDISK_INSTALLER_SHA256}, got {digest})."
        )
    return out


def install(
    *,
    archive: Path | None = None,
    install_dir: Path = IMDISK_INSTALL_DIR,
    progress_cb: ProgressCallback | None = None,
) -> None:
    if is_installed(install_dir):
        progress_cb and progress_cb("done", None)
        return

    if sys.platform != "win32":
        raise ImDiskError("ImDisk Toolkit installation is only available on Windows.")

    archive_path = archive or download_installer(progress_cb=progress_cb)
    progress_cb and progress_cb("extracting", None)
    with tempfile.TemporaryDirectory(prefix="aep-imdisk-") as td:
        extract_dir = Path(td)
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile as exc:
            raise ImDiskError(f"ImDisk Toolkit archive is not a valid zip file: {archive_path}") from exc

        install_bat = extract_dir / "install.bat"
        if not install_bat.is_file():
            raise ImDiskError("ImDisk Toolkit archive did not contain install.bat.")

        progress_cb and progress_cb("installing", None)
        rc = _run_elevated(build_install_cmd(install_bat), cwd=extract_dir)
        if rc != 0:
            raise ImDiskError(f"ImDisk Toolkit installer exited with code {rc}.")

    if not is_installed(install_dir):
        raise ImDiskError(
            f"ImDisk Toolkit installer finished, but {ramdyn_exe(install_dir)} was not found."
        )
    progress_cb and progress_cb("done", None)


def create_ramdisk(
    letter: str,
    size_gb: int,
    *,
    install_dir: Path = IMDISK_INSTALL_DIR,
) -> None:
    letter = normalize_drive_letter(letter)
    if not is_installed(install_dir):
        raise ImDiskError("ImDisk Toolkit is not installed.")
    if not is_drive_available(letter):
        raise ImDiskError(f"Drive {mountpoint(letter)} is already in use.")

    rc = _run_elevated(build_ramdyn_launch_cmd(letter, size_gb, install_dir=install_dir))
    if rc != 0:
        raise ImDiskError(f"RamDyn failed to create {mountpoint(letter)} (exit code {rc}).")

    _wait_for_drive(letter)
    rc = _run_elevated(build_format_cmd(letter))
    if rc != 0:
        raise ImDiskError(f"Formatting {mountpoint(letter)} exited with code {rc}.")
    _wait_for_formatted(letter)


def remove_ramdisk(letter: str, *, install_dir: Path = IMDISK_INSTALL_DIR) -> None:
    letter = normalize_drive_letter(letter)
    if not is_installed(install_dir):
        raise ImDiskError("ImDisk Toolkit is not installed.")
    if is_drive_available(letter):
        return

    rc = _run_elevated(build_remove_cmd(letter))
    if rc != 0:
        raise ImDiskError(f"Removing {mountpoint(letter)} exited with code {rc}.")


def is_drive_available(letter: str) -> bool:
    letter = normalize_drive_letter(letter)
    if sys.platform != "win32":
        return not Path(root_path(letter)).exists()
    return _get_drive_type(root_path(letter)) == DRIVE_NO_ROOT_DIR


def get_volume_status(letter: str) -> VolumeStatus:
    letter = normalize_drive_letter(letter)
    if is_drive_available(letter):
        return VolumeStatus(mounted=False)
    try:
        usage = shutil.disk_usage(root_path(letter))
    except OSError:
        return VolumeStatus(mounted=True)
    return VolumeStatus(mounted=True, total_bytes=usage.total, free_bytes=usage.free)


def _wait_for_drive(letter: str, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not is_drive_available(letter):
            return
        time.sleep(0.25)
    raise ImDiskError(f"Timed out waiting for {mountpoint(letter)} to become available.")


def _wait_for_formatted(letter: str, *, timeout_s: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            shutil.disk_usage(root_path(letter))
            return
        except OSError:
            time.sleep(0.5)
    raise ImDiskError(f"Timed out waiting for {mountpoint(letter)} to be formatted.")


def _get_drive_type(path: str) -> int:
    return int(ctypes.windll.kernel32.GetDriveTypeW(path))  # type: ignore[attr-defined]


def _run_elevated(argv: list[str], *, cwd: Path | None = None) -> int:
    if sys.platform != "win32":
        raise ImDiskError("Elevated execution is only available on Windows.")
    if not argv:
        raise ValueError("argv must not be empty")

    from ctypes import wintypes

    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    SW_HIDE = 0
    INFINITE = 0xFFFFFFFF

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", wintypes.ULONG),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", wintypes.LPCWSTR),
            ("hkeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIcon", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    params = subprocess.list2cmdline([str(part) for part in argv[1:]])
    info = SHELLEXECUTEINFOW()
    info.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = str(argv[0])
    info.lpParameters = params
    info.lpDirectory = str(cwd) if cwd else None
    info.nShow = SW_HIDE

    log.info("exec elevated: %s", subprocess.list2cmdline([str(part) for part in argv]))
    if not ctypes.windll.shell32.ShellExecuteExW(ctypes.byref(info)):  # type: ignore[attr-defined]
        err = ctypes.get_last_error()
        raise ImDiskError(f"Could not start elevated process (Windows error {err}).")

    try:
        ctypes.windll.kernel32.WaitForSingleObject(info.hProcess, INFINITE)  # type: ignore[attr-defined]
        code = wintypes.DWORD(0)
        if not ctypes.windll.kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code)):  # type: ignore[attr-defined]
            err = ctypes.get_last_error()
            raise ImDiskError(f"Could not read elevated process exit code (Windows error {err}).")
        return int(code.value)
    finally:
        ctypes.windll.kernel32.CloseHandle(info.hProcess)  # type: ignore[attr-defined]
