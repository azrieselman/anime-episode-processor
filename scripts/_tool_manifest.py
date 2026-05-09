"""Pinned external tool manifest.

Single source of truth for `fetch_tools.py` and `verify_tools.py`. The dev installer
unpacks every tool under `<repo>/tools/<subdir>/` so the bundled-tools resolution path
in `aep.adapters.base.ToolAdapter._resolve` finds them.

Updating a pin is a deliberate three-step process:
  1. Bump the URL + sha256 here AND the version in `aep.constants.PINNED_VERSIONS`.
  2. Run `python scripts/fetch_tools.py --force` and `verify_tools.py`.
  3. Commit. CI then re-verifies.

The Windows-essentials FFmpeg build (gyan.dev) ships ffmpeg/ffprobe + their DLLs in a
single archive; we extract only the executables and DLLs we actually call. MKVToolNix
similarly bundles all its CLIs.

Checksums below were computed by downloading the exact archive from each vendor's
immutable release URL (GitHub release CDN / mkvtoolnix.download) and running
`sha256sum` on the resulting bytes. To rotate a pin, change the URL + version, then
run `fetch_tools.py --update-hashes` to recompute and write back the new sha. The
verify-script aborts if any checksum is the placeholder, so you can't accidentally
ship a release without explicitly pinning every artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SHA256_TBD = "TBD-FILL-IN-AFTER-DOWNLOAD"


@dataclass(frozen=True)
class ToolPin:
    tool_id: str                 # matches PINNED_VERSIONS in aep.constants
    subdir: str                  # under tools/
    version: str                 # human-readable
    archive_url: str             # source of truth (vendor-controlled URL preferred)
    archive_sha256: str          # checksum of the *archive*, not files inside
    archive_format: Literal["zip", "7z"]
    # List of (path-in-archive, install-path-relative-to-tools-subdir).
    # Use forward slashes; `_install` translates per-platform.
    files: tuple[tuple[str, str], ...]


# ----- FFmpeg / FFprobe -----------------------------------------------------
#
# We pin the gyan.dev "essentials" build because:
#   * it includes NVENC (h264_nvenc, hevc_nvenc, av1_nvenc when driver supports it)
#   * it includes libx264 / libx265 / libsvtav1 / libaom-av1 (software fallbacks)
#   * it ships the matching ffprobe in the same archive, identical version
#
# The exact filename incorporates the tag; check
# https://www.gyan.dev/ffmpeg/builds/ for the current essentials filename.

FFMPEG_PIN = ToolPin(
    tool_id="ffmpeg",
    subdir="ffmpeg",
    version="n7.0.2",
    archive_url=(
        "https://github.com/GyanD/codexffmpeg/releases/download/7.0.2/"
        "ffmpeg-7.0.2-essentials_build.zip"
    ),
    archive_sha256="d5308d30872b2739cf53169df61faba8639d39a19b20b91e611c177ef676f64c",
    archive_format="zip",
    files=(
        ("ffmpeg-7.0.2-essentials_build/bin/ffmpeg.exe", "ffmpeg.exe"),
        ("ffmpeg-7.0.2-essentials_build/bin/ffprobe.exe", "ffprobe.exe"),
    ),
)


# ----- MKVToolNix ----------------------------------------------------------
#
# MKVToolNix portable archive ships .exe + a runtime DLL set. We extract the three
# CLIs we use; the rest (mkvextract, mkvtoolnix-gui) are unused.

MKVTOOLNIX_PIN = ToolPin(
    tool_id="mkvmerge",
    subdir="mkvtoolnix",
    version="85.0",
    archive_url=(
        "https://mkvtoolnix.download/windows/releases/85.0/"
        "mkvtoolnix-64-bit-85.0.7z"
    ),
    archive_sha256="753c1391af806f86815196ae6d259c1dd7552ca31844b6f59926e7537dfaffba",
    archive_format="7z",
    files=(
        ("mkvtoolnix/mkvmerge.exe", "mkvmerge.exe"),
        ("mkvtoolnix/mkvpropedit.exe", "mkvpropedit.exe"),
        ("mkvtoolnix/mkvinfo.exe", "mkvinfo.exe"),
        # MKVToolNix needs its DLLs/Qt runtime — we install the entire dir.
        # The fetcher honors a special "*" pattern to copy a directory tree.
        ("mkvtoolnix/*", "."),
    ),
)


# ----- Real-CUGAN ----------------------------------------------------------

REALCUGAN_PIN = ToolPin(
    tool_id="realcugan-ncnn-vulkan",
    subdir="realcugan-ncnn-vulkan",
    version="20220728",
    archive_url=(
        "https://github.com/nihui/realcugan-ncnn-vulkan/releases/download/"
        "20220728/realcugan-ncnn-vulkan-20220728-windows.zip"
    ),
    archive_sha256="c6e08d46c11704b1e3a1ada9ddd591cb5005f52f132136c8633ba25def400e01",
    archive_format="zip",
    files=(
        ("realcugan-ncnn-vulkan-20220728-windows/*", "."),
    ),
)


# ----- Real-ESRGAN ---------------------------------------------------------

REALESRGAN_PIN = ToolPin(
    tool_id="realesrgan-ncnn-vulkan",
    subdir="realesrgan-ncnn-vulkan",
    version="0.2.0",
    archive_url=(
        "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/"
        "v0.2.0/realesrgan-ncnn-vulkan-v0.2.0-windows.zip"
    ),
    archive_sha256="1bbbdb12d470af80b035c773682e144c6c2f6ece9210832a289af0a48ce3fa9a",
    archive_format="zip",
    files=(
        ("realesrgan-ncnn-vulkan-v0.2.0-windows/*", "."),
    ),
)


# ----- RIFE -----------------------------------------------------------------

RIFE_PIN = ToolPin(
    tool_id="rife-ncnn-vulkan",
    subdir="rife-ncnn-vulkan",
    version="20250112",
    archive_url=(
        "https://github.com/TNTwise/rife-ncnn-vulkan/releases/download/"
        "20250112/windows.zip"
    ),
    archive_sha256="42ed35e115b026f222386648920218cb8a9c7ae1e23698a7363bdd2e1455aba3",
    archive_format="zip",
    files=(
        ("rife-ncnn-vulkan-refs/heads/master-windows/*", "."),
    ),
)


# ----- waifu2x --------------------------------------------------------------

WAIFU2X_PIN = ToolPin(
    tool_id="waifu2x-ncnn-vulkan",
    subdir="waifu2x-ncnn-vulkan",
    version="20220728",
    archive_url=(
        "https://github.com/nihui/waifu2x-ncnn-vulkan/releases/download/"
        "20220728/waifu2x-ncnn-vulkan-20220728-windows.zip"
    ),
    archive_sha256="3f60ba0b26763c602cb75178c2051bf0c46f3cc9d13975a052a902773988a34b",
    archive_format="zip",
    files=(
        ("waifu2x-ncnn-vulkan-20220728-windows/*", "."),
    ),
)


ANIME4KCPP_PIN = ToolPin(
    tool_id="anime4kcpp",
    subdir="anime4kcpp",
    version="3.0.0",
    archive_url=(
        "https://github.com/TianZerL/Anime4KCPP/releases/download/"
        "v3.0.0/Anime4KCPP-CLI-v3.0.0-x64-MSVC.7z"
    ),
    archive_sha256="65d6a9b1befef0167f8b0c3e57ab10207584b3344e02637c0cf9afedda1d2164",
    archive_format="7z",
    files=(
        ("ac_cli.exe", "ac_cli.exe"),
        ("avcodec-60.dll", "avcodec-60.dll"),
        ("avformat-60.dll", "avformat-60.dll"),
        ("avutil-58.dll", "avutil-58.dll"),
        ("swresample-4.dll", "swresample-4.dll"),
        ("swscale-7.dll", "swscale-7.dll"),
    ),
)

ANIME4KCPP_VS_FILTER_PIN = ToolPin(
    tool_id="anime4kcpp-vs",
    subdir="anime4kcpp-filter-vs",
    version="3.0.0",
    archive_url=(
        "https://github.com/TianZerL/Anime4KCPP/releases/download/"
        "v3.0.0/Anime4KCPP-Filter-AVS-VS-v3.0.0-x86-x64-MSVC.7z"
    ),
    archive_sha256="be38d89d014151d19a15749f1bb70195939bc9ff63e8251f71cd065e2536d0db",
    archive_format="7z",
    files=(
        ("x64/ac_filter_avs_vs.dll", "ac_filter_avs_vs.dll"),
    ),
)

VAPOURSYNTH_PORTABLE_PIN = ToolPin(
    tool_id="vapoursynth-vspipe",
    subdir="vapoursynth",
    version="R74",
    archive_url=(
        "https://github.com/vapoursynth/vapoursynth/releases/download/"
        "R74/VapourSynth64-Portable-R74.zip"
    ),
    archive_sha256="53fb6eb4d59b26aaf26ea26fecbb9fc9698eb49b64c1cd4246dfa5d32aecfe46",
    archive_format="zip",
    files=(
        ("wheel/vapoursynth-74-cp312-abi3-win_amd64.whl", "vapoursynth-74-cp312-abi3-win_amd64.whl"),
    ),
)

FFMS2_VAPOURSYNTH_PIN = ToolPin(
    tool_id="ffms2-vapoursynth",
    subdir="ffms2-vs",
    version="2.40",
    archive_url="https://github.com/FFMS/ffms2/releases/download/2.40/ffms2-2.40-msvc.7z",
    archive_sha256="0da7454faaab87fc1515d72ffc2edeaee0e6090fa2f27a8986bdebed06197457",
    archive_format="7z",
    files=(
        ("ffms2-2.40-msvc/x64/ffms2.dll", "ffms2.dll"),
    ),
)


ALL_PINS: tuple[ToolPin, ...] = (
    FFMPEG_PIN,
    MKVTOOLNIX_PIN,
    REALCUGAN_PIN,
    REALESRGAN_PIN,
    RIFE_PIN,
    WAIFU2X_PIN,
    ANIME4KCPP_PIN,
    ANIME4KCPP_VS_FILTER_PIN,
    VAPOURSYNTH_PORTABLE_PIN,
    FFMS2_VAPOURSYNTH_PIN,
)
