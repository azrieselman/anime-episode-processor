"""Application-wide constants. Keep these centralized; do not scatter literals.

Trade-off: a single constants module is sometimes criticized as a "bag of stuff," but for
a desktop app where most constants are paths/identifiers shared across GUI + worker, it
prevents drift far better than per-module duplication.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "AnimeEpisodeProcessor"
APP_DISPLAY_NAME = "Anime Episode Processor"
APP_VENDOR = "AEP"
APP_ID = "com.aep.anime-episode-processor"

# Filesystem layout. On Windows the production install resolves these to %LOCALAPPDATA%\AEP,
# %APPDATA%\AEP, etc. (see aep.util.paths). For dev runs they fall back to ./runtime/.
ENV_RUNTIME_DIR = "AEP_RUNTIME_DIR"
ENV_TOOLS_DIR = "AEP_TOOLS_DIR"
ENV_PRESETS_DIR = "AEP_PRESETS_DIR"
ENV_LOG_LEVEL = "AEP_LOG_LEVEL"

DEFAULT_LOG_LEVEL = "INFO"

# Subdirectories under the runtime root.
DIR_LOGS = "logs"
DIR_JOBS = "jobs"
DIR_CACHE = "cache"
DIR_PRESETS_USER = "presets"  # user-editable presets (override built-in)
DIR_BENCH = "bench"
DIR_TEMP = "temp"

# Files
FILE_SETTINGS = "settings.json"
FILE_DB = "aep.db"
FILE_GLOBAL_LOG = "aep.log"

# Tool binary names (Windows-only filenames; the adapter layer resolves full paths).
BIN_FFMPEG = "ffmpeg.exe"
BIN_FFPROBE = "ffprobe.exe"
BIN_MKVMERGE = "mkvmerge.exe"
BIN_MKVPROPEDIT = "mkvpropedit.exe"
BIN_MKVINFO = "mkvinfo.exe"
BIN_REALCUGAN = "realcugan-ncnn-vulkan.exe"
BIN_REALESRGAN = "realesrgan-ncnn-vulkan.exe"
BIN_RIFE = "rife-ncnn-vulkan.exe"
BIN_WAIFU2X = "waifu2x-ncnn-vulkan.exe"
BIN_ANIME4KCPP = "ac_cli.exe"
BIN_VSPIPE = "vspipe.exe"

# Pinned tool versions. The adapter base class verifies these at startup; any mismatch is a
# loud error, not a warning, because behavior differences between FFmpeg builds (esp. NVENC
# parameter availability) are real and silent breakage is unacceptable.
PINNED_VERSIONS = {
    "ffmpeg": "n7.0.2",          # gyan.dev essentials build line
    "ffprobe": "n7.0.2",
    "mkvmerge": "85.0",
    "mkvpropedit": "85.0",
    "realcugan-ncnn-vulkan": "20220728",
    "realesrgan-ncnn-vulkan": "0.2.0",
    "rife-ncnn-vulkan": "20250112",
    "waifu2x-ncnn-vulkan": "20220728",
    "anime4kcpp": "3.2.0",
    "anime4kcpp-vs": "3.2.0",
    "vapoursynth-vspipe": "R74",
    "ffms2-vapoursynth": "2.40",
}

# Hard-coded UI sizing baselines (Qt logical px). Real DPI handling is automatic via Qt.
WINDOW_DEFAULT_WIDTH = 1400
WINDOW_DEFAULT_HEIGHT = 900
WINDOW_MIN_WIDTH = 1100
WINDOW_MIN_HEIGHT = 720

# Pipeline knobs (defaults; per-job overrides via plan)
DEFAULT_RING_BUFFER_FRAMES = 240
DEFAULT_TILE_SIZE = 256
DEFAULT_LOW_VRAM_TILE_SIZE = 128
DEFAULT_RIFE_THREADS = "10:10:10"
DEFAULT_SCENE_THRESHOLD = 0.4

# PNG compression level for ffmpeg's **libpng** path (postprocess, encode-from-frames,
# and decode when the preset sets ``decode.png_intermediate_codec`` to libpng).
# Default decode with ``intermediate_format=png`` uses MJPEG into ``.png`` names
# when ``decode.png_intermediate_codec`` is mjpeg; see ``FFmpegAdapter.build_decode_to_frames``.
# zlib level 0..9; 6 is zlib's default and the explicitly chosen sweet spot where
# intermediates use PNG:
#   * level 1 was the prior default — ~3× larger files than level 6, which
#     mattered when intermediates lived on a small RAM-disk (M6.5 batches).
#   * level 9 is ~10 % smaller than 6 but ~3× slower per frame in CPU encode,
#     which becomes the bottleneck behind a Vulkan upscaler.
# Wired into ffmpeg's PNG encoder (-compression_level) where libpng is used, and
# ncnn binaries' -f flag (rife/realcugan/realesrgan/waifu2x all accept the same 0..9 mapping).
PNG_COMPRESSION_LEVEL = 6

# Validation tolerances
DURATION_TOLERANCE_MS = 200


def is_windows() -> bool:
    return os.name == "nt"


def project_root() -> Path:
    """Resolve repo root in dev and bundle root in frozen builds."""
    if getattr(sys, "frozen", False):
        # PyInstaller exposes the extraction/bundle directory at sys._MEIPASS
        # (one-file and one-folder). Data files are typically rooted there.
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        # Fallback for frozen environments that do not provide _MEIPASS.
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]
