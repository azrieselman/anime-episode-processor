"""Application settings persistence (settings.json) using Pydantic for validation.

Trade-off: we use JSON instead of YAML for settings because settings are machine-edited
(via the GUI) more often than human-edited. Presets are the opposite — those use YAML.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from aep.constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_RIFE_THREADS,
    DEFAULT_RING_BUFFER_FRAMES,
    DEFAULT_TILE_SIZE,
    FILE_SETTINGS,
)
from aep.errors import ConfigError
from aep.util.paths import runtime_dir

log = logging.getLogger(__name__)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
DecodeHwaccelMode = Literal["auto", "off", "d3d11va"]
PipelineOrder = Literal["interpolate_first", "upscale_first"]


class GeneralSettings(BaseModel):
    log_level: LogLevel = DEFAULT_LOG_LEVEL  # type: ignore[assignment]
    output_dir: str | None = None  # null = next to source
    output_naming_template: str = "{source_stem}.{height}p.{fps}fps.aep.mkv"
    keep_temp_artifacts: bool = False
    confirm_overwrite: bool = True
    # When True, the broker starts un-paused and runs queued jobs as soon as
    # they're enqueued (legacy 1.0.0rc1 behavior). When False (default), the
    # broker boots paused; the user must click "Start Queue" in the GUI to
    # release dispatch. Power users who want fire-and-forget enqueue can flip
    # this in settings.json.
    auto_start_jobs: bool = False
    # When True, the Queue tab shows a Job ID column (hidden by default).
    show_queue_job_id_column: bool = False


class HardwareSettings(BaseModel):
    prefer_nvenc: bool = True
    max_concurrent_jobs: int = Field(default=1, ge=1, le=4)
    ring_buffer_frames: int = Field(default=DEFAULT_RING_BUFFER_FRAMES, ge=32, le=2048)
    default_tile_size: int = Field(default=DEFAULT_TILE_SIZE, ge=64, le=1024)
    # When an NCNN upscale stage sees more frames than this in one shot, we
    # split the input dir into N-frame chunks and run the binary once per chunk.
    # Rationale: long-running ncnn-vulkan invocations (~30k+ frames) hit
    # cumulative driver/GPU state issues on Windows that don't appear at
    # smaller sizes. 2000 is a comfortable middle ground (~80s per chunk on
    # RTX 3080 at tile=256), giving frequent recovery points without
    # per-chunk overhead dominating.
    ncnn_chunk_threshold: int = Field(default=2000, ge=200, le=20000)
    ncnn_chunk_size: int = Field(default=500, ge=100, le=5000)
    decode_hwaccel: DecodeHwaccelMode = "auto"
    anime4k_prefer_cuda: bool = True
    anime4k_threads: int = Field(default=4, ge=1, le=64)
    rife_threads: str = Field(
        default=DEFAULT_RIFE_THREADS,
        pattern=r"^[1-9]\d{0,2}:[1-9]\d{0,2}:[1-9]\d{0,2}$",
    )


class PathSettings(BaseModel):
    """Optional overrides for tool locations. Empty = use bundled tools."""
    ffmpeg_dir: str | None = None
    mkvtoolnix_dir: str | None = None
    realcugan_dir: str | None = None
    realesrgan_dir: str | None = None
    anime4kcpp_dir: str | None = None
    anime4kcpp_vs_filter_dir: str | None = None
    vapoursynth_dir: str | None = None
    rife_dir: str | None = None
    waifu2x_dir: str | None = None
    # If set and free space ≥ planner frame estimate, stages 04-07 (frame I/O heavy)
    # write their working frames here instead of the regular work dir. Intended
    # for ImDisk/ramdisk on Windows; an SSD scratch dir works equally well.
    # When unset or unavailable, the pipeline silently falls back to work_dir.
    ramdisk_path: str | None = None


class PipelineSettings(BaseModel):
    """Global order of NCNN stages between decode and postprocess.

    * interpolate_first — RIFE on decoded frames, then upscale (default; fewer pixels for RIFE).
    * upscale_first — legacy Waifu2x-Extension-GUI order (upscale then RIFE).
    """
    order: PipelineOrder = "interpolate_first"


class AppSettings(BaseModel):
    schema_version: int = 1
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    hardware: HardwareSettings = Field(default_factory=HardwareSettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    last_used_preset: str = "anime_balanced"
    # Increment `aep.app.hardware_defaults.HARDWARE_ENCODER_DEFAULTS_VERSION` when logic changes.
    hardware_encoder_defaults_version: int = 0


def settings_path() -> Path:
    return runtime_dir() / FILE_SETTINGS


def load_settings() -> AppSettings:
    p = settings_path()
    if not p.exists():
        log.info("no settings file at %s; using defaults", p)
        return AppSettings()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return AppSettings.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        # Don't silently overwrite a user's settings — surface the error.
        raise ConfigError(
            f"settings file is invalid: {p}",
            context={"reason": str(exc)},
        ) from exc


def save_settings(settings: AppSettings) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(p)  # atomic on Windows when target exists on same volume
    log.info("settings saved to %s", p)
