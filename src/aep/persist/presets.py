"""Preset loading & validation.

Presets are YAML files (one preset per file). Built-in presets ship in `presets/` at the
repo root; user presets live under `%LOCALAPPDATA%\\AEP\\presets\\`. User presets with
the same `id` as a built-in override the built-in (logged at INFO).

A preset is a complete description of how to process an episode. It is intentionally
verbose — defaults like NVENC tuning, tile sizes, and stream-mapping policies are spelled
out so a user can copy-edit one and understand it without source-diving. Hidden defaults
breed the kind of behavior bugs Waifu2x-Extension-GUI is criticized for.

Field descriptions and ``json_schema_extra["gui"]`` feed the Preset Designer GUI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from aep.errors import PresetError
from aep.util.paths import builtin_presets_dir, user_presets_dir

log = logging.getLogger(__name__)


def _gui(
    category: str,
    tier: Literal["simple", "advanced"],
    *,
    spin_decimals: int | None = None,
) -> dict[str, object]:
    """Metadata consumed by the schema-driven preset editor."""
    gui: dict[str, object] = {"category": category, "tier": tier}
    if spin_decimals is not None:
        gui["spin_decimals"] = spin_decimals
    return {"gui": gui}


# ----- Sub-models -----------------------------------------------------------

UpscalerEngine = Literal[
    "realcugan-ncnn-vulkan",
    "realesrgan-ncnn-vulkan",
    "waifu2x-ncnn-vulkan",
    "anime4kcpp",
    "anime4k-legacy",
    "anime4kcpp-vs",
    "none",
]
# RIFE model catalog evolves frequently (fork bundles dozens of variants);
# adapter-level validation warns for unknown values while still allowing them.
RifeVersion = str
EncoderName = Literal[
    "hevc_nvenc",
    "h264_nvenc",
    "av1_nvenc",
    "h264_qsv",
    "hevc_qsv",
    "av1_qsv",
    "h264_amf",
    "hevc_amf",
    "av1_amf",
    "libx264",
    "libx265",
]
AmfQuality = Literal["speed", "balanced", "quality", "high_quality"]
ContainerName = Literal["mkv", "mp4"]
ContentClass = Literal["anime_2d", "anime_compressed", "mixed", "auto"]
DecodeHwaccelMode = Literal["auto", "off", "d3d11va", "cuda"]
PngIntermediateCodec = Literal["mjpeg", "libpng"]
BatchingMode = Literal["manual", "auto"]


class UpscalerCfg(BaseModel):
    engine: UpscalerEngine = Field(
        default="realcugan-ncnn-vulkan",
        description="Which upscaling backend to run (NCNN Vulkan engines, Anime4K variants, or none).",
        json_schema_extra=_gui("upscaler", "simple"),
    )
    model: str = Field(
        default="models-pro",
        description="Engine-specific model identifier (e.g. Real-CUGAN tier, Waifu2x noise scale, Anime4K model name).",
        json_schema_extra=_gui("upscaler", "simple"),
    )
    scale: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Spatial upscale factor applied by the upscaler (1–4×).",
        json_schema_extra=_gui("upscaler", "simple"),
    )
    denoise: int = Field(
        default=3,
        ge=-1,
        le=3,
        description="Noise reduction strength where supported (e.g. CUGAN); -1 disables denoise.",
        json_schema_extra=_gui("upscaler", "simple"),
    )
    tile_size: int = Field(
        default=256,
        ge=64,
        le=1024,
        description="Tile size in pixels for tiled inference; smaller uses less VRAM but may show seams.",
        json_schema_extra=_gui("upscaler", "advanced"),
    )
    fp16: bool = Field(
        default=True,
        description="Use half-precision inference when supported (faster, slightly less precise).",
        json_schema_extra=_gui("upscaler", "simple"),
    )
    enabled: bool = Field(
        default=True,
        description="When false, the upscale stage is skipped (passthrough to later stages).",
        json_schema_extra=_gui("upscaler", "simple"),
    )
    tta: bool = Field(
        default=False,
        description="Test-time augmentation: higher quality, roughly 4× slower.",
        json_schema_extra=_gui("upscaler", "advanced"),
    )
    intermediate_format: Literal["png", "webp"] = Field(
        default="png",
        description=(
            "Frame file extension for the NCNN frame pipeline: png uses ``.png`` filenames; "
            "the on-disk payload is chosen by ``decode.png_intermediate_codec`` (MJPEG vs true PNG). "
            "webp uses lossless WebP."
        ),
        json_schema_extra=_gui("upscaler", "advanced"),
    )
    hdr_policy: Literal["skip", "allow_8bit_roundtrip"] = Field(
        default="allow_8bit_roundtrip",
        description=(
            "10-bit/HDR sources: skip disables upscaling with a warning; allow_8bit_roundtrip runs "
            "8-bit NCNN then re-quantizes on encode (HDR gamut is not preserved)."
        ),
        json_schema_extra=_gui("upscaler", "advanced"),
    )


class TargetResolution(BaseModel):
    """Either a named target ('1080p'/'1440p'/'2160p') or explicit width/height.

    Aspect ratio is preserved by the planner; we never letterbox without explicit consent.
    """

    mode: Literal["named", "explicit", "scale_only"] = Field(
        default="named",
        description="named: pick a standard height; explicit: set width/height; scale_only: upscaler/planner drives size.",
        json_schema_extra=_gui("resolution", "simple"),
    )
    named: Literal["720p", "1080p", "1440p", "2160p"] | None = Field(
        default="1440p",
        description="Target when mode is named (vertical resolution preset). Ignored for explicit/scale_only.",
        json_schema_extra=_gui("resolution", "simple"),
    )
    width: int | None = Field(
        default=None,
        description="Explicit output width in pixels when mode is explicit (aspect preserved).",
        json_schema_extra=_gui("resolution", "advanced"),
    )
    height: int | None = Field(
        default=None,
        description="Explicit output height in pixels when mode is explicit (aspect preserved).",
        json_schema_extra=_gui("resolution", "advanced"),
    )


class InterpolationCfg(BaseModel):
    enabled: bool = Field(
        default=True,
        description="Enable frame interpolation (e.g. RIFE) after upscale.",
        json_schema_extra=_gui("interpolation", "simple"),
    )
    engine: Literal["rife-ncnn-vulkan", "none"] = Field(
        default="rife-ncnn-vulkan",
        description="Interpolation backend; none skips interpolation.",
        json_schema_extra=_gui("interpolation", "simple"),
    )
    version: RifeVersion = Field(
        default="v4.22-lite",
        description="RIFE model/version string matching your bundled binary (unknown IDs may still run with a warning).",
        json_schema_extra=_gui("interpolation", "simple"),
    )
    target_fps: float | None = Field(
        default=60.0,
        description="Output frames per second; leave unset (null) to preserve source frame rate.",
        json_schema_extra=_gui("interpolation", "simple"),
    )
    multiplier: int | None = Field(
        default=None,
        description="Optional frame multiplier alternative to target_fps (e.g. 2 = double frames).",
        json_schema_extra=_gui("interpolation", "advanced"),
    )
    scene_cut_threshold: float = Field(
        default=0.4,
        description="RIFE scene-change sensitivity; lower detects harder cuts (fewer blended transitions).",
        json_schema_extra=_gui("interpolation", "advanced"),
    )
    duplicate_on_scene_cut: bool = Field(
        default=True,
        description="Duplicate frames at detected scene cuts to avoid blending across cuts.",
        json_schema_extra=_gui("interpolation", "advanced"),
    )
    fp16: bool = Field(
        default=True,
        description="Half-precision RIFE inference when supported.",
        json_schema_extra=_gui("interpolation", "simple"),
    )
    scene_detect_backend: Literal["pyscenedetect", "ffmpeg_scdet"] = Field(
        default="pyscenedetect",
        description="Backend for stage 03 scene-cut detection.",
        json_schema_extra=_gui("interpolation", "advanced"),
    )
    scene_change_threshold_percent: float = Field(
        default=10.0,
        ge=0.1,
        le=100.0,
        description="FFmpeg scdet sensitivity (percent); used when scene_detect_backend=ffmpeg_scdet.",
        json_schema_extra=_gui("interpolation", "advanced"),
    )


class EncoderCfg(BaseModel):
    name: EncoderName = Field(
        default="hevc_nvenc",
        description="Video encoder: NVENC, Intel QSV, AMD AMF, or CPU libx264/libx265.",
        json_schema_extra=_gui("encoder", "simple"),
    )
    nvenc_preset: str = Field(
        default="p6",
        description="NVENC preset p1 (fastest) … p7 (slowest/best quality).",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    nvenc_tune: str = Field(
        default="hq",
        description="NVENC tune string passed to ffmpeg (e.g. hq, ll, ull).",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    nvenc_rc: str = Field(
        default="vbr",
        description="NVENC rate-control mode (e.g. vbr, constqp).",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    nvenc_cq: int = Field(
        default=20,
        description="NVENC constant-quality target when using CQ-style modes (lower = higher quality).",
        json_schema_extra=_gui("encoder", "simple"),
    )
    nvenc_multipass: str = Field(
        default="fullres",
        description="NVENC multipass setting (e.g. fullres, qres, disabled per ffmpeg).",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    nvenc_spatial_aq: bool = Field(
        default=True,
        description="NVENC spatial adaptive quantization.",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    nvenc_temporal_aq: bool = Field(
        default=True,
        description="NVENC temporal adaptive quantization.",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    nvenc_bframes: int = Field(
        default=3,
        description="Number of B-frames for NVENC GOP structure.",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    nvenc_b_ref_mode: str = Field(
        default="middle",
        description="NVENC B-frame reference mode (e.g. middle, each).",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    qsv_preset: str = Field(
        default="medium",
        description="Intel QSV -preset (e.g. veryfast…veryslow, or driver-specific).",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    qsv_global_quality: int = Field(
        default=26,
        ge=1,
        le=51,
        description="Intel QSV -global_quality (lower = higher quality; scale depends on codec).",
        json_schema_extra=_gui("encoder", "simple"),
    )
    amf_quality: AmfQuality = Field(
        default="quality",
        description="AMD AMF -quality (speed, balanced, quality, high_quality).",
        json_schema_extra=_gui("encoder", "simple"),
    )
    amf_rc: str = Field(
        default="cqp",
        description="AMD AMF -rc mode (e.g. cqp, vbr_latency).",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    amf_qp_i: int = Field(
        default=22,
        ge=0,
        le=51,
        description="AMD AMF -qp_i when using CQP-style modes.",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    amf_qp_p: int = Field(
        default=22,
        ge=0,
        le=51,
        description="AMD AMF -qp_p when using CQP-style modes.",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    x_crf: int = Field(
        default=18,
        description="CRF for libx264/libx265 (lower = higher quality, larger files).",
        json_schema_extra=_gui("encoder", "simple"),
    )
    x_preset: str = Field(
        default="slow",
        description="libx264/x265 preset (ultrafast…veryslow).",
        json_schema_extra=_gui("encoder", "advanced"),
    )
    x_tune: str | None = Field(
        default="animation",
        description='x264 tune (e.g. "animation"); use empty/null for x265 or no tune.',
        json_schema_extra=_gui("encoder", "advanced"),
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Additional ffmpeg video encoder arguments as separate tokens (passed verbatim).",
        json_schema_extra=_gui("encoder", "advanced"),
    )


class StreamMappingCfg(BaseModel):
    """How non-video streams flow from source to output.

    Defaults are conservative: copy everything we can, transform nothing implicitly.
    """

    copy_audio: bool = Field(
        default=True,
        description="Copy audio streams without re-encoding when possible.",
        json_schema_extra=_gui("streams", "simple"),
    )
    copy_subtitles: bool = Field(
        default=True,
        description="Copy subtitle streams to the output container.",
        json_schema_extra=_gui("streams", "simple"),
    )
    copy_chapters: bool = Field(
        default=True,
        description="Copy chapter markers.",
        json_schema_extra=_gui("streams", "advanced"),
    )
    copy_attachments: bool = Field(
        default=True,
        description="Copy font attachments (MKV) and similar attachment streams.",
        json_schema_extra=_gui("streams", "advanced"),
    )
    copy_global_metadata: bool = Field(
        default=True,
        description="Copy container-level metadata.",
        json_schema_extra=_gui("streams", "advanced"),
    )
    copy_stream_metadata: bool = Field(
        default=True,
        description="Copy per-stream metadata.",
        json_schema_extra=_gui("streams", "advanced"),
    )
    copy_dispositions: bool = Field(
        default=True,
        description="Copy stream disposition flags (default, forced, hearing impaired, etc.).",
        json_schema_extra=_gui("streams", "advanced"),
    )
    burn_in_subtitles: bool = Field(
        default=False,
        description="Burn subtitles into the video (destructive; opt-in only).",
        json_schema_extra=_gui("streams", "simple"),
    )


class PostprocessCfg(BaseModel):
    enabled: bool = Field(
        default=False,
        description="Enable optional ffmpeg post-filters after upscale/interpolate.",
        json_schema_extra=_gui("postprocess", "simple"),
    )
    deband: bool = Field(
        default=False,
        description="Apply debanding filter (helps gradients after heavy compression).",
        json_schema_extra=_gui("postprocess", "simple"),
    )
    deblock: bool = Field(
        default=False,
        description="Apply deblocking filter.",
        json_schema_extra=_gui("postprocess", "advanced"),
    )
    grain_addback: int = Field(
        default=0,
        ge=0,
        le=32,
        description="Film-grain noise strength (0 = off; ffmpeg noise filter).",
        json_schema_extra=_gui("postprocess", "advanced"),
    )


class DecodeCfg(BaseModel):
    hwaccel: DecodeHwaccelMode = Field(
        default="auto",
        description=(
            "Decoder hardware acceleration: auto (D3D11VA on Windows, off elsewhere), off, "
            "d3d11va (DXVA/D3D11), or cuda (NVIDIA NVDEC via FFmpeg -hwaccel cuda; requires "
            "a CUDA-enabled FFmpeg build and drivers)."
        ),
        json_schema_extra=_gui("decode", "simple"),
    )
    png_intermediate_codec: PngIntermediateCodec = Field(
        default="mjpeg",
        description=(
            "When upscaler ``intermediate_format`` is png: mjpeg writes JPEG (MJPEG) with "
            "yuvj444p into ``.png`` filenames (smaller cache; NCNN still uses ``-f png``). "
            "libpng writes true lossless PNG (zlib level from app constants)."
        ),
        json_schema_extra=_gui("decode", "advanced"),
    )


class FrameDedupeCfg(BaseModel):
    """Optional ffmpeg scene-score pass to skip near-duplicate frames before NCNN stages."""

    enabled: bool = Field(
        default=False,
        description=(
            "When true (and the frame pipeline is active), decode is compacted using "
            "per-frame scene scores; RIFE/upscale run on fewer frames, then the full "
            "timeline is restored with duplicate neighbors before encode."
        ),
        json_schema_extra=_gui("decode", "advanced"),
    )
    threshold: float = Field(
        default=0.02,
        ge=0.0,
        le=1.0,
        description=(
            "Frames whose ffmpeg scene score is strictly below this (vs the previous frame) "
            "are treated as duplicates; frame 1 is never skipped. Scores are often very small "
            "(e.g. 1e-6–0.1); use values like 0.01–0.05 for conservative dedupe. Extremely small "
            "thresholds can mark almost every frame a duplicate if scores cluster near zero."
        ),
        json_schema_extra=_gui("decode", "advanced", spin_decimals=8),
    )
    protect_scene_cuts: bool = Field(
        default=True,
        description="Never skip frames adjacent to a batch-local scene-cut boundary.",
        json_schema_extra=_gui("decode", "advanced"),
    )


class BatchingCfg(BaseModel):
    """Per-batch chunking of the frame-heavy stages.

    Without batching, decode-serve writes every frame of the source to disk
    before upscale runs. For a 24-min 1080p episode at 4 bytes/px that's
    ~700 GB — not viable on a 32 GB ramdisk. Batching processes the source
    in `chunk_seconds`-long slices so peak intermediate storage equals one
    chunk's frames rather than the whole episode's. Encoded segments per
    chunk are concatenated in the mux stage.

    mode:
      * auto — choose unbatched vs batched and an effective chunk length from
        free RAM/scratch space and the planner's frame-byte estimate (see stage
        01 plan). `chunk_seconds` is the maximum slice target (cap).
      * manual — use `enabled` and `chunk_seconds` as fixed settings.

    boundary_policy:
      * keyframe — snap each batch boundary to the nearest source keyframe
        (≤ target time). Lets decode-serve seek by keyframe without re-decoding
        the prior chunk. This is the right default for almost everyone.
      * exact — use exact `chunk_seconds`-multiple boundaries even if they
        fall mid-GOP. Decode pays a re-decode-from-prior-keyframe penalty
        on each chunk, but the chunks are uniform sizes (useful for
        deterministic batch budgeting).
    """

    mode: BatchingMode = Field(
        default="auto",
        description="auto: size batches from free scratch/RAM space; manual: use enabled + chunk_seconds.",
        json_schema_extra=_gui("batching", "simple"),
    )
    enabled: bool = Field(
        default=True,
        description="manual mode only: when true, time-based batching is on. Ignored for batch decisions in auto mode.",
        json_schema_extra=_gui("batching", "simple"),
    )
    chunk_seconds: int = Field(
        default=30,
        ge=5,
        le=600,
        description="Duration of each batch slice in seconds (larger = more peak disk).",
        json_schema_extra=_gui("batching", "simple"),
    )
    boundary_policy: Literal["keyframe", "exact"] = Field(
        default="keyframe",
        description="keyframe aligns chunk cuts to source keyframes; exact forces rigid timing (may re-decode more).",
        json_schema_extra=_gui("batching", "advanced"),
    )


class PresetMeta(BaseModel):
    id: str = Field(
        ...,
        description="Stable preset identifier and YAML filename stem (no spaces recommended).",
        json_schema_extra=_gui("meta", "simple"),
    )
    name: str = Field(
        ...,
        description="Human-readable preset title shown in menus.",
        json_schema_extra=_gui("meta", "simple"),
    )
    description: str = Field(
        default="",
        description="Longer summary shown as tooltip / documentation.",
        json_schema_extra=_gui("meta", "simple"),
    )
    builtin: bool = Field(
        default=False,
        description="True when loaded from shipped presets; overridden on save for user copies.",
        json_schema_extra=_gui("meta", "advanced"),
    )
    suitable_for: list[ContentClass] = Field(
        default_factory=list,
        description="Tags describing intended content (anime 2D, compressed sources, mixed, auto-detect).",
        json_schema_extra=_gui("meta", "advanced"),
    )
    target_hardware: str = Field(
        default="RTX 3000 series, 8GB+ VRAM",
        description="Free-text hardware guidance for users choosing a preset.",
        json_schema_extra=_gui("meta", "simple"),
    )


class Preset(BaseModel):
    meta: PresetMeta = Field(
        ...,
        description="Preset identity and presentation metadata.",
        json_schema_extra=_gui("meta", "simple"),
    )
    container: ContainerName = Field(
        default="mkv",
        description="Output container format.",
        json_schema_extra=_gui("container", "simple"),
    )
    target_resolution: TargetResolution = Field(
        default_factory=TargetResolution,
        description="How output resolution is chosen relative to source aspect.",
        json_schema_extra=_gui("resolution", "simple"),
    )
    upscaler: UpscalerCfg = Field(
        default_factory=UpscalerCfg,
        description="Upscaler configuration.",
        json_schema_extra=_gui("upscaler", "simple"),
    )
    interpolation: InterpolationCfg = Field(
        default_factory=InterpolationCfg,
        description="Frame interpolation configuration.",
        json_schema_extra=_gui("interpolation", "simple"),
    )
    encoder: EncoderCfg = Field(
        default_factory=EncoderCfg,
        description="Video encoder configuration.",
        json_schema_extra=_gui("encoder", "simple"),
    )
    decode: DecodeCfg = Field(
        default_factory=DecodeCfg,
        description="Decoder / hwaccel configuration.",
        json_schema_extra=_gui("decode", "simple"),
    )
    frame_dedupe: FrameDedupeCfg = Field(
        default_factory=FrameDedupeCfg,
        description="Perceptual duplicate-frame skipping before the first NCNN stage.",
        json_schema_extra=_gui("decode", "advanced"),
    )
    streams: StreamMappingCfg = Field(
        default_factory=StreamMappingCfg,
        description="Stream copy/burn-in policy.",
        json_schema_extra=_gui("streams", "simple"),
    )
    postprocess: PostprocessCfg = Field(
        default_factory=PostprocessCfg,
        description="Optional post filters.",
        json_schema_extra=_gui("postprocess", "simple"),
    )
    batching: BatchingCfg = Field(
        default_factory=BatchingCfg,
        description="Batch chunking for intermediate frames.",
        json_schema_extra=_gui("batching", "simple"),
    )


# ----- Loader ---------------------------------------------------------------


def _load_yaml_file(path: Path) -> Preset:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PresetError(f"YAML parse failed: {path}", context={"reason": str(exc)}) from exc
    if not isinstance(raw, dict):
        raise PresetError(f"Preset must be a YAML mapping: {path}")
    try:
        return Preset.model_validate(raw)
    except ValidationError as exc:
        raise PresetError(f"Preset validation failed: {path}", context={"reason": str(exc)}) from exc


def list_presets() -> list[Preset]:
    """Built-ins first, then user presets override by id."""
    presets: dict[str, Preset] = {}

    builtin_dir = builtin_presets_dir()
    if builtin_dir.exists():
        for p in sorted(builtin_dir.glob("*.yaml")):
            preset = _load_yaml_file(p)
            preset.meta.builtin = True
            presets[preset.meta.id] = preset

    user_dir = user_presets_dir()
    for p in sorted(user_dir.glob("*.yaml")):
        preset = _load_yaml_file(p)
        preset.meta.builtin = False
        if preset.meta.id in presets:
            log.info("user preset overrides built-in: %s", preset.meta.id)
        presets[preset.meta.id] = preset

    return list(presets.values())


def load_preset(preset_id: str) -> Preset:
    for p in list_presets():
        if p.meta.id == preset_id:
            return p
    raise PresetError(f"preset not found: {preset_id}")


def save_user_preset(preset: Preset) -> Path:
    """Saves a preset to the user presets dir. Filename = id.yaml."""
    out = user_presets_dir() / f"{preset.meta.id}.yaml"
    payload = preset.model_dump(mode="json")
    out.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    log.info("user preset saved: %s", out)
    return out


def delete_user_preset(preset_id: str) -> bool:
    """Remove a user preset file if it exists. Built-ins cannot be deleted."""
    path = user_presets_dir() / f"{preset_id}.yaml"
    if not path.is_file():
        return False
    path.unlink()
    log.info("user preset deleted: %s", path)
    return True
