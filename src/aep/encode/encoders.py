"""FFmpeg argument builders per encoder.

Each builder takes:
  * encoder config (from the preset)
  * target geometry (width, height) — None means "preserve source size"
  * source pix_fmt — used to decide between 8-bit/10-bit output
  * frame rate target — informational; -fps_mode passthrough by default

Each returns a list of ffmpeg arguments. They produce arguments for the OUTPUT side of
the command (everything after `-i` / `-map`). The caller (FFmpegAdapter.build_*) prepends
the input/mapping section.

Why bake target geometry into a `scale=` filter here rather than in the upscaler stage?
* When the encoder runs without an upstream upscaler, we still need to honor a
  resolution change ("transcode 1080p → 720p" is a legitimate use).
* When the upscaler stage emits frames already at target geometry, the planner
  sets `target_size_changed_in_pipeline=False` and the encoder skips the scale filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from aep.persist.presets import EncoderCfg

log = logging.getLogger(__name__)


# Pixel formats we'll actually emit. NVENC HEVC supports 10-bit on Pascal+; h264_nvenc
# 10-bit support is limited (Hi10P is widely incompatible with hardware decoders).
PIX_FMT_8BIT = "yuv420p"
PIX_FMT_10BIT = "yuv420p10le"
# AMF encoders accept P010 for 10-bit, not yuv420p10le (see FFmpeg ff_amf_pix_fmts).
PIX_FMT_AMF_10BIT_INPUT = "p010le"


QSV_GLOBAL_PREFIX = ("-init_hw_device", "qsv=hw", "-filter_hw_device", "hw")
VULKAN_GLOBAL_PREFIX = ("-init_hw_device", "vulkan=vk:0", "-filter_hw_device", "vk")


@dataclass(frozen=True)
class EncodeBuildResult:
    """Bundle: argv chunk + chosen pix_fmt + notes for the manifest/UI.

    ``global_prefix`` is inserted by FFmpegAdapter **before** ``-i`` (e.g. QSV
    ``-init_hw_device``); encoder ``args`` remain after input mapping as today.
    """
    args: list[str]
    pix_fmt: str
    rationale: list[str]
    global_prefix: tuple[str, ...] = field(default_factory=tuple)


def _is_10bit_pix_fmt(pix_fmt: str | None) -> bool:
    if not pix_fmt:
        return False
    return any(t in pix_fmt for t in ("p10", "p12", "p16", "yuv420p10", "yuv422p10", "yuv444p10"))


def _scale_filter_if_needed(width: int | None, height: int | None) -> list[str]:
    """Emit a `scale=` filter only if both dimensions are set. We always force mod-2.
    `flags=lanczos` for a downscale; `bicubic` would be acceptable but lanczos is the
    sane default for real content even on 'scale up' pseudo-passes (rare).
    """
    if not (width and height):
        return []
    return [
        "-vf",
        f"scale={width}:{height}:flags=lanczos:force_original_aspect_ratio=disable",
    ]


def build_nvenc(
    cfg: EncoderCfg,
    *,
    family: str,                   # "h264" | "hevc" | "av1"
    target_width: int | None,
    target_height: int | None,
    source_pix_fmt: str | None,
    fps_mode: str = "passthrough",
) -> EncodeBuildResult:
    rationale: list[str] = []
    if family == "h264":
        codec = "h264_nvenc"
        # h264_nvenc 10-bit is non-portable; force 8-bit and note it.
        pix_fmt = PIX_FMT_8BIT
        if _is_10bit_pix_fmt(source_pix_fmt):
            rationale.append(
                "h264_nvenc: source is 10-bit but H.264 10-bit (Hi10P) hardware decode is rare; "
                "downconverting to 8-bit for compatibility."
            )
    elif family == "hevc":
        codec = "hevc_nvenc"
        pix_fmt = PIX_FMT_10BIT if _is_10bit_pix_fmt(source_pix_fmt) else PIX_FMT_8BIT
        if pix_fmt == PIX_FMT_10BIT:
            rationale.append("hevc_nvenc: preserving 10-bit (Main10) from source.")
    elif family == "av1":
        codec = "av1_nvenc"
        pix_fmt = PIX_FMT_10BIT if _is_10bit_pix_fmt(source_pix_fmt) else PIX_FMT_8BIT
        rationale.append(
            "av1_nvenc: experimental — requires Ada-class (RTX 4000+) hardware. "
            "On older GPUs, switch the preset's encoder to hevc_nvenc."
        )
    else:
        raise ValueError(f"unsupported NVENC family: {family}")

    args = [
        "-c:v", codec,
        "-preset", cfg.nvenc_preset,
        "-tune", cfg.nvenc_tune,
        "-rc", cfg.nvenc_rc,
        "-cq", str(cfg.nvenc_cq),
        "-b:v", "0",                    # constant-quality VBR — let cq drive
        "-pix_fmt", pix_fmt,
        "-multipass", cfg.nvenc_multipass,
        "-spatial_aq", "1" if cfg.nvenc_spatial_aq else "0",
        "-temporal_aq", "1" if cfg.nvenc_temporal_aq else "0",
        "-bf", str(cfg.nvenc_bframes),
        "-b_ref_mode", cfg.nvenc_b_ref_mode,
        "-rc-lookahead", str(cfg.nvenc_rc_lookahead),
        "-fps_mode", fps_mode,
    ]
    args = _scale_filter_if_needed(target_width, target_height) + args
    args += list(cfg.extra_args)

    rationale.append(
        f"NVENC {family}: preset={cfg.nvenc_preset} cq={cfg.nvenc_cq} "
        f"multipass={cfg.nvenc_multipass} spatial_aq={int(cfg.nvenc_spatial_aq)} "
        f"temporal_aq={int(cfg.nvenc_temporal_aq)} bf={cfg.nvenc_bframes}"
    )
    return EncodeBuildResult(args=args, pix_fmt=pix_fmt, rationale=rationale)


def _qsv_hw_vf(
    target_width: int | None,
    target_height: int | None,
    upload_fmt: str,
) -> str:
    """upload_fmt: nv12 or p010le for upload stage before ``format=qsv``."""
    parts: list[str] = []
    if target_width and target_height:
        parts.append(
            f"scale={target_width}:{target_height}:flags=lanczos:force_original_aspect_ratio=disable"
        )
    parts.append(f"format={upload_fmt}")
    parts.append("hwupload=extra_hw_frames=64")
    parts.append("format=qsv")
    return ",".join(parts)


def build_qsv(
    cfg: EncoderCfg,
    *,
    family: str,
    target_width: int | None,
    target_height: int | None,
    source_pix_fmt: str | None,
    fps_mode: str = "passthrough",
) -> EncodeBuildResult:
    rationale: list[str] = []
    if family == "h264":
        codec = "h264_qsv"
        pix_fmt = PIX_FMT_8BIT
        upload_fmt = "nv12"
        if _is_10bit_pix_fmt(source_pix_fmt):
            rationale.append(
                "h264_qsv: source is 10-bit; using 8-bit output for broad compatibility."
            )
    elif family == "hevc":
        codec = "hevc_qsv"
        use_10 = _is_10bit_pix_fmt(source_pix_fmt)
        pix_fmt = PIX_FMT_10BIT if use_10 else PIX_FMT_8BIT
        upload_fmt = "p010le" if use_10 else "nv12"
        if use_10:
            rationale.append("hevc_qsv: preserving 10-bit (Main10) where supported.")
    elif family == "av1":
        codec = "av1_qsv"
        pix_fmt = PIX_FMT_10BIT if _is_10bit_pix_fmt(source_pix_fmt) else PIX_FMT_8BIT
        upload_fmt = "p010le" if pix_fmt == PIX_FMT_10BIT else "nv12"
    else:
        raise ValueError(f"unsupported QSV family: {family}")

    vf = _qsv_hw_vf(target_width, target_height, upload_fmt)
    args = [
        "-vf", vf,
        "-c:v", codec,
        "-preset", cfg.qsv_preset,
        "-global_quality", str(cfg.qsv_global_quality),
        "-extbrc", "1" if cfg.qsv_extbrc else "0",
        "-bf", str(cfg.qsv_bf),
        "-low_power", "1" if cfg.qsv_low_power else "0",
        "-pix_fmt", pix_fmt,
        "-fps_mode", fps_mode,
    ]
    if cfg.qsv_extbrc and cfg.qsv_look_ahead_depth > 0:
        args += ["-look_ahead_depth", str(cfg.qsv_look_ahead_depth)]
    args += list(cfg.extra_args)
    rationale.append(
        f"Intel QSV {family}: preset={cfg.qsv_preset} global_quality={cfg.qsv_global_quality} "
        f"extbrc={int(cfg.qsv_extbrc)} look_ahead_depth={cfg.qsv_look_ahead_depth} "
        f"bf={cfg.qsv_bf} low_power={int(cfg.qsv_low_power)} pix_fmt={pix_fmt}"
    )
    return EncodeBuildResult(
        args=args,
        pix_fmt=pix_fmt,
        rationale=rationale,
        global_prefix=QSV_GLOBAL_PREFIX,
    )


def _amf_use_10bit(
    cfg: EncoderCfg,
    *,
    family: str,
    source_pix_fmt: str | None,
) -> bool:
    """Whether AMF HEVC/AV1 encode should emit 10-bit (p010le). h264_amf is always 8-bit."""
    if family == "h264":
        return False
    mode = (cfg.amf_bit_depth or "auto").lower()
    if mode == "8":
        return False
    if mode == "10":
        return True
    return _is_10bit_pix_fmt(source_pix_fmt)


def _amf_quality_args(quality: str) -> list[str]:
    """Map preset ``amf_quality`` to FFmpeg AMF ``-quality`` / ``-usage`` options.

    AMF encoders expose ``-quality`` (speed/balanced/quality), not ``-preset``.
    ``high_quality`` is a ``-usage`` preset on h264/hevc/av1_amf, not a quality tier.
    """
    q = (quality or "balanced").lower()
    if q == "high_quality":
        return ["-usage", "high_quality", "-quality", "quality"]
    return ["-quality", q]


def _amf_vf_parts(
    *,
    use_10: bool,
    decode_hwaccel: str,
    target_width: int | None,
    target_height: int | None,
) -> list[str]:
    """Build AMF encode vf chain for 10-bit surfaces and optional resize."""
    parts: list[str] = []
    hwaccel = (decode_hwaccel or "off").lower()
    if use_10 and hwaccel == "amf":
        parts.append(f"hwdownload,format={PIX_FMT_AMF_10BIT_INPUT}")
    if target_width and target_height:
        parts.append(
            f"scale={target_width}:{target_height}:flags=lanczos:force_original_aspect_ratio=disable"
        )
    if use_10 and hwaccel != "amf" and target_width and target_height:
        parts.append(f"format={PIX_FMT_AMF_10BIT_INPUT}")
    return parts


def build_amf(
    cfg: EncoderCfg,
    *,
    family: str,
    target_width: int | None,
    target_height: int | None,
    source_pix_fmt: str | None,
    fps_mode: str = "passthrough",
    decode_hwaccel: str = "off",
) -> EncodeBuildResult:
    rationale: list[str] = []
    if family == "h264":
        codec = "h264_amf"
        use_10 = False
        if cfg.amf_bit_depth == "10":
            rationale.append(
                "h264_amf: amf_bit_depth=10 ignored; H.264 AMF is always 8-bit."
            )
        elif _is_10bit_pix_fmt(source_pix_fmt):
            rationale.append(
                "h264_amf: source is 10-bit; using 8-bit output for compatibility."
            )
    elif family == "hevc":
        codec = "hevc_amf"
        use_10 = _amf_use_10bit(cfg, family=family, source_pix_fmt=source_pix_fmt)
        if use_10:
            if cfg.amf_bit_depth == "10":
                rationale.append("hevc_amf: amf_bit_depth=10 forces 10-bit (Main10) via p010le.")
            else:
                rationale.append("hevc_amf: preserving 10-bit (Main10) via p010le.")
        elif cfg.amf_bit_depth == "8" and _is_10bit_pix_fmt(source_pix_fmt):
            rationale.append("hevc_amf: amf_bit_depth=8 forces 8-bit output.")
    elif family == "av1":
        codec = "av1_amf"
        use_10 = _amf_use_10bit(cfg, family=family, source_pix_fmt=source_pix_fmt)
        if use_10:
            if cfg.amf_bit_depth == "10":
                rationale.append("av1_amf: amf_bit_depth=10 forces 10-bit via p010le.")
            else:
                rationale.append("av1_amf: preserving 10-bit via p010le.")
        elif cfg.amf_bit_depth == "8" and _is_10bit_pix_fmt(source_pix_fmt):
            rationale.append("av1_amf: amf_bit_depth=8 forces 8-bit output.")
    else:
        raise ValueError(f"unsupported AMF family: {family}")

    result_pix_fmt = PIX_FMT_10BIT if use_10 else PIX_FMT_8BIT
    amf_pix_fmt = PIX_FMT_AMF_10BIT_INPUT if use_10 else PIX_FMT_8BIT

    # AMF 10-bit encode fails encoder Init (error 4) when preanalysis is enabled.
    preanalysis = cfg.amf_preanalysis and not use_10
    if cfg.amf_preanalysis and use_10:
        rationale.append(
            f"{codec}: preanalysis disabled for 10-bit AMF encode (AMF driver limitation)."
        )

    vf_parts = _amf_vf_parts(
        use_10=use_10,
        decode_hwaccel=decode_hwaccel,
        target_width=target_width,
        target_height=target_height,
    )
    vf_prefix: list[str] = []
    if vf_parts:
        vf_prefix = ["-vf", ",".join(vf_parts)]

    args = vf_prefix + [
        "-c:v", codec,
        *_amf_quality_args(cfg.amf_quality),
        "-rc", cfg.amf_rc,
        "-preanalysis", "true" if preanalysis else "false",
        "-vbaq", "true" if cfg.amf_vbaq else "false",
        "-preencode", "true" if cfg.amf_preencode else "false",
        # Do not set ``-header_insertion_mode idr``: AMF/ffmpeg can hang on later
        # batched encode invocations (~batch 3+). Each batch is a fresh ffmpeg process
        # and AMF still emits a keyframe at frame 0 by default.
    ]
    if cfg.amf_g > 0:
        args += ["-g", str(cfg.amf_g)]
    args += ["-bf", str(cfg.amf_bf)]
    if cfg.amf_pa_lookahead_buffer_depth >= 0:
        args += [
            "-pa_lookahead_buffer_depth",
            str(cfg.amf_pa_lookahead_buffer_depth),
        ]
    if cfg.amf_pa_taq_mode >= 0:
        args += ["-pa_taq_mode", str(cfg.amf_pa_taq_mode)]
    rc_lower = cfg.amf_rc.lower()
    if rc_lower in {"cqp", "constqp", "qp"}:
        args += [
            "-qp_i", str(cfg.amf_qp_i),
            "-qp_p", str(cfg.amf_qp_p),
        ]
        if family == "h264":
            args += ["-qp_b", str(cfg.amf_qp_b)]
    elif rc_lower in {"vbr_peak", "vbr_latency"}:
        if cfg.amf_maxrate > 0:
            args += ["-maxrate", str(cfg.amf_maxrate)]
        if cfg.amf_bufsize > 0:
            args += ["-bufsize", str(cfg.amf_bufsize)]
    args += ["-pix_fmt", amf_pix_fmt, "-fps_mode", fps_mode]
    args += list(cfg.extra_args)
    rationale.append(
        f"AMD AMF {family}: quality={cfg.amf_quality} rc={cfg.amf_rc} "
        f"bit_depth={cfg.amf_bit_depth} preanalysis={int(preanalysis)} "
        f"vbaq={int(cfg.amf_vbaq)} preencode={int(cfg.amf_preencode)} "
        f"g={cfg.amf_g} bf={cfg.amf_bf} pix_fmt={amf_pix_fmt}"
    )
    return EncodeBuildResult(args=args, pix_fmt=result_pix_fmt, rationale=rationale)


def build_d3d12(
    cfg: EncoderCfg,
    *,
    family: str,
    target_width: int | None,
    target_height: int | None,
    source_pix_fmt: str | None,
    fps_mode: str = "passthrough",
) -> EncodeBuildResult:
    rationale: list[str] = []
    if family == "h264":
        codec = "h264_d3d12"
    elif family == "av1":
        codec = "av1_d3d12"
    else:
        raise ValueError(f"unsupported D3D12 family: {family}")

    pix_fmt = PIX_FMT_8BIT
    if _is_10bit_pix_fmt(source_pix_fmt):
        rationale.append(f"{codec}: source is 10-bit; using 8-bit output for compatibility.")

    args = _scale_filter_if_needed(target_width, target_height) + [
        "-c:v", codec,
        "-pix_fmt", "nv12",
    ]
    if cfg.d3d12_qp >= 0:
        args += ["-qp", str(cfg.d3d12_qp)]
    if cfg.d3d12_quality >= 0:
        args += ["-quality", str(cfg.d3d12_quality)]
    args += ["-fps_mode", fps_mode]
    args += list(cfg.extra_args)
    rationale.append(
        f"D3D12 {family}: qp={cfg.d3d12_qp} quality={cfg.d3d12_quality} pix_fmt=nv12"
    )
    return EncodeBuildResult(args=args, pix_fmt=pix_fmt, rationale=rationale)


def build_vulkan(
    cfg: EncoderCfg,
    *,
    family: str,
    target_width: int | None,
    target_height: int | None,
    source_pix_fmt: str | None,
    fps_mode: str = "passthrough",
) -> EncodeBuildResult:
    if family == "h264":
        codec = "h264_vulkan"
    elif family == "hevc":
        codec = "hevc_vulkan"
    elif family == "av1":
        codec = "av1_vulkan"
    else:
        raise ValueError(f"unsupported Vulkan family: {family}")

    rationale: list[str] = []
    pix_fmt = PIX_FMT_8BIT
    if _is_10bit_pix_fmt(source_pix_fmt):
        rationale.append(f"{codec}: source is 10-bit; forcing nv12 upload for compatibility.")

    vf_parts: list[str] = []
    if target_width and target_height:
        vf_parts.append(
            f"scale={target_width}:{target_height}:flags=lanczos:force_original_aspect_ratio=disable"
        )
    vf_parts.extend(["format=nv12", "hwupload"])
    args = [
        "-vf", ",".join(vf_parts),
        "-c:v", codec,
        "-pix_fmt", "vulkan",
    ]
    if cfg.vulkan_qp >= 0:
        args += ["-qp", str(cfg.vulkan_qp)]
    if cfg.vulkan_quality >= 0:
        args += ["-quality", str(cfg.vulkan_quality)]
    if cfg.vulkan_async_depth > 0:
        args += ["-async_depth", str(cfg.vulkan_async_depth)]
    args += ["-fps_mode", fps_mode]
    args += list(cfg.extra_args)
    rationale.append(
        f"Vulkan {family}: qp={cfg.vulkan_qp} quality={cfg.vulkan_quality} "
        f"async_depth={cfg.vulkan_async_depth} upload=nv12->hwupload"
    )
    return EncodeBuildResult(
        args=args,
        pix_fmt=pix_fmt,
        rationale=rationale,
        global_prefix=VULKAN_GLOBAL_PREFIX,
    )


def build_x264(
    cfg: EncoderCfg,
    *,
    target_width: int | None,
    target_height: int | None,
    source_pix_fmt: str | None,
    fps_mode: str = "passthrough",
) -> EncodeBuildResult:
    rationale: list[str] = []
    pix_fmt = PIX_FMT_8BIT  # libx264 8-bit is the default mainstream build
    if _is_10bit_pix_fmt(source_pix_fmt):
        rationale.append(
            "libx264: source is 10-bit; mainstream libx264 builds are 8-bit. Downconverting."
        )
    args = [
        "-c:v", "libx264",
        "-preset", cfg.x_preset,
        "-crf", str(cfg.x_crf),
        "-pix_fmt", pix_fmt,
        "-fps_mode", fps_mode,
    ]
    if cfg.x_tune:
        args += ["-tune", cfg.x_tune]
        if cfg.x_tune == "animation":
            rationale.append("libx264: tune=animation (favors line art and flat color).")
    args = _scale_filter_if_needed(target_width, target_height) + args
    args += list(cfg.extra_args)
    rationale.append(f"libx264: preset={cfg.x_preset} crf={cfg.x_crf} pix_fmt={pix_fmt}")
    return EncodeBuildResult(args=args, pix_fmt=pix_fmt, rationale=rationale)


def build_x265(
    cfg: EncoderCfg,
    *,
    target_width: int | None,
    target_height: int | None,
    source_pix_fmt: str | None,
    fps_mode: str = "passthrough",
) -> EncodeBuildResult:
    rationale: list[str] = []
    pix_fmt = PIX_FMT_10BIT if _is_10bit_pix_fmt(source_pix_fmt) else PIX_FMT_8BIT
    args = [
        "-c:v", "libx265",
        "-preset", cfg.x_preset,
        "-crf", str(cfg.x_crf),
        "-pix_fmt", pix_fmt,
        "-fps_mode", fps_mode,
    ]
    # x265 doesn't accept --tune animation; anime-favorable params via -x265-params.
    # If the preset already supplies -x265-params via extra_args, don't double-set.
    has_extra_x265_params = any(arg == "-x265-params" for arg in cfg.extra_args)
    params_from_cfg = (cfg.x265_params or "").strip()
    if not has_extra_x265_params and params_from_cfg:
        args += ["-x265-params", params_from_cfg]
        rationale.append("libx265: applying configured -x265-params string.")
    args = _scale_filter_if_needed(target_width, target_height) + args
    args += list(cfg.extra_args)
    rationale.append(f"libx265: preset={cfg.x_preset} crf={cfg.x_crf} pix_fmt={pix_fmt}")
    return EncodeBuildResult(args=args, pix_fmt=pix_fmt, rationale=rationale)


def build_encoder_args(
    cfg: EncoderCfg,
    *,
    target_width: int | None,
    target_height: int | None,
    source_pix_fmt: str | None,
    fps_mode: str = "passthrough",
    decode_hwaccel: str = "off",
) -> EncodeBuildResult:
    """Dispatch on encoder name."""
    name = cfg.name
    if name == "h264_nvenc":
        return build_nvenc(cfg, family="h264", target_width=target_width,
                           target_height=target_height, source_pix_fmt=source_pix_fmt,
                           fps_mode=fps_mode)
    if name == "hevc_nvenc":
        return build_nvenc(cfg, family="hevc", target_width=target_width,
                           target_height=target_height, source_pix_fmt=source_pix_fmt,
                           fps_mode=fps_mode)
    if name == "av1_nvenc":
        return build_nvenc(cfg, family="av1", target_width=target_width,
                           target_height=target_height, source_pix_fmt=source_pix_fmt,
                           fps_mode=fps_mode)
    if name == "libx264":
        return build_x264(cfg, target_width=target_width, target_height=target_height,
                          source_pix_fmt=source_pix_fmt, fps_mode=fps_mode)
    if name == "libx265":
        return build_x265(cfg, target_width=target_width, target_height=target_height,
                          source_pix_fmt=source_pix_fmt, fps_mode=fps_mode)
    if name == "h264_qsv":
        return build_qsv(cfg, family="h264", target_width=target_width,
                         target_height=target_height, source_pix_fmt=source_pix_fmt,
                         fps_mode=fps_mode)
    if name == "hevc_qsv":
        return build_qsv(cfg, family="hevc", target_width=target_width,
                         target_height=target_height, source_pix_fmt=source_pix_fmt,
                         fps_mode=fps_mode)
    if name == "av1_qsv":
        return build_qsv(cfg, family="av1", target_width=target_width,
                         target_height=target_height, source_pix_fmt=source_pix_fmt,
                         fps_mode=fps_mode)
    if name == "h264_amf":
        return build_amf(cfg, family="h264", target_width=target_width,
                         target_height=target_height, source_pix_fmt=source_pix_fmt,
                         fps_mode=fps_mode, decode_hwaccel=decode_hwaccel)
    if name == "hevc_amf":
        return build_amf(cfg, family="hevc", target_width=target_width,
                         target_height=target_height, source_pix_fmt=source_pix_fmt,
                         fps_mode=fps_mode, decode_hwaccel=decode_hwaccel)
    if name == "av1_amf":
        return build_amf(cfg, family="av1", target_width=target_width,
                         target_height=target_height, source_pix_fmt=source_pix_fmt,
                         fps_mode=fps_mode, decode_hwaccel=decode_hwaccel)
    if name == "h264_d3d12":
        return build_d3d12(cfg, family="h264", target_width=target_width,
                           target_height=target_height, source_pix_fmt=source_pix_fmt,
                           fps_mode=fps_mode)
    if name == "av1_d3d12":
        return build_d3d12(cfg, family="av1", target_width=target_width,
                           target_height=target_height, source_pix_fmt=source_pix_fmt,
                           fps_mode=fps_mode)
    if name == "h264_vulkan":
        return build_vulkan(cfg, family="h264", target_width=target_width,
                            target_height=target_height, source_pix_fmt=source_pix_fmt,
                            fps_mode=fps_mode)
    if name == "hevc_vulkan":
        return build_vulkan(cfg, family="hevc", target_width=target_width,
                            target_height=target_height, source_pix_fmt=source_pix_fmt,
                            fps_mode=fps_mode)
    if name == "av1_vulkan":
        return build_vulkan(cfg, family="av1", target_width=target_width,
                            target_height=target_height, source_pix_fmt=source_pix_fmt,
                            fps_mode=fps_mode)
    raise ValueError(f"unsupported encoder: {name}")
