"""FFmpeg adapter.

This is the workhorse adapter used by:
  * stage 08 (encode) for transcoding
  * stage 09 (mux) for stream-copy muxing
  * stage 04 (decode_serve)
  * stage 02 (sample_bench)

Design choices, briefly:

* Encoder availability is detected dynamically. NVENC may not be present (driver too old,
  or running under a non-NVIDIA GPU); the recommender consumes this info.
* We use `-progress pipe:1` rather than parsing stderr. Stderr is only used for warnings.
* We never pass `-y` (overwrite) by default. Callers that need overwrite must pass
  `allow_overwrite=True` in the command builder.
* All commands are logged in full (run_capture / run_streaming both log).
* No ffmpeg-python or other binding — we build argv lists ourselves. Bindings hide flags
  and version-skew bugs that bite in production.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.base import ToolAdapter, env_with_tool_dirs
from aep.constants import BIN_FFMPEG
from aep.errors import EncodeError, ToolNotFoundError
from aep.util.proc import run_capture, run_streaming

log = logging.getLogger(__name__)


_VERSION_RE = re.compile(r"ffmpeg version (\S+)")


# Encoders we care about. Presence is verified at runtime; this list is just the
# universe the rest of the app reasons about.
KNOWN_ENCODERS = {
    "h264_nvenc",
    "hevc_nvenc",
    "av1_nvenc",
    "h264_qsv",
    "hevc_qsv",
    "av1_qsv",
    "h264_amf",
    "hevc_amf",
    "av1_amf",
    "libx264",
    "libx265",
    "libsvtav1",   # software AV1 fallback; unsupported as default for anime
    "libaom-av1",  # too slow for episodic content; listed for completeness
}


@dataclass(frozen=True)
class EncoderInfo:
    name: str
    description: str
    is_hardware: bool
    family: str  # "h264" | "hevc" | "av1"


@dataclass(frozen=True)
class EncodeProgress:
    """One progress sample emitted during ffmpeg execution."""
    frame: int | None
    fps: float | None
    bitrate_kbps: float | None
    out_time_us: int | None
    speed: float | None       # multiple of realtime, e.g. 1.5x
    progress: str | None      # "continue" or "end"


class FFmpegAdapter(ToolAdapter):
    tool_id = "ffmpeg"
    bin_name = BIN_FFMPEG
    tools_subdir = "ffmpeg"

    # ----- versioning ----------------------------------------------------

    def _detect_version(self) -> str:
        result = run_capture([self.path, "-version"], timeout=15.0)
        m = _VERSION_RE.search(result.stdout)
        return m.group(1) if m else "unknown"

    def command_executable(self) -> str | Path:
        """Return executable token for argv construction.

        Unit tests that only assert argv shape should not require ffmpeg to be
        installed. Runtime execution paths still resolve and validate the tool.
        """
        try:
            return self.path
        except ToolNotFoundError:
            return self.bin_name

    # ----- encoder enumeration ------------------------------------------

    def list_encoders(self) -> list[EncoderInfo]:
        """Parse `ffmpeg -hide_banner -encoders`.

        Output format (each row):
            V..... libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10
        Type letter is the first column: V=video, A=audio, S=subtitle.
        """
        result = run_capture(
            [self.path, "-hide_banner", "-encoders"],
            env=env_with_tool_dirs(),
            timeout=15.0,
        )
        encoders: list[EncoderInfo] = []
        in_table = False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if not in_table:
                if stripped.startswith("------"):
                    in_table = True
                continue
            if not stripped:
                continue
            # First token is flags; second is name; rest is description.
            parts = stripped.split(maxsplit=2)
            if len(parts) < 2:
                continue
            flags, name = parts[0], parts[1]
            description = parts[2] if len(parts) > 2 else ""
            if not flags.startswith("V"):
                continue  # video only
            if name not in KNOWN_ENCODERS:
                continue
            family = (
                "h264" if "h264" in name else
                "hevc" if "hevc" in name or "x265" in name else
                "av1" if "av1" in name else
                "other"
            )
            is_hw = "_nvenc" in name or "_qsv" in name or "_amf" in name or "_vaapi" in name
            encoders.append(EncoderInfo(name=name, description=description,
                                        is_hardware=is_hw, family=family))
        return encoders

    def encoder_available(self, name: str) -> bool:
        return any(e.name == name for e in self.list_encoders())

    # ----- streaming execution with progress ----------------------------

    def run_with_progress(
        self,
        cmd: list[str | Path],
        *,
        cwd: Path | None = None,
    ) -> Iterator[EncodeProgress | str]:
        """Run an ffmpeg command and yield EncodeProgress samples or stderr lines.

        The caller is responsible for including `-progress pipe:1 -nostats` somewhere in
        the argv if they want progress events; otherwise this is a plain streaming run.
        """
        buf: dict[str, str] = {}
        env = env_with_tool_dirs()
        for stream, line in run_streaming(cmd, cwd=cwd, env=env):
            if stream == "stdout":
                # progress key=value
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                buf[key] = value
                if key == "progress":
                    yield _build_progress(buf)
                    if value == "end":
                        return
                    buf = {k: v for k, v in buf.items() if k == "progress"}
            else:
                # stderr: pass through to logs, but yield as informational lines
                yield line

    # ----- command builders ---------------------------------------------

    def build_passthrough_video_encode(
        self,
        *,
        source: Path,
        video_only_out: Path,
        encoder_args: list[str],
        decode_hwaccel: str = "off",
        progress: bool = True,
        allow_overwrite: bool = False,
        start_pts: float | None = None,  # M6.5: input-side seek for batched mode
        end_pts: float | None = None,    # M6.5: -t duration relative to start_pts
        global_prefix: list[str | Path] | None = None,
    ) -> list[str | Path]:
        """Build an ffmpeg command that re-encodes ONLY the source's first video stream
        to `video_only_out`. Audio/subs/chapters/attachments are NOT included in this
        output; they're merged later in the mux stage.

        We split video encoding from muxing because (1) it lets the mux stage choose
        between ffmpeg-mux and mkvmerge-mux without re-encoding, and (2) it makes resume
        possible — encoded video survives even if mux later fails.
        """
        cmd: list[str | Path] = [
            self.command_executable(),
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
        ]
        if global_prefix:
            cmd += list(global_prefix)
        if progress:
            cmd += ["-progress", "pipe:1", "-nostats"]
        if allow_overwrite:
            cmd += ["-y"]
        else:
            cmd += ["-n"]
        # Input-side -ss before -i for fast keyframe seek (see build_decode_to_frames).
        if start_pts is not None and start_pts > 0:
            cmd += ["-ss", f"{start_pts:.6f}"]
        cmd += _decode_input_args(str(source), decode_hwaccel=decode_hwaccel)
        cmd += [
            "-map", "0:v:0",  # first video stream only
            "-map_metadata", "-1",  # do not copy global metadata onto video-only artifact
            "-map_chapters", "-1",  # chapters belong to the final mux, not the intermediate
        ]
        if end_pts is not None:
            duration = end_pts - (start_pts or 0.0)
            if duration > 0:
                cmd += ["-t", f"{duration:.6f}"]
        cmd += encoder_args
        cmd += [str(video_only_out)]
        return cmd

    def build_remux_with_streams(
        self,
        *,
        encoded_video: Path,
        source: Path,
        output: Path,
        map_args: list[str],
        copy_args: list[str],
        global_metadata: bool,
        chapters: bool,
        allow_overwrite: bool = False,
    ) -> list[str | Path]:
        """Build a remux command that combines our encoded video with selected streams
        from the original source.

        Inputs:
            -i encoded_video   (input 0; video stream we just produced)
            -i source          (input 1; original; we pull audio/subs/chapters from here)

        `map_args` and `copy_args` come from the mux mapping engine.
        """
        cmd: list[str | Path] = [
            self.command_executable(),
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
        ]
        if allow_overwrite:
            cmd += ["-y"]
        else:
            cmd += ["-n"]
        cmd += ["-i", str(encoded_video), "-i", str(source)]
        cmd += map_args
        cmd += copy_args
        if global_metadata:
            cmd += ["-map_metadata", "1"]   # copy global metadata from source
        else:
            cmd += ["-map_metadata", "-1"]
        if chapters:
            cmd += ["-map_chapters", "1"]
        else:
            cmd += ["-map_chapters", "-1"]
        cmd += [str(output)]
        return cmd

    def build_decode_to_frames(
        self,
        *,
        source: Path,
        out_dir: Path,
        frame_format: str = "png",   # "png" or "webp"
        webp_lossless: bool = True,
        png_compression: int | None = None,  # 0..9; defaults to constants.PNG_COMPRESSION_LEVEL
        target_width: int | None = None,
        target_height: int | None = None,
        bt709_normalize: bool = True,  # zscale conversion to BT.709 limited 8-bit
        use_zscale: bool = True,
        start_number: int = 1,
        digits: int = 8,
        allow_overwrite: bool = True,
        decode_hwaccel: str = "off",
        start_pts: float | None = None,  # M6.5: -ss seek (input-side, before -i)
        end_pts: float | None = None,    # M6.5: -to end timestamp (output-side)
    ) -> list[str | Path]:
        """Build a command that decodes the source's primary video stream to a
        directory of numbered frames.

        We force a colorspace conversion to BT.709 limited 8-bit RGB by default
        because the NCNN-Vulkan binaries are 8-bit sRGB only — feeding them
        10-bit BT.2020 input would silently clip. The plan stage decides
        whether to take this path at all (HDR sources may be skipped per preset).

        Filename pattern: %0Nd.<ext> where N = ``digits``. NCNN binaries iterate
        the input dir in alpha order, so 8 digits is plenty for any episode.
        """
        if frame_format not in ("png", "webp"):
            raise ValueError(f"frame_format must be png or webp, got {frame_format!r}")
        from aep.constants import PNG_COMPRESSION_LEVEL
        if png_compression is None:
            png_compression = PNG_COMPRESSION_LEVEL

        cmd: list[str | Path] = [
            self.command_executable(),
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
        ]
        cmd += ["-y"] if allow_overwrite else ["-n"]
        # Input-side -ss is much faster than output-side: ffmpeg seeks to the
        # nearest preceding keyframe in the demuxer rather than decoding from 0.
        # Combined with output-side -to (after -i), this gives accurate batch
        # boundaries when start_pts is on a keyframe (the planner snaps to one).
        if start_pts is not None and start_pts > 0:
            cmd += ["-ss", f"{start_pts:.6f}"]
        cmd += _decode_input_args(str(source), decode_hwaccel=decode_hwaccel)
        cmd += ["-map", "0:v:0"]
        if end_pts is not None:
            # -to is interpreted relative to the seek point when -ss precedes -i,
            # so we pass duration = end - start to keep semantics explicit and
            # avoid surprising ffmpeg behavior across versions.
            duration = end_pts - (start_pts or 0.0)
            if duration > 0:
                cmd += ["-t", f"{duration:.6f}"]

        # Filter chain: optional resize, then colorspace normalization, then
        # explicit pix-fmt to RGB so the encoder is forced to 8-bit. zscale is
        # used over scale because it correctly handles BT.2020/PQ → BT.709 SDR.
        cmd += [
            "-vf",
            _decode_preprocess_vf_inner(
                target_width=target_width,
                target_height=target_height,
                bt709_normalize=bt709_normalize,
                use_zscale=use_zscale,
            ),
        ]

        # Per-format encoder flags.
        if frame_format == "png":
            cmd += ["-c:v", "png", "-compression_level", str(png_compression)]
            ext = "png"
        else:  # webp
            cmd += ["-c:v", "libwebp"]
            if webp_lossless:
                cmd += ["-lossless", "1", "-compression_level", "6"]
            else:
                # Even "high quality" WebP is lossy; we don't expose this path
                # by default but leaving the branch makes intent visible.
                cmd += ["-quality", "95"]
            ext = "webp"

        cmd += ["-an", "-sn", "-dn"]
        cmd += ["-start_number", str(start_number)]
        out_pattern = out_dir / f"%0{digits}d.{ext}"
        cmd += [str(out_pattern)]
        return cmd

    def build_decode_to_frames_with_scene_metadata_fused(
        self,
        *,
        source: Path,
        out_dir: Path,
        metadata_out: Path,
        frame_format: str = "png",
        webp_lossless: bool = True,
        png_compression: int | None = None,
        target_width: int | None = None,
        target_height: int | None = None,
        bt709_normalize: bool = True,
        use_zscale: bool = True,
        start_number: int = 1,
        digits: int = 8,
        allow_overwrite: bool = True,
        decode_hwaccel: str = "off",
        start_pts: float | None = None,
        end_pts: float | None = None,
    ) -> list[str | Path]:
        """Single decode: frame files + ``metadata=print`` scene scores via ``filter_complex``.

        One decode feeds ``split``: branch (1) matches ``build_decode_to_frames`` output;
        branch (2) matches ``build_scene_score_scan`` (``select`` + ``metadata=print``).
        Run with ``cwd=metadata_out.parent`` so ``metadata_out.name`` is basename-only.

        If ffmpeg rejects the graph, the caller falls back to decode + scan separately.
        """
        if frame_format not in ("png", "webp"):
            raise ValueError(f"frame_format must be png or webp, got {frame_format!r}")
        from aep.constants import PNG_COMPRESSION_LEVEL
        if png_compression is None:
            png_compression = PNG_COMPRESSION_LEVEL

        meta_name = metadata_out.name
        if not meta_name or "/" in meta_name or "\\" in meta_name:
            raise ValueError(
                "build_decode_to_frames_with_scene_metadata_fused: metadata_out must be "
                "a filename with no path separators (caller sets cwd to its parent)",
            )

        prep = _decode_preprocess_vf_inner(
            target_width=target_width,
            target_height=target_height,
            bt709_normalize=bt709_normalize,
            use_zscale=use_zscale,
        )
        fc = (
            f"[0:v]{prep}[rgb];"
            f"[rgb]split[enc][scn];"
            f"[scn]select=gt(scene+1\\,0),metadata=print:file={meta_name}[meta]"
        )

        cmd: list[str | Path] = [
            self.command_executable(),
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
        ]
        cmd += ["-y"] if allow_overwrite else ["-n"]
        if start_pts is not None and start_pts > 0:
            cmd += ["-ss", f"{start_pts:.6f}"]
        cmd += _decode_input_args(str(source), decode_hwaccel=decode_hwaccel)
        cmd += ["-map", "0:v:0"]
        if end_pts is not None:
            duration = end_pts - (start_pts or 0.0)
            if duration > 0:
                cmd += ["-t", f"{duration:.6f}"]
        cmd += ["-filter_complex", fc]
        cmd += ["-map", "[enc]", "-an", "-sn", "-dn"]
        if frame_format == "png":
            cmd += ["-c:v", "png", "-compression_level", str(png_compression)]
            ext = "png"
        else:
            cmd += ["-c:v", "libwebp"]
            if webp_lossless:
                cmd += ["-lossless", "1", "-compression_level", "6"]
            else:
                cmd += ["-quality", "95"]
            ext = "webp"
        cmd += ["-start_number", str(start_number)]
        out_pattern = out_dir / f"%0{digits}d.{ext}"
        cmd += [str(out_pattern)]
        cmd += ["-map", "[meta]", "-an", "-sn", "-dn", "-f", "null", "-"]
        return cmd

    def build_scene_score_scan(
        self,
        *,
        source: Path,
        metadata_out: Path,
        decode_hwaccel: str = "off",
        start_pts: float | None = None,
        end_pts: float | None = None,
    ) -> list[str | Path]:
        """Decode video to null while recording per-frame scene scores to ``metadata_out``.

        Uses the **select** filter's ``scene`` statistic (same idea as
        ``select='gt(scene,THRESHOLD)'`` in the ffmpeg docs). ``select`` stores the
        score in frame metadata as ``lavfi.scene_score``; **showinfo does not print
        that metadata**, so we chain **metadata=print:file=...** and parse the sidecar
        file (see ``aep.util.frame_dedupe.parse_metadata_print_scene_scores``).

        ``metadata_out`` must be a **basename-only** path segment (no directories):
        the caller runs ffmpeg with ``cwd=metadata_out.parent`` so Windows drive
        letters and ``:`` in paths do not break the ``-vf`` option parser.

        Mirrors ``build_decode_to_frames`` input timing (``-ss`` before ``-i``, ``-t``)
        so batch windows match decode-serve.
        """
        cmd: list[str | Path] = [
            self.command_executable(),
            "-hide_banner",
            "-nostdin",
            "-loglevel", "info",
            "-y",
        ]
        if start_pts is not None and start_pts > 0:
            cmd += ["-ss", f"{start_pts:.6f}"]
        cmd += _decode_input_args(str(source), decode_hwaccel=decode_hwaccel)
        cmd += ["-map", "0:v:0"]
        if end_pts is not None:
            duration = end_pts - (start_pts or 0.0)
            if duration > 0:
                cmd += ["-t", f"{duration:.6f}"]
        meta_name = metadata_out.name
        if not meta_name or "/" in meta_name or "\\" in meta_name:
            raise ValueError(
                "build_scene_score_scan: metadata_out must be a filename in cwd "
                "(no path separators); pass e.g. parent / 'aep_frame_dedupe_scene.txt'",
            )
        # Comma inside select expr separates filters — escape it (see ffmpeg select examples).
        cmd += [
            "-vf", f"select=gt(scene+1\\,0),metadata=print:file={meta_name}",
            "-an", "-sn", "-dn",
            "-f", "null",
            "-",
        ]
        return cmd

    def build_encode_from_frames(
        self,
        *,
        frame_dir: Path,
        frame_format: str,
        fps_num: int,
        fps_den: int,
        video_only_out: Path,
        encoder_args: list[str],
        start_number: int = 1,
        digits: int = 8,
        allow_overwrite: bool = True,
        progress: bool = False,
        png_compression: int | None = None,  # M6.5: only relevant if encoder_args re-emits PNG
        global_prefix: list[str | Path] | None = None,
    ) -> list[str | Path]:
        """Build a command that encodes a directory of numbered frames into a
        video-only intermediate file.

        Uses the image2 demuxer with an explicit framerate. We pass framerate
        as numerator/denominator so 24000/1001 (NTSC), 60000/1001, etc. survive
        round-trips with no rounding error.
        """
        if frame_format not in ("png", "webp"):
            raise ValueError(f"frame_format must be png or webp, got {frame_format!r}")
        cmd: list[str | Path] = [
            self.command_executable(),
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
        ]
        if global_prefix:
            cmd += list(global_prefix)
        if progress:
            cmd += ["-progress", "pipe:1", "-nostats"]
        cmd += ["-y"] if allow_overwrite else ["-n"]
        cmd += [
            "-framerate", f"{fps_num}/{fps_den}",
            "-start_number", str(start_number),
            "-i", str(frame_dir / f"%0{digits}d.{frame_format}"),
            "-map", "0:v:0",
            "-map_metadata", "-1",
            "-map_chapters", "-1",
        ]
        cmd += encoder_args
        cmd += [str(video_only_out)]
        return cmd

def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: str) -> float | None:
    if value in ("N/A", "", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_progress(buf: dict[str, str]) -> EncodeProgress:
    bitrate_raw = buf.get("bitrate", "").rstrip("kbits/s").strip()
    speed_raw = buf.get("speed", "").rstrip("x").strip()
    return EncodeProgress(
        frame=_to_int(buf.get("frame", "")),
        fps=_to_float(buf.get("fps", "")),
        bitrate_kbps=_to_float(bitrate_raw),
        out_time_us=_to_int(buf.get("out_time_us", "")),
        speed=_to_float(speed_raw),
        progress=buf.get("progress"),
    )


def raise_if_failed(returncode: int, stderr_tail: str) -> None:
    if returncode != 0:
        raise EncodeError(
            f"ffmpeg exited with code {returncode}",
            context={"stderr_tail": stderr_tail[-2000:]},
        )


def _decode_preprocess_vf_inner(
    *,
    target_width: int | None,
    target_height: int | None,
    bt709_normalize: bool,
    use_zscale: bool,
) -> str:
    """Comma-separated vf chain (no ``-vf`` / brackets) matching decode-to-frames color path."""
    parts: list[str] = []
    if target_width and target_height:
        parts.append(f"scale={target_width}:{target_height}:flags=bicubic")
    if bt709_normalize:
        if use_zscale:
            parts.append("zscale=t=bt709:m=bt709:p=bt709:r=limited,format=rgb24")
        else:
            parts.append("format=rgb24")
    else:
        parts.append("format=rgb24")
    return ",".join(parts)


# Decode modes that may fail at runtime (driver / build / source); stages retry with software decode.
DECODE_HWACCEL_WITH_SW_FALLBACK = frozenset({"d3d11va", "cuda"})


def decode_hwaccel_has_sw_fallback(decode_hwaccel: str) -> bool:
    return (decode_hwaccel or "off").lower() in DECODE_HWACCEL_WITH_SW_FALLBACK


def _decode_input_args(source: str, *, decode_hwaccel: str) -> list[str]:
    mode = (decode_hwaccel or "off").lower()
    if mode == "d3d11va":
        return [
            "-hwaccel", "d3d11va",
            "-i", source,
        ]
    if mode == "cuda":
        return [
            "-hwaccel", "cuda",
            "-i", source,
        ]
    return ["-i", source]
