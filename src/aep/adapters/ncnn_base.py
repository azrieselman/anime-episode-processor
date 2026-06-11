"""NCNN-Vulkan adapter base class.

All three of our learned-model binaries (realcugan-ncnn-vulkan,
realesrgan-ncnn-vulkan, rife-ncnn-vulkan, plus an experimental waifu2x adapter)
share a near-identical command-line surface:

    <bin> -i <input_dir> -o <output_dir> -g <gpu_id> -j <load:proc:save>
          -t <tile_size> -m <model_dir> [-n <noise>] [-s <scale>] [-x] [-f <fmt>]

…and a near-identical failure mode: Vulkan OOM emits ``vkAllocateMemory failed``
(or ``failed to allocate``/``out of device memory``) on stderr and the process
exits non-zero. This base class factors out:

* path resolution + version probe (delegated to ToolAdapter)
* model-dir resolution (NCNN ships models alongside the binary; we accept
  either an absolute model-dir or a name relative to the binary's parent)
* a hardened ``run_with_oom_fallback`` helper that shrinks tile size on OOM
  and consults a per-hardware-fingerprint hint cache so future jobs start at
  the known-good tile size
* image-format selection (PNG vs WebP-lossless)
* a small dataclass capturing the result so stage code stays terse

Everything that varies per tool — the exact subset of flags accepted, model
naming convention, default tile size — lives in the subclass.

We do NOT try to abstract away the directory-mode I/O. NCNN binaries are
strictly directory-in/directory-out; piping is not supported upstream and
working around it requires re-encoding PNG per frame in Python, which is both
slow and fragile. Disk it is.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aep.adapters.base import ToolAdapter, env_with_tool_dirs
from aep.errors import OOMError, ToolNotFoundError
from aep.util.paths import cache_dir
from aep.util.proc import ProcError, ProcInterrupted, run_capture, run_streaming

log = logging.getLogger(__name__)


# --- OOM detection --------------------------------------------------------
#
# These patterns come from inspecting actual stderr from the upstream binaries
# under VRAM pressure. We match generously because release builds occasionally
# log slightly different messages depending on the underlying Vulkan loader.

_OOM_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"vkAllocateMemory\s+failed", re.IGNORECASE),
    re.compile(r"out of device memory", re.IGNORECASE),
    re.compile(r"failed to allocate.*memory", re.IGNORECASE),
    re.compile(r"VK_ERROR_OUT_OF_DEVICE_MEMORY", re.IGNORECASE),
    re.compile(r"VK_ERROR_OUT_OF_HOST_MEMORY", re.IGNORECASE),
)


def stderr_indicates_oom(stderr: str) -> bool:
    return any(p.search(stderr) for p in _OOM_PATTERNS)


# --- Vulkan GPU fault detection ------------------------------------------
#
# Under cumulative driver/GPU state pressure (common on long batched RIFE runs
# on Windows) the ncnn binaries may log ``vkQueueSubmit failed -4`` (device
# lost) while continuing to emit frames — some of which are black/corrupt.
# The process often never exits non-zero, so stages must watch stderr and
# terminate + retry rather than trusting the exit code alone.

_VULKAN_GPU_FAULT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"vkQueueSubmit\s+failed", re.IGNORECASE),
    re.compile(r"VK_ERROR_DEVICE_LOST", re.IGNORECASE),
    re.compile(r"vkDeviceWaitIdle\s+failed", re.IGNORECASE),
    re.compile(r"vkAcquireNextImageKHR\s+failed", re.IGNORECASE),
)


def stderr_indicates_vulkan_gpu_fault(stderr: str) -> bool:
    return any(p.search(stderr) for p in _VULKAN_GPU_FAULT_PATTERNS)


# --- tile hint persistence -----------------------------------------------
#
# We persist (hardware_fp, tool_id, model_id, source_height) → known_good_tile
# so that after a single OOM-driven shrink, subsequent jobs on the same machine
# don't waste an entire run discovering the same bad starting tile size.

_HINT_FILE_NAME = "tile_hints.json"


def _hint_path() -> Path:
    return cache_dir() / _HINT_FILE_NAME


def _hint_key(*, hardware_fp: str, tool_id: str, model_id: str, source_height: int | None) -> str:
    # Bucket source heights into 360-line bins so a 1078-line and a 1080-line
    # source share a hint without colliding with 720p.
    bucket = (source_height // 360 * 360) if source_height else 0
    return f"{hardware_fp}|{tool_id}|{model_id}|{bucket}"


def load_tile_hint(*, hardware_fp: str, tool_id: str, model_id: str, source_height: int | None) -> int | None:
    path = _hint_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data.get(_hint_key(
        hardware_fp=hardware_fp, tool_id=tool_id,
        model_id=model_id, source_height=source_height,
    ))


def save_tile_hint(
    *,
    hardware_fp: str,
    tool_id: str,
    model_id: str,
    source_height: int | None,
    tile_size: int,
) -> None:
    path = _hint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}
    data[_hint_key(
        hardware_fp=hardware_fp, tool_id=tool_id,
        model_id=model_id, source_height=source_height,
    )] = tile_size
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


# --- result type ---------------------------------------------------------


@dataclass(frozen=True)
class NcnnRunResult:
    output_dir: Path
    frames_in: int
    frames_out: int
    tile_size_used: int
    duration_s: float
    attempts: int                 # 1 = first try succeeded; 2+ = OOM fallbacks
    rationale: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --- the adapter base ---------------------------------------------------


# Lossless image formats accepted by all NCNN binaries. PNG is the safe
# default. WebP-lossless is ~25-35% smaller and accepted by every binary in
# the supported version range; lossy variants are deliberately not exposed.
#
# M6.5 PNG-compression-level note: upstream NCNN-Vulkan binaries use libpng's
# default zlib level (6) when writing PNGs and expose no flag to override it.
# That happens to match constants.PNG_COMPRESSION_LEVEL exactly, so AEP's
# decode/postprocess (ffmpeg) and upscale/interpolate (ncnn) intermediates all
# land at level 6 without per-binary configuration. If a future NCNN release
# adds a compression-level flag we'd wire it in via build_argv's `extra=`.
SUPPORTED_FRAME_FORMATS = ("png", "webp")


class NcnnVulkanAdapter(ToolAdapter):
    """Common machinery for ncnn-vulkan model runners.

    Subclasses set:
      tool_id, bin_name, tools_subdir       (inherited from ToolAdapter)
      default_tile_size                     (per-tool VRAM-friendly default)
      tile_size_floor                       (refuse to shrink below this)
      supports_format_flag                  (rife <20221029 lacks `-f`; subclass overrides)
      version_re                            (compiled regex matching `<bin> -h` output)
      model_subdir_default                  (relative to binary dir; subclass-specific)
    """

    default_tile_size: int = 256
    tile_size_floor: int = 64
    supports_format_flag: bool = True
    version_re: re.Pattern[str] = re.compile(r"version\s+(\S+)", re.IGNORECASE)
    # Forks such as TNTwise/rife-ncnn-vulkan omit a "version X" line but often
    # echo the YYYYMMDD release tag in the help banner / examples.
    _date_tag_version_re: re.Pattern[str] = re.compile(r"\b(20\d{6})\b")
    model_subdir_default: str = "models"

    # --------------------------------------------------------- versioning

    def _detect_version(self) -> str:
        """NCNN binaries print usage to STDERR on `-h`/no-args and exit non-zero.

        We tolerate the non-zero exit because that's the documented behavior;
        we just want to scrape the banner line.
        """
        try:
            result = run_capture(
                [self.path, "-h"],
                env=env_with_tool_dirs(),
                timeout=10.0,
                check=False,
            )
        except ProcError as exc:
            result = exc.result
        # Some builds put the version in stdout, others stderr. Check both.
        for blob in (result.stdout, result.stderr):
            m = self.version_re.search(blob)
            if m:
                return m.group(1)
            m = self._date_tag_version_re.search(blob)
            if m:
                return m.group(1)
        return "unknown"

    # --------------------------------------------------------- model resolution

    def resolve_model_dir(self, model_id: str) -> Path:
        """Find the directory containing the model files for ``model_id``.

        NCNN binaries expect a directory containing matched ``.param``/``.bin``
        files. Conventional layout in the upstream releases:

            <bin_dir>/models/<model_id>/  (rife)
            <bin_dir>/models-<model_id>/  (cugan: models-pro, models-se)
            <bin_dir>/models/<model_id>/  (esrgan)

        This method tries common variants and returns the first that exists,
        or raises ``ToolNotFoundError`` if nothing matches. Subclasses can
        override for tool-specific layouts.
        """
        bin_dir = self.path.parent
        candidates = [
            bin_dir / model_id,                          # absolute-ish: "models-pro"
            bin_dir / self.model_subdir_default / model_id,
            bin_dir / f"models-{model_id}",              # CUGAN convention
            bin_dir / f"rife-{model_id}",                # alternate RIFE convention
        ]
        for c in candidates:
            if c.is_dir():
                return c.resolve()
        raise ToolNotFoundError(
            f"{self.tool_id} model dir not found for id={model_id!r}",
            context={"searched": [str(c) for c in candidates]},
        )

    # --------------------------------------------------------- argv builder

    def build_argv(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        model_dir: Path,
        tile_size: int,
        gpu_id: int = 0,
        load_proc_save: tuple[int, int, int] = (1, 2, 2),
        frame_format: str = "png",
        scale: int | None = None,
        denoise: int | None = None,
        tta: bool = False,
        extra: list[str] | None = None,
    ) -> list[str | Path]:
        """Compose the argv shared by every NCNN-Vulkan binary.

        Subclasses extend by passing ``extra=[...]`` for tool-specific flags.
        """
        if frame_format not in SUPPORTED_FRAME_FORMATS:
            raise ValueError(
                f"frame_format must be one of {SUPPORTED_FRAME_FORMATS}, got {frame_format!r}"
            )
        if tile_size < self.tile_size_floor:
            raise ValueError(
                f"tile_size {tile_size} below floor {self.tile_size_floor} for {self.tool_id}"
            )
        load, proc, save = load_proc_save
        argv: list[str | Path] = [
            self.path,
            "-i", str(input_dir),
            "-o", str(output_dir),
            "-g", str(gpu_id),
            "-j", f"{load}:{proc}:{save}",
            "-t", str(tile_size),
            "-m", str(model_dir),
        ]
        if scale is not None:
            argv += ["-s", str(scale)]
        if denoise is not None:
            argv += ["-n", str(denoise)]
        if tta:
            argv += ["-x"]
        if self.supports_format_flag:
            argv += ["-f", frame_format]
        if extra:
            argv += list(extra)
        return argv

    # --------------------------------------------------------- streaming run

    def run_with_oom_fallback(
        self,
        *,
        argv_factory: Callable[[int], list[str | Path]],
        initial_tile_size: int,
        hardware_fp: str,
        model_id: str,
        source_height: int | None,
        on_progress: Callable[[str], None] | None = None,
        should_interrupt: Callable[[], str | None] | None = None,
        max_attempts: int = 4,
    ) -> NcnnRunResult:
        """Run the binary, halving tile size on Vulkan OOM up to ``max_attempts`` times.

        ``argv_factory`` is a callable that produces argv given a tile size, so
        we never have to mutate an existing list. The caller is responsible for
        producing a fresh, empty output directory before each attempt — but in
        practice the tools overwrite cleanly so we don't enforce that.

        Persists a per-hardware-fingerprint tile hint after success so future
        jobs avoid the same OOM-driven discovery cycle.
        """
        rationale: list[str] = []
        warnings: list[str] = []

        # Consult hint cache; if present and lower than initial, start there.
        hinted = load_tile_hint(
            hardware_fp=hardware_fp, tool_id=self.tool_id,
            model_id=model_id, source_height=source_height,
        )
        if hinted is not None and hinted < initial_tile_size:
            rationale.append(
                f"using cached tile-size hint {hinted} (< requested {initial_tile_size})"
            )
            tile = hinted
        else:
            tile = initial_tile_size

        last_stderr = ""
        attempts = 0
        t0 = time.monotonic()
        env = env_with_tool_dirs()
        while attempts < max_attempts:
            attempts += 1
            argv = argv_factory(tile)
            log.info("ncnn run attempt=%d tile=%d", attempts, tile)
            try:
                stderr_lines: list[str] = []
                for stream, line in run_streaming(argv, env=env, should_interrupt=should_interrupt):
                    if stream == "stderr":
                        stderr_lines.append(line)
                        if on_progress:
                            on_progress(line)
                last_stderr = "\n".join(stderr_lines)
                # Successful exit fell out of the streaming loop without raising.
                save_tile_hint(
                    hardware_fp=hardware_fp, tool_id=self.tool_id,
                    model_id=model_id, source_height=source_height,
                    tile_size=tile,
                )
                rationale.append(f"completed at tile={tile} on attempt {attempts}")
                # frames_in/out are filled in by the caller (we don't list dirs here
                # so this base class stays I/O-light).
                return NcnnRunResult(
                    output_dir=Path("."),  # caller fills in
                    frames_in=0,
                    frames_out=0,
                    tile_size_used=tile,
                    duration_s=time.monotonic() - t0,
                    attempts=attempts,
                    rationale=rationale,
                    warnings=warnings,
                )
            except ProcError as exc:
                last_stderr = exc.result.stderr
                if stderr_indicates_oom(last_stderr):
                    new_tile = tile // 2
                    if new_tile < self.tile_size_floor:
                        warnings.append(
                            f"OOM at tile={tile}; cannot shrink below floor {self.tile_size_floor}"
                        )
                        break
                    warnings.append(
                        f"Vulkan OOM at tile={tile}; retrying at tile={new_tile}"
                    )
                    tile = new_tile
                    continue
                # Non-OOM failure: re-raise immediately, no retry.
                raise
            except ProcInterrupted:
                raise

        # Exhausted attempts (or hit floor).
        raise OOMError(
            f"{self.tool_id}: exhausted tile fallbacks after {attempts} attempts",
            context={
                "stderr_tail": last_stderr[-2000:],
                "final_tile": tile,
                "floor": self.tile_size_floor,
                "warnings": warnings,
            },
        )

    # --------------------------------------------------------- chunked execution

    def run_chunked(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        chunk_size: int,
        argv_factory: Callable[[Path, Path, int], list[str | Path]],
        initial_tile_size: int,
        hardware_fp: str,
        model_id: str,
        source_height: int | None,
        frame_format: str = "png",
        scratch_dir: Path | None = None,
        on_progress: Callable[[str], None] | None = None,
        should_interrupt: Callable[[], str | None] | None = None,
    ) -> NcnnRunResult:
        """Run the binary in chunks of ``chunk_size`` frames at a time.

        Long ncnn-vulkan invocations (30k+ frames) hit cumulative driver/GPU
        state issues on Windows that smaller invocations don't. We split the
        input directory into ``chunk_size``-frame slices, hardlink (or copy)
        the frames for each slice into a temp subdir, run the binary, and
        merge outputs back into ``output_dir`` preserving the original 8-digit
        filenames.

        ``argv_factory(in_dir, out_dir, tile)`` builds the argv for one chunk.
        The OOM-fallback wrapper still applies per chunk; tile hints persist
        across chunks via the same hint cache.
        """
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {chunk_size}")
        if not input_dir.is_dir():
            raise ValueError(f"input_dir not a directory: {input_dir}")

        # Enumerate frame files in deterministic alpha order. We respect the
        # caller's chosen format (png/webp); ignore other files quietly so a
        # stray ``.tmp`` doesn't break alignment.
        suffix = f".{frame_format}"
        all_frames = sorted(
            p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == suffix
        )
        if not all_frames:
            raise ValueError(
                f"run_chunked: no {frame_format} frames in {input_dir}"
            )

        scratch_root = scratch_dir or (output_dir.parent / f".chunks_{self.tool_id}")
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
        scratch_root.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        rationale: list[str] = []
        warnings: list[str] = []
        total_attempts = 0
        total_frames_written = 0
        last_tile_used = initial_tile_size
        n_chunks = (len(all_frames) + chunk_size - 1) // chunk_size
        rationale.append(
            f"chunked run: {len(all_frames)} frames / chunk_size={chunk_size} → {n_chunks} chunks"
        )
        t0 = time.monotonic()

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, len(all_frames))
            chunk_in = scratch_root / f"in_{chunk_idx:04d}"
            chunk_out = scratch_root / f"out_{chunk_idx:04d}"
            chunk_in.mkdir(parents=True, exist_ok=False)
            chunk_out.mkdir(parents=True, exist_ok=False)

            # Materialize input frames for this chunk. Try hardlink first
            # (zero-copy on NTFS/ext4 same-volume); fall back to copy.
            for src in all_frames[start:end]:
                dst = chunk_in / src.name
                try:
                    dst.hardlink_to(src)
                except (OSError, NotImplementedError):
                    shutil.copy2(src, dst)

            run_result = self.run_with_oom_fallback(
                argv_factory=lambda t, _ci=chunk_in, _co=chunk_out: argv_factory(_ci, _co, t),
                initial_tile_size=last_tile_used,
                hardware_fp=hardware_fp,
                model_id=model_id,
                source_height=source_height,
                on_progress=on_progress,
                should_interrupt=should_interrupt,
            )
            last_tile_used = run_result.tile_size_used
            total_attempts += run_result.attempts
            warnings.extend(
                f"chunk {chunk_idx + 1}/{n_chunks}: {w}" for w in run_result.warnings
            )
            rationale.extend(
                f"chunk {chunk_idx + 1}/{n_chunks}: {r}" for r in run_result.rationale
            )

            # Move chunk outputs into the final output_dir, preserving names.
            # ncnn writes outputs with the same filenames as inputs, so we get
            # 8-digit alignment for free.
            for produced in chunk_out.iterdir():
                if produced.is_file():
                    dst = output_dir / produced.name
                    if dst.exists():
                        dst.unlink()
                    produced.replace(dst)
                    total_frames_written += 1

            # Reclaim space immediately so the next chunk's hardlinks/copies
            # don't accumulate under the scratch root for the whole run.
            shutil.rmtree(chunk_in, ignore_errors=True)
            shutil.rmtree(chunk_out, ignore_errors=True)

        # Final cleanup of the scratch root.
        shutil.rmtree(scratch_root, ignore_errors=True)

        rationale.append(
            f"chunked run completed: {total_frames_written} frames, {total_attempts} cumulative attempts"
        )
        return NcnnRunResult(
            output_dir=output_dir,
            frames_in=len(all_frames),
            frames_out=total_frames_written,
            tile_size_used=last_tile_used,
            duration_s=time.monotonic() - t0,
            attempts=total_attempts,
            rationale=rationale,
            warnings=warnings,
        )


# --------------------------------------------------------- frame I/O helpers


def expected_frame_filenames(count: int, *, format: str = "png", start: int = 1, width: int = 8) -> list[str]:
    """Generate the deterministic filenames our pipeline uses for frames.

    NCNN binaries don't enforce a naming scheme — they just iterate the input
    directory in alpha order and write to the output directory using the same
    name. We use 8-digit zero-padded indices because a 24fps episode at 60fps
    interpolation can exceed 86,400 frames per chunk in pathological cases.
    """
    if format not in SUPPORTED_FRAME_FORMATS:
        raise ValueError(f"unsupported format: {format!r}")
    return [f"{i:0{width}d}.{format}" for i in range(start, start + count)]


def count_frames_in_dir(dir_path: Path, *, format: str | None = None) -> int:
    """Count files matching our frame format(s) in a directory."""
    if not dir_path.is_dir():
        return 0
    if format:
        suffixes = {f".{format}"}
    else:
        suffixes = {f".{f}" for f in SUPPORTED_FRAME_FORMATS}
    return sum(1 for p in dir_path.iterdir() if p.suffix.lower() in suffixes)


def empty_dir(dir_path: Path) -> Path:
    """Idempotently create an empty directory (used between OOM retries)."""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
