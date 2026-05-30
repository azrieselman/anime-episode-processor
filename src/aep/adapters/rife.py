"""RIFE ncnn-vulkan adapter.

Upstream binary: github.com/TNTwise/rife-ncnn-vulkan, pinned at 20250112.

Argv surface we use:
    rife-ncnn-vulkan.exe
        -i <in_dir> -o <out_dir>
        -m <model_dir>      e.g. .../rife-v4.6
        -n <num_frames>     OUTPUT frame count (when set, overrides -s)
        -t <tile_size>      RIFE rarely OOMs, but the flag exists
        -g <gpu_id>
        -j <load:proc:save>
        -f png|webp
        -u                  enable UHD mode (4K+)

Critical behavior — scene-cut handling:

The vanilla ncnn binary will happily interpolate across scene cuts, producing
the morphing-transition artifacts that make Waifu2x-Extension-GUI's RIFE
output look amateurish. The binary has no scene-cut hint flag.

Our solution is to slice the input into runs separated by scene cuts (passed
in from stage 03), run RIFE per-run, and at concat time insert ``multiplier-1``
copies of the boundary frame so the final frame count and timing line up
exactly with what a non-cut-aware RIFE pass would have produced — without the
morph.

This module provides the splicing primitives. The stage-level orchestrator in
``s06_interpolate`` uses them.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.ncnn_base import NcnnVulkanAdapter
from aep.constants import BIN_RIFE, DEFAULT_RIFE_THREADS, PINNED_VERSIONS
from aep.util.paths import tools_dir
from aep.util.win_pe_version import first_yyyymmdd_tag, pe_version_resource_strings

log = logging.getLogger(__name__)


# Model directories shipped in the TNTwise 20250112 Windows bundle.
# Keys in _KNOWN_VERSIONS strip the "rife-" prefix for preset readability.
_MODEL_DIR_NAMES: tuple[str, ...] = (
    "rife-HD",
    "rife-UHD",
    "rife-anime",
    "rife-v2.3",
    "rife-v2.4",
    "rife-v2",
    "rife-v3.0",
    "rife-v3.1",
    "rife-v3.6",
    "rife-v3.9",
    "rife-v4.1",
    "rife-v4.10",
    "rife-v4.11",
    "rife-v4.12-lite",
    "rife-v4.12",
    "rife-v4.13-lite",
    "rife-v4.13",
    "rife-v4.14-lite",
    "rife-v4.14",
    "rife-v4.15-lite",
    "rife-v4.15",
    "rife-v4.16-lite",
    "rife-v4.17-lite",
    "rife-v4.17",
    "rife-v4.18",
    "rife-v4.19",
    "rife-v4.2",
    "rife-v4.20",
    "rife-v4.21",
    "rife-v4.22-lite",
    "rife-v4.22",
    "rife-v4.23",
    "rife-v4.24",
    "rife-v4.25-heavy",
    "rife-v4.25-lite",
    "rife-v4.25",
    "rife-v4.26-large",
    "rife-v4.26",
    "rife-v4.3",
    "rife-v4.4",
    "rife-v4.5",
    "rife-v4.6",
    "rife-v4.7",
    "rife-v4.8",
    "rife-v4.9",
    "rife-v4",
    "rife",
)

_KNOWN_VERSIONS: dict[str, dict[str, object]] = {
    model_dir.removeprefix("rife-"): {
        "dir": model_dir,
        "lite": "-lite" in model_dir,
        "uhd_safe": True,
    }
    for model_dir in _MODEL_DIR_NAMES
}


@dataclass(frozen=True)
class RifeJob:
    input_dir: Path
    output_dir: Path
    version: str = "v4.22-lite"
    multiplier: int = 2                 # RIFE supports integer multipliers via -s
    tile_size: int = 0                  # 0 = no tiling (RIFE rarely needs it)
    gpu_id: int = 0
    fp16: bool = True
    uhd: bool = False                   # 4K+ source flag
    frame_format: str = "png"
    threads: str = DEFAULT_RIFE_THREADS


class RifeAdapter(NcnnVulkanAdapter):
    tool_id = "rife-ncnn-vulkan"
    bin_name = BIN_RIFE
    tools_subdir = "rife-ncnn-vulkan"

    # RIFE is mostly memory-light; tile size 0 = the binary picks per-frame.
    default_tile_size = 0
    tile_size_floor = 0                 # we accept 0 as "no tiling"
    supports_format_flag = True

    def _detect_version(self) -> str:
        pin = PINNED_VERSIONS.get(self.tool_id)
        v = super()._detect_version()
        if v != "unknown":
            return v
        pe_blob = "\n".join(pe_version_resource_strings(self.path))
        try:
            with self.path.open("rb") as f:
                sample = f.read(48_971_520)  # 45 MiB; enough for PE .rdata + strings
        except OSError:
            sample = b""
        latin = sample.decode("latin-1", errors="ignore")
        tag = first_yyyymmdd_tag(pe_blob, latin, prefer=pin)
        if tag:
            return tag
        if pin:
            pb = pin.encode("ascii")
            if pb in sample or pin.encode("utf-16le") in sample:
                return pin
        if pin and self._looks_like_bundled_rife():
            # Official layout: pinned fetch_tools artefact slot; binaries often lack
            # any scrapeable banner or PE string carrying the upstream tag verbatim.
            return pin
        return "unknown"

    def _looks_like_bundled_rife(self) -> bool:
        try:
            bundled = (tools_dir() / self.tools_subdir / self.bin_name).resolve()
            return self.path.resolve() == bundled
        except OSError:
            return False

    # --------------------------------------------------------- model dir

    def resolve_model_dir(self, version: str) -> Path:
        bin_dir = self.path.parent
        info = _KNOWN_VERSIONS.get(version)
        sub = info["dir"] if info else f"rife-{version}"
        candidate = bin_dir / sub
        if candidate.is_dir():
            return candidate.resolve()
        return super().resolve_model_dir(version)

    # --------------------------------------------------------- argv

    def build_rife_argv(self, job: RifeJob) -> list[str | Path]:
        model_dir = self.resolve_model_dir(job.version)
        argv: list[str | Path] = [
            self.path,
            "-i", str(job.input_dir),
            "-o", str(job.output_dir),
            "-m", str(model_dir),
            "-g", str(job.gpu_id),
            "-j", job.threads,
        ]
        # The TNTwise binary defaults to -s 2 (frame doubling); pass -s
        # explicitly so multipliers > 2 don't silently degrade to 2x. Stage 06
        # relies on the output count being exactly L*M, which only holds when
        # the multiplier reaches the binary.
        if job.multiplier > 1:
            argv += ["-s", str(job.multiplier)]
        if job.tile_size > 0:
            argv += ["-t", str(job.tile_size)]
        if job.uhd:
            argv += ["-u"]
        if self.supports_format_flag:
            argv += ["-f", job.frame_format]
        return argv

    # --------------------------------------------------------- planner help

    @staticmethod
    def validate_version(version: str) -> list[str]:
        warnings: list[str] = []
        info = _KNOWN_VERSIONS.get(version)
        if info is None:
            warnings.append(
                f"RIFE version {version!r} not in our catalog; behavior may differ."
            )
            return warnings
        if info["lite"]:
            warnings.append(
                f"RIFE {version} is a lite variant — faster but lower quality. "
                f"Prefer v4.22/v4.24-class non-lite models unless throughput is a hard constraint."
            )
        return warnings


# ---------------------------------------------------------- scene-cut splicing


@dataclass(frozen=True)
class FrameRun:
    """One contiguous run of input frames with no scene cut inside.

    `start_idx` and `end_idx` are 0-based, inclusive both ends (Python ranges
    are exclusive on the end, but cuts are conceptually "between frame N and
    N+1" so inclusive math reads more naturally).
    """
    start_idx: int
    end_idx: int

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx + 1


def split_by_scene_cuts(total_frames: int, scene_cuts: Iterable[int]) -> list[FrameRun]:
    """Split a frame range into runs separated by scene cuts.

    A scene cut at index N means frame N is the first frame of a new scene —
    so the previous run ends at N-1 and the next run starts at N.

    >>> split_by_scene_cuts(10, [3, 7])
    [FrameRun(start_idx=0, end_idx=2), FrameRun(start_idx=3, end_idx=6), FrameRun(start_idx=7, end_idx=9)]
    """
    if total_frames <= 0:
        return []
    cuts = sorted({c for c in scene_cuts if 0 < c < total_frames})
    runs: list[FrameRun] = []
    prev = 0
    for c in cuts:
        runs.append(FrameRun(start_idx=prev, end_idx=c - 1))
        prev = c
    runs.append(FrameRun(start_idx=prev, end_idx=total_frames - 1))
    return runs


def expected_output_count(total_input_frames: int, multiplier: int, scene_cut_count: int = 0) -> int:
    """The frame count a consolidated single-pass RIFE invocation should emit.

    The stage invokes RIFE once per batch on all `L` input frames and gets back
    `L*M` output frames numbered ``1..L*M``. Scene cuts no longer change the
    count: the post-process step *overwrites* the (M-1) morphed frames at each
    cut with hardlinks to the boundary frame, leaving the total length intact.

    `scene_cut_count` is accepted for backward compatibility with older callers
    but is ignored — left in the signature so a stale plan-dict doesn't crash.
    """
    del scene_cut_count  # retained for signature compatibility; intentionally unused.
    if total_input_frames <= 0 or multiplier <= 0:
        return 0
    return total_input_frames * multiplier


def local_cuts_from_global(
    global_cuts: Iterable[int],
    *,
    batch_offset: int | None = None,
    rife_input_base: int | None = None,
    in_count: int,
) -> list[int]:
    """Translate global source-frame cut indices into batch-local input frame indices.

    `global_cuts` are 0-based source-frame indices (the form `ctx.scene_cuts`
    holds). ``rife_input_base`` (or legacy ``batch_offset``) is the global index
    of local input frame 1 — for batched RIFE with overlap context this is one
    frame *before* the batch's first content frame. ``in_count`` is the number
    of RIFE input frames (including overlap).

    A cut at global index `g` is rewritten to local 1-based input index
    ``g - base + 1``. Cuts at local 1 (no prior input in this RIFE run) or
    past ``in_count`` are dropped.

    The result is sorted and deduped.
    """
    base = rife_input_base if rife_input_base is not None else batch_offset
    if base is None:
        base = 0
    if in_count <= 0:
        return []
    seen: set[int] = set()
    for g in global_cuts:
        local = int(g) - int(base) + 1
        if 2 <= local <= in_count:
            seen.add(local)
    return sorted(seen)


def morphed_output_range(local_cut: int, multiplier: int) -> tuple[int, int]:
    """Output frames RIFE morphs across a scene cut at local input frame `local_cut`.

    Output numbering is 1-based. Input frame `i` lands at output index
    ``(i-1)*M + 1``; the M-1 frames after it (``(i-1)*M + 2 .. i*M``) are
    interpolations toward input `i+1`. For a cut at input frame `c` (first
    frame of the new scene), the morphs we want to overwrite are the
    interpolations between input `c-1` (at output ``(c-2)*M + 1``) and input
    `c` (at output ``(c-1)*M + 1``) — that's outputs ``(c-2)*M + 2 .. (c-1)*M``.

    Returns (first_morphed_index, count). `count` is `M - 1`; both are 0
    when `multiplier <= 1` or `local_cut < 2`.
    """
    if multiplier <= 1 or local_cut < 2:
        return (0, 0)
    first = (local_cut - 2) * multiplier + 2
    return (first, multiplier - 1)


def replace_with_boundary_dup(
    out_dir: Path,
    *,
    boundary_idx: int,
    start_idx: int,
    count: int,
    format: str = "png",
) -> list[Path]:
    """Overwrite ``count`` consecutive output frames with hardlinks to the boundary frame.

    Frame numbering is 1-based and matches RIFE's output: file ``N`` is
    ``out_dir/{N:08d}.{format}``. The frame at ``boundary_idx`` is the source
    we're duplicating from (typically the preserved RIFE output of the input
    frame just before the scene cut). Frames at indices
    ``start_idx .. start_idx + count - 1`` are unlinked (if they exist) and
    re-created as hardlinks to the boundary frame; cross-volume / non-link
    filesystems fall through to a copy, mirroring the rest of the stage.

    Returns the list of overwritten paths.
    """
    if count <= 0:
        return []
    src = out_dir / f"{boundary_idx:08d}.{format}"
    if not src.is_file():
        raise FileNotFoundError(
            f"replace_with_boundary_dup: boundary frame missing: {src}",
        )
    overwritten: list[Path] = []
    for i in range(count):
        dst = out_dir / f"{start_idx + i:08d}.{format}"
        if dst.exists():
            try:
                dst.unlink()
            except OSError:
                # Fall through to copy2 below, which will overwrite.
                pass
        try:
            dst.hardlink_to(src)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
        overwritten.append(dst)
    return overwritten


def stage_run_for_rife(run: FrameRun, src_dir: Path, dest_dir: Path, *, format: str = "png") -> int:
    """Copy frames belonging to ``run`` from ``src_dir`` into ``dest_dir`` with a
    contiguous 1-based numbering RIFE expects.

    Returns the number of frames staged.

    .. deprecated::
        The consolidated stage 06 invokes RIFE on the entire batch at once and
        no longer splits inputs by scene cuts; this helper is kept only for
        the unit tests that pin the legacy splicing math, and will be removed
        in a follow-up cleanup once the new path has soaked.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(run.start_idx, run.end_idx + 1):
        src = src_dir / f"{i + 1:08d}.{format}"
        if not src.is_file():
            raise FileNotFoundError(f"missing input frame for RIFE staging: {src}")
        dst = dest_dir / f"{n + 1:08d}.{format}"
        # Hardlink when on same volume to avoid double-disk usage; fall back to copy.
        try:
            dst.hardlink_to(src)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dst)
        n += 1
    return n


def boundary_duplicates(*, last_src_frame_path: Path, count: int, start_index: int, dest_dir: Path, format: str = "png") -> list[Path]:
    """Insert ``count`` duplicates of ``last_src_frame_path`` into ``dest_dir``,
    numbered starting at ``start_index`` (1-based). Returns the created paths.

    .. deprecated::
        Stage 06 now overwrites morphed frames in place via
        :func:`replace_with_boundary_dup` instead of inserting extra duplicate
        frames between per-run RIFE outputs. Retained for the unit tests that
        pin the legacy splicing math; will be removed in a follow-up cleanup.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    for i in range(count):
        dst = dest_dir / f"{start_index + i:08d}.{format}"
        try:
            dst.hardlink_to(last_src_frame_path)
        except (OSError, NotImplementedError):
            shutil.copy2(last_src_frame_path, dst)
        created.append(dst)
    return created
