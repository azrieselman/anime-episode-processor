"""Anime4KCPP 2.5.x legacy CLI (`Anime4KCPP_CLI.exe`).

v2.5 accepts an input *directory* and output *directory*, which avoids per-frame
`-i`/`-o` pairs and keeps GPU init/model load inside a single process.

CLI mapping (see `Anime4KCPP_CLI -?` — **not** `-h`, which is OpenCL *platform* id):

* Always: ``-w`` (ACNet), ``-H`` (HDN), ``-L 1`` (HDN level), per product policy.
* ``scale`` → ``-z`` (zoom factor)
* Settings/hardware: ``anime4k_threads`` → ``-t``, ``anime4k_prefer_cuda`` → try
  ``-q -M cuda`` first, then ``-q -M opencl`` on failure; ``-d`` is GPU id (0).

Preset ``model`` / ``denoise`` / ``tta`` are validated like other Anime4K engines but
do not change the 2.5 CLI surface (single ACNet+HDN path).
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from aep.adapters.anime4kcpp_models import (
    DEFAULT_ANIME4K_MODEL,
    KNOWN_ANIME4K_MODELS,
)
from aep.adapters.base import env_with_tool_dirs
from aep.adapters.ncnn_base import NcnnRunResult, NcnnVulkanAdapter
from aep.constants import BIN_ANIME4KCPP_LEGACY
from aep.util.proc import ProcError, run_capture, run_streaming

log = logging.getLogger(__name__)

_GPGPU = Literal["cuda", "opencl"]
_VERSION_RE = re.compile(r"core\s+version:\s*(\d+\.\d+\.\d+)", re.IGNORECASE)


class Anime4kcppLegacyAdapter(NcnnVulkanAdapter):
    tool_id = "anime4k-legacy"
    bin_name = BIN_ANIME4KCPP_LEGACY
    tools_subdir = "anime4k-legacy"

    default_tile_size = 256
    tile_size_floor = 64
    supports_format_flag = False

    def _detect_version(self) -> str:
        try:
            result = run_capture(
                [self.path, "-V"],
                env=env_with_tool_dirs(),
                check=False,
                timeout=10.0,
            )
            blob = f"{result.stdout}\n{result.stderr}"
        except Exception:
            return "unknown"
        m = _VERSION_RE.search(blob)
        return m.group(1) if m else "unknown"

    def build_anime4k_legacy_argv(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        scale: int,
        threads: int,
        gpgpu: _GPGPU,
        gpu_id: int = 0,
        platform_id: int = 0,
    ) -> list[str | Path]:
        argv: list[str | Path] = [
            self.path,
            "-i",
            str(input_dir.resolve()),
            "-o",
            str(output_dir.resolve()),
            "-w",
            "-H",
            "-L",
            "1",
            "-z",
            str(float(scale)),
            "-t",
            str(threads),
            "-q",
            "-M",
            gpgpu,
            "-d",
            str(gpu_id),
        ]
        if platform_id != 0:
            argv.extend(["-h", str(platform_id)])
        return argv

    def run_frame_sequence(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        model_id: str,
        scale: int,
        prefer_cuda: bool,
        frame_format: str = "png",
        threads: int = 4,
        on_progress: Callable[[str], None] | None = None,
        should_interrupt: Callable[[], str | None] | None = None,
    ) -> NcnnRunResult:
        _ = model_id  # validated upstream; 2.5 CLI has no per-model selector
        frames = sorted(input_dir.glob(f"*.{frame_format}"))
        if not frames:
            raise ValueError(f"no input frames found in {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        attempts = 0
        warnings: list[str] = []
        rationale: list[str] = []
        env = env_with_tool_dirs()
        gpu_id = 0
        platform_id = 0

        def _run_one(gpgpu: _GPGPU) -> None:
            nonlocal attempts
            attempts += 1
            argv = self.build_anime4k_legacy_argv(
                input_dir=input_dir,
                output_dir=output_dir,
                scale=scale,
                threads=threads,
                gpgpu=gpgpu,
                gpu_id=gpu_id,
                platform_id=platform_id,
            )
            summary = (
                f"{argv[0]} -i <dir> -o <dir> -w -H -L 1 -z {float(scale)} "
                f"-t {threads} -q -M {gpgpu} -d {gpu_id}"
            )
            log.info("exec: %s", summary)
            for stream, line in run_streaming(
                argv,
                cwd=None,
                env=env,
                should_interrupt=should_interrupt,
            ):
                if stream == "stderr" and on_progress:
                    on_progress(line)

        primary: _GPGPU = "cuda" if prefer_cuda else "opencl"
        used: _GPGPU
        try:
            _run_one(primary)
            used = primary
        except ProcError as exc:
            stderr = (exc.result.stderr or "").lower()
            if primary == "cuda" and ("cuda" in stderr or "nvidia" in stderr):
                warnings.append("anime4k-legacy: CUDA failed; retrying with OpenCL.")
                _run_one("opencl")
                used = "opencl"
            else:
                raise
        rationale.append(f"anime4k-legacy gpgpu={used} directory mode")

        done = len(frames)
        if on_progress:
            on_progress(f"{done}/{done}")

        return NcnnRunResult(
            output_dir=output_dir,
            frames_in=done,
            frames_out=done,
            tile_size_used=0,
            duration_s=time.monotonic() - t0,
            attempts=attempts,
            rationale=rationale,
            warnings=warnings,
        )

    @staticmethod
    def validate_combination(model_id: str, scale: int, _denoise: int) -> list[str]:
        warnings: list[str] = []
        if model_id not in KNOWN_ANIME4K_MODELS:
            warnings.append(
                f"Anime4K model {model_id!r} is not in our catalog; "
                f"see Anime4KCPP wiki Model list (default: {DEFAULT_ANIME4K_MODEL})"
            )
        if scale < 1 or scale > 4:
            warnings.append("Anime4K scale should be within 1..4 for predictable output quality.")
        return warnings
