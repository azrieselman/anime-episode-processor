"""Anime4KCPP adapter.

Anime4KCPP v3 uses `ac_cli` and supports multiple processors (cpu/opencl/cuda).
We default to ACNet + HDN mode and select CUDA when available, otherwise OpenCL.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.base import env_with_tool_dirs
from aep.adapters.ncnn_base import NcnnRunResult, NcnnVulkanAdapter
from aep.constants import BIN_ANIME4KCPP
from aep.util.proc import ProcError, run_capture

_KNOWN_MODELS: frozenset[str] = frozenset({
    "acnet",
    "acnet-gan",
    "acnet-hdn",
    "acnet-hdn-gan",
})


@dataclass(frozen=True)
class Anime4kcppJob:
    input_path: Path
    output_path: Path
    model_id: str = "acnet-hdn-gan"
    scale: int = 2
    prefer_cuda: bool = True
    tile_size: int = 256
    gpu_id: int = 0
    tta: bool = False
    frame_format: str = "png"
    threads: int = 4


class Anime4kcppAdapter(NcnnVulkanAdapter):
    tool_id = "anime4kcpp"
    bin_name = BIN_ANIME4KCPP
    tools_subdir = "anime4kcpp"

    default_tile_size = 256
    tile_size_floor = 64
    supports_format_flag = False
    version_re = re.compile(r"(\d+\.\d+\.\d+)")

    def __init__(self, *, override_dir: Path | str | None = None) -> None:
        super().__init__(override_dir=override_dir)
        self._preferred_processor: str | None = None

    def _detect_version(self) -> str:
        try:
            result = run_capture([self.path, "--version"], check=False, timeout=10.0)
            blob = f"{result.stdout}\n{result.stderr}"
        except Exception:
            return "unknown"
        m = self.version_re.search(blob)
        return m.group(1) if m else "unknown"

    def _detect_preferred_processor(self, *, prefer_cuda: bool) -> str:
        if self._preferred_processor is not None:
            return self._preferred_processor
        try:
            proc_info = run_capture([self.path, "--lp"], check=False, timeout=10.0)
            proc_blob = f"{proc_info.stdout}\n{proc_info.stderr}".lower()
        except Exception:
            self._preferred_processor = "cpu"
            return self._preferred_processor

        has_cuda = "cuda" in proc_blob
        has_opencl = "opencl" in proc_blob
        if prefer_cuda and has_cuda:
            self._preferred_processor = "cuda"
        elif has_opencl:
            self._preferred_processor = "opencl"
        elif has_cuda:
            self._preferred_processor = "cuda"
        else:
            self._preferred_processor = "cpu"
        return self._preferred_processor

    def build_anime4kcpp_argv(
        self,
        job: Anime4kcppJob,
        *,
        tile_size_override: int | None = None,
    ) -> list[str | Path]:
        # Anime4KCPP does not expose NCNN-like tile controls. We keep the same
        # method signature as other adapters so stage 05 can share dispatch code.
        _unused_tile = tile_size_override if tile_size_override is not None else job.tile_size
        _ = _unused_tile
        preferred_processor = self._detect_preferred_processor(prefer_cuda=job.prefer_cuda)
        argv: list[str | Path] = [
            self.path,
            "-i", str(job.input_path),
            "-o", str(job.output_path),
            "-m", str(job.model_id),
            "-p", preferred_processor,
            "-d", str(job.gpu_id),
            "-f", str(job.scale),
        ]
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
        on_progress=None,
    ) -> NcnnRunResult:
        # NOTE: Anime4KCPP ac_cli.exe does NOT support directory input/output.
        # Keep this stage as one-process-per-frame (parallelized via workers).
        frames = sorted(input_dir.glob(f"*.{frame_format}"))
        if not frames:
            raise ValueError(f"no input frames found in {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.monotonic()
        attempts = 0
        warnings: list[str] = []
        rationale: list[str] = []
        env = env_with_tool_dirs()
        processor = self._detect_preferred_processor(prefer_cuda=prefer_cuda)
        rationale.append(f"anime4k processor={processor}")

        def _run_one(frame: Path, proc: str) -> None:
            out = output_dir / frame.name
            job = Anime4kcppJob(
                input_path=frame,
                output_path=out,
                model_id=model_id,
                scale=scale,
                prefer_cuda=prefer_cuda,
                threads=threads,
            )
            self._preferred_processor = proc
            argv = self.build_anime4kcpp_argv(job)
            run_capture(argv, env=env, timeout=120.0, check=True)

        start_idx = 0
        if frames and processor == "cuda":
            # Probe a single frame first so CUDA capability failures are detected
            # deterministically before we fan out worker processes.
            attempts += 1
            try:
                _run_one(frames[0], "cuda")
            except ProcError as exc:
                stderr = (exc.result.stderr or "").lower()
                if processor == "cuda" and ("cuda" in stderr or "nvidia" in stderr):
                    processor = "opencl"
                    warnings.append("anime4k: CUDA failed; retrying with OpenCL.")
                    attempts += 1
                    _run_one(frames[0], "opencl")
                else:
                    raise
            if on_progress:
                on_progress(f"1/{len(frames)}")
            start_idx = 1

        remaining = frames[start_idx:]
        done = start_idx
        if remaining:
            if threads <= 1:
                for frame in remaining:
                    attempts += 1
                    _run_one(frame, processor)
                    done += 1
                    if on_progress:
                        on_progress(f"{done}/{len(frames)}")
            else:
                with ThreadPoolExecutor(max_workers=threads) as pool:
                    futures = [pool.submit(_run_one, frame, processor) for frame in remaining]
                    attempts += len(futures)
                    for fut in as_completed(futures):
                        fut.result()
                        done += 1
                        if on_progress:
                            on_progress(f"{done}/{len(frames)}")

        return NcnnRunResult(
            output_dir=output_dir,
            frames_in=len(frames),
            frames_out=len(frames),
            tile_size_used=0,
            duration_s=time.monotonic() - t0,
            attempts=attempts,
            rationale=rationale,
            warnings=warnings,
        )

    @staticmethod
    def validate_combination(model_id: str, scale: int, denoise: int) -> list[str]:
        warnings: list[str] = []
        if model_id not in _KNOWN_MODELS:
            warnings.append(
                f"Anime4K model {model_id!r} is not in our catalog; "
                f"known models: {sorted(_KNOWN_MODELS)}"
            )
        if "acnet" not in model_id or "hdn" not in model_id:
            warnings.append(
                "Anime4K balanced default expects ACNet + HDN model (for example acnet-hdn-gan)."
            )
        if scale < 1 or scale > 4:
            warnings.append("Anime4K scale should be within 1..4 for predictable output quality.")
        return warnings
