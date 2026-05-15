"""Anime4KCPP adapter.

Anime4KCPP v3.2+ uses `ac_cli` with multi-file `-i`/`-o`, `-t` for threading, and
multiple processors (cpu/opencl/cuda). We default to ACNet F8B8 + mild denoise
(`acnet-f8b8-hdn`) and select CUDA when available, otherwise OpenCL.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from aep.adapters.anime4kcpp_models import (
    DEFAULT_ANIME4K_MODEL,
    KNOWN_ANIME4K_MODELS,
)
from aep.adapters.base import env_with_tool_dirs
from aep.adapters.ncnn_base import NcnnRunResult, NcnnVulkanAdapter
from aep.constants import BIN_ANIME4KCPP
from aep.util.proc import ProcError, run_capture, run_streaming

log = logging.getLogger(__name__)

# Python's subprocess invokes CreateProcessW directly (no cmd.exe shell), so the
# real Windows command-line ceiling is ~32,767 UTF-16 chars, not the 8,191-char
# cmd.exe limit a previous version of this file assumed. We hold a comfortable
# margin for list2cmdline quoting overhead and the fixed flags (`-m`, `-p`, etc.).
_MAX_ARGV_CHARS = 30000


@dataclass(frozen=True)
class Anime4kcppJob:
    input_path: Path
    output_path: Path
    model_id: str = DEFAULT_ANIME4K_MODEL
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

    def _argv_char_len(self, argv: Sequence[str | Path]) -> int:
        return sum(len(str(a)) for a in argv) + max(0, len(argv) - 1)

    def build_anime4kcpp_argv_batch(
        self,
        inputs: Sequence[Path],
        outputs: Sequence[Path],
        *,
        model_id: str,
        scale: int,
        processor: str,
        gpu_id: int,
        threads: int,
    ) -> list[str | Path]:
        if len(inputs) != len(outputs) or not inputs:
            raise ValueError("inputs and outputs must be same non-empty length")
        argv: list[str | Path] = [
            self.path,
            "-i",
            *[str(p) for p in inputs],
            "-o",
            *[str(p) for p in outputs],
            "-m",
            model_id,
            "-p",
            processor,
            "-d",
            str(gpu_id),
            "-f",
            str(float(scale)),
            "-t",
            str(threads),
        ]
        return argv

    def build_anime4kcpp_argv(
        self,
        job: Anime4kcppJob,
        *,
        tile_size_override: int | None = None,
        processor_override: str | None = None,
    ) -> list[str | Path]:
        # Anime4KCPP does not expose NCNN-like tile controls. We keep the same
        # method signature as other adapters so stage 05 can share dispatch code.
        _unused_tile = tile_size_override if tile_size_override is not None else job.tile_size
        _ = _unused_tile
        processor = processor_override or self._detect_preferred_processor(
            prefer_cuda=job.prefer_cuda,
        )
        return self.build_anime4kcpp_argv_batch(
            [job.input_path],
            [job.output_path],
            model_id=job.model_id,
            scale=job.scale,
            processor=processor,
            gpu_id=job.gpu_id,
            threads=job.threads,
        )

    def _chunk_input_output_pairs(
        self,
        pairs: list[tuple[Path, Path]],
        *,
        model_id: str,
        scale: int,
        gpu_id: int,
        threads: int,
        processor: str,
    ) -> list[list[tuple[Path, Path]]]:
        chunks: list[list[tuple[Path, Path]]] = []
        current: list[tuple[Path, Path]] = []
        for pair in pairs:
            trial = current + [pair]
            ins = [p[0] for p in trial]
            outs = [p[1] for p in trial]
            argv = self.build_anime4kcpp_argv_batch(
                ins,
                outs,
                model_id=model_id,
                scale=scale,
                processor=processor,
                gpu_id=gpu_id,
                threads=threads,
            )
            if self._argv_char_len(argv) <= _MAX_ARGV_CHARS or not current:
                current = trial
            else:
                chunks.append(current)
                current = [pair]
        if current:
            chunks.append(current)
        return chunks

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
        frames = sorted(input_dir.glob(f"*.{frame_format}"))
        if not frames:
            raise ValueError(f"no input frames found in {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # ac_cli has no list-file option (`-l` lists devices), so the only way
        # to batch multiple frames into one invocation is to pass them all as
        # positional `-i`/`-o` args. Each batch is bounded by the Windows
        # command-line ceiling, so the shorter the path strings the more
        # frames we can stuff into a single ac_cli.exe call — and per-call
        # CUDA init + model load is the dominant cost on Windows. We anchor
        # the subprocess at `cwd=input_dir` and reference inputs by bare
        # filename (~12 chars) and outputs by a relpath (~30 chars) instead
        # of ~120-char absolute paths. This typically multiplies the frames-
        # per-spawn by ~8–10x.
        rel_out_dir = Path(os.path.relpath(output_dir, start=input_dir))
        pairs: list[tuple[Path, Path]] = [
            (Path(f.name), rel_out_dir / f.name) for f in frames
        ]
        processor = self._detect_preferred_processor(prefer_cuda=prefer_cuda)
        gpu_id = 0
        chunks = self._chunk_input_output_pairs(
            pairs,
            model_id=model_id,
            scale=scale,
            gpu_id=gpu_id,
            threads=threads,
            processor=processor,
        )

        t0 = time.monotonic()
        attempts = 0
        warnings: list[str] = []
        rationale: list[str] = []
        env = env_with_tool_dirs()
        rationale.append(f"anime4k processor={processor}")
        rationale.append(
            f"anime4k batches: {len(chunks)} (avg {len(frames) / max(1, len(chunks)):.0f} frames/batch)"
        )

        def _run_batch(
            batch: list[tuple[Path, Path]],
            proc_name: str,
        ) -> None:
            """Run one ac_cli.exe invocation streaming its stderr.

            Streaming (vs `run_capture`) gives us two things:
              * `should_interrupt` is polled between yielded lines, so cancel
                /pause take effect mid-batch instead of having to wait out a
                long ac_cli call.
              * per-line stderr flows into `on_progress` for the GUI, instead
                of one event per batch boundary.
            """
            nonlocal attempts
            attempts += 1
            self._preferred_processor = proc_name
            ins = [p[0] for p in batch]
            outs = [p[1] for p in batch]
            argv = self.build_anime4kcpp_argv_batch(
                ins,
                outs,
                model_id=model_id,
                scale=scale,
                processor=proc_name,
                gpu_id=gpu_id,
                threads=threads,
            )
            n = len(ins)
            exec_summary = (
                f"{argv[0]} -i <{n} paths> -o <{n} paths> "
                f"-m {model_id} -p {proc_name} -d {gpu_id} -f {float(scale)} -t {threads}"
            )
            log.info("exec: %s", exec_summary)
            for stream, line in run_streaming(
                argv,
                cwd=input_dir,
                env=env,
                should_interrupt=should_interrupt,
            ):
                if stream == "stderr" and on_progress:
                    on_progress(line)

        done_frames = 0
        total = len(frames)

        for chunk_idx, chunk in enumerate(chunks):
            proc = processor
            try:
                _run_batch(chunk, proc)
            except ProcError as exc:
                stderr = (exc.result.stderr or "").lower()
                if (
                    chunk_idx == 0
                    and proc == "cuda"
                    and ("cuda" in stderr or "nvidia" in stderr)
                ):
                    processor = "opencl"
                    self._preferred_processor = "opencl"
                    warnings.append("anime4k: CUDA failed; retrying with OpenCL.")
                    rationale[0] = "anime4k processor=opencl"
                    _run_batch(chunk, "opencl")
                else:
                    raise
            done_frames += len(chunk)
            if on_progress:
                on_progress(f"{done_frames}/{total}")

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
