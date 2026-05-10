"""Anime4KCPP VapourSynth adapter.

This backend is separate from `ac_cli` and runs Anime4K through the
VapourSynth plugin path:
    source frames -> temporary lossless clip -> vspipe (ACUpscale) -> frames
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from aep.adapters.anime4kcpp_models import DEFAULT_ANIME4K_MODEL, KNOWN_ANIME4K_MODELS
from aep.adapters.base import ToolAdapter, env_with_tool_dirs
from aep.adapters.ffmpeg import FFmpegAdapter
from aep.adapters.ncnn_base import NcnnRunResult
from aep.util.paths import tools_dir
from aep.util.proc import ProcError, ProcResult, run_capture

log = logging.getLogger(__name__)


class VapourSynthAdapter(ToolAdapter):
    tool_id = "vapoursynth-vspipe"
    bin_name = "vspipe.exe"
    tools_subdir = "vapoursynth"
    version_re = re.compile(r"Core\s+R(\d+)")
    _configured = False

    def _resolve(self) -> Path:
        if self._override_dir:
            return super()._resolve()
        wheel = tools_dir() / self.tools_subdir / "vapoursynth-74-cp312-abi3-win_amd64.whl"
        if wheel.is_file():
            extracted = wheel.parent / self.bin_name
            if not extracted.is_file():
                self._extract_runtime_from_wheel(wheel, wheel.parent)
            self._ensure_python_config(wheel.parent)
        return super()._resolve()

    def _ensure_python_config(self, runtime_dir: Path) -> None:
        if VapourSynthAdapter._configured:
            return
        python_cmd = self._python_config_interpreter()
        if python_cmd is None:
            VapourSynthAdapter._configured = True
            return
        env = self.runtime_env(extra_dirs=[runtime_dir])
        run_capture(
            [
                python_cmd,
                "-c",
                "import vapoursynth as vs; vs.vapoursynth_config()",
            ],
            env=env,
            check=False,
            timeout=30.0,
        )
        VapourSynthAdapter._configured = True

    def _python_config_interpreter(self) -> str | None:
        # In PyInstaller GUI builds, sys.executable points to aep-gui.exe.
        # Running "aep-gui.exe -c ..." recursively launches more GUI windows.
        if getattr(sys, "frozen", False):
            sibling_python = Path(sys.executable).with_name("python.exe")
            if sibling_python.is_file():
                return str(sibling_python)
            log.warning(
                "vapoursynth config bootstrap skipped: no python.exe beside %s",
                sys.executable,
            )
            return None
        return sys.executable

    def runtime_env(self, *, extra_dirs: list[Path] | None = None) -> dict[str, str]:
        env = env_with_tool_dirs(extra_dirs=extra_dirs)
        # vspipe imports the `vapoursynth` Python module. The module lives under
        # tools/vapoursynth, so Python needs the parent tools dir on PYTHONPATH.
        tools_parent = str(tools_dir())
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            tools_parent
            if not existing
            else os.pathsep.join([tools_parent, existing])
        )
        return env

    def _extract_runtime_from_wheel(self, wheel_path: Path, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(wheel_path) as whl:
            for member in whl.namelist():
                if not member.startswith("vapoursynth/") or member.endswith("/"):
                    continue
                rel = member.split("/", 1)[1]
                out = dest_dir / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(whl.read(member))

    def _detect_version(self) -> str:
        env = self.runtime_env()

        def _parse_core_or_vspipe(blob: str) -> str | None:
            m = self.version_re.search(blob)
            if m:
                return f"R{m.group(1)}"
            m = re.search(r"VSPipe\s+R(\d+)", blob, re.IGNORECASE)
            return f"R{m.group(1)}" if m else None

        result = run_capture(
            [self.path, "--version"],
            check=False,
            timeout=10.0,
            env=env,
        )
        blob = f"{result.stdout}\n{result.stderr}"
        parsed = _parse_core_or_vspipe(blob)
        if parsed:
            return parsed
        # --version initializes a core instance; bundled layouts can fail until
        # PYTHONPATH / plugin paths match runtime_env(). `-h` is static text and
        # still exposes "VSPipe Rxx".
        result = run_capture([self.path, "-h"], check=False, timeout=10.0, env=env)
        blob = f"{result.stdout}\n{result.stderr}"
        parsed = _parse_core_or_vspipe(blob)
        return parsed if parsed else "unknown"


class Ffms2VapourSynthAdapter(ToolAdapter):
    tool_id = "ffms2-vapoursynth"
    bin_name = "ffms2.dll"
    tools_subdir = "ffms2-vs"

    def _detect_version(self) -> str:
        # FFMS2 plugin DLL does not expose a stable CLI version probe.
        return "2.40"


class Anime4kcppVsAdapter:
    """Runs Anime4KCPP via VapourSynth plugin + vspipe."""

    tool_id = "anime4kcpp-vs"

    def __init__(
        self,
        *,
        vspipe_override_dir: Path | str | None = None,
        plugin_override_dir: Path | str | None = None,
    ) -> None:
        self._vspipe = VapourSynthAdapter(override_dir=vspipe_override_dir)
        self._plugin_override = Path(plugin_override_dir) if plugin_override_dir else None
        self._ffmpeg = FFmpegAdapter()

    @property
    def version(self) -> str:
        return self._vspipe.version

    def _plugin_path(self) -> Path:
        if self._plugin_override:
            candidate = self._plugin_override / "ac_filter_avs_vs.dll"
            if candidate.is_file():
                return candidate.resolve()
            raise FileNotFoundError(f"Anime4KCPP VS plugin not found in {self._plugin_override}")
        bundled = tools_dir() / "anime4kcpp-filter-vs" / "ac_filter_avs_vs.dll"
        if bundled.is_file():
            return bundled.resolve()
        raise FileNotFoundError(f"Anime4KCPP VS plugin not found at {bundled}")

    def _ffms2_plugin_path(self) -> Path:
        bundled = tools_dir() / "ffms2-vs" / "ffms2.dll"
        if bundled.is_file():
            return bundled.resolve()
        raise FileNotFoundError(
            "FFMS2 VapourSynth source plugin missing at "
            f"{bundled}. Re-run scripts/fetch_tools.py --tool ffms2-vapoursynth"
        )

    def run_frame_sequence(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        model_id: str,
        scale: int,
        prefer_cuda: bool,
        frame_format: str = "png",
        gpu_id: int = 0,
        on_progress=None,
    ) -> NcnnRunResult:
        frames = sorted(input_dir.glob(f"*.{frame_format}"))
        if not frames:
            raise ValueError(f"no input frames found in {input_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        attempts = 0
        warnings: list[str] = []
        rationale: list[str] = []
        env = self._vspipe.runtime_env()
        plugin = self._plugin_path()
        ffms2_plugin = self._ffms2_plugin_path()

        temp_root = output_dir.parent / "_tmp_anime4k_vs"
        temp_root.mkdir(parents=True, exist_ok=True)
        # Keep heavy VS intermediates on the same filesystem as stage output,
        # so ramdisk-backed pipelines avoid writing these files to system temp.
        with tempfile.TemporaryDirectory(prefix="aep-anime4k-vs-", dir=str(temp_root)) as tmp:
            tmp_dir = Path(tmp)
            src_video = tmp_dir / "input.mkv"
            script = tmp_dir / "anime4k.vpy"
            ffmpeg_path = self._ffmpeg.path
            processors = ["cuda", "opencl", "cpu"] if prefer_cuda else ["opencl", "cuda", "cpu"]

            encode_cmd = [
                ffmpeg_path, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-framerate", "24000/1001",
                "-i", str(input_dir / f"%08d.{frame_format}"),
                "-map", "0:v:0",
                "-c:v", "ffv1",
                "-pix_fmt", "yuv420p",
                str(src_video),
            ]
            run_capture(encode_cmd, env=env, check=True, timeout=600.0)

            selected = processors[-1]
            last_error: ProcError | None = None
            for proc_name in processors:
                attempts += 1
                script.write_text(
                    "\n".join([
                        "import vapoursynth as vs",
                        "core = vs.core",
                        f'core.std.LoadPlugin(path={plugin.as_posix()!r})',
                        f'core.std.LoadPlugin(path={ffms2_plugin.as_posix()!r})',
                        f"src = core.ffms2.Source(source={src_video.as_posix()!r})",
                        (
                            "src = core.anime4kcpp.ACUpscale("
                            f"src, factor={float(scale)}, processor={proc_name!r}, "
                            f"device={int(gpu_id)}, model={model_id!r})"
                        ),
                        "src.set_output()",
                        "",
                    ]),
                    encoding="utf-8",
                )
                try:
                    self._pipe_vspipe_to_frames(
                        script_path=script,
                        output_dir=output_dir,
                        frame_format=frame_format,
                        ffmpeg_path=ffmpeg_path,
                        env=env,
                    )
                    selected = proc_name
                    break
                except ProcError as exc:
                    last_error = exc
                    warnings.append(f"anime4k-vs: processor {proc_name} failed, trying fallback.")
            else:
                assert last_error is not None
                raise last_error

            rationale.append(f"anime4k-vs processor={selected}")
        if on_progress:
            on_progress(f"{len(frames)}/{len(frames)}")
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

    def _pipe_vspipe_to_frames(
        self,
        *,
        script_path: Path,
        output_dir: Path,
        frame_format: str,
        ffmpeg_path: Path,
        env: dict[str, str],
    ) -> None:
        out_pattern = output_dir / f"%08d.{frame_format}"
        vspipe_cmd = [str(self._vspipe.path), "-c", "y4m", str(script_path), "-"]
        ffmpeg_cmd = [
            str(ffmpeg_path),
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-start_number",
            "1",
            str(out_pattern),
        ]
        creationflags = 0x08000000 if sys.platform == "win32" else 0
        p1 = subprocess.Popen(
            vspipe_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )
        p2 = subprocess.Popen(
            ffmpeg_cmd,
            stdin=p1.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
            creationflags=creationflags,
        )
        assert p1.stdout is not None
        p1.stdout.close()
        _, ffmpeg_stderr = p2.communicate()
        assert p1.stderr is not None
        vspipe_stderr = p1.stderr.read()
        vspipe_stdout = b""
        ffmpeg_stdout = b""
        p1.wait()
        if p1.returncode != 0:
            raise ProcError(
                ProcResult(
                    cmd=vspipe_cmd,
                    returncode=p1.returncode or 1,
                    stdout=(vspipe_stdout or b"").decode("utf-8", errors="replace"),
                    stderr=(vspipe_stderr or b"").decode("utf-8", errors="replace"),
                ),
            )
        if p2.returncode != 0:
            raise ProcError(
                ProcResult(
                    cmd=ffmpeg_cmd,
                    returncode=p2.returncode or 1,
                    stdout=(ffmpeg_stdout or b"").decode("utf-8", errors="replace"),
                    stderr=(ffmpeg_stderr or b"").decode("utf-8", errors="replace"),
                ),
            )

    @staticmethod
    def validate_combination(model_id: str, scale: int, denoise: int) -> list[str]:
        warnings: list[str] = []
        if model_id not in KNOWN_ANIME4K_MODELS:
            warnings.append(
                f"Anime4K model {model_id!r} is not in our catalog; "
                f"see Anime4KCPP wiki Model list (default: {DEFAULT_ANIME4K_MODEL})"
            )
        if scale < 1 or scale > 4:
            warnings.append("Anime4K scale should be within 1..4 for predictable output quality.")
        if denoise not in (-1, 0, 1, 2, 3):
            warnings.append("Anime4K denoise should stay in -1..3 to match preset compatibility.")
        return warnings
