"""Subprocess helpers.

All external-tool invocations go through `run_capture` or `run_streaming`. They:

* Always log the full command line (required by spec).
* Apply a default timeout (long for media work; callers can override).
* Use `shell=False` always — paths are passed as a list.
* Disable inherited stdin and use UTF-8 decoding.
* Raise typed errors with stdout/stderr attached.

For long-running tools (ffmpeg, NCNN binaries) callers want `run_streaming` to consume
stdout/stderr line-by-line for progress events. Wrappers in `aep.adapters` parse those
lines into structured pipeline events.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

log = logging.getLogger(__name__)
_PROC_STATS: ContextVar[dict[str, float] | None] = ContextVar("_PROC_STATS", default=None)


@dataclass(frozen=True)
class ProcResult:
    cmd: list[str]
    returncode: int
    stdout: str
    stderr: str


class ProcError(RuntimeError):
    def __init__(self, result: ProcResult) -> None:
        super().__init__(
            f"Command failed (exit {result.returncode}): {format_cmd(result.cmd)}"
        )
        self.result = result


class ProcInterrupted(RuntimeError):
    def __init__(self, reason: str, result: ProcResult) -> None:
        super().__init__(f"Command interrupted ({reason}): {format_cmd(result.cmd)}")
        self.reason = reason
        self.result = result


@contextmanager
def proc_stats_scope() -> Iterator[dict[str, float]]:
    stats = {"calls": 0.0, "streaming_calls": 0.0, "capture_calls": 0.0, "wall_s": 0.0}
    token = _PROC_STATS.set(stats)
    try:
        yield stats
    finally:
        _PROC_STATS.reset(token)


def format_cmd(cmd: list[str | os.PathLike[str]]) -> str:
    """Render a command list to a human-readable string with proper quoting."""
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _creation_flags() -> int:
    # Windows: prevent flashing console windows when GUI spawns workers.
    if sys.platform == "win32":
        # CREATE_NO_WINDOW = 0x08000000
        return 0x08000000
    return 0


def run_capture(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = 600.0,
    check: bool = True,
    input_text: str | None = None,
) -> ProcResult:
    """Run a command and capture stdout/stderr fully. Suitable for short tool calls
    (ffprobe, mkvmerge -J, version checks, etc.)."""
    str_cmd = [str(p) for p in cmd]
    log.info("exec: %s", format_cmd(str_cmd))

    started = time.monotonic()
    try:
        completed = subprocess.run(
            str_cmd,
            cwd=str(cwd) if cwd else None,
            env=env,
            input=input_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=_creation_flags(),
            check=False,
        )
    except FileNotFoundError as exc:
        raise ProcError(ProcResult(str_cmd, 127, "", str(exc))) from exc
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise ProcError(ProcResult(str_cmd, -1, stdout, stderr + "\n[timeout]")) from exc

    elapsed = time.monotonic() - started
    stats = _PROC_STATS.get()
    if stats is not None:
        stats["calls"] += 1
        stats["capture_calls"] += 1
        stats["wall_s"] += elapsed
    result = ProcResult(str_cmd, completed.returncode, completed.stdout, completed.stderr)
    if check and completed.returncode != 0:
        log.error("exit=%s stderr=%s", completed.returncode, completed.stderr.strip()[:500])
        raise ProcError(result)
    return result


def run_streaming(
    cmd: list[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    on_stdout: Any = None,
    on_stderr: Any = None,
    should_interrupt: Callable[[], str | None] | None = None,
) -> Iterator[tuple[str, str]]:
    """Run a command and yield (stream, line) tuples as output arrives.

    `stream` is "stdout" or "stderr". The process is waited on after the iterator is
    exhausted; non-zero exit raises ProcError.

    This is the preferred entry point for ffmpeg, NCNN-Vulkan binaries, and anything that
    emits progress over time.
    """
    str_cmd = [str(p) for p in cmd]
    log.info("exec: %s", format_cmd(str_cmd))

    started = time.monotonic()
    proc = subprocess.Popen(
        str_cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=_creation_flags(),
    )

    stdout_buf: list[str] = []
    stderr_buf: list[str] = []

    def _drain(stream: IO[str], tag: str, buf: list[str]) -> None:
        try:
            for line in stream:
                buf.append(line)
                _emit(tag, line.rstrip("\n"))
        finally:
            stream.close()

    yielded: list[tuple[str, str]] = []
    lock = threading.Lock()
    cond = threading.Condition(lock)

    def _emit(tag: str, line: str) -> None:
        with cond:
            yielded.append((tag, line))
            cond.notify_all()

    t_out = threading.Thread(
        target=_drain, args=(proc.stdout, "stdout", stdout_buf), daemon=True,
    )
    t_err = threading.Thread(
        target=_drain, args=(proc.stderr, "stderr", stderr_buf), daemon=True,
    )
    t_out.start()
    t_err.start()

    interrupted_reason: str | None = None
    try:
        while True:
            with cond:
                while not yielded and (t_out.is_alive() or t_err.is_alive()):
                    cond.wait(timeout=0.5)
                items = yielded[:]
                yielded.clear()
            for item in items:
                yield item
            if should_interrupt is not None:
                reason = should_interrupt()
                if reason:
                    interrupted_reason = reason
                    proc.terminate()
                    break
            if not (t_out.is_alive() or t_err.is_alive()):
                break
    finally:
        t_out.join()
        t_err.join()
        proc.wait()

    result = ProcResult(str_cmd, proc.returncode, "".join(stdout_buf), "".join(stderr_buf))
    elapsed = time.monotonic() - started
    stats = _PROC_STATS.get()
    if stats is not None:
        stats["calls"] += 1
        stats["streaming_calls"] += 1
        stats["wall_s"] += elapsed
    if interrupted_reason is not None:
        raise ProcInterrupted(interrupted_reason, result)
    if proc.returncode != 0:
        raise ProcError(result)
