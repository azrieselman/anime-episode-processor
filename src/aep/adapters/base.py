"""Adapter base class.

Every external binary is wrapped behind an adapter that:
* Resolves the binary path (settings override → bundled tools → PATH).
* Verifies it's installed.
* Captures and validates the version against PINNED_VERSIONS.
* Provides a typed surface for the rest of the app.

We deliberately do NOT call subprocess directly anywhere outside `aep.util.proc` and
adapter classes. This makes mocking adapters in unit tests straightforward.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from aep.constants import PINNED_VERSIONS
from aep.errors import ToolNotFoundError, ToolVersionMismatchError
from aep.util.paths import tools_dir

log = logging.getLogger(__name__)


class ToolAdapter:
    """Base class for external-tool adapters.

    Subclasses set:
      tool_id      — key into PINNED_VERSIONS
      bin_name     — Windows binary filename
      tools_subdir — subdir under tools/ where this tool lives (e.g. "ffmpeg")
    """

    tool_id: str = ""
    bin_name: str = ""
    tools_subdir: str = ""

    def __init__(self, *, override_dir: Path | str | None = None) -> None:
        self._override_dir = Path(override_dir) if override_dir else None
        self._resolved_path: Path | None = None
        self._version: str | None = None

    @property
    def path(self) -> Path:
        if self._resolved_path is None:
            self._resolved_path = self._resolve()
        return self._resolved_path

    @property
    def version(self) -> str:
        if self._version is None:
            self._version = self._detect_version()
        return self._version

    def _resolve(self) -> Path:
        # 1. explicit override
        if self._override_dir:
            candidate = self._override_dir / self.bin_name
            if candidate.is_file():
                return candidate.resolve()
            raise ToolNotFoundError(
                f"{self.bin_name} not found in override dir: {self._override_dir}"
            )
        # 2. bundled tools dir
        bundled = tools_dir() / self.tools_subdir / self.bin_name
        if bundled.is_file():
            return bundled.resolve()
        # 3. PATH (development convenience; production installs always have bundled tools)
        path_hit = shutil.which(self.bin_name) or shutil.which(self.bin_name.replace(".exe", ""))
        if path_hit:
            return Path(path_hit).resolve()
        raise ToolNotFoundError(
            f"{self.bin_name} not found",
            context={
                "searched_override": str(self._override_dir) if self._override_dir else None,
                "searched_bundled": str(bundled),
                "searched_path": True,
            },
        )

    def _detect_version(self) -> str:
        """Subclasses override to extract version from `<tool> --version` output."""
        raise NotImplementedError

    def verify(self, *, strict: bool = False) -> None:
        """Confirm the binary is present and (optionally) version-pinned.

        We default to non-strict so dev environments with slightly different builds still
        work; strict mode is enabled by the installer post-install verification step.
        """
        _ = self.path  # raises ToolNotFoundError
        try:
            v = self.version
        except Exception as exc:
            log.warning("could not detect version for %s: %s", self.tool_id, exc)
            return
        expected = PINNED_VERSIONS.get(self.tool_id)
        if not expected:
            return
        if not v.startswith(expected.lstrip("n")):
            msg = f"{self.tool_id} version {v!r} does not match pinned {expected!r}"
            if strict:
                raise ToolVersionMismatchError(msg, context={"got": v, "want": expected})
            log.warning(msg)


def env_with_tool_dirs(extra_dirs: list[Path] | None = None) -> dict[str, str]:
    """Returns an env dict where bundled tool dirs are prepended to PATH.

    Useful when a tool spawns helper executables (e.g. ffmpeg looking up DLLs).
    """
    env = os.environ.copy()
    parts: list[str] = []
    base = tools_dir()
    if base.exists():
        for child in base.iterdir():
            if child.is_dir():
                parts.append(str(child))
    if extra_dirs:
        parts.extend(str(p) for p in extra_dirs)
    if parts:
        env["PATH"] = os.pathsep.join(parts + [env.get("PATH", "")])
    return env
