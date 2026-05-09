"""Path resolution.

In production the runtime directory is %LOCALAPPDATA%\\AEP, and tools live under
%LOCALAPPDATA%\\AEP\\tools. In dev (env var AEP_RUNTIME_DIR set, or running from a
checkout), these resolve to ./runtime/ and ./tools/ respectively.

We never hardcode tool paths in business logic. All access goes through `tools_dir()` and
the adapter base class, which validates the binary exists and is the pinned version.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from aep.constants import (
    APP_NAME,
    APP_VENDOR,
    DIR_BENCH,
    DIR_CACHE,
    DIR_JOBS,
    DIR_LOGS,
    DIR_PRESETS_USER,
    DIR_TEMP,
    ENV_PRESETS_DIR,
    ENV_RUNTIME_DIR,
    ENV_TOOLS_DIR,
    is_windows,
    project_root,
)


@lru_cache(maxsize=1)
def runtime_dir() -> Path:
    override = os.environ.get(ENV_RUNTIME_DIR)
    if override:
        return Path(override).expanduser().resolve()
    if is_windows():
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / APP_VENDOR / APP_NAME
    # Dev/posix fallback
    return project_root() / "runtime"


@lru_cache(maxsize=1)
def tools_dir() -> Path:
    override = os.environ.get(ENV_TOOLS_DIR)
    if override:
        return Path(override).expanduser().resolve()
    # Prefer in-repo tools/ during development; in installed builds the installer copies
    # tools to %LOCALAPPDATA%\AEP\tools and the env var is set on first run.
    repo_tools = project_root() / "tools"
    if repo_tools.exists():
        return repo_tools
    return runtime_dir() / "tools"


@lru_cache(maxsize=1)
def builtin_presets_dir() -> Path:
    """Built-in presets shipped inside the package or repo. Read-only."""
    return project_root() / "presets"


@lru_cache(maxsize=1)
def user_presets_dir() -> Path:
    override = os.environ.get(ENV_PRESETS_DIR)
    if override:
        path = Path(override).expanduser().resolve()
    else:
        path = runtime_dir() / DIR_PRESETS_USER
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    path = runtime_dir() / DIR_LOGS
    path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_dir() -> Path:
    path = runtime_dir() / DIR_JOBS
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = runtime_dir() / DIR_CACHE
    path.mkdir(parents=True, exist_ok=True)
    return path


def bench_dir() -> Path:
    path = runtime_dir() / DIR_BENCH
    path.mkdir(parents=True, exist_ok=True)
    return path


def temp_dir() -> Path:
    path = runtime_dir() / DIR_TEMP
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_runtime_dirs() -> None:
    for func in (runtime_dir, logs_dir, jobs_dir, cache_dir, bench_dir, temp_dir, user_presets_dir):
        func()
