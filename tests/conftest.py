"""Shared test fixtures.

We make every test run with isolated runtime + tools dirs by setting AEP_RUNTIME_DIR,
AEP_TOOLS_DIR, and AEP_PRESETS_DIR for the duration of the test.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aep.constants import ENV_PRESETS_DIR, ENV_RUNTIME_DIR, ENV_TOOLS_DIR


@pytest.fixture
def tmp_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runtime = tmp_path / "runtime"
    tools = tmp_path / "tools"
    presets = tmp_path / "presets"
    for d in (runtime, tools, presets):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(ENV_RUNTIME_DIR, str(runtime))
    monkeypatch.setenv(ENV_TOOLS_DIR, str(tools))
    monkeypatch.setenv(ENV_PRESETS_DIR, str(presets))
    # Bust path lru_caches.
    from aep.util import paths
    for fn in (paths.runtime_dir, paths.tools_dir, paths.user_presets_dir, paths.builtin_presets_dir):
        fn.cache_clear()  # type: ignore[attr-defined]
    return runtime


@pytest.fixture
def builtin_presets_copied(tmp_runtime: Path) -> Path:
    """Copy the repo's built-in presets to the user presets dir for tests that need them."""
    repo_presets = Path(__file__).resolve().parents[1] / "presets"
    user_presets = Path.cwd() / Path(tmp_runtime).parent / "presets"
    if not user_presets.exists():
        user_presets = tmp_runtime.parent / "presets"
    for src in repo_presets.glob("*.yaml"):
        shutil.copy(src, user_presets / src.name)
    return user_presets
