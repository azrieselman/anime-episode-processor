from __future__ import annotations

import sys
from pathlib import Path

from aep import constants
from aep.util import paths


def test_project_root_uses_executable_parent_when_frozen(monkeypatch, tmp_path: Path) -> None:
    exe_path = tmp_path / "install" / "aep-gui.exe"
    exe_path.parent.mkdir(parents=True)
    exe_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)

    assert constants.project_root() == exe_path.parent


def test_builtin_presets_dir_resolves_from_frozen_bundle_root(
    monkeypatch, tmp_path: Path
) -> None:
    exe_path = tmp_path / "install" / "aep-gui.exe"
    exe_path.parent.mkdir(parents=True)
    exe_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    paths.builtin_presets_dir.cache_clear()

    assert paths.builtin_presets_dir() == exe_path.parent / "presets"


def test_builtin_presets_dir_prefers_meipass_when_present(
    monkeypatch, tmp_path: Path
) -> None:
    exe_path = tmp_path / "install" / "aep-gui.exe"
    internal_root = tmp_path / "install" / "_internal"
    exe_path.parent.mkdir(parents=True)
    internal_root.mkdir(parents=True)
    exe_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe_path))
    monkeypatch.setattr(sys, "_MEIPASS", str(internal_root), raising=False)
    paths.builtin_presets_dir.cache_clear()

    assert constants.project_root() == internal_root
    assert paths.builtin_presets_dir() == internal_root / "presets"
