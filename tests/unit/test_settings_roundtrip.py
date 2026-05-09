"""Settings load/save roundtrip and invalid-file handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from aep.errors import ConfigError
from aep.persist.settings import AppSettings, load_settings, save_settings, settings_path


def test_default_when_missing(tmp_runtime: Path) -> None:
    s = load_settings()
    assert isinstance(s, AppSettings)
    assert s.general.log_level in {"DEBUG", "INFO", "WARNING", "ERROR"}


def test_roundtrip(tmp_runtime: Path) -> None:
    s = AppSettings()
    s.general.log_level = "DEBUG"
    s.hardware.max_concurrent_jobs = 2
    s.hardware.decode_hwaccel = "d3d11va"
    s.hardware.rife_threads = "6:8:6"
    s.paths.anime4kcpp_dir = str(tmp_runtime / "anime4kcpp")
    s.paths.anime4kcpp_vs_filter_dir = str(tmp_runtime / "anime4kcpp-vs-filter")
    s.paths.vapoursynth_dir = str(tmp_runtime / "vapoursynth")
    save_settings(s)
    loaded = load_settings()
    assert loaded.general.log_level == "DEBUG"
    assert loaded.hardware.max_concurrent_jobs == 2
    assert loaded.hardware.decode_hwaccel == "d3d11va"
    assert loaded.hardware.rife_threads == "6:8:6"
    assert loaded.paths.anime4kcpp_dir == str(tmp_runtime / "anime4kcpp")
    assert loaded.paths.anime4kcpp_vs_filter_dir == str(tmp_runtime / "anime4kcpp-vs-filter")
    assert loaded.paths.vapoursynth_dir == str(tmp_runtime / "vapoursynth")


def test_invalid_file_raises(tmp_runtime: Path) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings()
