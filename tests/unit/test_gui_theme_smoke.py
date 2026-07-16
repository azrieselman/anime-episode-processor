"""Smoke tests for bundled GUI assets and Lens Dark theme (minimal Qt)."""

from __future__ import annotations

import os

import pytest


def test_bundled_app_ico_readable() -> None:
    """Windows shell expects a valid `.ico`; ensure package data is present."""
    from importlib import resources as ir

    root = ir.files("aep.gui.resources")
    ico = root.joinpath("app.ico")
    assert ico.is_file()
    data = ico.read_bytes()
    # ICONDIR: reserved=0, type=1 (icon), count>=1
    assert len(data) >= 22
    assert data[2:4] == b"\x01\x00"


def test_bundled_sidebar_nav_icons_present() -> None:
    """Main-window sidebar icons ship under ``gui/resources/sidebar``."""
    from importlib import resources as ir

    side = ir.files("aep.gui.resources").joinpath("sidebar")
    stems = (
        "queue.ico",
        "stream inspector.ico",
        "preset designer.ico",
        "logs.ico",
        "settings.ico",
    )
    for filename in stems:
        assert side.joinpath(filename).is_file(), filename


def test_lens_tokens_are_stable() -> None:
    from aep.gui.theme import TOKENS

    assert TOKENS.bg.startswith("#")
    assert TOKENS.accent == "#3B82F6"
    assert TOKENS.surface != TOKENS.bg


def test_lens_stylesheet_covers_signature_selectors() -> None:
    from aep.gui.theme import _build_stylesheet

    qss = _build_stylesheet()
    for needle in (
        "QListWidget#navSidebar",
        "QPushButton#primaryButton",
        "QFrame#dropArea",
        "sidebarBrand",
        "QListWidget#presetList",
        "QListWidget#presetList::item:hover",
        "QListWidget#presetList::item:selected:hover",
        "QCheckBox::indicator",
        "dropArea QLabel",
    ):
        assert needle in qss


def test_theme_apply_sets_lens_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """With high-contrast off, apply() installs Fusion + stylesheet once."""
    # Offscreen avoids needing a display; required for CI/headless.
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from aep.gui import theme

    app = QApplication.instance()
    if app is None:
        app = QApplication(["aep-theme-smoke"])

    monkeypatch.setattr(theme, "_is_high_contrast", lambda: False)
    # Reset module flag so the assertion is meaningful across re-runs.
    theme._lens_applied = False  # noqa: SLF001

    theme.apply(app)
    assert theme.is_lens_applied()
    assert "navSidebar" in app.styleSheet()
    assert not app.windowIcon().isNull()


def test_theme_apply_skips_qss_under_high_contrast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from aep.gui import theme

    app = QApplication.instance()
    if app is None:
        app = QApplication(["aep-theme-smoke-hc"])

    monkeypatch.setattr(theme, "_is_high_contrast", lambda: True)
    theme._lens_applied = False  # noqa: SLF001
    app.setStyleSheet("")

    theme.apply(app)
    assert not theme.is_lens_applied()
    assert app.styleSheet() == ""
