"""Minimal theming. Native Windows look by default.

We deliberately do NOT ship a custom dark stylesheet — Qt's Fusion + Windows palette
gives a consistent native feel. Semantic label colors use `palette(...)` stylesheet
references or QLabel foreground roles so light, dark, and high-contrast themes stay
readable.
"""

from __future__ import annotations

from importlib import resources as importlib_resources

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QLabel

# Page title typography (centralized so GA copy/layout tweaks stay in one place).
_PAGE_TITLE_STYLESHEET = "font-size: 18px; font-weight: 600;"


def apply(app: QApplication) -> None:
    """Apply app-wide theme hooks (native style by default)."""
    icon = load_window_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)


def make_page_title_label(text: str, parent=None) -> QLabel:
    lbl = QLabel(text, parent)
    lbl.setStyleSheet(_PAGE_TITLE_STYLESHEET)
    return lbl


def style_muted_detail_label(widget: QLabel, *, small: bool = False) -> None:
    """Secondary / de-emphasized text that tracks the palette (captions, hints)."""
    extra = " font-size: 11px;" if small else ""
    widget.setStyleSheet(f"color: palette(placeholder-text);{extra}")


def style_attention_status_label(widget: QLabel, *, italic: bool = False) -> None:
    """Draw attention without hard-coded amber/orange (e.g. queue paused)."""
    sl = " font-style: italic;" if italic else ""
    widget.setStyleSheet(f"color: palette(highlight);{sl}")


def style_inspector_note_label(widget: QLabel) -> None:
    """Ffprobe / media notes — distinct but theme-aware."""
    widget.setStyleSheet("color: palette(link);")


_cached_window_icon: QIcon | None = None

# Sidebar/tab icons bundled under ``aep.gui.resources.sidebar``.
_SIDEBAR_ICON_EXTENSIONS: tuple[str, ...] = (".png", ".ico", ".svg", ".webp")
_sidebar_icons_cache: dict[str, QIcon | None] = {}


def _sidebar_icon_basenames(name: str) -> tuple[str, ...]:
    """Try ``name`` plus hyphen ↔ space variants (e.g. ``job-config`` ↔ ``job config``)."""
    return tuple(dict.fromkeys((name, name.replace("-", " "), name.replace(" ", "-"))))


def load_sidebar_nav_icon(slug: str) -> QIcon | None:
    """Load a sidebar/tab icon from packaged ``gui/resources/sidebar`` if present.

    Tries ``<basename>.png`` / ``.ico`` / ``.svg`` / ``.webp`` for ``slug`` plus hyphen ↔ space
    variants so ``job-config`` matches ``job config.ico``.

    Args:
        slug: Stem used in code, typically kebab-case; may also match a spaced filename.

    Returns:
        Loaded icon, or ``None`` if no file was found or the image could not be read.
    """
    if slug in _sidebar_icons_cache:
        cached = _sidebar_icons_cache[slug]
        return None if cached is None or cached.isNull() else cached

    try:
        pkg_sidebar = importlib_resources.files("aep.gui.resources").joinpath("sidebar")
        for base in _sidebar_icon_basenames(slug):
            for ext in _SIDEBAR_ICON_EXTENSIONS:
                candidate = pkg_sidebar.joinpath(f"{base}{ext}")
                if not candidate.is_file():
                    continue
                with importlib_resources.as_file(candidate) as path:
                    icon = QIcon(str(path))
                    if icon.isNull():
                        continue
                    _sidebar_icons_cache[slug] = icon
                    return icon
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        pass

    _sidebar_icons_cache[slug] = None
    return None


def load_window_icon() -> QIcon:
    """Prefer bundled `gui/resources/app.ico`; fall back to a simple generated mark."""
    global _cached_window_icon  # noqa: PLW0603 -- single process-wide icon cache
    if _cached_window_icon is not None and not _cached_window_icon.isNull():
        return _cached_window_icon
    try:
        traj = importlib_resources.files("aep.gui.resources").joinpath("app.ico")
        with importlib_resources.as_file(traj) as p:
            if p.is_file():
                icon = QIcon(str(p))
                if not icon.isNull():
                    _cached_window_icon = icon
                    return icon
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        pass
    _cached_window_icon = _fallback_window_icon()
    return _cached_window_icon


def _fallback_window_icon() -> QIcon:
    """Multi-resolution placeholder when no bundled .ico is present."""
    icon = QIcon()
    fill = QColor(37, 99, 235)
    for size in (16, 32, 48, 64, 256):
        pix = QPixmap(size, size)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(fill)
        p.setPen(fill.darker(130))
        m = max(1, size // 16)
        p.drawRoundedRect(m, m, size - 2 * m, size - 2 * m, size // 5, size // 5)
        p.end()
        icon.addPixmap(pix)
    return icon
