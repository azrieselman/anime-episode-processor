"""Lens Dark theming for AEP.

Logo-grounded navy surfaces, cool blue accent, thin borders. One static QSS +
palette pass at startup - no animated chrome. High-contrast Windows skips the
custom stylesheet and keeps the native palette.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from importlib import resources as importlib_resources
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QStyleFactory


@dataclass(frozen=True, slots=True)
class LensTokens:
    """Static color tokens for Lens Dark (hex strings for QSS)."""

    bg: str = "#0B1220"
    surface: str = "#121A2B"
    surface_raised: str = "#1A2438"
    border: str = "#2A3548"
    text: str = "#E8EEF7"
    muted: str = "#8B9BB4"
    accent: str = "#3B82F6"
    accent_hover: str = "#2563EB"
    danger: str = "#F87171"
    success: str = "#34D399"
    warning: str = "#FBBF24"


TOKENS = LensTokens()

_PAGE_TITLE_STYLESHEET = (
    f"font-size: 18px; font-weight: 600; color: {TOKENS.text}; letter-spacing: 0.5px;"
)

_lens_applied: bool = False


def is_lens_applied() -> bool:
    """True when Lens Dark QSS/palette was applied this process."""
    return _lens_applied


def _is_high_contrast() -> bool:
    """Return True when Windows High Contrast is on (skip custom Lens QSS)."""
    try:
        import ctypes

        class HIGHCONTRASTW(ctypes.Structure):
            _fields_ = [
                ("cbSize", ctypes.c_uint),
                ("dwFlags", ctypes.c_uint),
                ("lpszDefaultScheme", ctypes.c_wchar_p),
            ]

        hc = HIGHCONTRASTW()
        hc.cbSize = ctypes.sizeof(HIGHCONTRASTW)
        if ctypes.windll.user32.SystemParametersInfoW(0x0042, hc.cbSize, ctypes.byref(hc), 0):
            # HCF_HIGHCONTRASTON = 0x0001
            return bool(hc.dwFlags & 0x0001)
    except Exception:
        pass
    return False


def _build_palette(tokens: LensTokens = TOKENS) -> QPalette:
    pal = QPalette()
    bg = QColor(tokens.bg)
    surface = QColor(tokens.surface)
    raised = QColor(tokens.surface_raised)
    text = QColor(tokens.text)
    muted = QColor(tokens.muted)
    accent = QColor(tokens.accent)
    border = QColor(tokens.border)

    pal.setColor(QPalette.ColorRole.Window, bg)
    pal.setColor(QPalette.ColorRole.WindowText, text)
    pal.setColor(QPalette.ColorRole.Base, surface)
    pal.setColor(QPalette.ColorRole.AlternateBase, raised)
    pal.setColor(QPalette.ColorRole.Text, text)
    pal.setColor(QPalette.ColorRole.PlaceholderText, muted)
    pal.setColor(QPalette.ColorRole.Button, raised)
    pal.setColor(QPalette.ColorRole.ButtonText, text)
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorRole.Highlight, accent)
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    pal.setColor(QPalette.ColorRole.Link, accent)
    pal.setColor(QPalette.ColorRole.LinkVisited, QColor(tokens.accent_hover))
    pal.setColor(QPalette.ColorRole.ToolTipBase, raised)
    pal.setColor(QPalette.ColorRole.ToolTipText, text)
    pal.setColor(QPalette.ColorRole.Light, border)
    pal.setColor(QPalette.ColorRole.Mid, border)
    pal.setColor(QPalette.ColorRole.Dark, bg)
    pal.setColor(QPalette.ColorRole.Shadow, QColor("#000000"))
    # Disabled
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, muted)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, muted)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, muted)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, border)
    return pal


def _cache_dir() -> Path:
    root = Path(tempfile.gettempdir()) / "aep-lens-theme"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_checkbox_icons(tokens: LensTokens = TOKENS) -> tuple[str, str, str, str]:
    """Paint indicator PNGs; return (cb_off, cb_on, rb_off, rb_on) paths."""
    cache = _cache_dir()
    unchecked = cache / "cb-unchecked.png"
    checked = cache / "cb-checked.png"
    radio_off = cache / "rb-unchecked.png"
    radio_checked = cache / "rb-checked.png"

    size = 16
    border = QColor(tokens.muted)
    fill = QColor(tokens.surface_raised)
    accent = QColor(tokens.accent)
    white = QColor("#FFFFFF")

    # Unchecked square
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(fill)
    p.setPen(border)
    p.drawRoundedRect(1, 1, size - 3, size - 3, 3, 3)
    p.end()
    pix.save(str(unchecked), "PNG")

    # Checked square + tick
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(accent)
    p.setPen(accent.darker(110))
    p.drawRoundedRect(1, 1, size - 3, size - 3, 3, 3)
    pen = p.pen()
    pen.setColor(white)
    pen.setWidth(2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    p.setPen(pen)
    p.drawLine(4, 8, 7, 11)
    p.drawLine(7, 11, 12, 5)
    p.end()
    pix.save(str(checked), "PNG")

    # Radio unchecked
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(fill)
    p.setPen(border)
    p.drawEllipse(1, 1, size - 3, size - 3)
    p.end()
    pix.save(str(radio_off), "PNG")

    # Radio checked (dot)
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(fill)
    p.setPen(border)
    p.drawEllipse(1, 1, size - 3, size - 3)
    p.setBrush(accent)
    p.setPen(accent)
    p.drawEllipse(5, 5, 6, 6)
    p.end()
    pix.save(str(radio_checked), "PNG")

    return str(unchecked), str(checked), str(radio_off), str(radio_checked)


def _qss_url(path: str) -> str:
    """Qt stylesheet url() with forward slashes."""
    return Path(path).resolve().as_posix()


def _build_stylesheet(
    tokens: LensTokens = TOKENS,
    *,
    checkbox_icons: tuple[str, str, str, str] | None = None,
) -> str:
    t = tokens
    if checkbox_icons is None:
        cb_rules = f"""
    QCheckBox::indicator, QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: 1px solid {t.muted};
        background-color: {t.surface_raised};
    }}
    QCheckBox::indicator {{
        border-radius: 3px;
    }}
    QRadioButton::indicator {{
        border-radius: 8px;
    }}
    QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
        background-color: {t.accent};
        border-color: {t.accent_hover};
    }}
"""
    else:
        cb_off, cb_on, rb_off, rb_on = checkbox_icons
        cb_off_u = _qss_url(cb_off)
        cb_on_u = _qss_url(cb_on)
        rb_off_u = _qss_url(rb_off)
        rb_on_u = _qss_url(rb_on)
        cb_rules = f"""
    QCheckBox::indicator {{
        width: 16px;
        height: 16px;
        border: none;
        background: transparent;
    }}
    QCheckBox::indicator:unchecked {{
        image: url("{cb_off_u}");
    }}
    QCheckBox::indicator:checked {{
        image: url("{cb_on_u}");
    }}
    QRadioButton::indicator {{
        width: 16px;
        height: 16px;
        border: none;
        background: transparent;
    }}
    QRadioButton::indicator:unchecked {{
        image: url("{rb_off_u}");
    }}
    QRadioButton::indicator:checked {{
        image: url("{rb_on_u}");
    }}
"""
    return f"""
    QMainWindow, QDialog {{
        background-color: {t.bg};
        color: {t.text};
    }}
    QWidget {{
        color: {t.text};
    }}
    QStackedWidget {{
        background-color: {t.bg};
        border: none;
    }}
    QMenuBar {{
        background-color: {t.surface};
        color: {t.text};
        border-bottom: 1px solid {t.border};
        padding: 2px 0;
    }}
    QMenuBar::item:selected {{
        background-color: {t.surface_raised};
    }}
    QMenu {{
        background-color: {t.surface};
        color: {t.text};
        border: 1px solid {t.border};
    }}
    QMenu::item:selected {{
        background-color: {t.accent};
        color: #FFFFFF;
    }}
    QStatusBar {{
        background-color: {t.surface};
        color: {t.muted};
        border-top: 1px solid {t.border};
    }}
    QWidget#sidebarRail {{
        background-color: {t.surface};
    }}
    QListWidget#navSidebar {{
        background-color: {t.surface};
        border: none;
        border-right: 1px solid {t.border};
        outline: none;
        padding: 6px 0;
        font-size: 13px;
    }}
    QListWidget#navSidebar::item {{
        color: {t.text};
        padding: 10px 12px 10px 16px;
        margin: 2px 8px;
        border-radius: 4px;
        border-left: 3px solid transparent;
    }}
    QListWidget#navSidebar::item:hover {{
        background-color: {t.surface_raised};
    }}
    QListWidget#navSidebar::item:selected {{
        background-color: {t.surface_raised};
        border-left: 3px solid {t.accent};
        color: {t.text};
    }}
    QLabel#sidebarBrand {{
        color: {t.text};
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        line-height: 1.15;
        padding: 0;
    }}
    QFrame#sidebarBrandFrame {{
        background-color: {t.surface};
        border: none;
        border-right: 1px solid {t.border};
        border-bottom: 1px solid {t.border};
    }}
    QToolButton {{
        background-color: transparent;
        color: {t.accent};
        border: none;
        border-radius: 4px;
        padding: 4px 8px;
        font-weight: 600;
        text-align: left;
    }}
    QToolButton:hover {{
        background-color: {t.surface_raised};
        color: {t.text};
    }}
    QToolButton:checked {{
        color: {t.text};
    }}
    QListWidget {{
        background-color: {t.surface};
        color: {t.text};
        border: 1px solid {t.border};
        outline: none;
        selection-background-color: {t.accent};
        selection-color: #FFFFFF;
    }}
    QListWidget::item {{
        color: {t.text};
        padding: 8px 10px;
    }}
    QListWidget::item:hover {{
        background-color: {t.surface_raised};
    }}
    QListWidget::item:selected {{
        background-color: {t.accent};
        color: #FFFFFF;
    }}
    QListWidget#presetList {{
        background-color: {t.surface};
        border: 1px solid {t.border};
        border-radius: 4px;
        outline: none;
        padding: 6px 4px;
        font-size: 13px;
        /* Keep selection paint colors aligned with ::item rules. */
        selection-background-color: {t.surface_raised};
        selection-color: {t.text};
    }}
    QListWidget#presetList::item {{
        color: {t.text};
        border-radius: 4px;
        /* Keep padding modest — large QSS padding often overflows the layout slot. */
        padding: 6px 10px;
        margin: 0 4px;
        min-height: 22px;
        border-left: 3px solid transparent;
        background-color: transparent;
    }}
    QListWidget#presetList::item:hover {{
        background-color: {t.surface_raised};
        color: {t.text};
    }}
    QListWidget#presetList::item:selected {{
        background-color: {t.surface_raised};
        color: {t.text};
        border-left: 3px solid {t.accent};
    }}
    QListWidget#presetList::item:selected:hover {{
        background-color: {t.surface_raised};
        color: {t.text};
        border-left: 3px solid {t.accent};
    }}
    QWidget#presetEditorRoot, QWidget#presetEditorInner {{
        background-color: {t.bg};
        color: {t.text};
    }}
    QLabel#presetHint {{
        color: {t.muted};
        font-size: 11px;
    }}
    QScrollArea {{
        border: none;
        background-color: {t.bg};
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: {t.bg};
    }}
    QFrame#dropArea {{
        border: 2px dashed {t.border};
        border-radius: 8px;
        background-color: {t.surface};
    }}
    QFrame#dropArea QLabel {{
        color: {t.text};
        background-color: transparent;
    }}
    QFrame#dropArea[drag="true"] {{
        border-color: {t.accent};
        background-color: {t.surface_raised};
    }}
    QFrame#dropArea[drag="true"] QLabel {{
        color: {t.text};
    }}
    QPushButton {{
        background-color: {t.surface_raised};
        color: {t.text};
        border: 1px solid {t.border};
        border-radius: 4px;
        padding: 6px 12px;
        min-height: 22px;
    }}
    QPushButton:hover {{
        background-color: {t.border};
        border-color: {t.muted};
    }}
    QPushButton:pressed {{
        background-color: {t.surface};
    }}
    QPushButton:disabled {{
        color: {t.muted};
        background-color: {t.surface};
        border-color: {t.border};
    }}
    QPushButton#primaryButton {{
        background-color: {t.accent};
        color: #FFFFFF;
        border: 1px solid {t.accent};
        font-weight: 600;
    }}
    QPushButton#primaryButton:hover {{
        background-color: {t.accent_hover};
        border-color: {t.accent_hover};
    }}
    QPushButton#primaryButton:disabled {{
        background-color: {t.surface_raised};
        color: {t.muted};
        border-color: {t.border};
        font-weight: 600;
    }}
    QGroupBox {{
        background-color: {t.surface};
        border: 1px solid {t.border};
        border-radius: 4px;
        margin-top: 12px;
        padding: 12px 10px 10px 10px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {t.text};
    }}
    QTabWidget::pane {{
        border: 1px solid {t.border};
        background-color: {t.surface};
        border-radius: 0 0 4px 4px;
    }}
    QTabBar::tab {{
        background-color: {t.bg};
        color: {t.muted};
        border: 1px solid {t.border};
        border-bottom: none;
        padding: 6px 14px;
        margin-right: 2px;
    }}
    QTabBar::tab:selected {{
        background-color: {t.surface};
        color: {t.text};
        border-top: 2px solid {t.accent};
    }}
    QTabBar::tab:hover:!selected {{
        color: {t.text};
        background-color: {t.surface_raised};
    }}
    QTableWidget, QTableView, QTreeView, QListView {{
        background-color: {t.surface};
        alternate-background-color: {t.surface_raised};
        color: {t.text};
        gridline-color: {t.border};
        border: 1px solid {t.border};
        selection-background-color: {t.accent};
        selection-color: #FFFFFF;
    }}
    QHeaderView::section {{
        background-color: {t.surface_raised};
        color: {t.muted};
        border: none;
        border-right: 1px solid {t.border};
        border-bottom: 1px solid {t.border};
        padding: 6px 8px;
        font-weight: 600;
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background-color: {t.surface};
        color: {t.text};
        border: 1px solid {t.border};
        border-radius: 4px;
        padding: 4px 8px;
        selection-background-color: {t.accent};
        selection-color: #FFFFFF;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
    QPlainTextEdit:focus, QTextEdit:focus {{
        border-color: {t.accent};
        background-color: {t.surface};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {t.surface};
        color: {t.text};
        border: 1px solid {t.border};
        selection-background-color: {t.accent};
        selection-color: #FFFFFF;
    }}
    QCheckBox, QRadioButton {{
        color: {t.text};
        spacing: 8px;
    }}
    {cb_rules}
    QScrollBar:vertical {{
        background: {t.bg};
        width: 10px;
        margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {t.border};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QSplitter::handle {{
        background-color: {t.border};
    }}
    QToolTip {{
        background-color: {t.surface_raised};
        color: {t.text};
        border: 1px solid {t.border};
        padding: 4px;
    }}
    QListWidget#navSidebar {{
        background-color: {t.surface};
        border: none;
        border-right: 1px solid {t.border};
        outline: none;
        padding: 6px 0;
        font-size: 13px;
    }}
    QListWidget#navSidebar::item {{
        color: {t.text};
        padding: 10px 12px 10px 16px;
        margin: 2px 8px;
        border-radius: 4px;
        border-left: 3px solid transparent;
    }}
    QListWidget#navSidebar::item:hover {{
        background-color: {t.surface_raised};
    }}
    QListWidget#navSidebar::item:selected {{
        background-color: {t.surface_raised};
        border-left: 3px solid {t.accent};
        color: {t.text};
    }}
    """


def apply(app: QApplication) -> None:
    """Apply Lens Dark (or native fallback under high contrast) and window icon."""
    global _lens_applied  # noqa: PLW0603

    icon = load_window_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    if _is_high_contrast():
        _lens_applied = False
        return

    fusion = QStyleFactory.create("Fusion")
    if fusion is not None:
        app.setStyle(fusion)
    app.setPalette(_build_palette())
    icons = _write_checkbox_icons()
    app.setStyleSheet(_build_stylesheet(checkbox_icons=icons))
    _lens_applied = True


def make_page_title_label(text: str, parent=None) -> QLabel:
    lbl = QLabel(text, parent)
    if is_lens_applied():
        lbl.setStyleSheet(_PAGE_TITLE_STYLESHEET)
    else:
        lbl.setStyleSheet("font-size: 18px; font-weight: 600;")
    return lbl


def style_muted_detail_label(widget: QLabel, *, small: bool = False) -> None:
    """Secondary / de-emphasized text."""
    extra = " font-size: 11px;" if small else ""
    if _lens_applied:
        widget.setStyleSheet(f"color: {TOKENS.muted};{extra}")
    else:
        widget.setStyleSheet(f"color: palette(placeholder-text);{extra}")


def style_attention_status_label(widget: QLabel, *, italic: bool = False) -> None:
    """Draw attention (e.g. queue paused, tools warning)."""
    sl = " font-style: italic;" if italic else ""
    if _lens_applied:
        widget.setStyleSheet(f"color: {TOKENS.warning};{sl}")
    else:
        widget.setStyleSheet(f"color: palette(highlight);{sl}")


def style_inspector_note_label(widget: QLabel) -> None:
    """Ffprobe / media notes - distinct but theme-aware."""
    if _lens_applied:
        widget.setStyleSheet(f"color: {TOKENS.accent};")
    else:
        widget.setStyleSheet("color: palette(link);")


def drop_area_stylesheet() -> str:
    """Stylesheet for the Queue drop zone (also covered app-wide when Lens applies)."""
    if _lens_applied:
        return (
            f"QFrame#dropArea {{ border: 2px dashed {TOKENS.border}; border-radius: 8px; "
            f"background-color: {TOKENS.surface}; }}"
            f"QFrame#dropArea QLabel {{ color: {TOKENS.text}; background-color: transparent; }}"
            f"QFrame#dropArea[drag=\"true\"] {{ border-color: {TOKENS.accent}; "
            f"background-color: {TOKENS.surface_raised}; }}"
            f"QFrame#dropArea[drag=\"true\"] QLabel {{ color: {TOKENS.text}; }}"
        )
    return (
        "QFrame#dropArea { border: 2px dashed palette(mid); border-radius: 8px; }"
        "QFrame#dropArea QLabel { color: palette(window-text); background: transparent; }"
        "QFrame#dropArea[drag=true] { border-color: palette(highlight); }"
    )


_cached_window_icon: QIcon | None = None

# Sidebar/tab icons bundled under ``aep.gui.resources.sidebar``.
_SIDEBAR_ICON_EXTENSIONS: tuple[str, ...] = (".png", ".ico", ".svg", ".webp")
_sidebar_icons_cache: dict[str, QIcon | None] = {}


def _sidebar_icon_basenames(name: str) -> tuple[str, ...]:
    """Try ``name`` plus hyphen <-> space variants (e.g. ``stream-inspector`` <-> ``stream inspector``)."""
    return tuple(dict.fromkeys((name, name.replace("-", " "), name.replace(" ", "-"))))


def load_sidebar_nav_icon(slug: str) -> QIcon | None:
    """Load a sidebar/tab icon from packaged ``gui/resources/sidebar`` if present.

    Tries ``<basename>.png`` / ``.ico`` / ``.svg`` / ``.webp`` for ``slug`` plus hyphen <-> space
    variants so ``stream-inspector`` matches ``stream inspector.ico``.

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
    fill = QColor(TOKENS.accent)
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
