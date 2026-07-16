"""Drop area widget — accepts file/folder drops."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from aep.gui import theme

VIDEO_SUFFIXES = {".mkv", ".mp4", ".m4v", ".webm", ".avi", ".mov", ".ts", ".m2ts"}


def _expand_paths(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_file() and sub.suffix.lower() in VIDEO_SUFFIXES:
                    out.append(sub)
        elif p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            out.append(p)
    return out


class DropArea(QFrame):
    paths_dropped = Signal(list)  # list[Path]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("dropArea")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(theme.drop_area_stylesheet())
        self.setMinimumHeight(72)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel(
            "Drop video files or folders here\n"
            "(.mkv, .mp4, .webm, etc.) — folders are scanned recursively"
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setObjectName("dropAreaLabel")
        layout.addWidget(self._label)

    def set_empty_emphasis(self, empty: bool) -> None:
        """When the queue is empty, make the drop zone the primary call to action."""
        if empty:
            self._label.setText(
                "Drop video files or folders to get started\n"
                "(.mkv, .mp4, .webm, etc.) — or use Add Files / Add Folder below"
            )
            self.setMinimumHeight(96)
        else:
            self._label.setText(
                "Drop video files or folders here\n"
                "(.mkv, .mp4, .webm, etc.) — folders are scanned recursively"
            )
            self.setMinimumHeight(72)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            self.setProperty("drag", True)
            self.style().polish(self)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self.setProperty("drag", False)
        self.style().polish(self)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        self.setProperty("drag", False)
        self.style().polish(self)
        urls = event.mimeData().urls()
        raw = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        expanded = _expand_paths(raw)
        if expanded:
            self.paths_dropped.emit(expanded)
        event.acceptProposedAction()
