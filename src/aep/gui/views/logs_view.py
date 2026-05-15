"""Logs view \u2014 tails the global aep.log or a per-job log.

The job dropdown lists every job whose workdir contains a job.log; choosing
one switches the tail to that file. The dropdown auto-refreshes every few
seconds so a newly-started job appears without restarting the GUI.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from aep.gui import theme
from aep.util.paths import jobs_dir, logs_dir

_GLOBAL_KEY = "__global__"


class LogsView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._fp_pos = 0
        self._global_path: Path = logs_dir() / "aep.log"
        self._path: Path = self._global_path

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        head = QHBoxLayout()
        head.addWidget(theme.make_page_title_label("Logs", self))
        head.addSpacing(12)

        head.addWidget(QLabel("Job:"))
        self._job_combo = QComboBox()
        self._job_combo.setMinimumWidth(220)
        self._job_combo.currentIndexChanged.connect(self._on_job_changed)
        head.addWidget(self._job_combo)

        head.addStretch(1)
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter (substring)\u2026")
        self._filter.textChanged.connect(self._apply_filter)
        head.addWidget(self._filter)
        self._clear_btn = QPushButton("Clear View")
        self._clear_btn.clicked.connect(self._clear_view)
        head.addWidget(self._clear_btn)
        root.addLayout(head)

        self._view = QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setMaximumBlockCount(20000)
        font = self._view.font()
        font.setFamily("Consolas")
        font.setPointSize(9)
        self._view.setFont(font)
        root.addWidget(self._view, 1)

        self._lines: list[str] = []

        # Tail loop
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tail)
        self._timer.start()

        # Job-list refresh loop \u2014 cheaper than tailing, runs less often.
        self._jobs_timer = QTimer(self)
        self._jobs_timer.setInterval(3000)
        self._jobs_timer.timeout.connect(self._refresh_job_list)
        self._jobs_timer.start()

        self._refresh_job_list()
        # Prime the view from the current source.
        self._reset_source(self._path)

    # --------------------------------------------------------------- jobs

    def _refresh_job_list(self) -> None:
        """Populate the dropdown with [Global] + every job that has a job.log.

        Preserves the current selection so the timer-driven refresh doesn't
        snap focus back to Global mid-read.
        """
        try:
            roots = sorted(jobs_dir().iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            roots = []

        # Build the desired list of (key, label) entries.
        desired: list[tuple[str, str]] = [(_GLOBAL_KEY, "Global (aep.log)")]
        for d in roots:
            if not d.is_dir():
                continue
            log_path = d / "job.log"
            if log_path.exists():
                # Show first 8 chars of the (UUID-shaped) job id for compactness.
                short = d.name[:8] if len(d.name) > 8 else d.name
                desired.append((d.name, f"Job {short}\u2026"))

        # Diff against current contents \u2014 only repopulate if something changed,
        # to avoid Qt firing currentIndexChanged on every refresh tick.
        current_keys = [
            self._job_combo.itemData(i) for i in range(self._job_combo.count())
        ]
        if current_keys == [k for k, _ in desired]:
            return

        previous = self._job_combo.currentData()
        self._job_combo.blockSignals(True)
        self._job_combo.clear()
        for key, label in desired:
            self._job_combo.addItem(label, userData=key)
        # Try to restore the previous selection.
        if previous is not None:
            for i in range(self._job_combo.count()):
                if self._job_combo.itemData(i) == previous:
                    self._job_combo.setCurrentIndex(i)
                    break
        self._job_combo.blockSignals(False)

    def _on_job_changed(self, _index: int) -> None:
        key = self._job_combo.currentData()
        if key == _GLOBAL_KEY or key is None:
            self._reset_source(self._global_path)
        else:
            self._reset_source(jobs_dir() / str(key) / "job.log")

    def _reset_source(self, path: Path) -> None:
        self._path = path
        self._fp_pos = 0
        self._view.clear()
        self._lines.clear()
        self._tail(initial=True)

    # --------------------------------------------------------------- tail

    def _tail(self, *, initial: bool = False) -> None:
        if not self._path.exists():
            return
        try:
            size = self._path.stat().st_size
        except OSError:
            return
        if size < self._fp_pos:
            # Rolled over (rotating handler). Reset.
            self._fp_pos = 0
            self._view.clear()
            self._lines.clear()
        if size == self._fp_pos:
            return
        try:
            with self._path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(self._fp_pos)
                new = f.read()
                self._fp_pos = f.tell()
        except OSError:
            return
        if not new:
            return
        new_lines = new.splitlines()
        self._lines.extend(new_lines)
        self._render(new_lines, append=not initial)

    def _apply_filter(self, _: str) -> None:
        self._render(self._lines, append=False)

    def _render(self, lines: list[str], *, append: bool) -> None:
        substr = self._filter.text().strip().lower()
        if append:
            for ln in lines:
                if substr and substr not in ln.lower():
                    continue
                self._view.appendPlainText(ln)
        else:
            self._view.clear()
            for ln in lines:
                if substr and substr not in ln.lower():
                    continue
                self._view.appendPlainText(ln)
        # Auto-scroll
        cur = self._view.textCursor()
        cur.movePosition(QTextCursor.MoveOperation.End)
        self._view.setTextCursor(cur)

    def _clear_view(self) -> None:
        self._view.clear()
        self._lines.clear()
