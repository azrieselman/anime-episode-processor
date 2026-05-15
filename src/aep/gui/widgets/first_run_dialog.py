"""First-run dialog: download and install pinned third-party tools.

The Verify Tools dialog reports status; this dialog *fixes* the most common
"nothing works" state on a fresh installer build by walking through every
:class:`ToolPin` in :mod:`aep.app.tools_fetcher`, downloading the archive,
verifying its SHA256, and extracting it under the configured tools root.

Threading model:

* The dialog spawns a single :class:`QThread` (``_FetchWorker``) that drives
  ``tools_fetcher.fetch_all`` to completion.
* Progress and cancel hooks are passed to the fetcher; they emit Qt signals
  that the dialog renders on the GUI thread.
* The cancel button flips a thread-safe flag that the fetcher polls between
  chunks, so a 100 MB download cancels in under a second instead of waiting
  for HTTP timeout.

The dialog is modal — there is nothing useful the user can do in the rest of
the app until tools are present. After ``done(QDialog.Accepted)`` the caller
re-runs the verification probe to confirm everything resolves.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aep.app.tools_fetcher import (
    ALL_PINS,
    FetchCancelled,
    FetchError,
    FetchProgress,
    ToolPin,
    fetch_all,
    is_installed,
)

log = logging.getLogger(__name__)


_STAGE_LABELS: dict[str, str] = {
    "starting": "Starting…",
    "downloading": "Downloading",
    "verifying": "Verifying SHA256",
    "extracting": "Extracting",
    "installing": "Installing",
    "done": "Installed",
    "skipped": "Already installed",
    "failed": "Failed",
}


class _FetchWorker(QObject):
    """Lives on a worker QThread; drives ``tools_fetcher.fetch_all``."""

    progress = Signal(object)        # FetchProgress
    finished_ok = Signal(int)        # number of pins installed
    failed = Signal(str)             # error message
    cancelled = Signal()

    def __init__(
        self,
        pins: list[ToolPin],
        install_root: Path | None,
        cancel_flag: threading.Event,
        *,
        force: bool = False,
    ) -> None:
        super().__init__()
        self._pins = pins
        self._install_root = install_root
        self._cancel_flag = cancel_flag
        self._force = force

    def run(self) -> None:
        def _on_progress(p: FetchProgress) -> None:
            self.progress.emit(p)

        try:
            installed = fetch_all(
                self._pins,
                install_root=self._install_root,
                force=self._force,
                progress_cb=_on_progress,
                cancel=self._cancel_flag.is_set,
            )
        except FetchCancelled:
            self.cancelled.emit()
            return
        except FetchError as exc:
            log.exception("first-run fetch failed")
            self.failed.emit(str(exc))
            return
        except Exception as exc:
            log.exception("first-run fetch crashed")
            self.failed.emit(f"unexpected error: {exc}")
            return
        self.finished_ok.emit(len(installed))


class FirstRunDialog(QDialog):
    """Modal dialog walking the user through the first-run tools download.

    Use :meth:`run_if_needed` as the canonical entry point — it short-circuits
    when every pin is already installed.
    """

    @staticmethod
    def run_if_needed(parent: QWidget | None = None, *, install_root: Path | None = None) -> bool:
        """Open the dialog if any pin is missing; return True when everything is present after.

        Returns False if the user cancelled or the fetch failed.
        """
        missing = [p for p in ALL_PINS if not is_installed(p, install_root)]
        if not missing:
            return True
        dlg = FirstRunDialog(parent, missing_pins=missing, install_root=install_root)
        result = dlg.exec()
        return result == QDialog.DialogCode.Accepted

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        missing_pins: list[ToolPin] | None = None,
        install_root: Path | None = None,
        window_title: str | None = None,
        heading: str | None = None,
        intro_html: str | None = None,
        force_refresh: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(window_title or "Anime Episode Processor — First-Run Setup")
        self.setModal(True)
        self.resize(720, 480)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        self._pins = list(missing_pins if missing_pins is not None else ALL_PINS)
        self._install_root = install_root
        self._force_refresh = force_refresh
        self._heading_text = heading or "Downloading required tools"
        self._intro_html = intro_html or (
            "AEP needs ffmpeg, mkvtoolnix, and a handful of NCNN-Vulkan upscalers "
            "(about 2 GB total) before it can process video.<br>"
            "Each archive is downloaded from its vendor's official release URL and "
            "verified against a pinned SHA256 checksum."
        )
        self._cancel_flag = threading.Event()
        self._worker: _FetchWorker | None = None
        self._thread: QThread | None = None
        self._row_for_tool: dict[str, int] = {}

        self._build_ui()
        self._populate_table()

    # ------------------------------------------------------------- UI

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QLabel(self._heading_text)
        header_font = QFont(header.font())
        header_font.setBold(True)
        header_font.setPointSize(header.font().pointSize() + 1)
        header.setFont(header_font)
        root.addWidget(header)

        intro = QLabel(self._intro_html)
        intro.setWordWrap(True)
        intro.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(intro)

        self._table = QTableWidget(self)
        self._table.setColumnCount(3)
        self._table.setHorizontalHeaderLabels(["Tool", "Status", "Progress"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        h = self._table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, 1)

        self._overall_label = QLabel(
            f"Ready to fetch {len(self._pins)} tool(s) into "
            f"{self._install_root_display()}.",
        )
        self._overall_label.setWordWrap(True)
        root.addWidget(self._overall_label)

        self._overall_bar = QProgressBar(self)
        self._overall_bar.setRange(0, max(1, len(self._pins)))
        self._overall_bar.setValue(0)
        root.addWidget(self._overall_bar)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._start_btn = QPushButton("Start Download", self)
        self._start_btn.setDefault(True)
        self._start_btn.clicked.connect(self._on_start)
        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._cancel_btn)
        root.addLayout(btn_row)

        self._buttons = QDialogButtonBox(self)
        self._buttons.setVisible(False)
        self._close_btn = self._buttons.addButton(
            QDialogButtonBox.StandardButton.Close,
        )
        self._close_btn.clicked.connect(self.accept)
        root.addWidget(self._buttons)

    def _install_root_display(self) -> str:
        from aep.util.paths import tools_dir
        return str(self._install_root or tools_dir())

    def _populate_table(self) -> None:
        self._table.setRowCount(len(self._pins))
        self._row_for_tool.clear()
        for row, pin in enumerate(self._pins):
            self._row_for_tool[pin.tool_id] = row
            id_item = QTableWidgetItem(f"{pin.tool_id} ({pin.version})")
            id_item.setToolTip(pin.archive_url)
            self._table.setItem(row, 0, id_item)
            self._table.setItem(row, 1, QTableWidgetItem("Pending"))

            bar = QProgressBar(self)
            bar.setRange(0, 0)  # indeterminate until we get a Content-Length
            bar.setVisible(False)
            self._table.setCellWidget(row, 2, bar)

    # ---------------------------------------------------------- handlers

    def _on_start(self) -> None:
        self._start_btn.setEnabled(False)
        self._cancel_flag.clear()

        thread = QThread(self)
        worker = _FetchWorker(
            self._pins,
            self._install_root,
            self._cancel_flag,
            force=self._force_refresh,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished_ok.connect(self._on_finished_ok)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(self._on_cancelled)
        # Tear down the QThread once any terminal signal fires.
        for sig in (worker.finished_ok, worker.failed, worker.cancelled):
            sig.connect(thread.quit)
            sig.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._worker = worker
        self._thread = thread
        thread.start()

    def _on_cancel(self) -> None:
        if self._thread is None or not self._thread.isRunning():
            self.reject()
            return
        self._cancel_flag.set()
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Cancelling…")

    # ---------------------------------------------------------- progress

    def _on_progress(self, p: FetchProgress) -> None:
        row = self._row_for_tool.get(p.tool_id)
        if row is None:
            return
        self._table.setItem(row, 1, QTableWidgetItem(_STAGE_LABELS.get(p.stage, p.stage)))
        bar = self._table.cellWidget(row, 2)
        if isinstance(bar, QProgressBar):
            bar.setVisible(True)
            if p.stage == "downloading" and p.bytes_total:
                # Switch to determinate mode on first byte-count callback.
                if bar.maximum() != p.bytes_total:
                    bar.setRange(0, max(1, p.bytes_total))
                bar.setValue(min(p.bytes_downloaded, p.bytes_total))
                bar.setFormat(
                    f"{_format_mib(p.bytes_downloaded)} / {_format_mib(p.bytes_total)}"
                )
            elif p.stage in ("done", "skipped"):
                bar.setRange(0, 1)
                bar.setValue(1)
                bar.setFormat(_STAGE_LABELS[p.stage])
            else:
                bar.setRange(0, 0)
                bar.setFormat(_STAGE_LABELS.get(p.stage, p.stage))

        if p.stage in ("done", "skipped"):
            self._overall_bar.setValue(min(p.pin_index + 1, self._overall_bar.maximum()))
            self._overall_label.setText(
                f"{p.pin_index + 1} of {p.pin_total} tools complete.",
            )
        elif p.stage == "downloading" and p.bytes_total:
            self._overall_label.setText(
                f"Downloading {p.tool_id}: "
                f"{_format_mib(p.bytes_downloaded)} / {_format_mib(p.bytes_total)} "
                f"(tool {p.pin_index + 1} of {p.pin_total})",
            )

    # ---------------------------------------------------------- terminal

    def _on_finished_ok(self, count: int) -> None:
        self._overall_bar.setValue(self._overall_bar.maximum())
        self._overall_label.setText(
            f"All {count} tool(s) installed successfully. "
            "You can close this dialog and start using AEP.",
        )
        self._show_close_only()

    def _on_failed(self, message: str) -> None:
        self._overall_label.setText(
            f"<b>Fetch failed.</b><br>{message}<br><br>"
            "Check your network connection or run "
            "<code>python scripts/fetch_tools.py</code> from a checkout to retry.",
        )
        self._show_close_only(reject=True)

    def _on_cancelled(self) -> None:
        self._overall_label.setText("Cancelled. Re-open this dialog any time to retry.")
        self._show_close_only(reject=True)

    def _show_close_only(self, *, reject: bool = False) -> None:
        self._start_btn.setVisible(False)
        self._cancel_btn.setVisible(False)
        self._buttons.setVisible(True)
        self._close_btn.setText("Close")
        if reject:
            self._close_btn.clicked.disconnect()
            self._close_btn.clicked.connect(self.reject)

    # ---------------------------------------------------------- close

    def reject(self) -> None:
        # If a download is in flight, ask the worker to stop before tearing down.
        if self._thread is not None and self._thread.isRunning():
            self._cancel_flag.set()
            self._thread.quit()
            self._thread.wait(2000)
        super().reject()


def _format_mib(byte_count: int) -> str:
    mib = byte_count / (1024 * 1024)
    return f"{mib:.1f} MiB"


__all__ = ["FirstRunDialog"]


if __name__ == "__main__":  # pragma: no cover — manual smoke test
    import sys

    app = QApplication(sys.argv)
    dlg = FirstRunDialog()
    dlg.exec()
