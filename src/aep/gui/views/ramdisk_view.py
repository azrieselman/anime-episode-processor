"""RamDisk management view backed by ImDisk Toolkit."""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aep.app import imdisk
from aep.app.services import AppServices
from aep.gui import theme
from aep.jobs.models import JobState
from aep.util.win_pe_version import pe_version_resource_strings


def _format_bytes(n: int) -> str:
    return f"{n / (1024**3):.1f} GiB"


def _format_free_space(path: str) -> str:
    if not path:
        return "Not configured"
    try:
        p = Path(path)
        probe = p
        while not probe.exists():
            if probe.parent == probe:
                return "Unavailable"
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        return f"{_format_bytes(usage.free)} free of {_format_bytes(usage.total)}"
    except OSError:
        return "Unavailable"


class _ImDiskWorker(QObject):
    progress = Signal(str, object)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(self, action: Callable[[], str]) -> None:
        super().__init__()
        self._action = action

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._action())
        except Exception as exc:
            self.failed.emit(str(exc))


class RamDiskView(QWidget):
    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self._services = services
        self._worker_thread: QThread | None = None
        self._worker: _ImDiskWorker | None = None
        self._pending_apply_letter: str | None = None
        self._pending_clear_letter: str | None = None
        self._build()
        self._refresh()

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.addWidget(theme.make_page_title_label("RamDisk", self))

        if sys.platform != "win32":
            msg = QLabel("RamDisk management is only available on Windows.")
            theme.style_muted_detail_label(msg)
            root.addWidget(msg)
            root.addStretch(1)
            return

        self._build_status_group(root)
        self._build_install_group(root)
        self._build_manage_group(root)
        root.addStretch(1)

    def _build_status_group(self, root: QVBoxLayout) -> None:
        status = QGroupBox("Status")
        form = QFormLayout(status)
        self._installed_label = QLabel("")
        self._version_label = QLabel("")
        self._scratch_path_label = QLabel("")
        self._scratch_space_label = QLabel("")
        form.addRow("ImDisk Toolkit:", self._installed_label)
        form.addRow("Version:", self._version_label)
        form.addRow("AEP scratch path:", self._scratch_path_label)
        form.addRow("Scratch space:", self._scratch_space_label)
        root.addWidget(status)

    def _build_install_group(self, root: QVBoxLayout) -> None:
        install = QGroupBox("Install ImDisk Toolkit")
        layout = QVBoxLayout(install)
        note = QLabel(
            "Downloads the pinned ImDisk Toolkit installer and runs install.bat with a UAC prompt."
        )
        theme.style_muted_detail_label(note, small=True)
        layout.addWidget(note)
        self._install_progress = QProgressBar()
        self._install_progress.setRange(0, 100)
        self._install_progress.setValue(0)
        self._install_status = QLabel("Ready")
        theme.style_muted_detail_label(self._install_status, small=True)
        row = QHBoxLayout()
        self._install_btn = QPushButton("Install ImDisk Toolkit")
        self._install_btn.clicked.connect(self._install_imdisk)
        row.addWidget(self._install_btn)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self._install_progress)
        layout.addWidget(self._install_status)
        root.addWidget(install)

    def _build_manage_group(self, root: QVBoxLayout) -> None:
        manage = QGroupBox("Manage RamDisk")
        form = QFormLayout(manage)
        note = QLabel("Creating and formatting a ramdisk may trigger a UAC prompt.")
        theme.style_muted_detail_label(note, small=True)
        form.addRow("", note)

        self._letter_combo = QComboBox()
        self._letter_combo.currentTextChanged.connect(self._refresh_selected_drive_status)
        form.addRow("Drive letter:", self._letter_combo)

        self._size_gb = QSpinBox()
        self._size_gb.setRange(1, 512)
        self._size_gb.setValue(32)
        self._size_gb.setSuffix(" GB")
        form.addRow("Size:", self._size_gb)

        self._selected_drive_label = QLabel("")
        theme.style_muted_detail_label(self._selected_drive_label, small=True)
        form.addRow("Selected drive:", self._selected_drive_label)

        btns = QHBoxLayout()
        self._create_btn = QPushButton("Create RamDisk")
        self._create_btn.clicked.connect(self._create_ramdisk)
        self._remove_btn = QPushButton("Remove RamDisk")
        self._remove_btn.clicked.connect(self._remove_ramdisk)
        btns.addWidget(self._create_btn)
        btns.addWidget(self._remove_btn)
        btns.addStretch(1)
        form.addRow("", btns)
        root.addWidget(manage)

    def _refresh(self) -> None:
        if sys.platform != "win32":
            return

        installed = imdisk.is_installed()
        self._installed_label.setText("Installed" if installed else "Not installed")
        self._version_label.setText(_imdisk_version_text() if installed else "--")

        settings = self._services.settings.get()
        scratch = settings.paths.ramdisk_path or ""
        self._scratch_path_label.setText(scratch or "Not configured")
        self._scratch_space_label.setText(_format_free_space(scratch))

        self._populate_drive_letters()
        self._install_btn.setEnabled(installed is False and self._worker_thread is None)
        self._install_btn.setVisible(installed is False)
        self._refresh_selected_drive_status()

    def _populate_drive_letters(self) -> None:
        current = self._letter_combo.currentText() or "R"
        self._letter_combo.blockSignals(True)
        self._letter_combo.clear()
        for code in range(ord("A"), ord("Z") + 1):
            letter = chr(code)
            status = imdisk.get_volume_status(letter)
            label = letter if not status.mounted else f"{letter} (mounted)"
            self._letter_combo.addItem(label, letter)
        idx = self._letter_combo.findData(current[0].upper())
        if idx < 0:
            idx = self._letter_combo.findData("R")
        self._letter_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._letter_combo.blockSignals(False)

    def _selected_letter(self) -> str:
        data = self._letter_combo.currentData()
        return str(data) if data else "R"

    def _refresh_selected_drive_status(self) -> None:
        if sys.platform != "win32" or not hasattr(self, "_letter_combo"):
            return
        letter = self._selected_letter()
        installed = imdisk.is_installed()
        status = imdisk.get_volume_status(letter)
        if status.mounted:
            self._selected_drive_label.setText(
                f"{imdisk.mountpoint(letter)} mounted"
                + (
                    f" ({_format_bytes(status.free_bytes)} free of {_format_bytes(status.total_bytes)})"
                    if status.total_bytes
                    else ""
                )
            )
        else:
            self._selected_drive_label.setText(f"{imdisk.mountpoint(letter)} available")

        busy = self._worker_thread is not None
        self._create_btn.setEnabled(installed and not status.mounted and not busy)
        self._remove_btn.setEnabled(installed and status.mounted and not busy)

    def _install_imdisk(self) -> None:
        def action() -> str:
            imdisk.install(progress_cb=self._on_worker_progress)
            return "ImDisk Toolkit installed."

        self._start_worker(action, "Installing ImDisk Toolkit…")

    def _create_ramdisk(self) -> None:
        letter = self._selected_letter()
        size_gb = self._size_gb.value()
        if not imdisk.is_drive_available(letter):
            QMessageBox.warning(self, "Create RamDisk", f"Drive {imdisk.mountpoint(letter)} is already in use.")
            self._refresh()
            return

        def action() -> str:
            imdisk.create_ramdisk(letter, size_gb)
            return f"RamDisk {imdisk.root_path(letter)} created and set as AEP scratch path."

        self._pending_apply_letter = letter
        self._start_worker(action, f"Creating {imdisk.mountpoint(letter)}…")

    def _remove_ramdisk(self) -> None:
        letter = self._selected_letter()
        scratch = self._services.settings.get().paths.ramdisk_path or ""
        if self._running_jobs_use_drive(letter, scratch):
            QMessageBox.warning(
                self,
                "Remove RamDisk",
                f"AEP has running jobs using {imdisk.mountpoint(letter)}. Stop them before removing the ramdisk.",
            )
            return

        answer = QMessageBox.question(
            self,
            "Remove RamDisk",
            f"Remove ramdisk {imdisk.mountpoint(letter)}? Any files on it will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        def action() -> str:
            imdisk.remove_ramdisk(letter)
            return f"RamDisk {imdisk.mountpoint(letter)} removed."

        self._pending_clear_letter = letter if scratch and _path_on_drive(scratch, letter) else None
        self._start_worker(action, f"Removing {imdisk.mountpoint(letter)}…")

    def _start_worker(self, action: Callable[[], str], status: str) -> None:
        if self._worker_thread is not None:
            return
        self._install_status.setText(status)
        self._install_progress.setRange(0, 0)
        self._set_busy(True)

        thread = QThread(self)
        worker = _ImDiskWorker(action)
        worker.moveToThread(thread)
        worker.progress.connect(self._on_progress_signal)
        worker.finished_ok.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.finished_ok.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished_ok.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        thread.started.connect(worker.run)
        self._worker_thread = thread
        self._worker = worker
        thread.start()

    def _on_worker_progress(self, stage: str, progress: imdisk.DownloadProgress | None) -> None:
        worker = self._worker
        if worker is not None:
            worker.progress.emit(stage, progress)

    def _on_progress_signal(self, stage: str, progress: imdisk.DownloadProgress | None) -> None:
        labels = {
            "downloading": "Downloading installer…",
            "extracting": "Extracting installer…",
            "installing": "Running install.bat…",
            "done": "Done.",
        }
        self._install_status.setText(labels.get(stage, stage))
        if stage == "downloading" and progress and progress.total_bytes:
            self._install_progress.setRange(0, 100)
            self._install_progress.setValue(int(progress.bytes_read * 100 / progress.total_bytes))
        else:
            self._install_progress.setRange(0, 0)

    def _on_worker_finished(self, message: str) -> None:
        if self._pending_apply_letter is not None:
            self._apply_ramdisk_setting(self._pending_apply_letter)
        if self._pending_clear_letter is not None:
            self._clear_ramdisk_setting()
        self._pending_apply_letter = None
        self._pending_clear_letter = None
        self._install_status.setText(message)
        self._install_progress.setRange(0, 100)
        self._install_progress.setValue(100)
        QMessageBox.information(self, "RamDisk", message)

    def _on_worker_failed(self, message: str) -> None:
        self._pending_apply_letter = None
        self._pending_clear_letter = None
        self._install_status.setText("Failed")
        self._install_progress.setRange(0, 100)
        self._install_progress.setValue(0)
        QMessageBox.critical(self, "RamDisk operation failed", message)

    def _clear_worker(self) -> None:
        self._worker_thread = None
        self._worker = None
        self._set_busy(False)
        self._refresh()

    def _set_busy(self, busy: bool) -> None:
        if not hasattr(self, "_install_btn"):
            return
        self._install_btn.setEnabled(not busy and not imdisk.is_installed())
        self._letter_combo.setEnabled(not busy)
        self._size_gb.setEnabled(not busy)
        self._refresh_selected_drive_status()

    def _apply_ramdisk_setting(self, letter: str) -> None:
        settings = imdisk.apply_ramdisk_path(self._services.settings.get(), letter)
        self._services.settings.update(settings)

    def _clear_ramdisk_setting(self) -> None:
        settings = imdisk.apply_ramdisk_path(self._services.settings.get(), None)
        self._services.settings.update(settings)

    def _running_jobs_use_drive(self, letter: str, scratch: str) -> bool:
        if not scratch or not _path_on_drive(scratch, letter):
            return False
        return any(job.state == JobState.RUNNING for job in self._services.jobs.list_jobs())


def _imdisk_version_text() -> str:
    strings = pe_version_resource_strings(imdisk.ramdyn_exe())
    return strings[0] if strings else "Installed"


def _path_on_drive(path: str, letter: str) -> bool:
    return path.strip().upper().startswith(f"{imdisk.normalize_drive_letter(letter)}:")
