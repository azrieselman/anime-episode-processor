"""Main application window.

Layout:
+----------------------------------------------------+
| File  Tools  Help                                   |
+----------------------------------------------------+
| [Queue] [Stream Inspector] [Presets]               |
| [Benchmark] [Logs] [RamDisk] [Settings]            |
+--------+-------------------------------------------+
| Stack of views (driven by the side rail)           |
+--------+-------------------------------------------+
| Status: N queued, N running …  | tools hint (if any)|
+----------------------------------------------------+

We use a `QStackedWidget` driven by a left sidebar (`QListWidget`). This is the standard
Windows-app layout (Settings app, Visual Studio Installer, etc.) and Qt does it natively
without custom drawing.
"""

from __future__ import annotations

import logging
import platform
import sys
import threading
import urllib.parse
from functools import partial

from PySide6.QtCore import QObject, QSize, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from aep.adapters.verification import check_all, has_any_issues, has_blocking_issues
from aep.app.services import AppServices
from aep.app.tools_fetcher import missing_pins as missing_tool_pins
from aep.app.update_checker import UpdateCheckResult, check_for_updates
from aep.constants import (
    APP_DISPLAY_NAME,
    WINDOW_DEFAULT_HEIGHT,
    WINDOW_DEFAULT_WIDTH,
    WINDOW_MIN_HEIGHT,
    WINDOW_MIN_WIDTH,
)
from aep.gui import theme
from aep.gui.preset_design import PresetDesignerView
from aep.gui.views.benchmark_view import BenchmarkView
from aep.gui.views.logs_view import LogsView
from aep.gui.views.queue_view import QueueView
from aep.gui.views.ramdisk_view import RamDiskView
from aep.gui.views.settings_view import SettingsView
from aep.gui.views.stream_inspector_view import StreamInspectorView
from aep.gui.widgets.first_run_dialog import FirstRunDialog
from aep.gui.widgets.verify_tools_dialog import VerifyToolsDialog
from aep.version import __version__

log = logging.getLogger(__name__)


# GitHub issue tracker for "Help → Report Issue". Kept module-local because
# only the issue-reporter handler uses it; promoting to constants.py would
# imply that other surfaces should link there too, which they shouldn't
# (CHANGELOG / README cover that).
ISSUES_NEW_URL = "https://github.com/azrieselman/anime-episode-processor/issues/new"


class _UpdateCheckWorker(QObject):
    """Runs :func:`check_for_updates` on a background :class:`QThread`."""

    finished = Signal(object)

    def __init__(self, version: str) -> None:
        super().__init__()
        self._version = version

    def run(self) -> None:
        self.finished.emit(check_for_updates(self._version))


class MainWindow(QMainWindow):
    def __init__(self, services: AppServices) -> None:
        super().__init__()
        self._services = services
        self.setWindowTitle(f"{APP_DISPLAY_NAME} {__version__}")
        self.setWindowIcon(theme.load_window_icon())
        self.resize(WINDOW_DEFAULT_WIDTH, WINDOW_DEFAULT_HEIGHT)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        self._build_menu()
        self._build_central()
        self._build_status_bar()

        # Run a one-shot tool verification in the background after the window
        # is up. If anything is missing or version-mismatched, surface the
        # Verify Tools dialog so the user finds out before launching a job
        # rather than after a stage crashes deep inside the pipeline. We use
        # singleShot(0, ...) so __init__ doesn't block on subprocess probes.
        QTimer.singleShot(0, self._kickoff_startup_verification)

        # Re-emit broker events on the GUI thread via a queued signal in views;
        # most views poll services directly when needed. The QueueView subscribes
        # for live updates because it is the busiest.

    # ----- construction ---------------------------------------------

    def _build_menu(self) -> None:
        mb = self.menuBar()
        file_menu = mb.addMenu("&File")

        add_act = QAction("&Add Files…", self)
        add_act.setShortcut(QKeySequence.StandardKey.Open)
        add_act.triggered.connect(self._on_add_files)
        file_menu.addAction(add_act)

        add_folder_act = QAction("Add Fol&der…", self)
        add_folder_act.triggered.connect(self._on_add_folder)
        file_menu.addAction(add_folder_act)

        file_menu.addSeparator()

        quit_act = QAction("E&xit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        tools_menu = mb.addMenu("&Tools")
        verify_act = QAction("&Verify Tools", self)
        verify_act.triggered.connect(self._on_verify_tools)
        tools_menu.addAction(verify_act)

        help_menu = mb.addMenu("&Help")
        report_issue_act = QAction("&Report Issue…", self)
        report_issue_act.setStatusTip(
            "Open a pre-filled GitHub issue with version + OS + GPU details."
        )
        report_issue_act.triggered.connect(self._on_report_issue)
        help_menu.addAction(report_issue_act)

        self._update_check_action = QAction("Check for &Updates…", self)
        self._update_check_action.setStatusTip("Compare this build to the newest GitHub Release.")
        self._update_check_action.triggered.connect(self._on_check_for_updates)
        help_menu.addAction(self._update_check_action)

        help_menu.addSeparator()

        about_act = QAction("&About…", self)
        about_act.triggered.connect(self._on_about)
        help_menu.addAction(about_act)

        # Ctrl+1 … Ctrl+7 — switch views by stack index (skips sidebar separator).
        for i in range(7):
            act = QAction(self)
            act.setShortcut(QKeySequence(f"Ctrl+{i + 1}"))
            act.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
            act.triggered.connect(partial(self._select_stack_index, i))
            self.addAction(act)

    def _build_central(self) -> None:
        central = QWidget(self)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        rail = QWidget(central)
        rail.setObjectName("sidebarRail")
        rail.setFixedWidth(220)
        rail.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(0)

        brand = QFrame(rail)
        brand.setObjectName("sidebarBrandFrame")
        brand_outer = QVBoxLayout(brand)
        brand_outer.setContentsMargins(12, 14, 12, 14)
        brand_outer.setSpacing(0)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        brand_row.setContentsMargins(0, 0, 0, 0)
        brand_icon = QLabel(brand)
        brand_icon.setPixmap(theme.load_window_icon().pixmap(QSize(48, 48)))
        brand_icon.setFixedSize(QSize(48, 48))
        brand_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand_row.addWidget(brand_icon, 0, Qt.AlignmentFlag.AlignVCenter)
        brand_label = QLabel("Anime\nEpisode\nProcessor", brand)
        brand_label.setObjectName("sidebarBrand")
        brand_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        brand_row.addWidget(brand_label, 0, Qt.AlignmentFlag.AlignVCenter)

        brand_outer.addLayout(brand_row, stretch=0)
        brand_outer.setAlignment(brand_row, Qt.AlignmentFlag.AlignHCenter)
        rail_layout.addWidget(brand)

        self._sidebar = QListWidget(rail)
        self._sidebar.setObjectName("navSidebar")
        self._sidebar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._sidebar.setSpacing(2)
        self._sidebar.setIconSize(QSize(20, 20))
        self._sidebar.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        rail_layout.addWidget(self._sidebar, 1)

        self._stack = QStackedWidget(central)
        # Sidebar row → stack index (1:1; no spacer rows).
        self._sidebar_stack_map: list[int | None] = []

        self._queue_view = QueueView(self._services, parent=self)
        self._inspector_view = StreamInspectorView(self._services, parent=self)
        self._preset_designer_view = PresetDesignerView(self._services, parent=self)
        self._benchmark_view = BenchmarkView(self._services, parent=self)
        self._logs_view = LogsView(parent=self)
        self._ramdisk_view = RamDiskView(self._services, parent=self)
        self._settings_view = SettingsView(self._services, parent=self)

        self._queue_view.selection_changed.connect(self._on_job_selected)
        self._preset_designer_view.presets_changed.connect(self._queue_view.reload_presets)

        style = self.style()
        nav: list[tuple[str, QWidget, str, QStyle.StandardPixmap]] = [
            ("Queue", self._queue_view, "queue", QStyle.StandardPixmap.SP_DirOpenIcon),
            (
                "Stream Inspector",
                self._inspector_view,
                "stream-inspector",
                QStyle.StandardPixmap.SP_FileDialogInfoView,
            ),
            (
                "Preset Designer",
                self._preset_designer_view,
                "preset-designer",
                QStyle.StandardPixmap.SP_FileIcon,
            ),
            ("Benchmark", self._benchmark_view, "benchmark", QStyle.StandardPixmap.SP_ComputerIcon),
            ("Logs", self._logs_view, "logs", QStyle.StandardPixmap.SP_FileDialogContentsView),
            ("RamDisk", self._ramdisk_view, "ramdisk", QStyle.StandardPixmap.SP_DriveHDIcon),
            ("Settings", self._settings_view, "settings", QStyle.StandardPixmap.SP_FileDialogListView),
        ]

        for label, widget, slug, spix in nav:
            custom = theme.load_sidebar_nav_icon(slug)
            icon = custom if custom is not None else style.standardIcon(spix)
            QListWidgetItem(icon, label, self._sidebar)
            stack_idx = self._stack.count()
            self._stack.addWidget(widget)
            self._sidebar_stack_map.append(stack_idx)

        self._sidebar.currentRowChanged.connect(self._on_sidebar_row_changed)
        self._sidebar.setCurrentRow(0)

        root.addWidget(rail)
        root.addWidget(self._stack, 1)
        self.setCentralWidget(central)

    def _build_status_bar(self) -> None:
        sb = QStatusBar(self)
        self._status_label = QLabel("Ready")
        sb.addWidget(self._status_label, 1)
        self._tools_warning_label = QLabel("")
        self._tools_warning_label.setVisible(False)
        theme.style_attention_status_label(self._tools_warning_label)
        sb.addPermanentWidget(self._tools_warning_label)
        self.setStatusBar(sb)
        self._queue_view.counts_changed.connect(self._update_status)

    # ----- handlers -------------------------------------------------

    def _on_sidebar_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._sidebar_stack_map):
            return
        stack_idx = self._sidebar_stack_map[row]
        if stack_idx is None:
            # Separator clicked (shouldn't normally happen); snap back.
            prev = self._stack.currentIndex()
            for i, mapped in enumerate(self._sidebar_stack_map):
                if mapped == prev:
                    self._sidebar.blockSignals(True)
                    self._sidebar.setCurrentRow(i)
                    self._sidebar.blockSignals(False)
                    return
            return
        self._stack.setCurrentIndex(stack_idx)

    def _select_stack_index(self, stack_idx: int) -> None:
        """Select the sidebar row that maps to ``stack_idx`` (Ctrl+1…Ctrl+7)."""
        for row, mapped in enumerate(self._sidebar_stack_map):
            if mapped == stack_idx:
                self._sidebar.setCurrentRow(row)
                return

    def _sync_tools_warning_banner(self) -> None:
        try:
            show = has_any_issues(check_all())
        except Exception:
            log.exception("tools re-check after Verify Tools failed")
            show = True
        if show:
            self._tools_warning_label.setText("Tools need attention — use Tools → Verify Tools")
            self._tools_warning_label.setVisible(True)
        else:
            self._tools_warning_label.setVisible(False)

    def _on_add_files(self) -> None:
        self._queue_view.prompt_add_files()

    def _on_add_folder(self) -> None:
        self._queue_view.prompt_add_folder()

    def _on_verify_tools(self) -> None:
        dlg = VerifyToolsDialog(self)
        self._status_label.setText("Tool verification…")
        log.info("Tools menu → verify")
        dlg.finished.connect(lambda *_: self._sync_tools_warning_banner())
        dlg.exec()
        self._status_label.setText("Ready")
        self._sync_tools_warning_banner()

    def _kickoff_startup_verification(self) -> None:
        """Probe pinned tools on a worker thread; nag the user if anything is off.

        Lives behind a one-shot guard so it runs at most once per process even
        if QTimer ever fires twice (e.g. event-loop oddities during testing).

        Routing:
          * If any required tool is missing entirely (blocking), open the
            modal :class:`FirstRunDialog` to download + install everything.
          * If everything resolves but versions drift, open the existing
            non-modal :class:`VerifyToolsDialog` so the user knows.
          * If everything is clean, do nothing.
        """
        if getattr(self, "_startup_verify_done", False):
            return
        self._startup_verify_done = True

        def worker() -> None:
            try:
                statuses = check_all()
            except Exception:
                log.exception("startup tool verification failed unexpectedly")
                return
            log.info(
                "startup tool verification: %s",
                ", ".join(f"{s.tool_id}={s.status}" for s in statuses),
            )
            if not has_any_issues(statuses):
                return
            blocking = has_blocking_issues(statuses)
            # This worker runs on a plain Python thread (no Qt event loop), so
            # singleShot without a QObject context may never deliver. Bind the
            # callback to the main-window QObject to guarantee GUI-thread dispatch.
            QTimer.singleShot(
                0,
                self,
                lambda blocking=blocking: self._show_startup_verification_dialog(blocking=blocking),
            )

        threading.Thread(
            target=worker,
            name="aep-startup-verify",
            daemon=True,
        ).start()

    def _show_startup_verification_dialog(self, *, blocking: bool) -> None:
        """Surface tool issues: missing → first-run fetch dialog, mismatched → Verify Tools."""
        if getattr(self, "_startup_dialog_open", False):
            log.info("startup verification dialog already open; skipping duplicate open")
            return

        if blocking and missing_tool_pins():
            log.info("startup verification: missing tools — opening FirstRunDialog")
            self._startup_dialog_open = True
            self._tools_warning_label.setText(
                "First-run setup: required tools are missing"
            )
            self._tools_warning_label.setVisible(True)
            try:
                installed = FirstRunDialog.run_if_needed(self)
                self._sync_tools_warning_banner()
                if installed:
                    log.info("first-run fetch completed; pipeline ready")
                else:
                    log.warning("first-run fetch did not complete; pipeline may not run")
            finally:
                self._startup_dialog_open = False
            return

        log.info("startup verification surfaced version drift; opening Verify Tools dialog")
        existing = getattr(self, "_startup_verify_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

        self._startup_dialog_open = True
        self._tools_warning_label.setText(
            "Startup check: tools need attention — see Verify Tools"
        )
        self._tools_warning_label.setVisible(True)

        dlg = VerifyToolsDialog(self)
        dlg.setModal(False)
        dlg.finished.connect(lambda *_: self._sync_tools_warning_banner())
        dlg.finished.connect(lambda *_: setattr(self, "_startup_dialog_open", False))
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

        self._startup_verify_dialog = dlg

    def _on_check_for_updates(self) -> None:
        if getattr(self, "_update_check_thread", None) is not None and self._update_check_thread.isRunning():
            log.info("Check for Updates: already running")
            return

        self._status_label.setText("Checking for updates…")
        self._update_check_action.setEnabled(False)

        thread = QThread(self)
        worker = _UpdateCheckWorker(__version__)
        worker.moveToThread(thread)

        worker.finished.connect(self._on_update_check_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)

        self._update_check_thread = thread
        thread.start()

    def _on_update_check_finished(self, result: UpdateCheckResult) -> None:
        """GUI-thread slot: present the update check outcome."""

        self._status_label.setText("Ready")
        self._update_check_action.setEnabled(True)

        if result.error:
            QMessageBox.warning(self, "Update check", result.error)
            return

        if result.latest_version is None:
            QMessageBox.information(
                self,
                "Update check",
                "Could not find a release with a readable version tag.",
            )
            return

        if result.is_update_available:
            choice = QMessageBox.question(
                self,
                "Update available",
                result.user_message_summary()
                + "\n\nOpen the GitHub release page in your browser?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if choice == QMessageBox.StandardButton.Yes and result.release_url:
                QDesktopServices.openUrl(QUrl(result.release_url))
        else:
            QMessageBox.information(self, "Update check", result.user_message_summary())

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_DISPLAY_NAME}",
            f"{APP_DISPLAY_NAME} v{__version__}\n\n"
            "Local Windows app for upscaling, interpolating, and re-encoding\n"
            "anime episodes with full subtitle/chapter/attachment preservation.\n\n"
            "Runs entirely on your PC.",
        )

    def _on_report_issue(self) -> None:
        """Open a templated GitHub issue URL with version + OS + GPU pre-filled.

        We assemble the body locally and let GitHub's ``issues/new`` form
        accept it via the ``body=`` querystring. Worst case (e.g. browser
        blocks long URLs) the user still sees the prefilled body to copy
        manually — :class:`QDesktopServices.openUrl` always returns synchronously,
        and we fall back to a clipboard copy with a status hint if it can't
        open.
        """
        body = self._build_issue_body()
        params = {
            "title": "[bug] <one-line summary>",
            "body": body,
            "labels": "bug,beta",
        }
        url = ISSUES_NEW_URL + "?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        log.info("Help → Report Issue: opening %s", ISSUES_NEW_URL)
        if not QDesktopServices.openUrl(QUrl(url)):
            QMessageBox.warning(
                self,
                "Could not open browser",
                f"Tried to open:\n\n{ISSUES_NEW_URL}\n\nThe pre-filled body has "
                "been copied to your clipboard so you can paste it after "
                "opening the page manually.",
            )
            from PySide6.QtWidgets import QApplication
            QApplication.clipboard().setText(body)

    @staticmethod
    def _build_issue_body() -> str:
        """Pre-filled markdown body for ``Help → Report Issue``.

        Kept narrow: app version, Python version, OS build, primary GPU.
        We deliberately do *not* try to dump settings.json / preset YAML —
        those can contain user filesystem paths the user may not want in a
        public issue. The user can attach those manually if relevant.
        """
        # Lazy-import the GPU probe so a missing nvidia-smi during normal
        # window construction never blocks Help menu rendering. The probe
        # itself is fast (<1s) and only runs when the user actually invokes
        # this action.
        try:
            from aep.adapters.nvidia import probe_nvidia
            probe = probe_nvidia()
            primary = probe.primary
            if primary is not None:
                gpu_str = (
                    f"{primary.name} (driver {primary.driver_version}, "
                    f"{primary.vram_total_mib} MiB VRAM)"
                )
            else:
                gpu_str = "no NVIDIA GPU detected"
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("GPU probe failed during Report Issue: %s", exc)
            gpu_str = "unknown (GPU probe failed)"

        py = "{}.{}.{}".format(*sys.version_info[:3])
        os_release = f"{platform.system()} {platform.release()} ({platform.version()})"

        return (
            "<!-- Please describe what happened and what you expected. -->\n\n"
            "## What happened\n\n"
            "_Describe the bug here._\n\n"
            "## Steps to reproduce\n\n"
            "1. \n"
            "2. \n"
            "3. \n\n"
            "## Logs\n\n"
            "_If applicable, attach `aep.log` from your runtime dir "
            "(`%LOCALAPPDATA%\\AEP\\logs\\aep.log` on a packaged install)._\n\n"
            "## Environment\n\n"
            f"- AEP version: `{__version__}`\n"
            f"- OS: `{os_release}`\n"
            f"- Python: `{py}` ({platform.python_implementation()})\n"
            f"- GPU: `{gpu_str}`\n"
        )

    def _on_job_selected(self, job_id: str | None) -> None:
        self._inspector_view.set_job(job_id)

    def _update_status(self, queued: int, running: int, completed: int, failed: int) -> None:
        queue_elapsed_s = self._services.jobs.get_queue_active_elapsed_s()
        self._status_label.setText(
            f"{queued} queued · {running} running · {completed} completed · {failed} failed"
            f" · queue active {self._format_elapsed(queue_elapsed_s)}"
        )

    @staticmethod
    def _format_elapsed(seconds: float | None) -> str:
        if seconds is None:
            return "--"
        total_seconds = max(0, int(seconds))
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"

    # ----- shutdown -------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        log.info("main window closing")
        super().closeEvent(event)
