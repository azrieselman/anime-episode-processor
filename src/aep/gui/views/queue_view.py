"""Queue view: drop area + table of jobs.

Refreshes on a `QTimer` rather than tying every cell to a broker signal — at the scale
of "tens of jobs", a 250 ms refresh is invisible to the user and dramatically simpler.
"""

from __future__ import annotations

import logging
import re
import time
from functools import partial
from pathlib import Path
from typing import Literal

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aep.app.services import AppServices
from aep.gui import theme
from aep.gui.widgets.drop_area import DropArea
from aep.jobs.models import Job, JobState
from aep.jobs.queue import QueuedDispatchOrder

log = logging.getLogger(__name__)


COLUMNS = ["File", "Job ID", "Preset", "State", "Stage", "Batches", "Runtime", "Error"]

_FILE_SORT_MODE = Literal["queue", "name_asc", "name_desc"]

_DEFAULT_COLUMN_WIDTHS = [280, 240, 140, 88, 140, 88, 88, 220]

_COL_JOB_ID = 1

# After Pause, wait until workers leave RUNNING before showing the idle banner.
_PAUSE_COMPLETION_POLL_MS = 150
_PAUSE_COMPLETION_TIMEOUT_S = 600.0

_STAGE_DISPLAY_NAMES: dict[str, str] = {
    "00_probe": "Probing",
    "01_plan": "Planning",
    "02_sample_bench": "Benchmarking",
    "03_scene_detect": "Scene Detection",
    "04_decode_serve": "Decoding",
    "05_upscale": "Upscaling",
    "06_interpolate": "Interpolating",
    "07_postprocess": "Post-Processing",
    "08_encode": "Encoding",
    "09_mux": "Muxing",
    "10_validate": "Validating",
}


def _display_stage_name(stage_name: str | None) -> str:
    if not stage_name:
        return ""
    if stage_name in _STAGE_DISPLAY_NAMES:
        return _STAGE_DISPLAY_NAMES[stage_name]
    normalized = re.sub(r"^\d+_", "", stage_name)
    normalized = normalized.replace("_", " ").strip()
    if not normalized:
        return stage_name
    return normalized.title()


def _job_stage_display(job: Job) -> str:
    """Stage column: live progress uses ``current_stage``; failed jobs show ``last_failed_stage``."""
    if job.state == JobState.FAILED and job.last_failed_stage:
        return _display_stage_name(job.last_failed_stage)
    return _display_stage_name(job.current_stage)


def _display_batch_progress(job: Job) -> str:
    plan = job.plan if isinstance(job.plan, dict) else {}
    raw = plan.get("batch_progress")
    if not isinstance(raw, dict):
        return "--"
    done = raw.get("done")
    total = raw.get("total")
    if not isinstance(done, int) or not isinstance(total, int) or total <= 0:
        return "--"
    done = max(0, min(done, total))
    return f"{done}/{total}"


class QueueView(QWidget):
    selection_changed = Signal(object)        # str | None (job id)
    counts_changed = Signal(int, int, int, int)  # queued, running, completed, failed

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self._services = services
        self._file_sort_mode: _FILE_SORT_MODE = "queue"
        self._build_ui()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(250)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start()
        self._sync_dispatch_order_with_broker()
        self._pause_watch_deadline: float | None = None

    def _dispatch_order_for_file_sort(self) -> QueuedDispatchOrder:
        if self._file_sort_mode == "queue":
            return "fifo"
        if self._file_sort_mode == "name_asc":
            return "name_asc"
        return "name_desc"

    def _sync_dispatch_order_with_broker(self) -> None:
        self._services.jobs.set_queued_dispatch_order(self._dispatch_order_for_file_sort())

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        root.addWidget(theme.make_page_title_label("Queue", self))

        self._drop = DropArea(self)
        self._drop.paths_dropped.connect(self._on_paths_dropped)
        root.addWidget(self._drop)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self._add_files_btn = QPushButton("Add Files…")
        self._add_files_btn.clicked.connect(self.prompt_add_files)
        self._add_folder_btn = QPushButton("Add Folder…")
        self._add_folder_btn.clicked.connect(self.prompt_add_folder)
        toolbar.addWidget(self._add_files_btn)
        toolbar.addWidget(self._add_folder_btn)

        toolbar.addSpacing(16)
        toolbar.addWidget(QLabel("Preset for new jobs:"))
        self._preset_combo = QComboBox()
        self._reload_presets()
        toolbar.addWidget(self._preset_combo)

        toolbar.addStretch(1)

        # Single Start/Pause control: toggles queue dispatch and pauses or
        # resumes every in-flight job (running workers hold pipeline contexts).
        self._queue_toggle_btn = QPushButton("Start")
        self._queue_toggle_btn.setStyleSheet("font-weight: 600;")
        self._queue_toggle_btn.clicked.connect(self._on_queue_toggle_clicked)
        toolbar.addWidget(self._queue_toggle_btn)
        toolbar.addSpacing(12)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        self._resume_btn = QPushButton("Resume")
        self._resume_btn.clicked.connect(self._on_resume_clicked)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove_clicked)
        self._clear_queue_btn = QPushButton("Clear Queue…")
        self._clear_queue_btn.clicked.connect(self._on_clear_queue_clicked)
        self._retry_btn = QPushButton("Retry Failed")
        self._retry_btn.clicked.connect(self._on_retry_clicked)
        for b in (
            self._cancel_btn,
            self._resume_btn,
            self._retry_btn,
            self._remove_btn,
            self._clear_queue_btn,
        ):
            toolbar.addWidget(b)

        root.addLayout(toolbar)

        # Status line under the toolbar shows whether dispatch is paused.
        # We keep it minimal to avoid stealing vertical space from the table.
        self._queue_status_label = QLabel("")
        theme.style_attention_status_label(self._queue_status_label, italic=True)
        root.addWidget(self._queue_status_label)
        self._refresh_queue_toggle()

        self._table = QTableWidget(0, len(COLUMNS), self)
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._table.setSortingEnabled(False)
        h = self._table.horizontalHeader()
        h.setStretchLastSection(False)
        for col in range(len(COLUMNS)):
            h.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
        for col, w in enumerate(_DEFAULT_COLUMN_WIDTHS):
            h.resizeSection(col, w)
        h.sectionClicked.connect(self._on_queue_header_section_clicked)
        self._update_file_sort_indicator()
        self._sync_job_id_column_visibility()
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        root.addWidget(self._table, 1)

    def reload_presets(self) -> None:
        """Refresh preset combo from disk (e.g. after Preset Designer save)."""
        self._reload_presets()

    def _reload_presets(self) -> None:
        self._preset_combo.clear()
        for p in self._services.presets.list():
            self._preset_combo.addItem(f"{p.meta.name}", p.meta.id)
        # Default selection to settings.last_used_preset if found.
        try:
            last = self._services.settings.get().last_used_preset
            idx = self._preset_combo.findData(last)
            if idx >= 0:
                self._preset_combo.setCurrentIndex(idx)
        except Exception:
            pass

    # ----- input ---------------------------------------------------

    def prompt_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add video files", "",
            "Video files (*.mkv *.mp4 *.m4v *.webm *.avi *.mov *.ts *.m2ts);;All files (*.*)",
        )
        if files:
            self._enqueue_paths([Path(f) for f in files])

    def prompt_add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Add folder")
        if folder:
            from aep.gui.widgets.drop_area import _expand_paths
            paths = _expand_paths([Path(folder)])
            self._enqueue_paths(paths)

    def _on_paths_dropped(self, paths: list[Path]) -> None:
        self._enqueue_paths(paths)

    def _enqueue_paths(self, paths: list[Path]) -> None:
        preset_id = self._preset_combo.currentData() or "anime_balanced"
        try:
            confirm_overwrite = bool(self._services.settings.get().general.confirm_overwrite)
        except Exception:
            confirm_overwrite = True
        apply_to_all: str | None = None
        for p in paths:
            decision = apply_to_all
            if confirm_overwrite and decision is None:
                try:
                    out_path = self._services.jobs.preview_output_path(p, preset_id)
                except Exception:
                    out_path = None
                if out_path is not None and out_path.exists():
                    decision = self._prompt_overwrite(p, out_path, multi=len(paths) > 1)
                    if decision in ("yes_all", "no_all"):
                        apply_to_all = decision
                    if decision in ("no", "no_all", "cancel"):
                        if decision == "cancel":
                            break
                        continue
            try:
                self._services.jobs.enqueue(p, preset_id)
            except Exception as exc:
                log.exception("failed to enqueue %s: %s", p, exc)
        # Persist last-used preset
        try:
            s = self._services.settings.get()
            s.last_used_preset = preset_id
            self._services.settings.update(s)
        except Exception:
            pass

    def _prompt_overwrite(self, source: Path, out_path: Path, *, multi: bool) -> str:
        """Ask the user whether to overwrite an existing output file.

        Returns one of: ``"yes"``, ``"no"``, ``"yes_all"``, ``"no_all"``,
        ``"cancel"``. ``yes_all``/``no_all`` are only offered when more than one
        file is being added in this batch.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Output already exists")
        box.setText(
            f"The output for <b>{source.name}</b> already exists:<br><br>"
            f"<code>{out_path}</code><br><br>"
            "Overwrite when this job runs?"
        )
        yes_btn = box.addButton("Overwrite", QMessageBox.ButtonRole.AcceptRole)
        no_btn = box.addButton("Skip", QMessageBox.ButtonRole.RejectRole)
        yes_all_btn = None
        no_all_btn = None
        if multi:
            yes_all_btn = box.addButton(
                "Overwrite All", QMessageBox.ButtonRole.AcceptRole,
            )
            no_all_btn = box.addButton(
                "Skip All", QMessageBox.ButtonRole.RejectRole,
            )
        cancel_btn = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(no_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is yes_btn:
            return "yes"
        if clicked is yes_all_btn:
            return "yes_all"
        if clicked is no_all_btn:
            return "no_all"
        if clicked is cancel_btn:
            return "cancel"
        return "no"

    # ----- table refresh ------------------------------------------

    def _refresh(self) -> None:
        self._sync_job_id_column_visibility()
        # Keep the queue-toggle UI in sync in case the broker state changes
        # from elsewhere (e.g. settings.auto_start_jobs at startup).
        self._refresh_queue_toggle()
        jobs = list(self._services.jobs.list_jobs())
        jobs = self._ordered_jobs(jobs)
        # Track currently selected job id to preserve selection across rebuilds.
        sel_id = self._current_selected_job_id()

        self._table.setRowCount(len(jobs))
        counts = {state: 0 for state in JobState}
        for row, job in enumerate(jobs):
            counts[job.state] = counts.get(job.state, 0) + 1
            self._fill_row(row, job)

        # Restore selection.
        if sel_id is not None:
            for row in range(self._table.rowCount()):
                if self._table.item(row, 0).data(Qt.ItemDataRole.UserRole) == sel_id:
                    self._table.selectRow(row)
                    break

        self.counts_changed.emit(
            counts.get(JobState.QUEUED, 0),
            counts.get(JobState.RUNNING, 0),
            counts.get(JobState.COMPLETED, 0),
            counts.get(JobState.FAILED, 0),
        )
        self._clear_queue_btn.setEnabled(len(jobs) > 0)

    def _ordered_jobs(self, jobs: list[Job]) -> list[Job]:
        if self._file_sort_mode == "queue":
            return jobs
        key = lambda j: (Path(j.source_path).name.lower(), j.id)
        if self._file_sort_mode == "name_asc":
            return sorted(jobs, key=key)
        return sorted(jobs, key=key, reverse=True)

    def _update_file_sort_indicator(self) -> None:
        h = self._table.horizontalHeader()
        if self._file_sort_mode == "queue":
            h.setSortIndicatorShown(False)
        elif self._file_sort_mode == "name_asc":
            h.setSortIndicatorShown(True)
            h.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        else:
            h.setSortIndicatorShown(True)
            h.setSortIndicator(0, Qt.SortOrder.DescendingOrder)

    def _sync_job_id_column_visibility(self) -> None:
        try:
            show = bool(self._services.settings.get().general.show_queue_job_id_column)
        except Exception:
            show = False
        self._table.setColumnHidden(_COL_JOB_ID, not show)

    def _on_queue_header_section_clicked(self, logical_index: int) -> None:
        if logical_index != 0:
            return
        if self._file_sort_mode == "queue":
            self._file_sort_mode = "name_asc"
        elif self._file_sort_mode == "name_asc":
            self._file_sort_mode = "name_desc"
        else:
            self._file_sort_mode = "queue"
        self._sync_dispatch_order_with_broker()
        self._update_file_sort_indicator()
        self._refresh()

    def _fill_row(self, row: int, job: Job) -> None:
        file_item = QTableWidgetItem(Path(job.source_path).name)
        file_item.setToolTip(job.source_path)
        file_item.setData(Qt.ItemDataRole.UserRole, job.id)

        job_id_item = QTableWidgetItem(job.id)
        job_id_item.setToolTip(job.id)

        preset_item = QTableWidgetItem(job.preset_id)
        state_item = QTableWidgetItem(job.state.value)
        stage_item = QTableWidgetItem(_job_stage_display(job))
        progress_item = QTableWidgetItem(_display_batch_progress(job))
        runtime_s = self._services.jobs.get_job_active_elapsed_s(job.id)
        runtime_item = QTableWidgetItem(self._format_elapsed(runtime_s))
        error_item = QTableWidgetItem(job.error or "")

        self._table.setItem(row, 0, file_item)
        self._table.setItem(row, _COL_JOB_ID, job_id_item)
        self._table.setItem(row, 2, preset_item)
        self._table.setItem(row, 3, state_item)
        self._table.setItem(row, 4, stage_item)
        self._table.setItem(row, 5, progress_item)
        self._table.setItem(row, 6, runtime_item)
        self._table.setItem(row, 7, error_item)

    # ----- selection / actions ------------------------------------

    def _current_selected_job_id(self) -> str | None:
        items = self._table.selectedItems()
        if not items:
            return None
        return self._table.item(items[0].row(), 0).data(Qt.ItemDataRole.UserRole)

    def _on_selection_changed(self) -> None:
        self.selection_changed.emit(self._current_selected_job_id())

    def _on_cancel_clicked(self) -> None:
        jid = self._current_selected_job_id()
        if jid:
            self._services.jobs.cancel(jid)

    def _on_resume_clicked(self) -> None:
        jid = self._current_selected_job_id()
        if jid:
            self._services.jobs.resume(jid)

    def _on_remove_clicked(self) -> None:
        jid = self._current_selected_job_id()
        if jid:
            self._services.jobs.remove(jid)

    def _on_clear_queue_clicked(self) -> None:
        jobs = self._services.jobs.list_jobs()
        if not jobs:
            return
        ret = QMessageBox.question(
            self,
            "Clear queue?",
            "Remove all jobs and delete their work folders (and ramdisk artifacts)? "
            "Running jobs will be stopped and returned to the queue.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ret != QMessageBox.StandardButton.Yes:
            return
        try:
            self._services.jobs.clear_queue()
        except Exception as exc:
            log.exception("clear_queue failed: %s", exc)
            QMessageBox.warning(self, "Clear queue failed", str(exc))

    def _on_retry_clicked(self) -> None:
        jid = self._current_selected_job_id()
        if not jid:
            QMessageBox.information(
                self,
                "Retry Failed",
                "Select a job in the queue first.",
            )
            return
        job = self._services.jobs.get(jid)
        ret = self._services.jobs.retry_failed(jid)
        if ret is None:
            st = job.state.value if job else "missing"
            QMessageBox.warning(
                self,
                "Cannot retry",
                "Only jobs in the Failed state can be retried. "
                f"This job is: {st}.",
            )
            return
        if self._services.jobs.is_queue_paused():
            QMessageBox.information(
                self,
                "Job re-queued",
                "The failed job was returned to the queue. "
                "Click **Start** so the dispatcher can run it "
                "(the queue is paused after jobs finish or fail).",
            )

    # ----- queue-level start/pause --------------------------------------

    def _on_queue_toggle_clicked(self) -> None:
        """Toggle queue dispatch plus all active pipeline workers."""
        if self._services.jobs.is_queue_paused():
            jobs = self._services.jobs.list_jobs()
            # Checkpoint pauses (`PausedError`) leave rows PAUSED until toolbar
            # Resume or this toggle. Re-queue those *before* ``start_queue`` so
            # dispatch is never briefly unpaused with zero QUEUED rows (the
            # dispatcher would sleep and jobs that only show as queued after
            # resume could appear stuck).
            for job in jobs:
                if job.state == JobState.PAUSED:
                    self._services.jobs.resume(job.id)
            self._services.jobs.start_queue()
            self._pause_watch_deadline = None
            self._queue_toggle_btn.setEnabled(True)
            self._refresh_queue_toggle()
        else:
            jobs = self._services.jobs.list_jobs()
            running_ids = frozenset(j.id for j in jobs if j.state == JobState.RUNNING)
            self._services.jobs.pause_queue()
            for jid in running_ids:
                self._services.jobs.pause(jid)
            if running_ids:
                self._pause_watch_deadline = time.monotonic() + _PAUSE_COMPLETION_TIMEOUT_S
                self._queue_toggle_btn.setEnabled(False)
                self._queue_status_label.setText(
                    "Pausing — waiting for workers to reach a safe stop…",
                )
                QTimer.singleShot(
                    _PAUSE_COMPLETION_POLL_MS,
                    partial(self._finish_pause_when_workers_idle, running_ids),
                )
            else:
                self._refresh_queue_toggle()

    def _finish_pause_when_workers_idle(self, pending: frozenset[str]) -> None:
        """Complete queue Pause UX once jobs leave RUNNING or a timeout hits."""
        deadline = self._pause_watch_deadline
        now = time.monotonic()
        timed_out = deadline is not None and now >= deadline
        try:
            still_running = self._services.jobs.any_job_in_ids_running(pending)
        except Exception:
            still_running = False

        if still_running and not timed_out:
            QTimer.singleShot(
                _PAUSE_COMPLETION_POLL_MS,
                partial(self._finish_pause_when_workers_idle, pending),
            )
            return

        self._pause_watch_deadline = None
        self._queue_toggle_btn.setEnabled(True)
        self._refresh_queue_toggle()
        if still_running and timed_out:
            base = self._queue_status_label.text()
            self._queue_status_label.setText(
                base + " Some workers are still finishing a long step; states update when they stop.",
            )

    def _refresh_queue_toggle(self) -> None:
        """Sync the toggle button label + status text to broker state."""
        try:
            paused = self._services.jobs.is_queue_paused()
            queue_elapsed_s = self._services.jobs.get_queue_active_elapsed_s()
        except Exception:
            paused = True
            queue_elapsed_s = 0.0
        queue_elapsed = self._format_elapsed(queue_elapsed_s)
        if paused:
            self._queue_toggle_btn.setText("Start")
            self._queue_status_label.setText(
                f"Paused — active runtime {queue_elapsed}. Click Start to resume dispatch and workers.",
            )
        else:
            self._queue_toggle_btn.setText("Pause")
            self._queue_status_label.setText(f"Running — active runtime {queue_elapsed}.")

    @staticmethod
    def _format_elapsed(seconds: float | None) -> str:
        if seconds is None:
            return "--"
        total_seconds = max(0, int(seconds))
        hours, rem = divmod(total_seconds, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02}:{minutes:02}:{secs:02}"
