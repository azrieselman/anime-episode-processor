"""Benchmark tab for quick segment-level pipeline comparisons."""

from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from aep.app.services import AppServices
from aep.bench.models import BenchmarkRequest, BenchmarkResult
from aep.gui import theme


class _BenchmarkWorker(QObject):
    progress = Signal(str)
    stage_event = Signal(object)
    ffmpeg_line = Signal(str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        services: AppServices,
        request: BenchmarkRequest,
        cancel_event: threading.Event,
    ) -> None:
        super().__init__()
        self._services = services
        self._request = request
        self._cancel_event = cancel_event

    def run(self) -> None:
        try:
            self.progress.emit("Benchmark started…")

            def _on_event(event) -> None:  # type: ignore[no-untyped-def]
                self.stage_event.emit(event)
                extra = event.extra if isinstance(event.extra, dict) else {}
                line = extra.get("ffmpeg_line")
                if isinstance(line, str) and line:
                    self.ffmpeg_line.emit(line)

            result = self._services.benchmark.run(
                self._request,
                cancel_event=self._cancel_event,
                on_event=_on_event,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class BenchmarkView(QWidget):
    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self._services = services
        self._worker_thread: QThread | None = None
        self._worker: _BenchmarkWorker | None = None
        self._cancel_event: threading.Event | None = None
        self._last_result: BenchmarkResult | None = None
        self._recent_results: list[BenchmarkResult] = []
        self._source_duration_s: float | None = None
        self._vmaf_available = False
        self._build()
        self._reload_presets()
        self._refresh_vmaf_banner()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(theme.make_page_title_label("Benchmark", self))
        self._vmaf_banner = QLabel("")
        theme.style_muted_detail_label(self._vmaf_banner, small=True)
        root.addWidget(self._vmaf_banner)

        input_group = QGroupBox("Input")
        input_form = QFormLayout(input_group)
        source_row = QHBoxLayout()
        self._source_edit = QLineEdit()
        self._source_edit.setPlaceholderText("Select a source video file")
        self._source_edit.editingFinished.connect(self._probe_source_duration)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_source)
        source_row.addWidget(self._source_edit, 1)
        source_row.addWidget(browse)
        input_form.addRow("Source file:", source_row)
        self._start_spin = QDoubleSpinBox()
        self._start_spin.setRange(0.0, 86_400.0)
        self._start_spin.setDecimals(3)
        self._start_spin.setSuffix(" s")
        self._start_spin.valueChanged.connect(self._sync_segment_limits)
        input_form.addRow("Segment start:", self._start_spin)
        self._duration_spin = QDoubleSpinBox()
        self._duration_spin.setRange(0.1, 3600.0)
        self._duration_spin.setDecimals(3)
        self._duration_spin.setValue(30.0)
        self._duration_spin.setSuffix(" s")
        input_form.addRow("Segment duration:", self._duration_spin)
        root.addWidget(input_group)

        config_group = QGroupBox("Configuration")
        config_form = QFormLayout(config_group)
        self._preset_combo = QComboBox()
        config_form.addRow("Preset:", self._preset_combo)
        self._scope_combo = QComboBox()
        self._scope_combo.addItem("Full pipeline", "full")
        self._scope_combo.addItem("Encode only", "encode_only")
        config_form.addRow("Scope:", self._scope_combo)
        self._verbose_ffmpeg = QCheckBox("Verbose FFmpeg output")
        self._verbose_ffmpeg.setChecked(True)
        config_form.addRow("", self._verbose_ffmpeg)
        self._compute_vmaf = QCheckBox("Compute VMAF for encoded segment")
        self._compute_vmaf.setChecked(True)
        config_form.addRow("", self._compute_vmaf)
        root.addWidget(config_group)

        actions = QHBoxLayout()
        self._run_btn = QPushButton("Run Benchmark")
        self._run_btn.clicked.connect(self._run_benchmark)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_benchmark)
        self._cancel_btn.setEnabled(False)
        self._export_btn = QPushButton("Export JSON…")
        self._export_btn.clicked.connect(self._export_result)
        self._export_btn.setEnabled(False)
        actions.addWidget(self._run_btn)
        actions.addWidget(self._cancel_btn)
        actions.addWidget(self._export_btn)
        actions.addStretch(1)
        root.addLayout(actions)

        self._status_label = QLabel("Ready.")
        theme.style_muted_detail_label(self._status_label)
        root.addWidget(self._status_label)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_summary_tab(), "Summary")
        self._tabs.addTab(self._build_stages_tab(), "Stages")
        self._tabs.addTab(self._build_encode_samples_tab(), "Encode Speed")
        self._tabs.addTab(self._build_ffmpeg_log_tab(), "FFmpeg Log")
        root.addWidget(self._tabs, 1)

        recent_group = QGroupBox("Recent Runs (session)")
        recent_layout = QVBoxLayout(recent_group)
        self._recent_table = QTableWidget(0, 6)
        self._recent_table.setHorizontalHeaderLabels(
            ["Run ID", "Preset", "Scope", "Duration (s)", "VMAF", "Proc Wall (s)"],
        )
        self._recent_table.verticalHeader().setVisible(False)
        self._recent_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._recent_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._recent_table.itemSelectionChanged.connect(self._update_delete_run_btn)
        recent_layout.addWidget(self._recent_table)
        recent_actions = QHBoxLayout()
        self._delete_run_btn = QPushButton("Delete run data…")
        self._delete_run_btn.setEnabled(False)
        self._delete_run_btn.clicked.connect(self._delete_selected_run)
        recent_actions.addStretch(1)
        recent_actions.addWidget(self._delete_run_btn)
        recent_layout.addLayout(recent_actions)
        root.addWidget(recent_group)

    def _build_summary_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._summary_run_id = QLabel("--")
        self._summary_proc_calls = QLabel("--")
        self._summary_proc_wall = QLabel("--")
        self._summary_vmaf = QLabel("--")
        self._summary_encoded = QLabel("--")
        self._summary_warnings = QLabel("--")
        theme.style_muted_detail_label(self._summary_warnings, small=True)
        form.addRow("Run ID:", self._summary_run_id)
        form.addRow("Process calls:", self._summary_proc_calls)
        form.addRow("Process wall time:", self._summary_proc_wall)
        form.addRow("VMAF:", self._summary_vmaf)
        form.addRow("Encoded segment:", self._summary_encoded)
        form.addRow("Warnings:", self._summary_warnings)
        return w

    def _build_stages_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._stages_table = QTableWidget(0, 5)
        self._stages_table.setHorizontalHeaderLabels(
            ["Stage", "Runs", "Duration (s)", "Proc Calls", "Proc Wall (s)"],
        )
        self._stages_table.verticalHeader().setVisible(False)
        self._stages_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._stages_table)
        return w

    def _build_encode_samples_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._encode_table = QTableWidget(0, 5)
        self._encode_table.setHorizontalHeaderLabels(
            ["Frame", "FPS", "Speed (x)", "Out Time (us)", "Progress"],
        )
        self._encode_table.verticalHeader().setVisible(False)
        self._encode_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._encode_table)
        return w

    def _build_ffmpeg_log_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self._ffmpeg_log = QPlainTextEdit()
        self._ffmpeg_log.setReadOnly(True)
        self._ffmpeg_log.setMaximumBlockCount(50_000)
        layout.addWidget(self._ffmpeg_log)
        return w

    def _refresh_vmaf_banner(self) -> None:
        try:
            available = self._services.benchmark.probe_vmaf_available()
        except Exception:
            available = False
        self._vmaf_available = available
        self._compute_vmaf.setEnabled(available)
        if available:
            self._vmaf_banner.setText("VMAF is available (libvmaf detected in ffmpeg).")
        else:
            self._vmaf_banner.setText(
                "VMAF is unavailable: libvmaf filter was not found in ffmpeg. Benchmark can still run.",
            )
            self._compute_vmaf.setChecked(False)

    def _reload_presets(self) -> None:
        self._preset_combo.clear()
        for preset in self._services.presets.list():
            self._preset_combo.addItem(preset.meta.name, preset.meta.id)
        idx = self._preset_combo.findData("anime_balanced")
        if idx >= 0:
            self._preset_combo.setCurrentIndex(idx)

    def _browse_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select source video",
            "",
            "Video Files (*.mkv *.mp4 *.avi *.mov *.wmv *.m4v *.webm *.ts *.m2ts);;All Files (*)",
        )
        if path:
            self._source_edit.setText(path)
            self._probe_source_duration()

    def _probe_source_duration(self) -> None:
        path = Path(self._source_edit.text().strip())
        self._source_duration_s = None
        if not path.is_file():
            self._sync_segment_limits()
            return
        try:
            info = self._services.media.analyze(path)
            self._source_duration_s = info.fmt.duration_s if info.fmt is not None else None
        except Exception as exc:
            self._status_label.setText(f"Duration probe failed: {exc}")
        self._sync_segment_limits()

    def _sync_segment_limits(self) -> None:
        if self._source_duration_s is None or self._source_duration_s <= 0:
            self._start_spin.setMaximum(86_400.0)
            self._duration_spin.setMaximum(3600.0)
            return
        max_start = max(0.0, float(self._source_duration_s) - 0.1)
        self._start_spin.setMaximum(max_start)
        if self._start_spin.value() > max_start:
            self._start_spin.setValue(max_start)
        remaining = max(0.1, float(self._source_duration_s) - self._start_spin.value())
        self._duration_spin.setMaximum(remaining)
        if self._duration_spin.value() > remaining:
            self._duration_spin.setValue(remaining)

    def _run_benchmark(self) -> None:
        source_path = Path(self._source_edit.text().strip())
        if not source_path.is_file():
            QMessageBox.warning(self, "Benchmark", "Select a valid source file first.")
            return
        preset_id = str(self._preset_combo.currentData() or "")
        if not preset_id:
            QMessageBox.warning(self, "Benchmark", "Select a preset before running.")
            return
        request = BenchmarkRequest(
            source_path=source_path,
            preset_id=preset_id,
            scope=str(self._scope_combo.currentData()),
            start_s=float(self._start_spin.value()),
            duration_s=float(self._duration_spin.value()),
            verbose_ffmpeg=self._verbose_ffmpeg.isChecked(),
            compute_vmaf=self._compute_vmaf.isChecked(),
        )

        self._ffmpeg_log.clear()
        self._status_label.setText("Starting benchmark…")
        self._set_busy(True)

        cancel_event = threading.Event()
        thread = QThread(self)
        worker = _BenchmarkWorker(
            services=self._services,
            request=request,
            cancel_event=cancel_event,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_worker_progress)
        worker.stage_event.connect(self._on_stage_event)
        worker.ffmpeg_line.connect(self._on_ffmpeg_line)
        worker.finished_ok.connect(self._on_worker_finished)
        worker.failed.connect(self._on_worker_failed)
        worker.finished_ok.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished_ok.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)

        self._cancel_event = cancel_event
        self._worker = worker
        self._worker_thread = thread
        thread.start()

    def _cancel_benchmark(self) -> None:
        if self._cancel_event is not None:
            self._cancel_event.set()
            self._status_label.setText("Cancelling benchmark…")

    def _on_worker_progress(self, message: str) -> None:
        self._status_label.setText(message)

    def _on_stage_event(self, event) -> None:  # type: ignore[no-untyped-def]
        self._status_label.setText(f"{event.stage}: {event.kind}")

    def _on_ffmpeg_line(self, line: str) -> None:
        self._ffmpeg_log.appendPlainText(line)

    def _on_worker_finished(self, result: BenchmarkResult) -> None:
        self._last_result = result
        self._export_btn.setEnabled(True)
        self._status_label.setText("Benchmark completed.")
        self._render_result(result)
        self._recent_results.insert(0, result)
        self._recent_results = self._recent_results[:5]
        self._render_recent_runs()

        # Ensure the final log file is loaded in case throttled signals dropped lines.
        try:
            if result.ffmpeg_log_path.is_file():
                self._ffmpeg_log.setPlainText(result.ffmpeg_log_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    def _on_worker_failed(self, message: str) -> None:
        self._status_label.setText("Benchmark failed.")
        QMessageBox.critical(self, "Benchmark failed", message)

    def _clear_worker(self) -> None:
        self._worker_thread = None
        self._worker = None
        self._cancel_event = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._run_btn.setEnabled(not busy)
        self._cancel_btn.setEnabled(busy)
        self._source_edit.setEnabled(not busy)
        self._preset_combo.setEnabled(not busy)
        self._scope_combo.setEnabled(not busy)
        self._start_spin.setEnabled(not busy)
        self._duration_spin.setEnabled(not busy)
        self._verbose_ffmpeg.setEnabled(not busy)
        self._compute_vmaf.setEnabled(not busy and self._vmaf_available)

    def _render_result(self, result: BenchmarkResult) -> None:
        perf = result.perf_profile or {}
        self._summary_run_id.setText(result.run_id)
        self._summary_proc_calls.setText(str(perf.get("total_proc_calls", "--")))
        proc_wall = perf.get("total_proc_wall_s")
        self._summary_proc_wall.setText(
            f"{float(proc_wall):.3f}" if isinstance(proc_wall, (int, float)) else "--",
        )
        if result.vmaf is None:
            self._summary_vmaf.setText("N/A")
        elif result.vmaf.harmonic_mean is None:
            self._summary_vmaf.setText(f"{result.vmaf.mean:.3f}")
        else:
            self._summary_vmaf.setText(
                f"{result.vmaf.mean:.3f} (harmonic {result.vmaf.harmonic_mean:.3f})",
            )
        self._summary_encoded.setText(
            str(result.encoded_video_path) if result.encoded_video_path is not None else "--",
        )
        self._summary_warnings.setText("; ".join(result.warnings) if result.warnings else "--")

        stages = perf.get("stages") if isinstance(perf, dict) else {}
        stage_items = stages.items() if isinstance(stages, dict) else []
        self._stages_table.setRowCount(0)
        for stage_name, stage_doc in stage_items:
            if not isinstance(stage_doc, dict):
                continue
            row = self._stages_table.rowCount()
            self._stages_table.insertRow(row)
            perf_doc = stage_doc.get("metrics", {}).get("perf", {})
            self._stages_table.setItem(row, 0, QTableWidgetItem(str(stage_name)))
            self._stages_table.setItem(row, 1, QTableWidgetItem(str(stage_doc.get("runs", 1))))
            self._stages_table.setItem(
                row,
                2,
                QTableWidgetItem(f"{float(stage_doc.get('duration_s', 0.0)):.3f}"),
            )
            self._stages_table.setItem(
                row,
                3,
                QTableWidgetItem(str(perf_doc.get("proc_calls", 0) if isinstance(perf_doc, dict) else 0)),
            )
            self._stages_table.setItem(
                row,
                4,
                QTableWidgetItem(
                    f"{float(perf_doc.get('proc_wall_s', 0.0)):.3f}"
                    if isinstance(perf_doc, dict)
                    else "0.000"
                ),
            )

        self._encode_table.setRowCount(0)
        for sample in result.encode_samples:
            if not isinstance(sample, dict):
                continue
            row = self._encode_table.rowCount()
            self._encode_table.insertRow(row)
            self._encode_table.setItem(row, 0, QTableWidgetItem(str(sample.get("frame", ""))))
            self._encode_table.setItem(row, 1, QTableWidgetItem(str(sample.get("fps", ""))))
            self._encode_table.setItem(row, 2, QTableWidgetItem(str(sample.get("speed", ""))))
            self._encode_table.setItem(row, 3, QTableWidgetItem(str(sample.get("out_time_us", ""))))
            self._encode_table.setItem(row, 4, QTableWidgetItem(str(sample.get("progress", ""))))

    def _update_delete_run_btn(self) -> None:
        self._delete_run_btn.setEnabled(bool(self._recent_table.selectionModel().selectedRows()))

    def _selected_recent_run(self) -> BenchmarkResult | None:
        rows = self._recent_table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._recent_results):
            return None
        return self._recent_results[idx]

    def _delete_selected_run(self) -> None:
        run = self._selected_recent_run()
        if run is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete benchmark run data",
            (
                f"Delete on-disk data for run {run.run_id}?\n\n"
                f"{run.workdir}\n\n"
                "This cannot be undone."
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._services.benchmark.delete_run_data(run)
        except Exception as exc:
            QMessageBox.critical(self, "Delete failed", str(exc))
            return

        self._recent_results = [item for item in self._recent_results if item.run_id != run.run_id]
        if self._last_result is not None and self._last_result.run_id == run.run_id:
            self._last_result = None
            self._export_btn.setEnabled(False)
            self._clear_result_display()
        self._render_recent_runs()
        self._update_delete_run_btn()
        self._status_label.setText(f"Deleted benchmark data for {run.run_id}.")

    def _clear_result_display(self) -> None:
        self._summary_run_id.setText("--")
        self._summary_proc_calls.setText("--")
        self._summary_proc_wall.setText("--")
        self._summary_vmaf.setText("--")
        self._summary_encoded.setText("--")
        self._summary_warnings.setText("--")
        self._stages_table.setRowCount(0)
        self._encode_table.setRowCount(0)
        self._ffmpeg_log.clear()

    def _render_recent_runs(self) -> None:
        self._recent_table.setRowCount(0)
        for run in self._recent_results:
            row = self._recent_table.rowCount()
            self._recent_table.insertRow(row)
            perf = run.perf_profile if isinstance(run.perf_profile, dict) else {}
            wall = perf.get("total_proc_wall_s")
            vmaf_text = (
                f"{run.vmaf.mean:.3f}"
                if run.vmaf is not None
                else "N/A"
            )
            self._recent_table.setItem(row, 0, QTableWidgetItem(run.run_id))
            self._recent_table.setItem(row, 1, QTableWidgetItem(run.request.preset_id))
            self._recent_table.setItem(row, 2, QTableWidgetItem(run.request.scope))
            self._recent_table.setItem(row, 3, QTableWidgetItem(f"{run.request.duration_s:.3f}"))
            self._recent_table.setItem(row, 4, QTableWidgetItem(vmaf_text))
            self._recent_table.setItem(
                row,
                5,
                QTableWidgetItem(
                    f"{float(wall):.3f}" if isinstance(wall, (int, float)) else "--",
                ),
            )

    def _export_result(self) -> None:
        if self._last_result is None:
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Export benchmark result",
            f"{self._last_result.run_id}.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not target:
            return
        try:
            self._services.benchmark.export_result(self._last_result, Path(target))
            self._status_label.setText(f"Exported: {target}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
