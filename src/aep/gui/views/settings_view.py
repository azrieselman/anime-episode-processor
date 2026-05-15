"""Settings view — minimal editor for AppSettings.

Tools detection lives in its own dialog (Tools → Verify Tools); this view focuses on the fields the user is most
likely to want to tweak now.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aep.app.services import AppServices
from aep.gui import theme

_TOOL_DIR_FIELDS: tuple[tuple[str, str], ...] = (
    ("ffmpeg_dir", "FFmpeg directory:"),
    ("mkvtoolnix_dir", "MKVToolNix directory:"),
    ("realcugan_dir", "Real-CUGAN directory:"),
    ("realesrgan_dir", "Real-ESRGAN directory:"),
    ("anime4kcpp_dir", "Anime4KCPP directory:"),
    ("anime4kcpp_vs_filter_dir", "Anime4KCPP VS filter directory:"),
    ("vapoursynth_dir", "VapourSynth (vspipe) directory:"),
    ("rife_dir", "RIFE directory:"),
    ("waifu2x_dir", "waifu2x directory:"),
)


def _format_free_space(path: str) -> str:
    """Best-effort free-space label for a path; never raises.

    Empty/missing path → empty string. Unreadable path (permission denied,
    not yet created on a removable drive) → "unavailable". Otherwise
    "<n.n> GiB free".
    """
    if not path:
        return ""
    try:
        p = Path(path)
        # Walk up to the nearest existing ancestor so we can still report a
        # number for a drive whose subfolder hasn't been created yet.
        probe = p
        while not probe.exists():
            if probe.parent == probe:
                return "unavailable"
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        return f"{usage.free / (1024**3):.1f} GiB free"
    except OSError:
        return "unavailable"


def _normalize_rife_threads(value: str) -> str:
    parts = value.strip().split(":")
    if len(parts) != 3 or not all(part.isdecimal() for part in parts):
        raise ValueError("RIFE threads must use the format load:process:save, for example 10:10:10.")
    nums = [int(part) for part in parts]
    if any(n < 1 for n in nums):
        raise ValueError("RIFE thread counts must be positive integers.")
    return f"{nums[0]}:{nums[1]}:{nums[2]}"


class SettingsView(QWidget):
    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self._services = services
        self._build()
        self._load()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(theme.make_page_title_label("Settings", self))

        general = QGroupBox("General")
        gf = QFormLayout(general)
        self._log_level = QComboBox()
        self._log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        gf.addRow("Log level:", self._log_level)

        out_row = QHBoxLayout()
        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("(leave blank to write next to source)")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_output)
        out_row.addWidget(self._output_dir, 1)
        out_row.addWidget(browse)
        gf.addRow("Default output directory:", out_row)

        self._naming = QLineEdit()
        gf.addRow("Output naming template:", self._naming)
        self._keep_temp = QCheckBox("Keep intermediate artifacts after success")
        gf.addRow("", self._keep_temp)
        self._confirm_overwrite = QCheckBox("Confirm before overwriting existing files")
        gf.addRow("", self._confirm_overwrite)
        self._show_queue_job_id = QCheckBox("Show job ID column in queue table")
        self._show_queue_job_id.setToolTip(
            "Display each job's internal ID in the Queue tab. Useful for logs and support."
        )
        gf.addRow("", self._show_queue_job_id)
        root.addWidget(general)

        hw = QGroupBox("Hardware")
        hf = QFormLayout(hw)
        self._prefer_nvenc = QCheckBox("Prefer NVENC when sensible")
        hf.addRow("", self._prefer_nvenc)
        self._decode_hwaccel = QComboBox()
        self._decode_hwaccel.addItem("Auto (D3D11VA on Windows)", "auto")
        self._decode_hwaccel.addItem("Off (software decode)", "off")
        self._decode_hwaccel.addItem("DirectX D3D11VA", "d3d11va")
        self._decode_hwaccel.addItem("NVIDIA NVDEC (CUDA)", "cuda")
        hf.addRow("Decode acceleration:", self._decode_hwaccel)
        self._max_jobs = QSpinBox()
        self._max_jobs.setRange(1, 4)
        hf.addRow("Max concurrent jobs:", self._max_jobs)
        self._ring_frames = QSpinBox()
        self._ring_frames.setRange(32, 2048)
        self._ring_frames.setSingleStep(16)
        hf.addRow("Ring buffer frames:", self._ring_frames)
        self._tile = QSpinBox()
        self._tile.setRange(64, 1024)
        self._tile.setSingleStep(32)
        hf.addRow("Default tile size:", self._tile)

        # NCNN chunking — long ncnn-vulkan runs hit driver state issues on
        # Windows beyond ~30k frames; chunking at threshold/size keeps each
        # invocation short with frequent recovery points. See HardwareSettings.
        self._ncnn_chunk_threshold = QSpinBox()
        self._ncnn_chunk_threshold.setRange(200, 20000)
        self._ncnn_chunk_threshold.setSingleStep(100)
        self._ncnn_chunk_threshold.setToolTip(
            "Split NCNN upscale runs into chunks once a stage exceeds this many frames."
        )
        hf.addRow("NCNN chunk threshold (frames):", self._ncnn_chunk_threshold)
        self._ncnn_chunk_size = QSpinBox()
        self._ncnn_chunk_size.setRange(100, 5000)
        self._ncnn_chunk_size.setSingleStep(50)
        self._ncnn_chunk_size.setToolTip(
            "Frames per NCNN chunk. Smaller = more recovery points, more per-chunk overhead."
        )
        hf.addRow("NCNN chunk size (frames):", self._ncnn_chunk_size)
        self._anime4k_prefer_cuda = QCheckBox("Anime4K: prefer CUDA (fallback to OpenCL)")
        hf.addRow("", self._anime4k_prefer_cuda)
        self._anime4k_threads = QSpinBox()
        self._anime4k_threads.setRange(1, 64)
        self._anime4k_threads.setToolTip(
            "Anime4KCPP CLI -t: thread count for multi-frame image batches "
            "(internal library parallelism; applies to the ac_cli engine only)."
        )
        hf.addRow("Anime4K CLI threads (-t):", self._anime4k_threads)
        self._rife_threads = QLineEdit()
        self._rife_threads.setPlaceholderText("10:10:10")
        self._rife_threads.setToolTip(
            "RIFE -j thread split in load:process:save format, e.g. 10:10:10."
        )
        hf.addRow("RIFE threads (-j):", self._rife_threads)
        root.addWidget(hw)

        pipeline = QGroupBox("Pipeline")
        plf = QFormLayout(pipeline)
        self._pipeline_order = QComboBox()
        self._pipeline_order.addItem(
            "Interpolation → Upscaling (default, faster)",
            "interpolate_first",
        )
        self._pipeline_order.addItem(
            "Upscaling → Interpolation",
            "upscale_first",
        )
        self._pipeline_order.setToolTip(
            "Order of RIFE and upscaler between decode and postprocess. "
            "Interpolation first runs RIFE on smaller frames (usually faster)."
        )
        plf.addRow("NCNN frame stages:", self._pipeline_order)
        root.addWidget(pipeline)

        # ----- Paths group -----------------------------------------------
        # Ramdisk + per-tool directory overrides. Empty values fall back to
        # bundled tools (or to the workdir for ramdisk).
        paths = QGroupBox("Paths")
        pf = QFormLayout(paths)

        ramdisk_row = QHBoxLayout()
        self._ramdisk_path = QLineEdit()
        self._ramdisk_path.setPlaceholderText(
            "(leave blank to use the regular work directory)"
        )
        # Refresh free-space hint as the user types or browses.
        self._ramdisk_path.textChanged.connect(self._refresh_ramdisk_free_space)
        rd_browse = QPushButton("Browse…")
        rd_browse.clicked.connect(self._browse_ramdisk)
        ramdisk_row.addWidget(self._ramdisk_path, 1)
        ramdisk_row.addWidget(rd_browse)
        pf.addRow("Ramdisk / scratch directory:", ramdisk_row)
        self._ramdisk_free_label = QLabel("")
        theme.style_muted_detail_label(self._ramdisk_free_label, small=True)
        pf.addRow("", self._ramdisk_free_label)

        # 6 tool-dir overrides — each gets a Browse button. Stored on self
        # under names matching PathSettings field names so _load/_save can
        # iterate _TOOL_DIR_FIELDS uniformly without per-field scaffolding.
        self._tool_dir_edits: dict[str, QLineEdit] = {}
        for attr, label in _TOOL_DIR_FIELDS:
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setPlaceholderText("(blank = use bundled tool)")
            browse_btn = QPushButton("Browse…")
            browse_btn.clicked.connect(
                lambda _checked, a=attr, lbl=label: self._browse_tool_dir(a, lbl)
            )
            row.addWidget(edit, 1)
            row.addWidget(browse_btn)
            pf.addRow(label, row)
            self._tool_dir_edits[attr] = edit
        root.addWidget(paths)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._save)
        self._reset_btn = QPushButton("Reset to defaults")
        self._reset_btn.clicked.connect(self._reset)
        btns.addWidget(self._reset_btn)
        btns.addWidget(self._save_btn)
        root.addLayout(btns)
        root.addStretch(1)

    def _browse_output(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose default output directory")
        if d:
            self._output_dir.setText(d)

    def _browse_ramdisk(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose ramdisk / scratch directory")
        if d:
            self._ramdisk_path.setText(d)

    def _browse_tool_dir(self, attr: str, label: str) -> None:
        d = QFileDialog.getExistingDirectory(self, f"Choose {label.rstrip(':').lower()}")
        if d:
            self._tool_dir_edits[attr].setText(d)

    def _refresh_ramdisk_free_space(self) -> None:
        self._ramdisk_free_label.setText(_format_free_space(self._ramdisk_path.text().strip()))

    def _load(self) -> None:
        s = self._services.settings.get()
        self._log_level.setCurrentText(s.general.log_level)
        self._output_dir.setText(s.general.output_dir or "")
        self._naming.setText(s.general.output_naming_template)
        self._keep_temp.setChecked(s.general.keep_temp_artifacts)
        self._confirm_overwrite.setChecked(s.general.confirm_overwrite)
        self._show_queue_job_id.setChecked(s.general.show_queue_job_id_column)
        self._prefer_nvenc.setChecked(s.hardware.prefer_nvenc)
        idx = self._decode_hwaccel.findData(s.hardware.decode_hwaccel)
        self._decode_hwaccel.setCurrentIndex(idx if idx >= 0 else 0)
        self._max_jobs.setValue(s.hardware.max_concurrent_jobs)
        self._ring_frames.setValue(s.hardware.ring_buffer_frames)
        self._tile.setValue(s.hardware.default_tile_size)
        self._ncnn_chunk_threshold.setValue(s.hardware.ncnn_chunk_threshold)
        self._ncnn_chunk_size.setValue(s.hardware.ncnn_chunk_size)
        self._anime4k_prefer_cuda.setChecked(s.hardware.anime4k_prefer_cuda)
        self._anime4k_threads.setValue(s.hardware.anime4k_threads)
        self._rife_threads.setText(s.hardware.rife_threads)
        pidx = self._pipeline_order.findData(s.pipeline.order)
        self._pipeline_order.setCurrentIndex(pidx if pidx >= 0 else 0)
        self._ramdisk_path.setText(s.paths.ramdisk_path or "")
        for attr, _label in _TOOL_DIR_FIELDS:
            self._tool_dir_edits[attr].setText(getattr(s.paths, attr) or "")
        self._refresh_ramdisk_free_space()

    def _save(self) -> None:
        try:
            rife_threads = _normalize_rife_threads(self._rife_threads.text())
        except ValueError as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        s = self._services.settings.get().model_copy(deep=True)
        s.general.log_level = self._log_level.currentText()  # type: ignore[assignment]
        s.general.output_dir = self._output_dir.text().strip() or None
        s.general.output_naming_template = self._naming.text().strip()
        s.general.keep_temp_artifacts = self._keep_temp.isChecked()
        s.general.confirm_overwrite = self._confirm_overwrite.isChecked()
        s.general.show_queue_job_id_column = self._show_queue_job_id.isChecked()
        s.hardware.prefer_nvenc = self._prefer_nvenc.isChecked()
        s.hardware.decode_hwaccel = str(self._decode_hwaccel.currentData())  # type: ignore[assignment]
        s.hardware.max_concurrent_jobs = self._max_jobs.value()
        s.hardware.ring_buffer_frames = self._ring_frames.value()
        s.hardware.default_tile_size = self._tile.value()
        s.hardware.ncnn_chunk_threshold = self._ncnn_chunk_threshold.value()
        s.hardware.ncnn_chunk_size = self._ncnn_chunk_size.value()
        s.hardware.anime4k_prefer_cuda = self._anime4k_prefer_cuda.isChecked()
        s.hardware.anime4k_threads = self._anime4k_threads.value()
        s.hardware.rife_threads = rife_threads
        s.pipeline.order = str(self._pipeline_order.currentData())  # type: ignore[assignment]
        s.paths.ramdisk_path = self._ramdisk_path.text().strip() or None
        for attr, _label in _TOOL_DIR_FIELDS:
            setattr(
                s.paths,
                attr,
                self._tool_dir_edits[attr].text().strip() or None,
            )
        try:
            self._services.settings.update(s)
            QMessageBox.information(self, "Settings saved", "Restart for log-level changes to take full effect.")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _reset(self) -> None:
        from aep.persist.settings import AppSettings
        self._services.settings.update(AppSettings())
        self._load()
