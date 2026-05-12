"""Per-job config view.

Editable for QUEUED jobs only. Once a job leaves the queue we lock the form
because the broker has already loaded its preset; later mutations would not
be picked up. The widgets cover the four override fields that move the most:

  * upscaler.scale       -- output multiplier (1..4)
  * upscaler.denoise     -- CUGAN noise strength (-1..3)
  * interpolation.target_fps  -- frame rate target (or "preserve source")
  * encoder.name         -- which video encoder to drive

Any change is persisted as a sparse `preset_overrides` dict on the Job row;
fields that match the base preset are dropped so the row stays minimal.
"""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aep.app.services import AppServices
from aep.gui import theme
from aep.jobs.models import JobState

log = logging.getLogger(__name__)


# Encoder choices mirror EncoderName Literal in persist/presets.py. Kept in a
# single module-level constant so test assertions and the dropdown stay aligned.
_ENCODER_CHOICES: list[str] = [
    "hevc_nvenc",
    "h264_nvenc",
    "av1_nvenc",
    "hevc_qsv",
    "h264_qsv",
    "av1_qsv",
    "hevc_amf",
    "h264_amf",
    "av1_amf",
    "libx264",
    "libx265",
]
_DECODE_HWACCEL_CHOICES: list[tuple[str, str]] = [
    ("Auto", "auto"),
    ("Off (software decode)", "off"),
    ("DirectX D3D11VA", "d3d11va"),
    ("NVIDIA NVDEC (CUDA)", "cuda"),
]


class JobConfigView(QWidget):
    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self._services = services
        self._job_id: str | None = None
        # Cached preset values for the currently selected job; used both to
        # seed widgets and to compute the sparse diff on save.
        self._base_preset: dict[str, Any] = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        root.addWidget(theme.make_page_title_label("Job Configuration", self))

        info = QGroupBox("Selected Job")
        form = QFormLayout(info)
        self._source_label = QLabel("—")
        self._preset_label = QLabel("—")
        self._output_label = QLabel("—")
        self._state_label = QLabel("—")
        form.addRow("Source:", self._source_label)
        form.addRow("Preset:", self._preset_label)
        form.addRow("Planned Output:", self._output_label)
        form.addRow("State:", self._state_label)
        root.addWidget(info)

        # ----- editable per-job overrides --------------------------------
        overrides_box = QGroupBox("Per-Job Overrides")
        ov_form = QFormLayout(overrides_box)

        self._scale_spin = QSpinBox()
        self._scale_spin.setRange(1, 4)
        ov_form.addRow("Upscaler scale (×):", self._scale_spin)

        self._denoise_spin = QSpinBox()
        self._denoise_spin.setRange(-1, 3)
        self._denoise_spin.setSpecialValueText("none (-1)")
        ov_form.addRow("Upscaler denoise:", self._denoise_spin)

        # target_fps is a Float | None; render as a checkbox + spinbox so
        # "preserve source" (None) is a first-class state, not an out-of-band
        # spinbox sentinel value.
        fps_row = QHBoxLayout()
        self._fps_enabled = QCheckBox("Set target FPS")
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 480.0)
        self._fps_spin.setDecimals(2)
        self._fps_spin.setSingleStep(1.0)
        self._fps_enabled.toggled.connect(self._fps_spin.setEnabled)
        fps_row.addWidget(self._fps_enabled)
        fps_row.addWidget(self._fps_spin, 1)
        fps_wrap = QWidget()
        fps_wrap.setLayout(fps_row)
        ov_form.addRow("Interpolation FPS:", fps_wrap)

        self._encoder_combo = QComboBox()
        self._encoder_combo.addItems(_ENCODER_CHOICES)
        ov_form.addRow("Encoder:", self._encoder_combo)
        self._decode_hwaccel_combo = QComboBox()
        for label, value in _DECODE_HWACCEL_CHOICES:
            self._decode_hwaccel_combo.addItem(label, value)
        ov_form.addRow("Decode acceleration:", self._decode_hwaccel_combo)

        button_row = QHBoxLayout()
        self._save_btn = QPushButton("Save Overrides")
        self._reset_btn = QPushButton("Reset to Preset")
        self._save_btn.clicked.connect(self._on_save)
        self._reset_btn.clicked.connect(self._on_reset)
        button_row.addWidget(self._save_btn)
        button_row.addWidget(self._reset_btn)
        button_row.addStretch(1)
        button_wrap = QWidget()
        button_wrap.setLayout(button_row)
        ov_form.addRow("", button_wrap)

        self._status_label = QLabel("")
        theme.style_muted_detail_label(self._status_label)
        ov_form.addRow("", self._status_label)

        root.addWidget(overrides_box)

        # ----- read-only preset dump ------------------------------------
        preset_box = QGroupBox("Resolved Preset (after overrides)")
        pv = QVBoxLayout(preset_box)
        self._preset_text = QPlainTextEdit()
        self._preset_text.setReadOnly(True)
        self._preset_text.setPlaceholderText("Select a job to view the resolved preset.")
        pv.addWidget(self._preset_text)
        root.addWidget(preset_box, 1)

        self._set_overrides_enabled(False)

    # ----- helpers ------------------------------------------------------

    def _set_overrides_enabled(self, enabled: bool) -> None:
        for w in (
            self._scale_spin, self._denoise_spin, self._fps_enabled,
            self._fps_spin, self._encoder_combo, self._decode_hwaccel_combo, self._save_btn,
            self._reset_btn,
        ):
            w.setEnabled(enabled)
        # When disabled, the FPS spinbox's enable state is independently
        # driven by the checkbox; force it off.
        if not enabled:
            self._fps_spin.setEnabled(False)

    def _seed_widgets(self, base: dict[str, Any], overrides: dict[str, Any] | None) -> None:
        """Populate widgets from the base preset, then layer overrides on top."""
        merged = _deep_merge(base, overrides or {})

        up = merged.get("upscaler", {}) or {}
        self._scale_spin.setValue(int(up.get("scale", 2)))
        self._denoise_spin.setValue(int(up.get("denoise", 3)))

        ic = merged.get("interpolation", {}) or {}
        target_fps = ic.get("target_fps")
        if target_fps is None:
            self._fps_enabled.setChecked(False)
            # Still seed a sensible default so the spinbox isn't 1.0 when the
            # user toggles it on for the first time.
            self._fps_spin.setValue(60.0)
        else:
            self._fps_enabled.setChecked(True)
            self._fps_spin.setValue(float(target_fps))
        self._fps_spin.setEnabled(self._fps_enabled.isChecked())

        enc_name = (merged.get("encoder", {}) or {}).get("name", "hevc_nvenc")
        if enc_name in _ENCODER_CHOICES:
            self._encoder_combo.setCurrentText(enc_name)
        else:
            # Foreign encoder: prepend it so the user can see what's set
            # without clobbering it on save.
            self._encoder_combo.insertItem(0, enc_name)
            self._encoder_combo.setCurrentIndex(0)

        decode_mode = (merged.get("decode", {}) or {}).get("hwaccel", "auto")
        idx = self._decode_hwaccel_combo.findData(decode_mode)
        self._decode_hwaccel_combo.setCurrentIndex(idx if idx >= 0 else 0)

    def _collect_widget_values(self) -> dict[str, Any]:
        """Snapshot widget state into a fully-populated dict (no diffing yet)."""
        target_fps: float | None = (
            float(self._fps_spin.value()) if self._fps_enabled.isChecked() else None
        )
        return {
            "upscaler": {
                "scale": int(self._scale_spin.value()),
                "denoise": int(self._denoise_spin.value()),
            },
            "interpolation": {
                "target_fps": target_fps,
            },
            "encoder": {
                "name": self._encoder_combo.currentText(),
            },
            "decode": {
                "hwaccel": str(self._decode_hwaccel_combo.currentData()),
            },
        }

    def _diff_against_base(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        """Return a sparse override dict containing only fields that differ from base."""
        out: dict[str, Any] = {}
        for section, fields in snapshot.items():
            base_section = self._base_preset.get(section, {}) or {}
            differing: dict[str, Any] = {}
            for k, v in fields.items():
                if base_section.get(k) != v:
                    differing[k] = v
            if differing:
                out[section] = differing
        return out

    def _refresh_preset_text(self, overrides: dict[str, Any] | None) -> None:
        import json
        merged = _deep_merge(self._base_preset, overrides or {})
        self._preset_text.setPlainText(json.dumps(merged, indent=2))

    # ----- public API ---------------------------------------------------

    def set_job(self, job_id: str | None) -> None:
        self._job_id = job_id
        self._status_label.setText("")
        if not job_id:
            self._source_label.setText("—")
            self._preset_label.setText("—")
            self._output_label.setText("—")
            self._state_label.setText("—")
            self._preset_text.setPlainText("")
            self._base_preset = {}
            self._set_overrides_enabled(False)
            return

        job = self._services.jobs.get(job_id)
        if job is None:
            self._set_overrides_enabled(False)
            return

        self._source_label.setText(job.source_path)
        self._preset_label.setText(job.preset_id)
        self._output_label.setText(job.output_path or "(auto: <source>.aep.mkv)")
        self._state_label.setText(job.state.value)

        try:
            preset = self._services.presets.get(job.preset_id)
            self._base_preset = preset.model_dump(mode="json")
        except Exception as exc:
            self._base_preset = {}
            self._preset_text.setPlainText(f"# could not load preset: {exc}")
            self._set_overrides_enabled(False)
            return

        self._seed_widgets(self._base_preset, job.preset_overrides)
        self._refresh_preset_text(job.preset_overrides)
        # Editable only while the job is still queued. Other states (running
        # / done / failed / cancelled) get a read-only view of what was used.
        self._set_overrides_enabled(job.state == JobState.QUEUED)
        if job.state != JobState.QUEUED:
            self._status_label.setText(
                f"Read-only: job is {job.state.value}; only queued jobs can be edited."
            )

    # ----- handlers -----------------------------------------------------

    def _on_save(self) -> None:
        if not self._job_id:
            return
        snapshot = self._collect_widget_values()
        diff = self._diff_against_base(snapshot)
        result = self._services.jobs.set_preset_overrides(self._job_id, diff or None)
        if result is None:
            QMessageBox.warning(self, "Job not found", "The selected job no longer exists.")
            return
        if result.state != JobState.QUEUED:
            QMessageBox.warning(
                self, "Cannot edit",
                f"Job is in state {result.state.value}; overrides were not saved.",
            )
            return
        self._refresh_preset_text(diff or None)
        if diff:
            self._status_label.setText(f"Saved {sum(len(v) for v in diff.values())} override(s).")
        else:
            self._status_label.setText("Cleared all overrides (matches preset).")
        log.info("job %s overrides saved: %s", self._job_id, diff)

    def _on_reset(self) -> None:
        if not self._job_id:
            return
        # Repopulate widgets from the bare preset (no overrides), then save.
        self._seed_widgets(self._base_preset, None)
        self._services.jobs.set_preset_overrides(self._job_id, None)
        self._refresh_preset_text(None)
        self._status_label.setText("Overrides cleared.")


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Same shape as broker._deep_merge; duplicated to avoid GUI->jobs import.

    The GUI must not import from `aep.jobs.broker` (services seam), and the
    helper is small enough that keeping a local copy is cheaper than building
    a third common module just for this.
    """
    out = dict(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
