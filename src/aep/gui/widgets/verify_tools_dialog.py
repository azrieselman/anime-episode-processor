"""Tools → Verify Tools dialog.

Shows a row per adapter with: tool id, resolved binary path, detected version,
expected pinned version, and a status (OK / mismatch / missing). The dialog runs
its probes synchronously when opened — every adapter's `--version` invocation is
fast (~200ms each), so blocking the GUI for ~1s on open is acceptable and
simpler than threading.

The dialog also surfaces hardware probe results (NVIDIA, encoders) because in
practice users open this dialog when something is going wrong with NVENC, and
having all of "what tools are present" + "what GPU is detected" in one view is
the fastest path to diagnosis.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from aep.adapters.verification import (
    DEFAULT_ADAPTERS,
    ToolStatus,
    check_adapter,
)
from aep.app.pinned_tools_refresh import pins_to_refresh
from aep.bench.hardware import probe_hardware
from aep.constants import PINNED_VERSIONS
from aep.gui.widgets.first_run_dialog import FirstRunDialog

log = logging.getLogger(__name__)

# Re-exported for backward compatibility — some code still imports ToolStatus
# from this module; keep that working without touching every call site.
__all__ = ["ToolStatus", "VerifyToolsDialog"]


class VerifyToolsDialog(QDialog):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Verify Tools")
        self.resize(820, 480)
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        header = QLabel("External tools used by the pipeline")
        header_font = QFont(header.font())
        header_font.setBold(True)
        header.setFont(header_font)
        root.addWidget(header)

        self._table = QTableWidget(self)
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["Tool", "Status", "Detected version", "Pinned version", "Path"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        h_header = self._table.horizontalHeader()
        h_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        h_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        root.addWidget(self._table, 1)

        # Hardware summary
        self._hw_label = QLabel("(probing hardware…)")
        self._hw_label.setWordWrap(True)
        self._hw_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self._hw_label)

        # Buttons
        btns = QDialogButtonBox(self)
        refresh_btn = QPushButton("Re-probe", self)
        refresh_btn.clicked.connect(self._refresh)
        btns.addButton(refresh_btn, QDialogButtonBox.ButtonRole.ActionRole)

        self._fetch_pins_btn = QPushButton("Download / refresh pinned tools…", self)
        self._fetch_pins_btn.clicked.connect(self._on_download_refresh_pins)
        btns.addButton(self._fetch_pins_btn, QDialogButtonBox.ButtonRole.ActionRole)

        close_btn = btns.addButton(QDialogButtonBox.StandardButton.Close)
        close_btn.clicked.connect(self.accept)
        root.addWidget(btns)

    def _collect_adapter_statuses(self) -> list[ToolStatus]:
        statuses: list[ToolStatus] = []
        for cls in DEFAULT_ADAPTERS:
            try:
                statuses.append(check_adapter(cls()))
            except Exception as exc:
                statuses.append(ToolStatus(
                    tool_id=cls.tool_id,
                    path="(error)",
                    version="",
                    expected=PINNED_VERSIONS.get(cls.tool_id, ""),
                    status="missing",
                    note=str(exc),
                ))
        return statuses

    def _on_download_refresh_pins(self) -> None:
        statuses = self._collect_adapter_statuses()
        pins = pins_to_refresh(statuses)
        if not pins:
            QMessageBox.information(
                self,
                "Tools up to date",
                "Nothing is missing or version-mismatched relative to this build's pinned tools.",
            )
            self._refresh()
            return

        dlg = FirstRunDialog(
            self,
            missing_pins=pins,
            window_title="Anime Episode Processor — Refresh pinned tools",
            heading="Re-downloading pinned tool archives",
            intro_html=(
                "This build expects specific tool versions. Missing or outdated installs will be "
                "replaced from the official vendor URLs (SHA256-verified), which may download up "
                "to a few gigabytes."
            ),
            force_refresh=True,
        )
        dlg.exec()
        self._refresh()

    def _refresh(self) -> None:
        log.info("VerifyToolsDialog: re-probing")

        statuses = self._collect_adapter_statuses()

        self._table.setRowCount(len(statuses))
        for row, st in enumerate(statuses):
            self._table.setItem(row, 0, QTableWidgetItem(st.tool_id))
            status_item = QTableWidgetItem(_status_label(st.status))
            status_item.setForeground(_status_color(st.status))
            self._table.setItem(row, 1, status_item)
            ver_item = QTableWidgetItem(st.version)
            if st.note:
                ver_item.setToolTip(st.note)
            self._table.setItem(row, 2, ver_item)
            self._table.setItem(row, 3, QTableWidgetItem(st.expected or "-"))
            self._table.setItem(row, 4, QTableWidgetItem(st.path))

        # Hardware probe
        try:
            hw = probe_hardware()
            self._hw_label.setText(_format_hardware(hw))
        except Exception as exc:
            self._hw_label.setText(f"hardware probe failed: {exc}")

        pin_count = len(pins_to_refresh(statuses))
        self._fetch_pins_btn.setEnabled(pin_count > 0)


def _status_label(status: str) -> str:
    return {
        "ok": "OK",
        "mismatch": "VERSION MISMATCH",
        "missing": "MISSING",
        "version_unknown": "VERSION UNKNOWN",
    }.get(status, status)


def _status_color(status: str) -> QColor:
    return {
        "ok": QColor(40, 140, 50),
        "mismatch": QColor(200, 130, 0),
        "missing": QColor(180, 30, 30),
        "version_unknown": QColor(120, 120, 120),
    }.get(status, QColor(0, 0, 0))


def _format_hardware(hw) -> str:
    gpu = hw.gpu
    parts = [
        "<b>Hardware</b><br>",
        f"CPU: {hw.cpu.logical_cores} logical cores",
    ]
    if hw.cpu.ram_total_mib:
        parts.append(f", {hw.cpu.ram_total_mib // 1024} GiB RAM")
    parts.append("<br>")
    parts.append(f"Primary vendor: {gpu.primary_vendor}<br>")
    if gpu.has_nvidia:
        parts.append(
            f"NVIDIA: arch={gpu.arch or '?'}, "
            f"VRAM={gpu.vram_total_mib} MiB, "
            f"driver={gpu.driver_version or '?'}<br>"
            f"NVENC: H.264={gpu.nvenc_h264}, HEVC={gpu.nvenc_hevc}, AV1={gpu.nvenc_av1}<br>"
        )
    else:
        parts.append("NVIDIA NVENC: not active on this profile<br>")
    parts.append(
        f"Intel QSV: H.264={gpu.qsv_h264}, HEVC={gpu.qsv_hevc}, AV1={gpu.qsv_av1}<br>"
    )
    parts.append(
        f"AMD AMF: H.264={gpu.amf_h264}, HEVC={gpu.amf_hevc}, AV1={gpu.amf_av1}<br>"
    )
    parts.append(f"FFmpeg: version={hw.ffmpeg_version or '?'}, encoders={len(hw.ffmpeg_encoders)} known")
    return "".join(parts)
