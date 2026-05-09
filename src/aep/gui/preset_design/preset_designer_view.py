"""Preset Designer — schema-driven editing tab."""

from __future__ import annotations

import copy
import logging
from typing import Any

from pydantic import ValidationError
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from aep.app.services import AppServices
from aep.gui import theme
from aep.gui.preset_design.editor_builder import EditorBinding, build_preset_editor
from aep.persist.presets import Preset
from aep.util.paths import user_presets_dir

log = logging.getLogger(__name__)


class PresetDesignerView(QWidget):
    """Edit presets with categorized fields; save writes user YAML (overrides built-ins by id)."""

    presets_changed = Signal()

    def __init__(self, services: AppServices, parent=None) -> None:
        super().__init__(parent)
        self._services = services
        self._bindings: list[EditorBinding] = []
        self._working: dict[str, Any] = {}
        self._selected_id: str | None = None
        self._dirty = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        head = QHBoxLayout()
        head.addWidget(theme.make_page_title_label("Preset Designer", self))
        head.addStretch(1)

        self._reload_btn = QPushButton("Reload")
        self._reload_btn.clicked.connect(self._reload_list_only)
        head.addWidget(self._reload_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_save)
        head.addWidget(self._save_btn)

        self._dup_btn = QPushButton("Duplicate…")
        self._dup_btn.clicked.connect(self._on_duplicate)
        head.addWidget(self._dup_btn)

        self._new_btn = QPushButton("New custom…")
        self._new_btn.clicked.connect(self._on_new_custom)
        head.addWidget(self._new_btn)

        self._delete_btn = QPushButton("Delete user copy")
        self._delete_btn.clicked.connect(self._on_delete_user)
        head.addWidget(self._delete_btn)

        self._open_dir_btn = QPushButton("Open Presets Folder")
        self._open_dir_btn.clicked.connect(self._open_dir)
        head.addWidget(self._open_dir_btn)

        root.addLayout(head)

        self._banner = QLabel("")
        theme.style_muted_detail_label(self._banner, small=True)
        self._banner.setWordWrap(True)
        root.addWidget(self._banner)

        self._validation_label = QLabel("")
        theme.style_attention_status_label(self._validation_label, italic=True)
        self._validation_label.setWordWrap(True)
        root.addWidget(self._validation_label)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_select)
        splitter.addWidget(self._list)

        self._editor_wrap = QWidget()
        self._editor_layout = QVBoxLayout(self._editor_wrap)
        self._editor_layout.setContentsMargins(0, 0, 0, 0)
        splitter.addWidget(self._editor_wrap)
        splitter.setSizes([280, 820])
        root.addWidget(splitter, 1)

        editor_root, self._bindings = build_preset_editor(on_changed=self._mark_dirty)
        self._editor_layout.addWidget(editor_root, 1)

        self._reload_list_only()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._validation_label.clear()

    def _reload_list_only(self) -> None:
        sel = self._selected_id
        self._list.blockSignals(True)
        self._list.clear()
        for p in self._services.presets.list():
            label = f"{p.meta.name}{'  (built-in)' if p.meta.builtin else ''}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, p.meta.id)
            item.setToolTip(p.meta.description)
            self._list.addItem(item)
        self._list.blockSignals(False)
        if sel:
            for row in range(self._list.count()):
                it = self._list.item(row)
                if it and it.data(Qt.ItemDataRole.UserRole) == sel:
                    self._list.setCurrentItem(it)
                    break
            else:
                if self._list.count() > 0:
                    self._list.setCurrentRow(0)
        elif self._list.count() > 0:
            self._list.setCurrentRow(0)
        self.presets_changed.emit()

    def _on_select(self, current: QListWidgetItem | None, _previous=None) -> None:
        if self._dirty and self._selected_id and current is not None:
            ret = QMessageBox.question(
                self,
                "Discard changes?",
                "You have unsaved edits. Switch presets anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ret != QMessageBox.StandardButton.Yes:
                self._list.blockSignals(True)
                for row in range(self._list.count()):
                    it = self._list.item(row)
                    if it and it.data(Qt.ItemDataRole.UserRole) == self._selected_id:
                        self._list.setCurrentItem(it)
                        break
                self._list.blockSignals(False)
                return

        if not current:
            self._selected_id = None
            self._working = {}
            self._banner.clear()
            return

        pid = current.data(Qt.ItemDataRole.UserRole)
        self._selected_id = pid
        self._dirty = False
        self._validation_label.clear()
        try:
            p = self._services.presets.get(pid)
        except Exception as exc:
            self._banner.setText("")
            self._validation_label.setText(f"Could not load preset: {exc}")
            self._working = {}
            return

        self._working = p.model_dump(mode="python")
        self._apply_bindings()
        if p.meta.builtin:
            self._banner.setText(
                "This preset is built-in. Saving writes a file under your user presets folder "
                f"({user_presets_dir()}) with the same id, which overrides the built-in for this app."
            )
        else:
            self._banner.setText("User preset — Save updates your presets folder.")

    def _apply_bindings(self) -> None:
        for b in self._bindings:
            b.apply(self._working)

    def _commit_bindings(self) -> None:
        for b in self._bindings:
            b.commit(self._working)

    def _on_save(self) -> None:
        if not self._selected_id:
            QMessageBox.information(self, "Save", "Select a preset first.")
            return
        self._commit_bindings()
        try:
            preset = Preset.model_validate(self._working)
        except ValidationError as exc:
            self._validation_label.setText(self._format_validation_error(exc))
            QMessageBox.warning(self, "Validation failed", str(exc))
            return

        preset.meta.builtin = False
        try:
            self._services.presets.save(preset)
        except OSError as exc:
            QMessageBox.warning(self, "Save failed", str(exc))
            return

        self._dirty = False
        self._validation_label.setText("Saved.")
        log.info("preset designer saved: %s", preset.meta.id)
        self._reload_list_only()
        self.presets_changed.emit()

    def _on_duplicate(self) -> None:
        if not self._working:
            QMessageBox.information(self, "Duplicate", "Select a preset to duplicate.")
            return
        self._commit_bindings()
        base_id = str(self._working.get("meta", {}).get("id", "preset"))
        nid, ok = self._prompt_text(self, "Duplicate preset", "New preset id (filename stem):", f"{base_id}_copy")
        if not ok or not nid.strip():
            return
        nid = nid.strip()
        name, ok2 = self._prompt_text(self, "Duplicate preset", "Display name:", f"{nid}")
        if not ok2:
            return
        clone = copy.deepcopy(self._working)
        clone.setdefault("meta", {})["id"] = nid
        clone["meta"]["name"] = name.strip() or nid
        clone["meta"]["builtin"] = False
        try:
            preset = Preset.model_validate(clone)
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation failed", str(exc))
            return
        self._services.presets.save(preset)
        self._reload_list_only()
        self._select_by_id(nid)
        self.presets_changed.emit()

    def _on_new_custom(self) -> None:
        nid, ok = self._prompt_text(self, "New preset", "New preset id:", "my_preset")
        if not ok or not nid.strip():
            return
        nid = nid.strip()
        name, ok2 = self._prompt_text(self, "New preset", "Display name:", nid)
        if not ok2:
            return
        try:
            base = self._services.presets.get("anime_balanced")
        except Exception:
            plist = self._services.presets.list()
            if not plist:
                QMessageBox.warning(self, "New preset", "No presets available to seed from.")
                return
            base = plist[0]
        fresh = base.model_dump(mode="python")
        fresh["meta"]["id"] = nid
        fresh["meta"]["name"] = name.strip() or nid
        fresh["meta"]["builtin"] = False
        fresh["meta"]["description"] = ""
        try:
            preset = Preset.model_validate(fresh)
        except ValidationError as exc:
            QMessageBox.warning(self, "Validation failed", str(exc))
            return
        self._services.presets.save(preset)
        self._reload_list_only()
        self._select_by_id(nid)
        self.presets_changed.emit()

    def _on_delete_user(self) -> None:
        if not self._selected_id:
            return
        pid = self._selected_id
        ans = QMessageBox.question(
            self,
            "Delete user preset",
            f'Remove user YAML for "{pid}" if it exists? Built-in presets cannot be deleted.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        deleted = self._services.presets.delete_user(pid)
        if deleted:
            self._reload_list_only()
            self.presets_changed.emit()
        else:
            QMessageBox.information(
                self,
                "Not deleted",
                "No user YAML found for this id (built-in presets ship with the app).",
            )

    def _select_by_id(self, preset_id: str) -> None:
        for row in range(self._list.count()):
            it = self._list.item(row)
            if it and it.data(Qt.ItemDataRole.UserRole) == preset_id:
                self._list.setCurrentItem(it)
                return

    @staticmethod
    def _format_validation_error(exc: ValidationError) -> str:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err.get("loc", ()))
            msg = err.get("msg", "")
            parts.append(f"{loc}: {msg}")
        return "; ".join(parts[:8]) + (" …" if len(parts) > 8 else "")

    @staticmethod
    def _prompt_text(parent: QWidget, title: str, label: str, default: str) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog, QLineEdit

        text, ok = QInputDialog.getText(
            parent, title, label, QLineEdit.EchoMode.Normal, default,
        )
        return text, ok

    def _open_dir(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_presets_dir())))
