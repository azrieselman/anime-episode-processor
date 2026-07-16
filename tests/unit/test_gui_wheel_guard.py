"""Smoke tests for wheel-value guard on form controls."""

from __future__ import annotations

import os

import pytest


@pytest.fixture
def qapp(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(["aep-wheel-guard-smoke"])
    return app


def test_disable_wheel_value_changes_blocks_spin_and_combo(qapp) -> None:
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QComboBox, QSpinBox, QVBoxLayout, QWidget

    from aep.gui.widgets.wheel_guard import disable_wheel_value_changes

    root = QWidget()
    layout = QVBoxLayout(root)
    spin = QSpinBox()
    spin.setRange(0, 100)
    spin.setValue(10)
    combo = QComboBox()
    combo.addItems(["a", "b", "c"])
    combo.setCurrentIndex(0)
    layout.addWidget(spin)
    layout.addWidget(combo)
    disable_wheel_value_changes(root)

    assert spin.focusPolicy() == Qt.FocusPolicy.StrongFocus
    assert combo.focusPolicy() == Qt.FocusPolicy.StrongFocus

    def _wheel(widget: QWidget) -> None:
        event = QWheelEvent(
            QPointF(1, 1),
            QPointF(1, 1),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        qapp.sendEvent(widget, event)

    _wheel(spin)
    _wheel(combo)
    assert spin.value() == 10
    assert combo.currentIndex() == 0
