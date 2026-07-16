"""Prevent mouse-wheel from changing spin box / combo box values.

Scrolling over a QSpinBox, QDoubleSpinBox, or QComboBox normally nudges the
value. On long forms that is easy to do by accident while scrolling the page.
This guard eats those wheel events (and forwards them to an enclosing scroll
area when present) so values only change via click, keyboard, or the spin
arrows.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollArea,
    QWidget,
)

_WHEEL_TARGETS = (QAbstractSpinBox, QComboBox)


class _WheelValueGuard(QObject):
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if event.type() != QEvent.Type.Wheel:
            return False
        if not isinstance(watched, _WHEEL_TARGETS):
            return False
        parent = watched.parentWidget()
        while parent is not None:
            if isinstance(parent, QScrollArea):
                QApplication.sendEvent(parent.viewport(), event)
                return True
            parent = parent.parentWidget()
        event.ignore()
        return True


_GUARD: _WheelValueGuard | None = None


def disable_wheel_value_changes(root: QWidget) -> None:
    """Stop wheel events from changing spin/combo values under *root*."""
    global _GUARD
    if _GUARD is None:
        _GUARD = _WheelValueGuard()
    targets: list[QWidget] = []
    if isinstance(root, _WHEEL_TARGETS):
        targets.append(root)
    targets.extend(root.findChildren(QAbstractSpinBox))
    targets.extend(root.findChildren(QComboBox))
    for widget in targets:
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        widget.installEventFilter(_GUARD)
