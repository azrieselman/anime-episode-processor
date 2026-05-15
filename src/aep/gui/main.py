"""GUI entry point — `aep-gui.exe`."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from aep.app.bootstrap import bootstrap
from aep.constants import APP_DISPLAY_NAME, APP_ID, APP_VENDOR
from aep.gui import theme
from aep.gui.app_window import MainWindow
from aep.gui.win_shell import set_explicit_app_user_model_id


def main() -> int:
    # Windows taskbar icon follows the process AppUserModelID; set before any UI exists.
    set_explicit_app_user_model_id(APP_ID)

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setOrganizationName(APP_VENDOR)
    app.setOrganizationDomain(APP_ID)
    theme.apply(app)

    services = bootstrap()
    try:
        win = MainWindow(services)
        win.show()
        rc = app.exec()
    finally:
        services.stop()
    return rc


if __name__ == "__main__":
    sys.exit(main())
