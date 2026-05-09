"""Windows shell integration for the GUI process."""

from __future__ import annotations

import ctypes
import logging
import sys

log = logging.getLogger(__name__)


def set_explicit_app_user_model_id(app_id: str) -> None:
    """Tell Windows which AppUserModelID owns this process (taskbar pinning / icon).

    Without this, `python.exe` (or an unpackaged host) often owns the taskbar button,
    so `QApplication.setWindowIcon` does not affect the taskbar icon the user sees.

    Must run before any top-level windows exist (see MSDN for
    SetCurrentProcessExplicitAppUserModelID).
    """
    if sys.platform != "win32":
        return
    if not app_id.strip():
        return
    try:
        hr = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        if hr != 0:
            log.debug("SetCurrentProcessExplicitAppUserModelID returned hr=%s", hr)
    except Exception:
        log.exception("failed to set Windows AppUserModelID")
