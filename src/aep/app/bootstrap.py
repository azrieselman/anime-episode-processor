"""Application bootstrap.

Single entry point for "set up the runtime, configure logging, init the DB, build
services." Used by both `aep-gui.exe` and `aep-cli`.
"""

from __future__ import annotations

import logging

from aep.app.hardware_defaults import apply_hardware_encoder_defaults
from aep.app.services import AppServices
from aep.logging_setup import configure_logging
from aep.persist.db import init_db
from aep.persist.settings import load_settings, save_settings
from aep.util.paths import ensure_runtime_dirs, logs_dir

log = logging.getLogger(__name__)


def bootstrap() -> AppServices:
    ensure_runtime_dirs()
    settings = load_settings()
    prev_hw_ver = settings.hardware_encoder_defaults_version
    settings = apply_hardware_encoder_defaults(settings)
    if settings.hardware_encoder_defaults_version != prev_hw_ver:
        save_settings(settings)
    configure_logging(logs_dir(), level=settings.general.log_level)
    init_db()
    services = AppServices()
    services.start()
    log.info("aep bootstrap complete")
    return services
