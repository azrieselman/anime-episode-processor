from __future__ import annotations

import logging
import shutil
from pathlib import Path

from aep.util.paths import jobs_dir

log = logging.getLogger(__name__)


def cleanup_job_artifacts(job_id: str, *, ramdisk_path: Path | None = None) -> None:
    workdir = jobs_dir() / job_id
    _safe_rmtree(workdir)
    if ramdisk_path is not None:
        _safe_rmtree(ramdisk_path / job_id)


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        log.warning("failed to remove path %s: %s", path, exc)
