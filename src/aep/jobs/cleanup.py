from __future__ import annotations

import logging
import shutil
from pathlib import Path

from aep.persist.db import connect
from aep.util.paths import jobs_dir

log = logging.getLogger(__name__)


def cleanup_job_artifacts(job_id: str, *, ramdisk_path: Path | None = None) -> None:
    workdir = jobs_dir() / job_id
    _safe_rmtree(workdir)
    if ramdisk_path is not None:
        _safe_rmtree(ramdisk_path / job_id)
    # The stage_cache rows we just orphaned point at directories on disk that
    # no longer exist. If we leave them in place, the next run of this job
    # gets a cache hit on (e.g.) 00_probe, the runner skips run() but the
    # rehydration of ctx.media_info silently no-ops (probe.json was deleted),
    # and 01_plan then fails with "requires 00_probe to have populated
    # ctx.media_info". Clearing the rows forces a clean re-execution.
    try:
        with connect() as conn:
            conn.execute("DELETE FROM stage_cache WHERE job_id=?", (job_id,))
    except Exception:
        log.warning(
            "failed to clear stage_cache for job %s", job_id, exc_info=True,
        )


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path, ignore_errors=False)
    except OSError as exc:
        log.warning("failed to remove path %s: %s", path, exc)
