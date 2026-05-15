"""Stage cache.

Stage outputs are content-addressed by:
  blake2(source_fingerprint + stage_name + tool_versions + stage_params)

If the same key has already produced outputs that still exist on disk and pass a sanity
check, the runner skips the stage. This is what makes "resume after restart" work without
custom resume code in every stage.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from aep.persist.db import connect
from aep.util.hashing import hash_json

log = logging.getLogger(__name__)


def compute_cache_key(
    *,
    source_fingerprint: str,
    stage_name: str,
    tool_versions: dict[str, str],
    params: dict[str, object],
) -> str:
    payload = {
        "source": source_fingerprint,
        "stage": stage_name,
        "tools": dict(sorted(tool_versions.items())),
        "params": params,
    }
    return hash_json(payload)


def lookup(job_id: str, stage_name: str) -> tuple[str, Path] | None:
    with connect() as conn:
        row = conn.execute(
            "SELECT cache_key, output_dir FROM stage_cache WHERE job_id=? AND stage_name=?",
            (job_id, stage_name),
        ).fetchone()
        if not row:
            return None
        return row["cache_key"], Path(row["output_dir"])


def record(job_id: str, stage_name: str, cache_key: str, output_dir: Path) -> None:
    now = datetime.now(UTC).isoformat()
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO stage_cache(job_id, stage_name, cache_key, output_dir, completed_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, stage_name, cache_key, str(output_dir), now),
        )
    log.debug("stage cache recorded: %s/%s", job_id, stage_name)


def write_stage_manifest(stage_dir: Path, payload: dict) -> None:
    p = stage_dir / "stage.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def read_stage_manifest(stage_dir: Path) -> dict | None:
    p = stage_dir / "stage.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("stage manifest unreadable: %s", p)
        return None
