"""Structured logging.

We emit two streams:

* Human log (`aep.log`, rotating, 5 MB × 5)
* JSONL log (`aep.jsonl`, one JSON object per line, for tooling)

Per-job logs are configured later by the job worker (see `aep.jobs.worker_main`); they
add a file handler scoped to that job's log directory.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any

from aep.constants import DEFAULT_LOG_LEVEL, ENV_LOG_LEVEL


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in {
                "args", "asctime", "created", "exc_info", "exc_text", "filename",
                "funcName", "levelname", "levelno", "lineno", "module", "msecs",
                "message", "msg", "name", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
            }:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(log_dir: Path, *, level: str | None = None) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)

    chosen_level = (level or os.environ.get(ENV_LOG_LEVEL) or DEFAULT_LOG_LEVEL).upper()
    numeric_level = getattr(logging, chosen_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Clear handlers (important when reconfiguring during tests / GUI restarts).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    # Console handler — short, human format.
    console = logging.StreamHandler(stream=sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # Rotating human log.
    human_path = log_dir / "aep.log"
    human = logging.handlers.RotatingFileHandler(
        human_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    human.setLevel(numeric_level)
    human.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s [%(threadName)s]: %(message)s",
    ))
    root.addHandler(human)

    # JSONL log for tooling.
    jsonl_path = log_dir / "aep.jsonl"
    jsonl = logging.handlers.RotatingFileHandler(
        jsonl_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    jsonl.setLevel(numeric_level)
    jsonl.setFormatter(_JsonFormatter())
    root.addHandler(jsonl)

    logging.captureWarnings(True)
    logging.getLogger("aep").info("logging configured", extra={"level": chosen_level, "dir": str(log_dir)})


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ----- per-job log files ----------------------------------------------------


class _ThreadFilter(logging.Filter):
    """Pass only records emitted from a specific thread.

    We attach a per-job FileHandler to the *root* logger (so every log call
    inside the pipeline ends up in the per-job file) and use this filter to
    keep one job's lines out of another job's file when concurrent jobs run
    on different worker threads.
    """

    def __init__(self, thread_id: int) -> None:
        super().__init__()
        self._thread_id = thread_id

    def filter(self, record: logging.LogRecord) -> bool:
        return record.thread == self._thread_id


def attach_job_log_handler(
    log_path: Path,
    *,
    level: str | int | None = None,
    thread_id: int | None = None,
) -> logging.handlers.RotatingFileHandler:
    """Attach a per-job rotating file handler to the root logger.

    Returns the handler so the caller can `detach_job_log_handler` it on job
    exit. When `thread_id` is given, only records emitted from that thread
    end up in the file; this matches the broker's threaded worker pool so
    concurrent jobs each get their own log file without cross-talk.

    Format mirrors the global aep.log so a quick eyeball comparison works,
    plus the thread name (handy when triaging cross-thread issues).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    chosen_level = level if level is not None else logging.getLogger().level
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8",
    )
    handler.setLevel(chosen_level)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(name)s [%(threadName)s]: %(message)s",
    ))
    if thread_id is not None:
        handler.addFilter(_ThreadFilter(thread_id))
    logging.getLogger().addHandler(handler)
    return handler


def detach_job_log_handler(handler: logging.Handler) -> None:
    """Remove a per-job handler and close it.

    Always safe to call — never raises even if the handler was already
    removed; we don't want logging cleanup to mask a real job error.
    """
    try:
        logging.getLogger().removeHandler(handler)
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass
