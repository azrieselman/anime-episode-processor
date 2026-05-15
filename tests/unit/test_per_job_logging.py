"""Tests for per-job log file attach/detach + thread filtering."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import pytest

from aep.logging_setup import attach_job_log_handler, detach_job_log_handler


@pytest.fixture(autouse=True)
def _enable_info_logs():
    # The real app calls configure_logging() which sets root to INFO; tests
    # don't, so records would be filtered before reaching our handler.
    root = logging.getLogger()
    prior = root.level
    root.setLevel(logging.INFO)
    yield
    root.setLevel(prior)


def test_attach_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "job.log"
    handler = attach_job_log_handler(log_file, level=logging.INFO)
    try:
        logging.getLogger("aep.test").info("hello world")
        # Force the handler to flush so we can read the file.
        handler.flush()
    finally:
        detach_job_log_handler(handler)

    text = log_file.read_text(encoding="utf-8")
    assert "hello world" in text


def test_thread_filter_isolates_jobs(tmp_path: Path) -> None:
    """Two handlers on different threads must not see each other's logs."""
    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"

    barrier = threading.Barrier(2)
    handlers: dict[str, logging.Handler] = {}

    def worker(name: str, log_path: Path) -> None:
        h = attach_job_log_handler(
            log_path, level=logging.INFO, thread_id=threading.get_ident(),
        )
        handlers[name] = h
        # Both threads log the same message string but tagged with their name;
        # if filtering works each file only contains its own thread's lines.
        barrier.wait()
        for i in range(20):
            logging.getLogger("aep.test").info("from-%s-%d", name, i)
        h.flush()

    t_a = threading.Thread(target=worker, args=("A", log_a))
    t_b = threading.Thread(target=worker, args=("B", log_b))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    for h in handlers.values():
        detach_job_log_handler(h)

    text_a = log_a.read_text(encoding="utf-8")
    text_b = log_b.read_text(encoding="utf-8")
    assert "from-A-0" in text_a and "from-A-19" in text_a
    assert "from-B-0" in text_b and "from-B-19" in text_b
    # Crucially: no cross-talk.
    assert "from-B" not in text_a
    assert "from-A" not in text_b


def test_detach_is_safe_on_double_call(tmp_path: Path) -> None:
    h = attach_job_log_handler(tmp_path / "x.log")
    detach_job_log_handler(h)
    # Second call should not raise.
    detach_job_log_handler(h)


def test_detach_removes_handler_from_root(tmp_path: Path) -> None:
    root = logging.getLogger()
    before = len(root.handlers)
    h = attach_job_log_handler(tmp_path / "x.log")
    assert len(root.handlers) == before + 1
    detach_job_log_handler(h)
    assert len(root.handlers) == before
