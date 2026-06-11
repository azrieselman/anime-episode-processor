from __future__ import annotations

import logging
from pathlib import Path

import pytest

from aep.jobs.broker import JobBroker
from aep.jobs.models import Job
from aep.pipeline.events import StageEvent, stage_event_log_text


def test_stage_event_log_text_prefers_message() -> None:
    ev = StageEvent("job1", "06_interpolate", "log", message="  frame 1/100  ")
    assert stage_event_log_text(ev) == "frame 1/100"


def test_stage_event_log_text_falls_back_to_ffmpeg_line() -> None:
    ev = StageEvent(
        "job1",
        "08_encode",
        "log",
        message="",
        extra={"ffmpeg_line": "frame=  42 fps=12.3"},
    )
    assert stage_event_log_text(ev) == "frame=  42 fps=12.3"


def test_broker_writes_stage_log_events_at_debug(
    tmp_runtime: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = Job(source_path=str(tmp_path / "in.mkv"))
    broker = JobBroker()
    ev = StageEvent(job.id, "06_interpolate", "log", message="frame 100/200")

    with caplog.at_level(logging.DEBUG, logger="aep.jobs.broker"):
        broker._on_stage_event(job, ev)

    assert "[06_interpolate] frame 100/200" in caplog.text
