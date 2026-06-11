"""Pipeline event sink.

Stages emit progress events through an EventSink rather than calling Qt signals directly.
That keeps stage code Qt-free and unit-testable. The GUI's JobBroker subscribes its own
sink that re-emits events on Qt signals.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageEvent:
    job_id: str
    stage: str
    kind: str            # "started"|"progress"|"log"|"warning"|"error"|"completed"|"skipped"
    message: str = ""
    progress: float | None = None  # 0..1
    fps: float | None = None
    eta_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


EventCallback = Callable[[StageEvent], None]


def emit_tool_log(events: EventSink, job_id: str, stage: str, line: str) -> None:
    """Forward a subprocess stderr/stdout line as a stage ``log`` event."""
    text = line.strip()
    if text:
        events.emit(StageEvent(job_id, stage, "log", message=text))


def stage_event_log_text(event: StageEvent) -> str | None:
    """Return the tool-output text carried by a ``log`` event, if any."""
    text = (event.message or "").strip()
    if text:
        return text
    extra = event.extra if isinstance(event.extra, dict) else {}
    raw = extra.get("ffmpeg_line")
    if isinstance(raw, str):
        text = raw.strip()
        return text or None
    return None


class EventSink:
    """Fan-out event emitter; supports multiple subscribers."""

    def __init__(self) -> None:
        self._subs: list[EventCallback] = []

    def subscribe(self, cb: EventCallback) -> None:
        self._subs.append(cb)

    def emit(self, event: StageEvent) -> None:
        for cb in self._subs:
            try:
                cb(event)
            except Exception:
                # Subscribers must be defensive; we still log.
                import logging
                logging.getLogger(__name__).exception(
                    "event subscriber raised; continuing", extra={"event": event.__dict__}
                )
