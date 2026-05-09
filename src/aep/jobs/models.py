"""Job data model.

State machine:
   queued → running → (paused ⇄ running) → completed
                                          → failed
                                          → cancelled

States are persisted to sqlite; `progress` is a 0..1 float updated by stage events.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Job(BaseModel):
    id: str = Field(default_factory=_new_id)
    source_path: str
    output_path: str | None = None
    preset_id: str = "anime_balanced"
    state: JobState = JobState.QUEUED
    progress: float = 0.0
    error: str | None = None
    current_stage: str | None = None
    last_failed_stage: str | None = None
    resume_from_stage: str | None = None
    retry_count: int = 0
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    plan: dict[str, Any] = Field(default_factory=dict)
    probe: dict[str, Any] | None = None
    # Sparse per-job preset overrides. Deep-merged onto the loaded preset's
    # JSON dump before the pipeline context is built. None = no override
    # (use the preset as-is). Stored only for the fields the JobConfigView
    # exposes for editing -- everything else stays preset-controlled.
    preset_overrides: dict[str, Any] | None = None

    @property
    def source(self) -> Path:
        return Path(self.source_path)

    def is_terminal(self) -> bool:
        return self.state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED)
