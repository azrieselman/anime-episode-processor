"""Stage protocol + supporting types.

A Stage is a small, well-defined piece of work:
1. `plan(ctx)` — pure-ish; computes a StagePlan (cache key + parameters). No side effects
   beyond logging.
2. `run(ctx, plan, events)` — does the work; emits events; returns a StageResult.
3. `can_skip(ctx)` — checked by the runner; usually defers to the cache.
4. `rollback(ctx, plan)` — best-effort cleanup if a later stage fails or job is cancelled.

Stages are stateless aside from the ctx they receive. Construction params are config; per-
job state lives in PipelineContext.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink


@dataclass
class StagePlan:
    stage_name: str
    cache_key: str
    params: dict[str, Any] = field(default_factory=dict)
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class StageResult:
    stage_name: str
    success: bool
    duration_s: float
    artifacts: dict[str, Path] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
    skipped: bool = False
    notes: list[str] = field(default_factory=list)


@runtime_checkable
class Stage(Protocol):
    name: str

    def plan(self, ctx: PipelineContext) -> StagePlan: ...
    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult: ...
    def can_skip(self, ctx: PipelineContext) -> bool: ...
    def rollback(self, ctx: PipelineContext, plan: StagePlan) -> None: ...


class BaseStage:
    """Optional convenience base. Stages can implement Stage Protocol directly without
    inheriting; this just gives a sensible default for `can_skip` and `rollback`.
    """

    name: str = "base"

    def can_skip(self, ctx: PipelineContext) -> bool:
        return False

    def rollback(self, ctx: PipelineContext, plan: StagePlan) -> None:
        return None
