"""Placeholder stage.

Used to fill stage slots that have no concrete implementation yet. Lets the runner walk
the full ordered stage list and emit normal events even when a stage is a no-op.

A placeholder is NOT a no-op in the destructive sense — it never deletes anything — but
it produces no real artifacts. It's marked with `notes=["placeholder"]` so validation
and tests can detect "we shouldn't be in production with placeholders left."
"""

from __future__ import annotations

import time

from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult


class PlaceholderStage(BaseStage):
    def __init__(self, name: str) -> None:
        self.name = name

    def plan(self, ctx: PipelineContext) -> StagePlan:
        cache_key = compute_cache_key(
            source_fingerprint="placeholder",
            stage_name=self.name,
            tool_versions={},
            params={"placeholder": True, "preset": ctx.preset_id},
        )
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params={"placeholder": True},
            notes=["placeholder stage; no work performed"],
        )

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        events.emit(StageEvent(ctx.job_id, self.name, "log",
                               message="placeholder stage executing (no-op)"))
        # Cooperative cancel point even for placeholders
        ctx.check_cancel()
        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            notes=["placeholder"],
        )
