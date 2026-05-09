"""Pipeline runner unit tests using fake stages.

We exercise: cancellation, error → rollback, cache-hit skipping, success accounting.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from aep.errors import CancelledError, PausedError, PipelineError
from aep.persist.db import init_db
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.cache import record as cache_record
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink
from aep.pipeline.runner import PipelineRunner, build_default_stages
from aep.pipeline.stage import BaseStage, StagePlan, StageResult


class _FakeStage(BaseStage):
    def __init__(self, name: str, *, raise_on_run: Exception | None = None,
                 mark_cached: bool = False) -> None:
        self.name = name
        self._raise = raise_on_run
        self._mark_cached = mark_cached
        self.ran = False
        self.rolled_back = False

    def plan(self, ctx: PipelineContext) -> StagePlan:
        key = compute_cache_key(
            source_fingerprint="fake", stage_name=self.name,
            tool_versions={}, params={"x": 1},
        )
        return StagePlan(stage_name=self.name, cache_key=key, params={"x": 1})

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        self.ran = True
        if self._raise:
            raise self._raise
        return StageResult(stage_name=self.name, success=True, duration_s=0.01)

    def rollback(self, ctx: PipelineContext, plan: StagePlan) -> None:
        self.rolled_back = True


def _make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        job_id="testjob",
        source_path=tmp_path / "src.mkv",
        workdir=tmp_path / "work",
        output_path=tmp_path / "out.mkv",
        preset_id="anime_balanced",
        preset_data={},
    )


def test_runs_all_stages(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    ctx = _make_ctx(tmp_path)
    a, b, c = _FakeStage("a"), _FakeStage("b"), _FakeStage("c")
    runner = PipelineRunner([a, b, c])
    results = runner.run(ctx, EventSink())
    assert all(s.ran for s in (a, b, c))
    assert all(results[k].success for k in ("a", "b", "c"))


def test_cancel_propagates(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    ctx = _make_ctx(tmp_path)

    class _CancelMidStage(BaseStage):
        name = "cancel"
        def plan(self, ctx: PipelineContext) -> StagePlan:
            return StagePlan(stage_name=self.name, cache_key="k", params={})
        def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
            ctx.cancel_event.set()
            ctx.check_cancel()
            return StageResult(stage_name=self.name, success=True, duration_s=0)

    runner = PipelineRunner([_CancelMidStage()])
    with pytest.raises(CancelledError):
        runner.run(ctx, EventSink())


def test_failure_triggers_rollback_and_stops(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    ctx = _make_ctx(tmp_path)
    a = _FakeStage("a")
    boom = _FakeStage("boom", raise_on_run=RuntimeError("kapow"))
    c = _FakeStage("c")
    runner = PipelineRunner([a, boom, c])
    with pytest.raises(PipelineError):
        runner.run(ctx, EventSink())
    assert a.ran is True
    assert boom.ran is True
    assert boom.rolled_back is True
    assert c.ran is False  # later stages are not invoked


def test_cache_hit_skips_execution(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    ctx = _make_ctx(tmp_path)
    a = _FakeStage("a")
    plan = a.plan(ctx)
    cache_record(ctx.job_id, a.name, plan.cache_key, ctx.workdir / a.name)
    runner = PipelineRunner([a])
    results = runner.run(ctx, EventSink())
    assert a.ran is False
    assert results["a"].cached is True


def test_pause_then_cancel(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    ctx = _make_ctx(tmp_path)
    ctx.pause_event.set()

    class _Trivial(BaseStage):
        name = "t"
        def plan(self, ctx: PipelineContext) -> StagePlan:
            return StagePlan(stage_name=self.name, cache_key="k", params={})
        def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
            return StageResult(stage_name=self.name, success=True, duration_s=0)

    runner = PipelineRunner([_Trivial()])

    def _later() -> None:
        ctx.cancel_event.set()

    threading.Timer(0.2, _later).start()
    with pytest.raises(CancelledError):
        runner.run(ctx, EventSink())


def test_resume_from_stage_skips_earlier(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    ctx = _make_ctx(tmp_path)
    ctx.extras["resume_from_stage"] = "b"
    a, b, c = _FakeStage("a"), _FakeStage("b"), _FakeStage("c")
    runner = PipelineRunner([a, b, c])
    runner.run(ctx, EventSink())
    assert a.ran is False
    assert b.ran is True
    assert c.ran is True


def test_paused_error_bubbles(tmp_runtime: Path, tmp_path: Path) -> None:
    init_db()
    ctx = _make_ctx(tmp_path)

    class _PauseMidStage(BaseStage):
        name = "pause"

        def plan(self, ctx: PipelineContext) -> StagePlan:
            return StagePlan(stage_name=self.name, cache_key="k", params={})

        def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
            raise PausedError("pause requested")

    runner = PipelineRunner([_PauseMidStage()])
    with pytest.raises(PausedError):
        runner.run(ctx, EventSink())


def test_build_default_stages_interpolate_first_orders_interp_before_upscale() -> None:
    names = [s.name for s in build_default_stages(order="interpolate_first")]
    assert names.index("06_interpolate") < names.index("05_upscale")


def test_build_default_stages_default_is_interpolate_first() -> None:
    names = [s.name for s in build_default_stages()]
    assert names.index("06_interpolate") < names.index("05_upscale")


def test_build_default_stages_upscale_first_orders_upscale_before_interp() -> None:
    names = [s.name for s in build_default_stages(order="upscale_first")]
    assert names.index("05_upscale") < names.index("06_interpolate")
