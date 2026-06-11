"""Tests for RIFE Vulkan GPU-fault detection and retry in stage 06."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from aep.errors import StageError
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stages.s06_interpolate import _MAX_RIFE_GPU_FAULT_ATTEMPTS, InterpolateStage
from aep.util.proc import ProcError, ProcInterrupted, ProcResult


class _FakeRifeAdapter:
    def build_rife_argv(self, job: object) -> list[str]:
        return ["rife-ncnn-vulkan.exe", "-i", "in", "-o", "out"]


def _ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        job_id="job1",
        source_path=tmp_path / "src.mkv",
        workdir=tmp_path / "work",
        output_path=tmp_path / "out.mkv",
        preset_id="default",
        preset_data={},
        cancel_event=threading.Event(),
        pause_event=threading.Event(),
    )


def _events() -> tuple[EventSink, list[StageEvent]]:
    captured: list[StageEvent] = []

    class _Sink(EventSink):
        def emit(self, event: StageEvent) -> None:
            captured.append(event)

    return _Sink(), captured


def _streaming_gpu_fault_then_success(
    calls: list[int],
) -> object:
    def _fake_run_streaming(cmd, **kwargs) -> Iterator[tuple[str, str]]:
        calls.append(1)
        if len(calls) == 1:
            yield ("stderr", "vkQueueSubmit failed -4")
            reason = kwargs.get("should_interrupt")
            if reason is not None:
                reason()
            raise ProcInterrupted(
                "gpu_fault",
                ProcResult([str(x) for x in cmd], -1, "", "vkQueueSubmit failed -4"),
            )
        if False:  # pragma: no cover
            yield ("stdout", "")

    return _fake_run_streaming


def _streaming_persistent_gpu_fault() -> object:
    def _fake_run_streaming(cmd, **kwargs) -> Iterator[tuple[str, str]]:
        yield ("stderr", "vkQueueSubmit failed -4")
        reason = kwargs.get("should_interrupt")
        if reason is not None:
            reason()
        raise ProcInterrupted(
            "gpu_fault",
            ProcResult([str(x) for x in cmd], -1, "", "vkQueueSubmit failed -4"),
        )

    return _fake_run_streaming


def test_run_rife_retries_after_vulkan_gpu_fault(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    monkeypatch.setattr(
        "aep.pipeline.stages.s06_interpolate.run_streaming",
        _streaming_gpu_fault_then_success(calls),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    events, captured = _events()
    stage = InterpolateStage(rife=_FakeRifeAdapter())  # type: ignore[arg-type]

    stage._run_rife(
        adapter=_FakeRifeAdapter(),  # type: ignore[arg-type]
        version="v4.22-lite",
        multiplier=2,
        input_dir=tmp_path / "in",
        output_dir=out_dir,
        frame_format="png",
        threads="1:2:2",
        ctx=_ctx(tmp_path),
        events=events,
    )

    assert len(calls) == 2
    retry_logs = [
        e.message for e in captured
        if e.kind == "log" and e.message and "retrying interpolation" in e.message
    ]
    assert len(retry_logs) == 1


def test_run_rife_raises_after_exhausted_gpu_fault_retries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "aep.pipeline.stages.s06_interpolate.run_streaming",
        _streaming_persistent_gpu_fault(),
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    events, _ = _events()
    stage = InterpolateStage(rife=_FakeRifeAdapter())  # type: ignore[arg-type]

    with pytest.raises(StageError, match="Vulkan GPU fault persisted"):
        stage._run_rife(
            adapter=_FakeRifeAdapter(),  # type: ignore[arg-type]
            version="v4.22-lite",
            multiplier=2,
            input_dir=tmp_path / "in",
            output_dir=out_dir,
            frame_format="png",
            threads="1:2:2",
            ctx=_ctx(tmp_path),
            events=events,
        )


def test_run_rife_retries_when_stderr_shows_fault_but_exit_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[int] = []

    def _fake_run_streaming(cmd, **kwargs) -> Iterator[tuple[str, str]]:
        calls.append(1)
        if len(calls) == 1:
            yield ("stderr", "vkQueueSubmit failed -4")
            return
        if False:  # pragma: no cover
            yield ("stdout", "")

    monkeypatch.setattr("aep.pipeline.stages.s06_interpolate.run_streaming", _fake_run_streaming)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    events, captured = _events()
    stage = InterpolateStage(rife=_FakeRifeAdapter())  # type: ignore[arg-type]

    stage._run_rife(
        adapter=_FakeRifeAdapter(),  # type: ignore[arg-type]
        version="v4.22-lite",
        multiplier=2,
        input_dir=tmp_path / "in",
        output_dir=out_dir,
        frame_format="png",
        threads="1:2:2",
        ctx=_ctx(tmp_path),
        events=events,
    )

    assert len(calls) == 2
    assert _MAX_RIFE_GPU_FAULT_ATTEMPTS == 3


def test_run_rife_retries_on_proc_error_with_gpu_fault_stderr(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls: list[int] = []

    def _fake_run_streaming(cmd, **kwargs) -> Iterator[tuple[str, str]]:
        calls.append(1)
        if len(calls) == 1:
            raise ProcError(
                ProcResult([str(x) for x in cmd], 1, "", "vkQueueSubmit failed -4"),
            )
        if False:  # pragma: no cover
            yield ("stdout", "")

    monkeypatch.setattr("aep.pipeline.stages.s06_interpolate.run_streaming", _fake_run_streaming)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    events, _ = _events()
    stage = InterpolateStage(rife=_FakeRifeAdapter())  # type: ignore[arg-type]

    stage._run_rife(
        adapter=_FakeRifeAdapter(),  # type: ignore[arg-type]
        version="v4.22-lite",
        multiplier=2,
        input_dir=tmp_path / "in",
        output_dir=out_dir,
        frame_format="png",
        threads="1:2:2",
        ctx=_ctx(tmp_path),
        events=events,
    )

    assert len(calls) == 2


def test_run_rife_succeeds_without_retry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []

    def _fake_run_streaming(cmd, **kwargs) -> Iterator[tuple[str, str]]:
        calls.append(1)
        yield ("stderr", "processing frame 10/100")
        if False:  # pragma: no cover
            yield ("stdout", "")

    monkeypatch.setattr("aep.pipeline.stages.s06_interpolate.run_streaming", _fake_run_streaming)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    events, captured = _events()
    stage = InterpolateStage(rife=_FakeRifeAdapter())  # type: ignore[arg-type]

    stage._run_rife(
        adapter=_FakeRifeAdapter(),  # type: ignore[arg-type]
        version="v4.22-lite",
        multiplier=2,
        input_dir=tmp_path / "in",
        output_dir=out_dir,
        frame_format="png",
        threads="1:2:2",
        ctx=_ctx(tmp_path),
        events=events,
    )

    assert len(calls) == 1
    assert not any(
        e.message and "retrying interpolation" in e.message
        for e in captured
        if e.kind == "log"
    )
