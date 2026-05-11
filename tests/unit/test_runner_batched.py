"""Runner-level tests for the M6.5 batched pipeline path.

We exercise the runner's three-phase decomposition (pre-batch → per-batch ×N
→ post-batch) using fake stages that share names with the real pipeline so
the partitioning logic in ``PipelineRunner.run()`` actually triggers.

Coverage:
  * batched flow runs per-batch stages once per batch, in order
  * each batch's pts_window is pushed into ctx.plan and cleared after
  * encoded segments accumulate in ctx.encoded_segments in batch-index order
  * cleanup_batch_dir is invoked after each batch's segment is copied
  * RAM-disk gate failure aborts before any per-batch stage runs
  * empty/missing batches list keeps the legacy single-pass behavior
  * mid-batch failure leaves ``_active_batch_idx`` cleared so the next job
    starts clean
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aep.errors import PipelineError
from aep.persist.db import init_db
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink
from aep.pipeline.runner import PipelineRunner
from aep.pipeline.stage import BaseStage, StagePlan, StageResult


class _NamedStage(BaseStage):
    """Fake stage that records every (batch_idx, pts_window) it sees.

    The name is whatever the test passes in — by using real per-batch stage
    names ("04_decode_serve", "08_encode") we trigger the runner's batched
    partitioning. Stage 08 also writes a fake video.mkv into stage_dir() so
    the runner's segment-copy step finds something to ship out.
    """

    def __init__(self, name: str, *, raise_on_batch: int | None = None) -> None:
        self.name = name
        self.calls: list[tuple[int | None, tuple[float, float] | None]] = []
        self._raise_on_batch = raise_on_batch

    def plan(self, ctx: PipelineContext) -> StagePlan:
        # Cache key must vary per batch so the persistent cache doesn't
        # collide; in batched mode the runner suppresses the cache anyway.
        idx = ctx._active_batch_idx
        return StagePlan(
            stage_name=self.name,
            cache_key=f"{self.name}-{idx}",
            params={"idx": idx},
            inputs=[],
            outputs=[],
        )

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        idx = ctx._active_batch_idx
        window = ctx.plan.get("decode", {}).get("pts_window")
        self.calls.append((idx, tuple(window) if window else None))

        if self._raise_on_batch is not None and idx == self._raise_on_batch:
            raise RuntimeError(f"stage {self.name} blew up on batch {idx}")

        # 08_encode writes the segment that the runner expects to copy out.
        if self.name == "08_encode":
            stage_dir = ctx.stage_dir(self.name)
            video = stage_dir / "video.mkv"
            video.write_bytes(b"fake mkv contents for batch " + str(idx).encode())
        return StageResult(stage_name=self.name, success=True, duration_s=0.0)

    def rollback(self, ctx: PipelineContext, plan: StagePlan) -> None:
        return None


def _ctx_with_ramdisk(tmp_path: Path, ramdisk: Path) -> PipelineContext:
    ramdisk.mkdir(parents=True, exist_ok=True)
    return PipelineContext(
        job_id="batchjob",
        source_path=tmp_path / "src.mkv",
        workdir=tmp_path / "work",
        output_path=tmp_path / "out.mkv",
        preset_id="p",
        preset_data={},
        ramdisk_path=ramdisk,
    )


def _batch_plan(*windows: tuple[float, float]) -> list[dict[str, Any]]:
    return [
        {
            "index": i,
            "start_pts": s,
            "end_pts": e,
            "frame_count_estimate": 720,
            "est_bytes": 0,  # 0 = skip RAM-disk gate (verified separately)
            "duration_s": e - s,
        }
        for i, (s, e) in enumerate(windows)
    ]


# --------------------------------------------------------------------- batched

def test_batched_runs_per_batch_in_order(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    init_db()
    ramdisk = tmp_path / "ram"
    ctx = _ctx_with_ramdisk(tmp_path, ramdisk)
    ctx.plan = {"batches": _batch_plan((0.0, 30.0), (30.0, 60.0), (60.0, 90.0))}

    decode = _NamedStage("04_decode_serve")
    encode = _NamedStage("08_encode")
    runner = PipelineRunner([decode, encode])
    runner.run(ctx, EventSink())

    # Both per-batch stages saw exactly 3 calls, with monotonic batch indices
    # and matching pts windows.
    assert [c[0] for c in decode.calls] == [0, 1, 2]
    assert [c[0] for c in encode.calls] == [0, 1, 2]
    assert decode.calls[0][1] == (0.0, 30.0)
    assert decode.calls[2][1] == (60.0, 90.0)

    # Segments were copied out in order, into <workdir>/batch_segments/.
    assert len(ctx.encoded_segments) == 3
    assert ctx.encoded_segments[0].name == "segment_00.mkv"
    assert ctx.encoded_segments[2].name == "segment_02.mkv"
    for seg in ctx.encoded_segments:
        assert seg.exists() and seg.stat().st_size > 0

    # _active_batch_idx is cleared after the loop and pts_window is removed.
    assert ctx._active_batch_idx is None
    assert "pts_window" not in ctx.plan.get("decode", {})


def test_batched_cleans_up_each_batch_dir(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    init_db()
    ramdisk = tmp_path / "ram"
    ctx = _ctx_with_ramdisk(tmp_path, ramdisk)
    ctx.plan = {"batches": _batch_plan((0.0, 30.0), (30.0, 60.0))}

    runner = PipelineRunner([_NamedStage("04_decode_serve"), _NamedStage("08_encode")])
    runner.run(ctx, EventSink())

    # cleanup_batch_dir removes each <ramdisk>/<job_id>/batch_NN/.
    job_root = ramdisk / ctx.job_id
    leftovers = [p for p in job_root.glob("batch_*") if p.is_dir()] if job_root.exists() else []
    assert leftovers == []


def test_empty_batches_uses_single_pass(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    """When ctx.plan['batches'] is empty (or missing), legacy single-pass runs."""
    init_db()
    ctx = PipelineContext(
        job_id="legacy",
        source_path=tmp_path / "src.mkv",
        workdir=tmp_path / "work",
        output_path=tmp_path / "out.mkv",
        preset_id="p",
        preset_data={},
    )
    ctx.plan = {}  # no "batches" key

    decode = _NamedStage("04_decode_serve")
    encode = _NamedStage("08_encode")
    runner = PipelineRunner([decode, encode])
    runner.run(ctx, EventSink())

    # Each stage ran exactly once with no batch index.
    assert decode.calls == [(None, None)]
    assert encode.calls == [(None, None)]
    assert ctx.encoded_segments == []


def test_batched_raises_on_ramdisk_undersize(
    tmp_runtime: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The RAM-disk gate fires before any per-batch stage executes."""
    from collections import namedtuple

    init_db()
    ramdisk = tmp_path / "ram"
    ctx = _ctx_with_ramdisk(tmp_path, ramdisk)
    # est_bytes set to a non-zero value triggers the gate.
    ctx.plan = {"batches": [{
        "index": 0, "start_pts": 0.0, "end_pts": 30.0,
        "frame_count_estimate": 720, "est_bytes": 10 * 1024 * 1024 * 1024,
        "duration_s": 30.0,
    }]}

    Usage = namedtuple("Usage", ["total", "used", "free"])
    monkeypatch.setattr(
        "aep.pipeline.context.shutil.disk_usage",
        lambda _p: Usage(total=20 * 1024 ** 3, used=20 * 1024 ** 3, free=1 * 1024 * 1024),
    )

    decode = _NamedStage("04_decode_serve")
    runner = PipelineRunner([decode, _NamedStage("08_encode")])
    with pytest.raises(PipelineError, match="RAM-disk insufficient"):
        runner.run(ctx, EventSink())
    # No per-batch stage ran.
    assert decode.calls == []
    # State cleared even on failure.
    assert ctx._active_batch_idx is None


def test_mid_batch_failure_clears_active_batch(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    init_db()
    ramdisk = tmp_path / "ram"
    ctx = _ctx_with_ramdisk(tmp_path, ramdisk)
    ctx.plan = {"batches": _batch_plan((0.0, 30.0), (30.0, 60.0))}

    encode = _NamedStage("08_encode", raise_on_batch=1)
    runner = PipelineRunner([_NamedStage("04_decode_serve"), encode])
    with pytest.raises(Exception):
        runner.run(ctx, EventSink())

    # Cleanup of in-flight state happens via finally:
    assert ctx._active_batch_idx is None
    assert "pts_window" not in ctx.plan.get("decode", {})
    # First batch's segment was successfully captured before batch 1 failed.
    assert len(ctx.encoded_segments) == 1
    assert ctx.encoded_segments[0].name == "segment_00.mkv"


def test_resume_rehydrates_prior_segments_and_skips_completed_batches(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    """Resuming a batched job must not drop already-encoded segments.

    Historically, a resumed run built a fresh PipelineContext with an empty
    `encoded_segments` list, so stage 09 concatenation would see only the
    post-resume segments and produce a truncated output (caught by validate).
    """
    init_db()
    ramdisk = tmp_path / "ram"
    ctx = _ctx_with_ramdisk(tmp_path, ramdisk)
    ctx.plan = {"batches": _batch_plan((0.0, 30.0), (30.0, 60.0), (60.0, 90.0))}

    # Simulate a prior run that already produced batch 0's segment.
    seg_dir = ctx.workdir / "batch_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / "segment_00.mkv").write_bytes(b"done0")

    # Resume from 08_encode: runner should skip earlier stages (04_decode_serve)
    # and also skip already completed batches (batch 0), while rehydrating
    # encoded_segments from disk.
    ctx.extras["resume_from_stage"] = "08_encode"

    decode = _NamedStage("04_decode_serve")
    encode = _NamedStage("08_encode")
    runner = PipelineRunner([decode, encode])
    runner.run(ctx, EventSink())

    # decode should be skipped for the resumed batch (batch 0), but the runner
    # intentionally restores the full per-batch chain for subsequent batches.
    assert [c[0] for c in decode.calls] == [1, 2]
    # encode should run only for remaining batches (1 and 2).
    assert [c[0] for c in encode.calls] == [1, 2]

    # All segments should be present in order (0 recovered + 1,2 produced).
    assert [p.name for p in ctx.encoded_segments] == [
        "segment_00.mkv",
        "segment_01.mkv",
        "segment_02.mkv",
    ]
    for seg in ctx.encoded_segments:
        assert seg.exists() and seg.stat().st_size > 0


def test_malformed_batch_plan_fails_fast(
    tmp_runtime: Path, tmp_path: Path,
) -> None:
    init_db()
    ramdisk = tmp_path / "ram"
    ctx = _ctx_with_ramdisk(tmp_path, ramdisk)
    ctx.plan = {"batches": [{"index": "not-an-int", "start_pts": 0.0, "end_pts": 30.0}]}

    runner = PipelineRunner([_NamedStage("04_decode_serve"), _NamedStage("08_encode")])
    with pytest.raises(PipelineError, match="malformed batch"):
        runner.run(ctx, EventSink())
