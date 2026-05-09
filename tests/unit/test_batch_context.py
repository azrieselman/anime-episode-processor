"""Tests for the M6.5 PipelineContext batched storage helpers.

Covers:
  * batch_dir() returns a RAM-disk-rooted path and creates it
  * batch_dir() hard-fails when ramdisk_path is None
  * assert_ramdisk_has_room_for() raises when free space is insufficient
  * assert_ramdisk_has_room_for() passes when there's enough room
  * assert_ramdisk_has_room_for() skips the check when est_bytes is 0
  * cleanup_batch_dir() removes the batch dir; idempotent on missing dir
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pytest

from aep.errors import PipelineError
from aep.pipeline.batches import BatchSpec
from aep.pipeline.context import PipelineContext

_FakeUsage = namedtuple("FakeUsage", ["total", "used", "free"])


def _ctx(workdir: Path, ramdisk: Path | None) -> PipelineContext:
    return PipelineContext(
        job_id="abc123",
        source_path=workdir / "src.mkv",
        workdir=workdir,
        output_path=workdir / "out.mkv",
        preset_id="anime_balanced",
        preset_data={},
        ramdisk_path=ramdisk,
    )


def test_batch_dir_creates_path_under_ramdisk(tmp_path: Path) -> None:
    ramdisk = tmp_path / "ram"
    ramdisk.mkdir()
    ctx = _ctx(tmp_path / "work", ramdisk)
    d = ctx.batch_dir(3, "05_upscale")
    assert d == ramdisk / "abc123" / "batch_03" / "05_upscale"
    assert d.is_dir()


def test_batch_dir_zero_padded_index_format(tmp_path: Path) -> None:
    ramdisk = tmp_path / "ram"
    ramdisk.mkdir()
    ctx = _ctx(tmp_path / "work", ramdisk)
    assert ctx.batch_dir(0, "x").name == "x"
    assert ctx.batch_dir(0, "x").parent.name == "batch_00"
    assert ctx.batch_dir(15, "x").parent.name == "batch_15"


def test_batch_dir_hard_fails_without_ramdisk(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path / "work", ramdisk=None)
    with pytest.raises(PipelineError, match="ramdisk_path"):
        ctx.batch_dir(0, "05_upscale")


def test_assert_ramdisk_room_passes_when_free_space_sufficient(tmp_path: Path) -> None:
    ramdisk = tmp_path / "ram"
    ramdisk.mkdir()
    ctx = _ctx(tmp_path / "work", ramdisk)
    batch = BatchSpec(
        index=0, start_pts=0.0, end_pts=30.0,
        frame_count_estimate=1800, est_bytes=100 * 1024 * 1024,  # 100 MiB
    )
    # 200 MiB free > 130 MiB required (100 MiB × 1.3).
    with patch(
        "aep.pipeline.context.shutil.disk_usage",
        return_value=_FakeUsage(total=1 << 30, used=0, free=200 * 1024 * 1024),
    ):
        ctx.assert_ramdisk_has_room_for(batch)


def test_assert_ramdisk_room_raises_on_insufficient_free_space(tmp_path: Path) -> None:
    ramdisk = tmp_path / "ram"
    ramdisk.mkdir()
    ctx = _ctx(tmp_path / "work", ramdisk)
    batch = BatchSpec(
        index=2, start_pts=60.0, end_pts=90.0,
        frame_count_estimate=1800, est_bytes=100 * 1024 * 1024,
    )
    # Only 100 MiB free < 130 MiB required.
    with patch(
        "aep.pipeline.context.shutil.disk_usage",
        return_value=_FakeUsage(total=1 << 30, used=0, free=100 * 1024 * 1024),
    ), pytest.raises(PipelineError) as excinfo:
        ctx.assert_ramdisk_has_room_for(batch)
    msg = str(excinfo.value)
    assert "batch 02" in msg
    assert "RAM-disk insufficient" in msg
    assert "MiB" in msg
    # Context is preserved so callers / log scrapers can introspect.
    assert excinfo.value.context["batch_idx"] == 2
    assert excinfo.value.context["free_bytes"] == 100 * 1024 * 1024


def test_assert_ramdisk_room_hard_fails_when_no_ramdisk(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path / "work", ramdisk=None)
    batch = BatchSpec(0, 0.0, 30.0, 1800, 1)
    with pytest.raises(PipelineError, match="ramdisk_path"):
        ctx.assert_ramdisk_has_room_for(batch)


def test_assert_ramdisk_room_skips_when_estimate_unknown(tmp_path: Path) -> None:
    """est_bytes=0 means the planner couldn't compute; gate should not block."""
    ramdisk = tmp_path / "ram"
    ramdisk.mkdir()
    ctx = _ctx(tmp_path / "work", ramdisk)
    batch = BatchSpec(0, 0.0, 30.0, 0, 0)
    # Even with 0 bytes free, this is a no-op.
    with patch(
        "aep.pipeline.context.shutil.disk_usage",
        return_value=_FakeUsage(total=1 << 30, used=1 << 30, free=0),
    ):
        ctx.assert_ramdisk_has_room_for(batch)  # must not raise


def test_cleanup_batch_dir_removes_existing_dir(tmp_path: Path) -> None:
    ramdisk = tmp_path / "ram"
    ramdisk.mkdir()
    ctx = _ctx(tmp_path / "work", ramdisk)
    d = ctx.batch_dir(7, "05_upscale")
    (d / "frame_0001.png").write_bytes(b"x")
    assert d.exists()
    ctx.cleanup_batch_dir(7)
    assert not (ramdisk / "abc123" / "batch_07").exists()


def test_cleanup_batch_dir_idempotent_on_missing(tmp_path: Path) -> None:
    ramdisk = tmp_path / "ram"
    ramdisk.mkdir()
    ctx = _ctx(tmp_path / "work", ramdisk)
    # No batch_dir() call yet — directory doesn't exist. Cleanup must not raise.
    ctx.cleanup_batch_dir(0)
    ctx.cleanup_batch_dir(99)


def test_cleanup_batch_dir_noop_without_ramdisk(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path / "work", ramdisk=None)
    # No ramdisk → unbatched mode. cleanup is a no-op (no exception).
    ctx.cleanup_batch_dir(0)
