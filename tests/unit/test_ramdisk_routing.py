"""Tests for `PipelineContext.stage_dir()` ramdisk routing and `_ramdisk_usable()`.

The ramdisk routing logic only applies to a small set of frame-heavy stages
(`_RAMDISK_STAGES`); other stages always use the regular workdir.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

from aep.pipeline.context import (
    _RAMDISK_STAGES,
    PipelineContext,
    _ramdisk_usable,
)

# --- helpers ------------------------------------------------------------


def _ctx(
    *,
    workdir: Path,
    output_path: Path,
    ramdisk_path: Path | None = None,
    ramdisk_estimate_bytes: int = 0,
) -> PipelineContext:
    return PipelineContext(
        job_id="job-abc",
        source_path=workdir / "src.mkv",
        workdir=workdir,
        output_path=output_path,
        preset_id="test",
        preset_data={},
        ramdisk_path=ramdisk_path,
        ramdisk_estimate_bytes=ramdisk_estimate_bytes,
    )


# --- stage_dir routing -------------------------------------------------


def test_no_ramdisk_uses_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    out = tmp_path / "out.mkv"
    ctx = _ctx(workdir=workdir, output_path=out, ramdisk_path=None)
    d = ctx.stage_dir("05_upscale")
    assert d == workdir / "05_upscale"
    assert d.is_dir()


def test_ramdisk_set_routes_frame_stage_onto_ramdisk(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    rd = tmp_path / "rd"
    out = tmp_path / "out.mkv"
    ctx = _ctx(workdir=workdir, output_path=out, ramdisk_path=rd)
    d = ctx.stage_dir("05_upscale")
    # Layout: <ramdisk>/<job_id>/<stage_name>/
    assert d == rd / "job-abc" / "05_upscale"
    assert d.is_dir()


def test_ramdisk_set_but_non_frame_stage_uses_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    rd = tmp_path / "rd"
    out = tmp_path / "out.mkv"
    ctx = _ctx(workdir=workdir, output_path=out, ramdisk_path=rd)
    # Encode/mux/validate are NOT in _RAMDISK_STAGES
    d = ctx.stage_dir("08_encode")
    assert d == workdir / "08_encode"
    assert rd.exists() is False or not (rd / "job-abc" / "08_encode").exists()


def test_all_known_frame_stages_route_to_ramdisk(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    rd = tmp_path / "rd"
    out = tmp_path / "out.mkv"
    ctx = _ctx(workdir=workdir, output_path=out, ramdisk_path=rd)
    for stage in _RAMDISK_STAGES:
        d = ctx.stage_dir(stage)
        assert d == rd / "job-abc" / stage, f"stage {stage} should route to ramdisk"


def test_insufficient_free_space_falls_back_to_workdir(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    rd = tmp_path / "rd"
    out = tmp_path / "out.mkv"
    # Estimate larger than any plausible disk → forces fallback even with safety multiplier.
    ctx = _ctx(
        workdir=workdir,
        output_path=out,
        ramdisk_path=rd,
        ramdisk_estimate_bytes=10**18,  # 1 EB
    )
    d = ctx.stage_dir("05_upscale")
    assert d == workdir / "05_upscale"


def test_estimate_zero_trusts_user_and_uses_ramdisk(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    rd = tmp_path / "rd"
    out = tmp_path / "out.mkv"
    ctx = _ctx(
        workdir=workdir,
        output_path=out,
        ramdisk_path=rd,
        ramdisk_estimate_bytes=0,
    )
    d = ctx.stage_dir("05_upscale")
    # No estimate → no free-space guard applied.
    assert d == rd / "job-abc" / "05_upscale"


def test_stage_dir_creates_directory_on_demand(tmp_path: Path) -> None:
    workdir = tmp_path / "wd"
    out = tmp_path / "out.mkv"
    ctx = _ctx(workdir=workdir, output_path=out)
    # workdir doesn't pre-exist
    assert not workdir.exists()
    d = ctx.stage_dir("01_plan")
    assert d.exists() and d.is_dir()


# --- _ramdisk_usable() helper directly --------------------------------


def test_ramdisk_usable_no_estimate_returns_true(tmp_path: Path) -> None:
    rd = tmp_path / "rd"
    assert _ramdisk_usable(rd, estimate_bytes=0) is True
    # Side effect: directory was created
    assert rd.is_dir()


def test_ramdisk_usable_huge_estimate_returns_false(tmp_path: Path) -> None:
    rd = tmp_path / "rd"
    # 1 EB requirement is impossible on any sane test environment
    assert _ramdisk_usable(rd, estimate_bytes=10**18) is False


def test_ramdisk_usable_small_estimate_returns_true(tmp_path: Path) -> None:
    rd = tmp_path / "rd"
    # 1 KB × 1.5 = 1.5 KB; surely available.
    assert _ramdisk_usable(rd, estimate_bytes=1024) is True


def test_ramdisk_usable_path_is_a_file_returns_false(tmp_path: Path) -> None:
    """If the configured 'ramdisk' path is actually a file, we must fall back."""
    rd = tmp_path / "not-a-dir"
    rd.write_text("oops")
    # mkdir(exist_ok=True) on an existing non-dir raises FileExistsError → caught.
    assert _ramdisk_usable(rd, estimate_bytes=0) is False


def test_ramdisk_usable_disk_usage_failure_falls_back(tmp_path: Path) -> None:
    rd = tmp_path / "rd"
    with mock.patch.object(shutil, "disk_usage", side_effect=OSError("simulated")):
        assert _ramdisk_usable(rd, estimate_bytes=1024) is False
