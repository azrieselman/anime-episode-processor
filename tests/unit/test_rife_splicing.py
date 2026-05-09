"""Tests for RIFE scene-cut splicing primitives.

The math here is the load-bearing piece of M3: if `expected_output_count`
disagrees with what `s06_interpolate` actually produces, validation fails
loudly and the run is wasted. We pin the formula here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aep.adapters.rife import (
    FrameRun,
    boundary_duplicates,
    expected_output_count,
    local_cuts_from_global,
    morphed_output_range,
    replace_with_boundary_dup,
    split_by_scene_cuts,
    stage_run_for_rife,
)

# ---------------------------------------------------------- split_by_scene_cuts


def test_split_no_cuts_returns_single_run() -> None:
    runs = split_by_scene_cuts(10, [])
    assert runs == [FrameRun(0, 9)]


def test_split_single_cut() -> None:
    runs = split_by_scene_cuts(10, [3])
    assert runs == [FrameRun(0, 2), FrameRun(3, 9)]


def test_split_multiple_cuts_in_order() -> None:
    runs = split_by_scene_cuts(10, [3, 7])
    assert runs == [FrameRun(0, 2), FrameRun(3, 6), FrameRun(7, 9)]


def test_split_dedupes_and_sorts_cuts() -> None:
    runs = split_by_scene_cuts(10, [7, 3, 3, 7])
    assert runs == [FrameRun(0, 2), FrameRun(3, 6), FrameRun(7, 9)]


def test_split_drops_out_of_range_cuts() -> None:
    # Cut at 0 (degenerate) and at total (past last frame) are dropped.
    runs = split_by_scene_cuts(10, [0, 5, 10, 15])
    assert runs == [FrameRun(0, 4), FrameRun(5, 9)]


def test_split_empty_total_frames() -> None:
    assert split_by_scene_cuts(0, [3]) == []


def test_frame_run_length() -> None:
    assert FrameRun(0, 9).length == 10
    assert FrameRun(5, 5).length == 1


# ---------------------------------------------------------- expected_output_count


def test_expected_output_count_no_cuts_simple_multiplier() -> None:
    # 100 frames × 2 = 200, single RIFE pass.
    assert expected_output_count(100, 2, 0) == 200
    assert expected_output_count(100, 3, 0) == 300


def test_expected_output_count_ignores_scene_cut_count() -> None:
    # The consolidated stage runs RIFE once on the whole batch and overwrites
    # morphed frames in place — cuts don't change the total output length.
    assert expected_output_count(100, 2, 3) == 200
    assert expected_output_count(100, 4, 3) == 400
    # Default arg (no cuts kwarg passed) matches the cuts-supplied result.
    assert expected_output_count(100, 4) == 400


def test_expected_output_count_multiplier_one_is_identity() -> None:
    # Multiplier=1 → no interpolation regardless of cuts.
    assert expected_output_count(100, 1, 5) == 100


def test_expected_output_count_zero_frames_or_multiplier() -> None:
    assert expected_output_count(0, 2, 3) == 0
    assert expected_output_count(100, 0, 3) == 0


# ---------------------------------------------------------- local_cuts_from_global


def test_local_cuts_from_global_translates_offset() -> None:
    # Batch covers source frames 240..479; global cuts at 250 and 260 should
    # land at local 1-based input indices 11 and 21. Cut at 5 (in the prior
    # batch) is dropped.
    out = local_cuts_from_global(
        [5, 250, 260], batch_offset=240, in_count=240,
    )
    assert out == [11, 21]


def test_local_cuts_from_global_filters_first_and_last_frame() -> None:
    # Global cut at offset itself maps to local 1 (input frame 1 starts a new
    # scene → no morph region inside the batch) and must be dropped — that's
    # the case that used to make per-run RIFE see a 1-frame run.
    # A cut past the last input frame is also dropped.
    out = local_cuts_from_global(
        [240, 480, 481], batch_offset=240, in_count=240,
    )
    assert out == []


def test_local_cuts_from_global_dedupes_and_sorts() -> None:
    out = local_cuts_from_global(
        [10, 5, 5, 10, 8], batch_offset=0, in_count=20,
    )
    # local = g - 0 + 1 → [11, 6, 6, 11, 9] → unique sorted [6, 9, 11]
    assert out == [6, 9, 11]


def test_local_cuts_from_global_empty_batch_returns_empty() -> None:
    assert local_cuts_from_global([5, 6], batch_offset=0, in_count=0) == []


# ---------------------------------------------------------- morphed_output_range


def test_morphed_output_range_basic() -> None:
    # Cut at local input 10, M=3:
    #   input 9 → output (9-1)*3+1 = 25
    #   input 10 → output (10-1)*3+1 = 28
    #   morphs between them = outputs 26..27
    # Helper returns (first, count) = ((c-2)*M + 2, M - 1) = (26, 2).
    assert morphed_output_range(10, 3) == (26, 2)


def test_morphed_output_range_multiplier_two() -> None:
    # M=2: exactly one morph frame per cut. For c=4:
    #   input 3 → output 5; input 4 → output 7; morph at output 6.
    #   first = (4-2)*2 + 2 = 6.
    assert morphed_output_range(4, 2) == (6, 1)


def test_morphed_output_range_first_possible_cut() -> None:
    # c=2 is the smallest legal cut: input 1 → output 1, input 2 → output M+1,
    # morphs at outputs 2..M. first = (2-2)*M + 2 = 2.
    assert morphed_output_range(2, 3) == (2, 2)
    assert morphed_output_range(2, 2) == (2, 1)


def test_morphed_output_range_no_op_cases() -> None:
    # Multiplier 1 → nothing to overwrite.
    assert morphed_output_range(5, 1) == (0, 0)
    # local_cut < 2 means there's no preceding scene to anchor on.
    assert morphed_output_range(1, 3) == (0, 0)
    assert morphed_output_range(0, 3) == (0, 0)


# ---------------------------------------------------------- replace_with_boundary_dup


def test_replace_with_boundary_dup_overwrites_existing(tmp_path: Path) -> None:
    # Create 6 distinct frames named 1..6.
    for i in range(1, 7):
        (tmp_path / f"{i:08d}.png").write_text(f"frame{i}")
    overwritten = replace_with_boundary_dup(
        tmp_path, boundary_idx=3, start_idx=4, count=2, format="png",
    )
    assert len(overwritten) == 2
    # 4 and 5 now match frame 3.
    assert (tmp_path / "00000004.png").read_text() == "frame3"
    assert (tmp_path / "00000005.png").read_text() == "frame3"
    # 6 untouched.
    assert (tmp_path / "00000006.png").read_text() == "frame6"
    # 3 (the source) untouched.
    assert (tmp_path / "00000003.png").read_text() == "frame3"


def test_replace_with_boundary_dup_count_zero_is_no_op(tmp_path: Path) -> None:
    (tmp_path / "00000001.png").write_text("ONE")
    out = replace_with_boundary_dup(
        tmp_path, boundary_idx=1, start_idx=2, count=0, format="png",
    )
    assert out == []
    assert list(tmp_path.iterdir()) == [tmp_path / "00000001.png"]


def test_replace_with_boundary_dup_missing_boundary_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        replace_with_boundary_dup(
            tmp_path, boundary_idx=99, start_idx=1, count=1, format="png",
        )


# ---------------------------------------------------------- stage_run_for_rife


def test_stage_run_for_rife_renumbers_to_1_based(tmp_path: Path) -> None:
    src = tmp_path / "src"; src.mkdir()
    dest = tmp_path / "dest"; dest.mkdir()
    # Source has frames 1..10, we stage [3..6] (0-based).
    for i in range(1, 11):
        (src / f"{i:08d}.png").write_text(f"frame{i}")
    n = stage_run_for_rife(FrameRun(start_idx=3, end_idx=6), src, dest)
    assert n == 4
    # Dest contains 1-based contiguous frames 1..4 corresponding to source 4..7.
    for j, expected_src in zip(range(1, 5), range(4, 8)):
        content = (dest / f"{j:08d}.png").read_text()
        assert content == f"frame{expected_src}"


def test_stage_run_for_rife_missing_frame_raises(tmp_path: Path) -> None:
    src = tmp_path / "src"; src.mkdir()
    dest = tmp_path / "dest"
    (src / "00000001.png").touch()
    # Asking for indices 2-3 but only 1 exists.
    with pytest.raises(FileNotFoundError):
        stage_run_for_rife(FrameRun(start_idx=1, end_idx=2), src, dest)


# ---------------------------------------------------------- boundary_duplicates


def test_boundary_duplicates_writes_n_copies(tmp_path: Path) -> None:
    src_frame = tmp_path / "last.png"
    src_frame.write_text("LAST")
    dest = tmp_path / "out"; dest.mkdir()
    # Insert 3 dupes starting at output index 100.
    created = boundary_duplicates(
        last_src_frame_path=src_frame, count=3, start_index=100, dest_dir=dest,
    )
    assert len(created) == 3
    for i in (100, 101, 102):
        assert (dest / f"{i:08d}.png").read_text() == "LAST"
    assert not (dest / "00000103.png").exists()


def test_boundary_duplicates_count_zero_no_op(tmp_path: Path) -> None:
    src_frame = tmp_path / "last.png"
    src_frame.write_text("LAST")
    dest = tmp_path / "out"; dest.mkdir()
    created = boundary_duplicates(
        last_src_frame_path=src_frame, count=0, start_index=100, dest_dir=dest,
    )
    assert created == []
    assert list(dest.iterdir()) == []
