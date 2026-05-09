"""Tests for the chunked-execution helper on ``NcnnVulkanAdapter``.

We don't invoke any real binary. Instead we monkeypatch ``run_with_oom_fallback``
to a fake that creates the expected output files (mirroring what an ncnn
binary would produce) and returns a synthetic ``NcnnRunResult``. That lets us
verify chunking arithmetic, file aggregation, scratch cleanup, and result
accumulation without any GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from aep.adapters.ncnn_base import NcnnRunResult, NcnnVulkanAdapter


class _FakeAdapter(NcnnVulkanAdapter):
    """Minimal subclass; we only need the chunked machinery, not real path resolution."""
    tool_id = "fake-ncnn-vulkan"
    bin_name = "fake.exe"
    tools_subdir = "fake-ncnn-vulkan"
    default_tile_size = 256
    tile_size_floor = 64


def _make_input_dir(tmp_path: Path, n_frames: int, fmt: str = "png") -> Path:
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(1, n_frames + 1):
        (in_dir / f"{i:08d}.{fmt}").write_bytes(b"frame")
    # A non-frame file to confirm it's ignored.
    (in_dir / "notes.txt").write_text("ignored")
    return in_dir


def _patch_run(
    adapter: NcnnVulkanAdapter, *, tile_used: int = 256, attempts: int = 1,
) -> list[tuple[Path, Path, int]]:
    """Replace ``run_with_oom_fallback`` with a stub that copies inputs to outputs.

    Returns a list that records (input_dir, output_dir, tile_size) per call so
    tests can assert on the chunking pattern.
    """
    calls: list[tuple[Path, Path, int]] = []

    def _fake_run(
        *,
        argv_factory: Callable[[int], list],
        initial_tile_size: int,
        hardware_fp: str,
        model_id: str,
        source_height: int | None,
        on_progress=None,
        should_interrupt=None,
        max_attempts: int = 4,
    ) -> NcnnRunResult:
        # Invoke the factory to discover what input/output dirs the caller wired up.
        argv = argv_factory(initial_tile_size)
        # Argv layout from build_argv: [bin, "-i", in, "-o", out, ...]
        in_dir = Path(argv[argv.index("-i") + 1])
        out_dir = Path(argv[argv.index("-o") + 1])
        calls.append((in_dir, out_dir, initial_tile_size))
        # Simulate the binary writing one output per input frame, same names.
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in in_dir.iterdir():
            if src.is_file() and src.suffix == ".png":
                (out_dir / src.name).write_bytes(b"output")
        return NcnnRunResult(
            output_dir=out_dir, frames_in=0, frames_out=0,
            tile_size_used=tile_used, duration_s=0.01, attempts=attempts,
            rationale=[f"completed at tile={tile_used}"],
            warnings=[],
        )

    adapter.run_with_oom_fallback = _fake_run  # type: ignore[method-assign]
    return calls


def _factory_for(adapter: NcnnVulkanAdapter):
    """Build an argv_factory matching the run_chunked signature."""
    def af(in_dir: Path, out_dir: Path, tile: int) -> list:
        # Use a model dir alongside the input dir so resolve_model_dir isn't needed.
        return [
            "fake.exe",
            "-i", str(in_dir),
            "-o", str(out_dir),
            "-t", str(tile),
            "-m", str(in_dir),  # placeholder
            "-s", "2",
        ]
    return af


# ---------------------------------------------------------- chunking arithmetic


def test_chunked_splits_into_correct_number_of_chunks(tmp_path: Path) -> None:
    in_dir = _make_input_dir(tmp_path, n_frames=23)
    out_dir = tmp_path / "out"
    adapter = _FakeAdapter(override_dir=tmp_path)
    calls = _patch_run(adapter)

    result = adapter.run_chunked(
        input_dir=in_dir, output_dir=out_dir,
        chunk_size=10, argv_factory=_factory_for(adapter),
        initial_tile_size=256, hardware_fp="hwfp", model_id="m",
        source_height=1080, frame_format="png",
    )

    # 23 frames / 10 per chunk = ceil → 3 chunks
    assert len(calls) == 3
    assert result.frames_in == 23
    assert result.frames_out == 23
    assert result.attempts == 3   # one attempt per chunk in the fake


def test_chunked_writes_all_frames_to_output_with_original_names(tmp_path: Path) -> None:
    in_dir = _make_input_dir(tmp_path, n_frames=7)
    out_dir = tmp_path / "out"
    adapter = _FakeAdapter(override_dir=tmp_path)
    _patch_run(adapter)

    adapter.run_chunked(
        input_dir=in_dir, output_dir=out_dir,
        chunk_size=3, argv_factory=_factory_for(adapter),
        initial_tile_size=256, hardware_fp="hwfp", model_id="m",
        source_height=1080, frame_format="png",
    )

    written = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    assert written == [f"{i:08d}.png" for i in range(1, 8)]


def test_chunked_cleans_up_scratch_dir(tmp_path: Path) -> None:
    in_dir = _make_input_dir(tmp_path, n_frames=5)
    out_dir = tmp_path / "out"
    adapter = _FakeAdapter(override_dir=tmp_path)
    _patch_run(adapter)

    adapter.run_chunked(
        input_dir=in_dir, output_dir=out_dir,
        chunk_size=2, argv_factory=_factory_for(adapter),
        initial_tile_size=256, hardware_fp="hwfp", model_id="m",
        source_height=1080, frame_format="png",
    )

    # Default scratch dir is sibling of out_dir; it must be gone after success.
    scratch_root = out_dir.parent / f".chunks_{adapter.tool_id}"
    assert not scratch_root.exists()


def test_chunked_explicit_scratch_dir_used(tmp_path: Path) -> None:
    in_dir = _make_input_dir(tmp_path, n_frames=5)
    out_dir = tmp_path / "out"
    scratch = tmp_path / "ramdisk_scratch"
    adapter = _FakeAdapter(override_dir=tmp_path)
    _patch_run(adapter)

    adapter.run_chunked(
        input_dir=in_dir, output_dir=out_dir,
        chunk_size=2, argv_factory=_factory_for(adapter),
        initial_tile_size=256, hardware_fp="hwfp", model_id="m",
        source_height=1080, frame_format="png",
        scratch_dir=scratch,
    )
    # Same cleanup rule for explicit scratch dirs.
    assert not scratch.exists()


def test_chunked_zero_frames_raises(tmp_path: Path) -> None:
    in_dir = tmp_path / "empty"; in_dir.mkdir()
    out_dir = tmp_path / "out"
    adapter = _FakeAdapter(override_dir=tmp_path)
    _patch_run(adapter)

    with pytest.raises(ValueError, match="no png frames"):
        adapter.run_chunked(
            input_dir=in_dir, output_dir=out_dir,
            chunk_size=10, argv_factory=_factory_for(adapter),
            initial_tile_size=256, hardware_fp="hwfp", model_id="m",
            source_height=1080, frame_format="png",
        )


def test_chunked_invalid_chunk_size_raises(tmp_path: Path) -> None:
    in_dir = _make_input_dir(tmp_path, n_frames=3)
    adapter = _FakeAdapter(override_dir=tmp_path)
    _patch_run(adapter)

    with pytest.raises(ValueError, match="chunk_size"):
        adapter.run_chunked(
            input_dir=in_dir, output_dir=tmp_path / "out",
            chunk_size=0, argv_factory=_factory_for(adapter),
            initial_tile_size=256, hardware_fp="hwfp", model_id="m",
            source_height=1080, frame_format="png",
        )


def test_chunked_propagates_tile_size_across_chunks(tmp_path: Path) -> None:
    """If the OOM-fallback for chunk 1 settles at tile=128, chunk 2 should start there."""
    in_dir = _make_input_dir(tmp_path, n_frames=6)
    out_dir = tmp_path / "out"
    adapter = _FakeAdapter(override_dir=tmp_path)

    # Patch with a tile_used of 128 so all chunks settle at 128.
    calls = _patch_run(adapter, tile_used=128, attempts=2)

    adapter.run_chunked(
        input_dir=in_dir, output_dir=out_dir,
        chunk_size=2, argv_factory=_factory_for(adapter),
        initial_tile_size=256, hardware_fp="hwfp", model_id="m",
        source_height=1080, frame_format="png",
    )

    # First chunk starts at the requested 256; subsequent chunks reuse the
    # discovered tile=128 to avoid re-discovering OOM each chunk.
    initial_tiles = [t for (_in, _out, t) in calls]
    assert initial_tiles[0] == 256
    assert all(t == 128 for t in initial_tiles[1:])


def test_chunked_ignores_non_frame_files(tmp_path: Path) -> None:
    """A stray .txt next to frames must not break chunk boundaries."""
    in_dir = _make_input_dir(tmp_path, n_frames=4)
    # The fixture already adds notes.txt; sanity-check it doesn't get processed.
    out_dir = tmp_path / "out"
    adapter = _FakeAdapter(override_dir=tmp_path)
    _patch_run(adapter)

    result = adapter.run_chunked(
        input_dir=in_dir, output_dir=out_dir,
        chunk_size=10, argv_factory=_factory_for(adapter),
        initial_tile_size=256, hardware_fp="hwfp", model_id="m",
        source_height=1080, frame_format="png",
    )
    assert result.frames_in == 4
    assert result.frames_out == 4
