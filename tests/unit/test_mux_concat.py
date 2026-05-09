"""Concat-demuxer path in stage 09 mux.

When the runner produces per-batch encoded segments, ``MuxStage._encoded_video_path``
must concatenate them into a single intermediate before handing off to the
existing mux flow. We verify the concat-list file format and the ffmpeg
argv shape; the actual ffmpeg invocation is patched.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from aep.pipeline.context import PipelineContext
from aep.pipeline.stages.s09_mux import MuxStage


def _make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        job_id="muxjob",
        source_path=tmp_path / "src.mkv",
        workdir=tmp_path / "work",
        output_path=tmp_path / "out.mkv",
        preset_id="p",
        preset_data={},
    )


def test_concat_list_format_and_argv(tmp_path: Path) -> None:
    ctx = _make_ctx(tmp_path)
    seg_dir = ctx.workdir / "batch_segments"
    seg_dir.mkdir(parents=True)
    # Three placeholder segments \u2014 contents irrelevant; we patch run_capture.
    for i in range(3):
        p = seg_dir / f"segment_{i:02d}.mkv"
        p.write_bytes(b"x" * 16)
        ctx.encoded_segments.append(p)

    stage = MuxStage()

    # Pretend ffmpeg always succeeds and creates the output file.
    captured: dict[str, list] = {}

    def _fake_run(cmd, *, timeout=None, check=False):
        captured["cmd"] = list(cmd)
        # Find the output path = last argv element; create it.
        Path(cmd[-1]).write_bytes(b"concat output")
        from aep.util.proc import ProcResult
        return ProcResult(cmd=[str(c) for c in cmd], returncode=0, stdout="", stderr="")

    with patch("aep.pipeline.stages.s09_mux.run_capture", side_effect=_fake_run):
        out = stage._encoded_video_path(ctx)

    # Output landed where we expect.
    assert out == seg_dir / "video_concat.mkv"
    assert out.exists()

    # Concat list contains every segment, quoted, in order.
    list_text = (seg_dir / "concat.txt").read_text(encoding="utf-8")
    lines = [ln for ln in list_text.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 3
    for i, ln in enumerate(lines):
        assert ln.startswith("file '")
        assert ln.endswith("'")
        # POSIX-form path \u2014 no backslashes even on Windows.
        assert "\\" not in ln
        assert f"segment_{i:02d}.mkv" in ln

    # Argv shape: -f concat -safe 0 -i <list> -c copy ...
    cmd = captured["cmd"]
    assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "concat"
    assert "-safe" in cmd and cmd[cmd.index("-safe") + 1] == "0"
    assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
    assert str(seg_dir / "concat.txt") in cmd


def test_concat_idempotent_when_output_fresh(tmp_path: Path) -> None:
    """A second call returns the cached output without re-running ffmpeg."""
    ctx = _make_ctx(tmp_path)
    seg_dir = ctx.workdir / "batch_segments"
    seg_dir.mkdir(parents=True)
    seg = seg_dir / "segment_00.mkv"
    seg.write_bytes(b"seg")
    ctx.encoded_segments.append(seg)

    # Pre-create a fresher concat output \u2014 newer than the segment.
    out = seg_dir / "video_concat.mkv"
    out.write_bytes(b"already done")
    import os
    future = seg.stat().st_mtime + 100
    os.utime(out, (future, future))

    stage = MuxStage()
    with patch("aep.pipeline.stages.s09_mux.run_capture") as mock_run:
        result = stage._encoded_video_path(ctx)
    mock_run.assert_not_called()
    assert result == out


def test_no_segments_falls_back_to_legacy_path(tmp_path: Path) -> None:
    """Without batch segments the stage uses the canonical 08_encode/video.mkv."""
    ctx = _make_ctx(tmp_path)
    enc_dir = ctx.workdir / "08_encode"
    enc_dir.mkdir(parents=True)
    canonical = enc_dir / "video.mkv"
    canonical.write_bytes(b"single-pass output")

    stage = MuxStage()
    result = stage._encoded_video_path(ctx)
    assert result == canonical
