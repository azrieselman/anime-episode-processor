"""PNG compression level wiring (M6.5).

constants.PNG_COMPRESSION_LEVEL = 6 must flow into every place AEP emits
PNG frames through ffmpeg: decode (s04), postprocess (s07), and \u2014 should
the encoder ever re-emit PNGs \u2014 build_encode_from_frames.

NCNN binaries (upscale/interpolate) write libpng's default zlib level (6)
internally; we don't test that here because there's no flag to assert on.
"""
from __future__ import annotations

from pathlib import Path

from aep.adapters.ffmpeg import FFmpegAdapter
from aep.constants import PNG_COMPRESSION_LEVEL


def test_constant_is_six() -> None:
    assert PNG_COMPRESSION_LEVEL == 6


def test_decode_to_frames_uses_constant(tmp_path: Path) -> None:
    """build_decode_to_frames must default png_compression to the constant."""
    ff = FFmpegAdapter()
    cmd = ff.build_decode_to_frames(
        source=tmp_path / "in.mkv",
        out_dir=tmp_path / "out",
        frame_format="png",
    )
    # Locate the -compression_level flag and check the immediately following arg.
    assert "-compression_level" in cmd
    idx = cmd.index("-compression_level")
    assert str(cmd[idx + 1]) == str(PNG_COMPRESSION_LEVEL)


def test_decode_to_frames_explicit_override(tmp_path: Path) -> None:
    """An explicit png_compression argument still wins."""
    ff = FFmpegAdapter()
    cmd = ff.build_decode_to_frames(
        source=tmp_path / "in.mkv",
        out_dir=tmp_path / "out",
        frame_format="png",
        png_compression=3,
    )
    idx = cmd.index("-compression_level")
    assert str(cmd[idx + 1]) == "3"


def test_postprocess_stage_uses_constant() -> None:
    """s07 postprocess hardcodes -compression_level via PNG_COMPRESSION_LEVEL.

    We read the source file rather than execute the stage \u2014 the stage runs
    a real ffmpeg subprocess, but the literal in the file is the contract.
    """
    src = Path(__file__).resolve().parents[2] / "src/aep/pipeline/stages/s07_postprocess.py"
    text = src.read_text(encoding="utf-8")
    assert "PNG_COMPRESSION_LEVEL" in text
    # Make sure the old hardcoded "1" is gone from the PNG branch.
    assert '"-compression_level", "1"' not in text


def test_decode_pts_window_seek_args(tmp_path: Path) -> None:
    """start_pts/end_pts emit -ss before -i and -t after -i (input-side seek)."""
    ff = FFmpegAdapter()
    cmd = [str(c) for c in ff.build_decode_to_frames(
        source=tmp_path / "in.mkv",
        out_dir=tmp_path / "out",
        frame_format="png",
        start_pts=10.0,
        end_pts=40.0,
    )]
    # -ss precedes -i.
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    assert ss_idx < i_idx
    assert cmd[ss_idx + 1].startswith("10.")
    # -t (duration = end - start = 30s) appears after -i.
    t_idx = cmd.index("-t")
    assert t_idx > i_idx
    assert cmd[t_idx + 1].startswith("30.")


def test_decode_sdr_omits_zscale(tmp_path: Path) -> None:
    """8-bit SDR path must not require zscale (optional in minimal ffmpeg builds)."""
    ff = FFmpegAdapter()
    cmd = [str(c) for c in ff.build_decode_to_frames(
        source=tmp_path / "in.mkv",
        out_dir=tmp_path / "out",
        frame_format="png",
        use_zscale=False,
    )]
    vf = cmd[cmd.index("-vf") + 1]
    assert "zscale" not in vf
    assert "format=rgb24" in vf


def test_decode_pts_window_zero_start(tmp_path: Path) -> None:
    """start_pts=0 means no seek \u2014 don't emit -ss (faster ffmpeg startup)."""
    ff = FFmpegAdapter()
    cmd = [str(c) for c in ff.build_decode_to_frames(
        source=tmp_path / "in.mkv",
        out_dir=tmp_path / "out",
        frame_format="png",
        start_pts=0.0,
        end_pts=30.0,
    )]
    assert "-ss" not in cmd
    # -t is still present with the full duration.
    t_idx = cmd.index("-t")
    assert cmd[t_idx + 1].startswith("30.")


def test_passthrough_encode_pts_window(tmp_path: Path) -> None:
    """build_passthrough_video_encode also accepts the seek window for source-mode batches."""
    ff = FFmpegAdapter()
    cmd = [str(c) for c in ff.build_passthrough_video_encode(
        source=tmp_path / "in.mkv",
        video_only_out=tmp_path / "out.mkv",
        encoder_args=["-c:v", "libx264"],
        start_pts=5.5,
        end_pts=10.5,
    )]
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    t_idx = cmd.index("-t")
    assert ss_idx < i_idx < t_idx
    assert cmd[ss_idx + 1].startswith("5.")
    assert cmd[t_idx + 1].startswith("5.")  # duration = 10.5 - 5.5 = 5.0


def test_decode_hwaccel_flags_on_decode_to_frames(tmp_path: Path) -> None:
    ff = FFmpegAdapter()
    cmd = [str(c) for c in ff.build_decode_to_frames(
        source=tmp_path / "in.mkv",
        out_dir=tmp_path / "out",
        frame_format="png",
        decode_hwaccel="d3d11va",
    )]
    assert "-hwaccel" in cmd
    idx = cmd.index("-hwaccel")
    assert cmd[idx + 1] == "d3d11va"


def test_decode_hwaccel_cuda_on_decode_to_frames(tmp_path: Path) -> None:
    ff = FFmpegAdapter()
    cmd = [str(c) for c in ff.build_decode_to_frames(
        source=tmp_path / "in.mkv",
        out_dir=tmp_path / "out",
        frame_format="png",
        decode_hwaccel="cuda",
    )]
    assert "-hwaccel" in cmd
    idx = cmd.index("-hwaccel")
    assert cmd[idx + 1] == "cuda"


def test_decode_hwaccel_flags_on_passthrough_encode(tmp_path: Path) -> None:
    ff = FFmpegAdapter()
    cmd = [str(c) for c in ff.build_passthrough_video_encode(
        source=tmp_path / "in.mkv",
        video_only_out=tmp_path / "out.mkv",
        encoder_args=["-c:v", "libx264"],
        decode_hwaccel="d3d11va",
    )]
    assert "-hwaccel" in cmd
    idx = cmd.index("-hwaccel")
    assert cmd[idx + 1] == "d3d11va"


def test_decode_hwaccel_cuda_on_passthrough_encode(tmp_path: Path) -> None:
    ff = FFmpegAdapter()
    cmd = [str(c) for c in ff.build_passthrough_video_encode(
        source=tmp_path / "in.mkv",
        video_only_out=tmp_path / "out.mkv",
        encoder_args=["-c:v", "libx264"],
        decode_hwaccel="cuda",
    )]
    assert "-hwaccel" in cmd
    idx = cmd.index("-hwaccel")
    assert cmd[idx + 1] == "cuda"


def test_fused_decode_emits_filter_complex_and_dual_maps(tmp_path: Path) -> None:
    """Fused decode+scene metadata must mirror decode preprocess + scan in one graph."""
    from aep.util.frame_dedupe import SCENE_SCORE_META_BASENAME

    ff = FFmpegAdapter()
    meta = tmp_path / SCENE_SCORE_META_BASENAME
    cmd = [str(c) for c in ff.build_decode_to_frames_with_scene_metadata_fused(
        source=tmp_path / "in.mkv",
        out_dir=tmp_path / "frames",
        metadata_out=meta,
        frame_format="png",
        start_pts=1.0,
        end_pts=3.0,
    )]
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:v]" in fc and "[rgb]" in fc
    assert "split[enc][scn]" in fc
    assert f"metadata=print:file={SCENE_SCORE_META_BASENAME}" in fc
    assert "select=gt(scene+1" in fc
    assert cmd.count("-map") >= 2
    assert "-map" in cmd and "[enc]" in cmd and "[meta]" in cmd
    assert cmd[-3:] == ["-f", "null", "-"]
    ss_idx = cmd.index("-ss")
    i_idx = cmd.index("-i")
    assert ss_idx < i_idx
    t_idx = cmd.index("-t")
    assert t_idx > i_idx
    assert cmd[t_idx + 1].startswith("2.")
