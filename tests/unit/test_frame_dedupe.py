"""Unit tests for aep.util.frame_dedupe."""

from __future__ import annotations

from pathlib import Path

from aep.util import frame_dedupe as fd


def test_parse_metadata_print_scene_scores() -> None:
    text = """
frame:0   pts:0       pts_time:0.000000
lavfi.scene_score=0.000000
checksum=abc

frame:1   pts:1 pts_time:0.04
lavfi.scene_score=0.00015
other_key=x
"""
    scores = fd.parse_metadata_print_scene_scores(text)
    assert scores[0] == 0.0
    assert scores[1] == 0.00015


def test_load_scene_score_scan_results_prefers_meta(tmp_path: Path) -> None:
    meta = tmp_path / fd.SCENE_SCORE_META_BASENAME
    meta.write_text("frame:0\nlavfi.scene_score=0.01\n", encoding="utf-8")
    out = fd.load_scene_score_scan_results(meta_path=meta, stderr="n:0 scene_score:0.99 showinfo")
    assert out == {0: 0.01}


def test_parse_showinfo_scene_scores() -> None:
    stderr = """
[Parsed_showinfo_0 @ 0x1] n:   0 pts:0 pts_time:0 ... scene_score:0.000000
[Parsed_showinfo_0 @ 0x1] n:   1 pts:1 pts_time:0.04 ... scene_score:0.15
[Parsed_showinfo_0 @ 0x1] n:   2 pts:2 ... lavfi.scene_score=0.02 more text
"""
    scores = fd.parse_showinfo_scene_scores(stderr)
    assert scores == {0: 0.0, 1: 0.15, 2: 0.02}


def test_parse_showinfo_ignores_lines_without_n() -> None:
    stderr = """
[Parsed_showinfo_0] lavfi.scene_score=0.99
[Parsed_showinfo_0 @ 0x1] n:   0 pts:0 scene_score:0.11
"""
    assert fd.parse_showinfo_scene_scores(stderr) == {0: 0.11}


def test_parse_showinfo_last_line_wins_for_same_n() -> None:
    stderr = """
[Parsed_showinfo_0] n:   1 pts:1 scene_score:0.0
[Parsed_showinfo_0] n:   1 pts:1 scene_score:0.2
"""
    assert fd.parse_showinfo_scene_scores(stderr) == {1: 0.2}


def test_skip_indices_from_scores_respects_first_frame_and_protected() -> None:
    scores = [0.0, 0.001, 0.9, 0.0001, 0.5]
    full_count = 5
    threshold = 0.01
    protected = {3}
    skip = fd.skip_indices_from_scores(
        scores,
        full_count=full_count,
        threshold=threshold,
        protected=protected,
    )
    assert 1 not in skip
    assert 3 not in skip
    assert 2 in skip
    assert 4 in skip
    assert 5 not in skip


def test_local_cuts_compact_from_full() -> None:
    kept_order = [1, 2, 4, 5]
    local_full = [3, 5]
    out = fd.local_cuts_compact_from_full(local_full, kept_order, l_prime=4)
    assert out == [3, 4]


def test_expand_rife_output_dir(tmp_path: Path) -> None:
    comp = tmp_path / "c"
    dest = tmp_path / "d"
    comp.mkdir()
    fmt = "png"
    for i in range(1, 7):
        (comp / f"{i:08d}.{fmt}").write_bytes(b"x")
    kept_order = [1, 3, 4]
    fd.expand_rife_output_dir(
        compact_rife_dir=comp,
        dest_dir=dest,
        kept_order=kept_order,
        full_count=5,
        multiplier=2,
        frame_format=fmt,
    )
    assert dest.joinpath("00000001.png").read_bytes() == b"x"
    assert dest.joinpath("00000002.png").read_bytes() == b"x"
    assert dest.joinpath("00000003.png").read_bytes() == b"x"
    assert dest.joinpath("00000004.png").read_bytes() == b"x"
    assert dest.joinpath("00000009.png").read_bytes() == b"x"
    assert dest.joinpath("00000010.png").read_bytes() == b"x"


def test_expand_upscale_output_dir(tmp_path: Path) -> None:
    comp = tmp_path / "u"
    dest = tmp_path / "v"
    comp.mkdir()
    fmt = "png"
    for i in range(1, 4):
        (comp / f"{i:08d}.{fmt}").write_bytes(b"y")
    kept_order = [1, 3, 4]
    fd.expand_upscale_output_dir(
        compact_up_dir=comp,
        dest_dir=dest,
        kept_order=kept_order,
        full_count=5,
        frame_format=fmt,
    )
    assert dest.joinpath("00000001.png").read_bytes() == b"y"
    assert dest.joinpath("00000002.png").read_bytes() == b"y"
    assert dest.joinpath("00000003.png").read_bytes() == b"y"
    assert dest.joinpath("00000004.png").read_bytes() == b"y"
    assert dest.joinpath("00000005.png").read_bytes() == b"y"


def test_compact_decode_directory(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    fmt = "png"
    for i in range(1, 6):
        (frames / f"{i:08d}.{fmt}").write_text(str(i))
    kept, dedir = fd.compact_decode_directory(
        frames_dir=frames,
        full_count=5,
        skip={2, 4},
        frame_format=fmt,
    )
    assert kept == [1, 3, 5]
    assert dedir.is_dir()
    assert (dedir / "00000002.png").read_text() == "2"
    assert (dedir / "00000004.png").read_text() == "4"
    assert (frames / "00000001.png").read_text() == "1"
    assert (frames / "00000002.png").read_text() == "3"
    assert (frames / "00000003.png").read_text() == "5"


def test_write_load_dedupe_map(tmp_path: Path) -> None:
    doc = {"full_decode_count": 10, "compact_decode_count": 8, "kept_order": [1, 2]}
    p = fd.write_dedupe_map(tmp_path, doc)
    assert p.is_file()
    loaded = fd.load_dedupe_map(tmp_path)
    assert loaded == doc


def test_scene_cut_protect_indices() -> None:
    prot = fd.scene_cut_protect_indices([5, 10], batch_offset=0, full_count=12)
    assert prot == {5, 6, 7, 10, 11, 12}
