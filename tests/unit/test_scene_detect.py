"""Tests for scene-detection helpers and stage compatibility."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from aep.encode.scene_detect import (
    SceneCut,
    cuts_to_frame_indices,
    detect_scene_cuts_ffmpeg_scdet,
    parse_scdet_log,
    scdet_timestr_to_seconds,
)
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink
from aep.pipeline.stages.s03_scene_detect import SceneDetectStage, _map_legacy_threshold, _write_report

# ---------------------------------------------------------- cuts_to_frame_indices


def test_cuts_to_frame_indices_drops_out_of_range() -> None:
    cuts = [
        SceneCut(frame_index=0, pts_time_s=0.0),      # dropped
        SceneCut(frame_index=24, pts_time_s=1.0),     # kept
        SceneCut(frame_index=2400, pts_time_s=100.0),  # dropped if total=1000
    ]
    idxs = cuts_to_frame_indices(cuts, total_frames=1000)
    assert idxs == [24]


def test_cuts_to_frame_indices_dedupes_and_sorts() -> None:
    cuts = [
        SceneCut(frame_index=96, pts_time_s=4.0),
        SceneCut(frame_index=24, pts_time_s=1.0),
        SceneCut(frame_index=24, pts_time_s=1.001),
    ]
    idxs = cuts_to_frame_indices(cuts)
    assert idxs == [24, 96]


# ---------------------------------------------------------- threshold remap


def test_map_legacy_threshold_maps_0_4_to_27() -> None:
    assert _map_legacy_threshold(0.4) == 27.0


def test_map_legacy_threshold_keeps_explicit_pyscenedetect_value() -> None:
    assert _map_legacy_threshold(30.0) == 30.0


# ---------------------------------------------------------- ffmpeg scdet parsing


def test_scdet_timestr_plain_seconds() -> None:
    assert scdet_timestr_to_seconds("1.500000") == 1.5


def test_scdet_timestr_hms() -> None:
    assert abs(scdet_timestr_to_seconds("0:01:30.500000") - 90.5) < 1e-9


def test_scdet_timestr_ms() -> None:
    assert abs(scdet_timestr_to_seconds("1:30.25") - 90.25) < 1e-9


def test_parse_scdet_log_with_ffmpeg_prefix() -> None:
    text = (
        "foo\n"
        "[Parsed_scdet_0 @ 0xdeadbeef] lavfi.scd.score: 12.300, lavfi.scd.time: 2.000000\n"
        "bar\n"
    )
    assert parse_scdet_log(text) == [(12.3, 2.0)]


def test_parse_scdet_log_multiple_lines() -> None:
    text = (
        "lavfi.scd.score: 10.000, lavfi.scd.time: 0:00:01.000000\n"
        "lavfi.scd.score: 11.000, lavfi.scd.time: 3.500000\n"
    )
    assert parse_scdet_log(text) == [(10.0, 1.0), (11.0, 3.5)]


def test_ffmpeg_scdet_vf_includes_downscale_before_scdet() -> None:
    cmds: list[list[str | Path]] = []

    def fake_run_capture(cmd: list[str | Path], **_: object) -> SimpleNamespace:
        cmds.append(cmd)
        return SimpleNamespace(stderr="", stdout="")

    with patch("aep.util.proc.run_capture", side_effect=fake_run_capture):
        detect_scene_cuts_ffmpeg_scdet(
            Path("/fake.mkv"),
            ffmpeg_executable="ffmpeg",
            video_stream_index=0,
            threshold_percent=10.0,
            fps=Fraction(24, 1),
            scale_width=320,
        )
    assert cmds
    vf_i = cmds[0].index("-vf")
    assert cmds[0][vf_i + 1] == "scale=320:-1,scdet=t=10.0"


def test_ffmpeg_scdet_vf_full_res_when_scale_width_zero() -> None:
    cmds: list[list[str | Path]] = []

    def fake_run_capture(cmd: list[str | Path], **_: object) -> SimpleNamespace:
        cmds.append(cmd)
        return SimpleNamespace(stderr="", stdout="")

    with patch("aep.util.proc.run_capture", side_effect=fake_run_capture):
        detect_scene_cuts_ffmpeg_scdet(
            Path("/fake.mkv"),
            ffmpeg_executable="ffmpeg",
            video_stream_index=0,
            threshold_percent=12.5,
            fps=Fraction(24, 1),
            scale_width=0,
        )
    vf_i = cmds[0].index("-vf")
    assert cmds[0][vf_i + 1] == "scdet=t=12.5"


# ---------------------------------------------------------- report compatibility


def test_write_report_shape_compatible() -> None:
    with TemporaryDirectory() as td:
        out = Path(td) / "scene_cuts.json"
        _write_report(
            out,
            cuts=[24, 96],
            threshold=0.4,
            fps="24000/1001",
            total_frames=1000,
            raw_count=2,
            active=True,
            scene_detect_backend="pyscenedetect",
            scene_change_threshold_percent=10.0,
        )
        payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {
        "active",
        "threshold",
        "fps",
        "total_frames",
        "raw_cut_count",
        "frame_indices",
        "scene_detect_backend",
        "scene_change_threshold_percent",
    }
    assert payload["frame_indices"] == [24, 96]


# ---------------------------------------------------------- stage no-op


class _StubFmt:
    filename = "source.mkv"
    duration_s = 60.0


class _StubMedia:
    primary_video = None
    fmt = _StubFmt()


def test_stage_noop_when_interpolation_disabled() -> None:
    with TemporaryDirectory() as td:
        root = Path(td)
        ctx = PipelineContext(
            job_id="job1",
            source_path=root / "in.mkv",
            workdir=root / "work",
            output_path=root / "out.mkv",
            preset_id="default",
            preset_data={
                "meta": {"id": "default", "name": "default"},
                "interpolation": {"enabled": False, "scene_cut_threshold": 0.4},
            },
            media_info=_StubMedia(),  # type: ignore[arg-type]
            plan={},
        )
        stage = SceneDetectStage()
        plan = stage.plan(ctx)
        result = stage.run(ctx, plan, EventSink())
        report = json.loads(plan.outputs[0].read_text(encoding="utf-8"))

    assert result.success is True
    assert ctx.scene_cuts == []
    assert report["active"] is False
    assert report["frame_indices"] == []
    assert report["raw_cut_count"] == 0

def test_cuts_to_frame_indices_respects_total_frames_upper_bound() -> None:
    cuts = [
        SceneCut(frame_index=24, pts_time_s=1.0),
        SceneCut(frame_index=1000, pts_time_s=41.66),
        SceneCut(frame_index=999, pts_time_s=41.62),
    ]
    assert cuts_to_frame_indices(cuts, total_frames=1000) == [24, 999]
