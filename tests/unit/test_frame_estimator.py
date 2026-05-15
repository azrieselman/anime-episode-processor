"""Unit tests for s01_plan._estimate_frame_bytes.

The estimator drives the ramdisk free-space guard: it must return a
*conservative* peak-bytes figure when inputs are known and 0 when any
required input is missing. Routing treats 0 as "trust the user".
"""

from __future__ import annotations

from aep.media.models import FormatInfo, MediaInfo, StreamInfo
from aep.pipeline.stages.s01_plan import _BYTES_PER_PIXEL, _estimate_frame_bytes


def _media(
    *,
    duration_s: float | None = 60.0,
    width: int | None = 1920,
    height: int | None = 1080,
    nb_frames: int | None = None,
    r_frame_rate: str | None = "24000/1001",
    avg_frame_rate: str | None = "24000/1001",
    include_video: bool = True,
) -> MediaInfo:
    fmt = FormatInfo(
        filename="x.mkv",
        format_name="matroska",
        duration_s=duration_s,
    )
    streams: list[StreamInfo] = []
    if include_video:
        streams.append(
            StreamInfo(
                index=0,
                kind="video",
                codec_name="h264",
                width=width,
                height=height,
                nb_frames=nb_frames,
                r_frame_rate=r_frame_rate,
                avg_frame_rate=avg_frame_rate,
            )
        )
    return MediaInfo(source_path="x.mkv", fmt=fmt, streams=streams)


_FLAT_PLAN: dict = {
    "upscale": {"active": False, "scale": 1},
    "interpolate": {"active": False, "multiplier": 1},
}


# ---------------------------------------------------------------- happy paths


def test_uses_nb_frames_when_present() -> None:
    media = _media(nb_frames=1440, duration_s=60.0)
    got = _estimate_frame_bytes(
        media=media, target_w=1920, target_h=1080, m3_plan=_FLAT_PLAN,
    )
    expected = int(1440 * 1920 * 1080 * _BYTES_PER_PIXEL)
    assert got == expected


def test_falls_back_to_duration_times_fps() -> None:
    # 60s × ~23.976 fps → 1438 frames (int truncation).
    media = _media(nb_frames=None, duration_s=60.0, r_frame_rate="24000/1001")
    got = _estimate_frame_bytes(
        media=media, target_w=1920, target_h=1080, m3_plan=_FLAT_PLAN,
    )
    expected_frames = int((24000 / 1001) * 60.0)  # 1438
    expected = int(expected_frames * 1920 * 1080 * _BYTES_PER_PIXEL)
    assert got == expected
    assert got > 0


# ---------------------------------------------------------------- edge cases


def test_missing_duration_and_nb_frames_returns_zero() -> None:
    media = _media(nb_frames=None, duration_s=None)
    got = _estimate_frame_bytes(
        media=media, target_w=1920, target_h=1080, m3_plan=_FLAT_PLAN,
    )
    assert got == 0


def test_missing_geometry_returns_zero() -> None:
    # Target geometry not pinned and source has no width/height → unknown.
    media = _media(nb_frames=1000, width=None, height=None)
    got = _estimate_frame_bytes(
        media=media, target_w=None, target_h=None, m3_plan=_FLAT_PLAN,
    )
    assert got == 0


def test_missing_primary_video_returns_zero() -> None:
    media = _media(include_video=False)
    got = _estimate_frame_bytes(
        media=media, target_w=1920, target_h=1080, m3_plan=_FLAT_PLAN,
    )
    assert got == 0


# ---------------------------------------------------------------- multipliers


def test_upscale_scale_multiplies_geometry_when_target_unpinned() -> None:
    # Source 960x540, scale=2 with no target geometry → 1920x1080.
    media = _media(nb_frames=100, width=960, height=540)
    plan = {
        "upscale": {"active": True, "scale": 2},
        "interpolate": {"active": False, "multiplier": 1},
    }
    got = _estimate_frame_bytes(
        media=media, target_w=None, target_h=None, m3_plan=plan,
    )
    expected = int(100 * (960 * 2) * (540 * 2) * _BYTES_PER_PIXEL)
    assert got == expected


def test_upscale_scale_ignored_when_target_geometry_pinned() -> None:
    # If the planner committed to explicit target geometry, the upscaler is
    # constrained to land on that — applying the scale multiplier on top would
    # double-count.
    media = _media(nb_frames=100, width=960, height=540)
    plan = {
        "upscale": {"active": True, "scale": 2},
        "interpolate": {"active": False, "multiplier": 1},
    }
    got = _estimate_frame_bytes(
        media=media, target_w=1920, target_h=1080, m3_plan=plan,
    )
    expected = int(100 * 1920 * 1080 * _BYTES_PER_PIXEL)
    assert got == expected


def test_interpolation_multiplier_doubles_frame_count() -> None:
    media = _media(nb_frames=100, width=1920, height=1080)
    plan = {
        "upscale": {"active": False, "scale": 1},
        "interpolate": {"active": True, "multiplier": 2},
    }
    got = _estimate_frame_bytes(
        media=media, target_w=1920, target_h=1080, m3_plan=plan,
    )
    expected = int((100 * 2) * 1920 * 1080 * _BYTES_PER_PIXEL)
    assert got == expected


def test_zero_or_garbage_multiplier_treated_as_one() -> None:
    media = _media(nb_frames=100, width=1920, height=1080)
    plan = {
        "upscale": {"active": False, "scale": 1},
        # Garbage value should not crash and should not zero the output.
        "interpolate": {"active": True, "multiplier": "abc"},
    }
    got = _estimate_frame_bytes(
        media=media, target_w=1920, target_h=1080, m3_plan=plan,
    )
    assert got > 0
