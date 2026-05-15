"""Tests for :mod:`aep.app.pinned_tools_refresh`."""

from __future__ import annotations

import pytest

from aep.adapters.verification import ToolStatus
from aep.app import pinned_tools_refresh
from aep.app.tools_fetcher import ALL_PINS


def test_pin_for_adapter_tool_id_maps_shared_archives() -> None:
    ff = pinned_tools_refresh.pin_for_adapter_tool_id("ffmpeg")
    fp = pinned_tools_refresh.pin_for_adapter_tool_id("ffprobe")
    assert ff is not None and fp is not None
    assert ff.tool_id == "ffmpeg" and fp.tool_id == "ffmpeg"

    m1 = pinned_tools_refresh.pin_for_adapter_tool_id("mkvmerge")
    m2 = pinned_tools_refresh.pin_for_adapter_tool_id("mkvpropedit")
    m3 = pinned_tools_refresh.pin_for_adapter_tool_id("mkvinfo")
    assert m1 is m2 is m3


def test_pins_to_refresh_dedupes_ffprobe_vs_ffmpeg_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pinned_tools_refresh,
        "missing_pins",
        lambda install_root=None: [],
    )
    st = [
        ToolStatus(
            tool_id="ffmpeg",
            path="x",
            version="wrong",
            expected="n7.0.2",
            status="mismatch",
        ),
        ToolStatus(
            tool_id="ffprobe",
            path="y",
            version="wrong",
            expected="n7.0.2",
            status="mismatch",
        ),
    ]
    pins = pinned_tools_refresh.pins_to_refresh(st)
    ffmpeg_pins = [p for p in pins if p.tool_id == "ffmpeg"]
    assert len(ffmpeg_pins) == 1


def test_pins_to_refresh_ignores_version_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pinned_tools_refresh,
        "missing_pins",
        lambda install_root=None: [],  # dev machines often lack binaries; isolate mapping logic
    )
    st = [
        ToolStatus(
            tool_id="ffmpeg",
            path="x",
            version="(probe failed)",
            expected="n7.0.2",
            status="version_unknown",
            note="boom",
        ),
    ]
    assert pinned_tools_refresh.pins_to_refresh(st) == []


def test_pins_to_refresh_single_archive_mismatch_skips_manifest_missing_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One logical archive out of date: do not reinstall unrelated missing manifest pins."""

    monkeypatch.setattr(
        pinned_tools_refresh,
        "missing_pins",
        lambda install_root=None: list(ALL_PINS),
    )
    st = [
        ToolStatus(
            tool_id="ffmpeg",
            path=r"C:\ffmpeg.exe",
            version="wrong",
            expected="n7.0.2",
            status="mismatch",
        ),
    ]
    pins = pinned_tools_refresh.pins_to_refresh(st)
    assert len(pins) == 1
    assert pins[0].tool_id == "ffmpeg"


def test_pins_to_refresh_two_archives_out_of_date_keeps_manifest_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Third pin (e.g. Real-CUGAN) missing on disk but not in DEFAULT_ADAPTERS probe table.
    extra_missing = ALL_PINS[2]
    monkeypatch.setattr(
        pinned_tools_refresh,
        "missing_pins",
        lambda install_root=None: [extra_missing],
    )
    st = [
        ToolStatus(
            tool_id="ffmpeg",
            path="x",
            version="bad",
            expected="n7.0.2",
            status="mismatch",
        ),
        ToolStatus(
            tool_id="mkvmerge",
            path="y",
            version="bad",
            expected="85.0",
            status="mismatch",
        ),
    ]
    pins = pinned_tools_refresh.pins_to_refresh(st)
    assert any(p.tool_id == "ffmpeg" for p in pins)
    assert any(p.tool_id == "mkvmerge" for p in pins)
    assert any(p.tool_id == extra_missing.tool_id for p in pins)
    assert len(pins) == 3
