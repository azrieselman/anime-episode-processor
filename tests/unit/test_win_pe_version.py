from __future__ import annotations

from aep.util.win_pe_version import first_yyyymmdd_tag


def test_first_yyyymmdd_tag_prefers_manifest_pin_even_if_other_dates_present() -> None:
    blob = "upstream 20990131 noise 20250112 tail"
    assert first_yyyymmdd_tag(blob, prefer="20250112") == "20250112"


def test_first_yyyymmdd_tag_rejects_calendar_garbage() -> None:
    assert first_yyyymmdd_tag("not a valid 20991345 date anywhere") is None


def test_first_yyyymmdd_tag_finds_plain_embedded_stamp() -> None:
    blob = (
        r"releases/download/windows.zip#\nupstream notes for 20250112 build.\n"
    )
    assert first_yyyymmdd_tag(blob, prefer=None) == "20250112"
