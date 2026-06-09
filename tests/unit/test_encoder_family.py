from __future__ import annotations

import pytest

from aep.encode.encoder_family import encode_name_for, encoder_family, software_name_for


@pytest.mark.parametrize(
    "name,expected",
    [
        ("hevc_nvenc", "nvenc"),
        ("h264_qsv", "qsv"),
        ("av1_amf", "amf"),
        ("libx264", "x264"),
        ("libx265", "x265"),
    ],
)
def test_encoder_family_maps_known_names(name: str, expected: str) -> None:
    assert encoder_family(name) == expected


def test_encoder_family_raises_for_unknown_name() -> None:
    with pytest.raises(ValueError):
        encoder_family("vp9_vaapi")


def test_encode_name_for_builds_vendor_encoder_name() -> None:
    assert encode_name_for("qsv", "hevc") == "hevc_qsv"


def test_software_name_for_maps_h264_and_hevc() -> None:
    assert software_name_for("h264") == "libx264"
    assert software_name_for("hevc") == "libx265"

