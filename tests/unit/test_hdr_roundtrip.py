"""Tests for the HDR transfer-characteristic validation in `_validate`.

Three operating modes:
  1. No HDR involvement on source       → no transfer enforcement
  2. allow_8bit_roundtrip (planner committed BT.709 8-bit)
       - output transfer in BT.709-equivalent set    → pass + note
       - residual smpte2084 / arib-std-b67           → fail with hdr_transfer_residual
  3. preserve / hdr_policy=skip
       - source transfer must round-trip; drift to BT.709 → fail with hdr_transfer_lost
"""

from __future__ import annotations

from aep.media.models import (
    Disposition,
    FormatInfo,
    MediaInfo,
    StreamInfo,
)
from aep.pipeline.stages.s10_validate import _validate


def _v(
    *,
    width: int = 3840,
    height: int = 2160,
    pix_fmt: str = "yuv420p10le",
    color_transfer: str | None = None,
) -> StreamInfo:
    return StreamInfo(
        index=0,
        kind="video",
        codec_name="hevc",
        width=width,
        height=height,
        pix_fmt=pix_fmt,
        color_transfer=color_transfer,
    )


def _a(idx: int) -> StreamInfo:
    return StreamInfo(
        index=idx, kind="audio", codec_name="aac", disposition=Disposition()
    )


def _media(
    *,
    video: StreamInfo,
    duration: float = 1440.0,
    filename: str = "/fake/out.mkv",
) -> MediaInfo:
    return MediaInfo(
        source_path=filename,
        fmt=FormatInfo(filename=filename, format_name="matroska,webm",
                       duration_s=duration),
        streams=[video, _a(1)],
        chapters=[],
        attachments=[],
        is_matroska=True,
    )


def _plan(
    *,
    target_w: int | None = 3840,
    target_h: int | None = 2160,
    hdr: dict | None = None,
) -> dict:
    return {
        "container": "mkv",
        "stream_mapping": {
            "audio": [{"source_index": 0}],
            "subtitles": [],
            "skipped": [],
            "copy_chapters": True,
        },
        "target_geometry": {"width": target_w, "height": target_h, "preserved": False},
        "preset": {"streams": {"copy_attachments": True}},
        "hdr": hdr or {
            "was_10bit": False,
            "was_hdr_transfer": False,
            "source_pix_fmt": "yuv420p",
            "source_color_transfer": None,
            "policy": "n/a",
            "target_pix_fmt": None,
            "target_color_transfer": None,
            "roundtripped_to_8bit": False,
        },
    }


# --- Case 1: no HDR involvement -------------------------------------------------


def test_no_hdr_involvement_ignores_output_transfer():
    """Source was SDR; output transfer is irrelevant to HDR validation."""
    src_v = _v(pix_fmt="yuv420p", color_transfer="bt709")
    out_v = _v(pix_fmt="yuv420p", color_transfer="bt709")
    src = _media(video=src_v)
    out = _media(video=out_v)
    report = _validate(src, out, _plan())
    codes = {f.code for f in report.failures}
    assert "hdr_transfer_residual" not in codes
    assert "hdr_transfer_lost" not in codes


def test_no_hdr_involvement_even_with_unexpected_transfer():
    """A weird output transfer on a non-HDR source isn't this validator's problem."""
    src_v = _v(pix_fmt="yuv420p", color_transfer="bt709")
    out_v = _v(pix_fmt="yuv420p", color_transfer="smpte2084")  # garbage tag
    src = _media(video=src_v)
    out = _media(video=out_v)
    report = _validate(src, out, _plan())
    codes = {f.code for f in report.failures}
    # Source had no HDR involvement so we don't enforce.
    assert "hdr_transfer_residual" not in codes
    assert "hdr_transfer_lost" not in codes


# --- Case 2: allow_8bit_roundtrip ----------------------------------------------


def test_roundtrip_to_bt709_passes_with_note():
    src_v = _v(pix_fmt="yuv420p10le", color_transfer="smpte2084")
    out_v = _v(pix_fmt="yuv420p", color_transfer="bt709")
    hdr = {
        "was_10bit": True,
        "was_hdr_transfer": True,
        "source_pix_fmt": "yuv420p10le",
        "source_color_transfer": "smpte2084",
        "policy": "allow_8bit_roundtrip",
        "target_pix_fmt": "yuv420p",
        "target_color_transfer": "bt709",
        "roundtripped_to_8bit": True,
    }
    report = _validate(_media(video=src_v), _media(video=out_v), _plan(hdr=hdr))
    codes = {f.code for f in report.failures}
    assert "hdr_transfer_residual" not in codes
    assert any("roundtripped" in n for n in report.notes)


def test_roundtrip_with_unspecified_output_transfer_passes():
    """Empty/unspecified output transfer is acceptable under the equivalent set."""
    src_v = _v(pix_fmt="yuv420p10le", color_transfer="smpte2084")
    out_v = _v(pix_fmt="yuv420p", color_transfer=None)
    hdr = {
        "was_10bit": True,
        "was_hdr_transfer": True,
        "source_pix_fmt": "yuv420p10le",
        "source_color_transfer": "smpte2084",
        "policy": "allow_8bit_roundtrip",
        "target_pix_fmt": "yuv420p",
        "target_color_transfer": "bt709",
        "roundtripped_to_8bit": True,
    }
    report = _validate(_media(video=src_v), _media(video=out_v), _plan(hdr=hdr))
    codes = {f.code for f in report.failures}
    assert "hdr_transfer_residual" not in codes


def test_roundtrip_with_residual_smpte2084_fails():
    """If the encoder retained the HDR tag despite roundtrip, that's a bug."""
    src_v = _v(pix_fmt="yuv420p10le", color_transfer="smpte2084")
    out_v = _v(pix_fmt="yuv420p", color_transfer="smpte2084")  # bug: tag carried over
    hdr = {
        "was_10bit": True,
        "was_hdr_transfer": True,
        "source_pix_fmt": "yuv420p10le",
        "source_color_transfer": "smpte2084",
        "policy": "allow_8bit_roundtrip",
        "target_pix_fmt": "yuv420p",
        "target_color_transfer": "bt709",
        "roundtripped_to_8bit": True,
    }
    report = _validate(_media(video=src_v), _media(video=out_v), _plan(hdr=hdr))
    codes = {f.code for f in report.failures}
    assert "hdr_transfer_residual" in codes


def test_roundtrip_with_residual_hlg_fails():
    """HLG (arib-std-b67) is also outside the BT.709-equivalent set."""
    src_v = _v(pix_fmt="yuv420p10le", color_transfer="arib-std-b67")
    out_v = _v(pix_fmt="yuv420p", color_transfer="arib-std-b67")
    hdr = {
        "was_10bit": True,
        "was_hdr_transfer": True,
        "source_pix_fmt": "yuv420p10le",
        "source_color_transfer": "arib-std-b67",
        "policy": "allow_8bit_roundtrip",
        "target_pix_fmt": "yuv420p",
        "target_color_transfer": "bt709",
        "roundtripped_to_8bit": True,
    }
    report = _validate(_media(video=src_v), _media(video=out_v), _plan(hdr=hdr))
    codes = {f.code for f in report.failures}
    assert "hdr_transfer_residual" in codes


# --- Case 3: preserve / skip ---------------------------------------------------


def test_preserve_with_drift_to_bt709_fails():
    """Source HDR + policy=skip → encoder must round-trip the transfer tag."""
    src_v = _v(pix_fmt="yuv420p10le", color_transfer="smpte2084")
    out_v = _v(pix_fmt="yuv420p10le", color_transfer="bt709")  # silent drift
    hdr = {
        "was_10bit": True,
        "was_hdr_transfer": True,
        "source_pix_fmt": "yuv420p10le",
        "source_color_transfer": "smpte2084",
        "policy": "skip",
        "target_pix_fmt": None,
        "target_color_transfer": "smpte2084",
        "roundtripped_to_8bit": False,
    }
    report = _validate(_media(video=src_v), _media(video=out_v), _plan(hdr=hdr))
    codes = {f.code for f in report.failures}
    assert "hdr_transfer_lost" in codes


def test_preserve_with_matching_transfer_passes():
    src_v = _v(pix_fmt="yuv420p10le", color_transfer="smpte2084")
    out_v = _v(pix_fmt="yuv420p10le", color_transfer="smpte2084")
    hdr = {
        "was_10bit": True,
        "was_hdr_transfer": True,
        "source_pix_fmt": "yuv420p10le",
        "source_color_transfer": "smpte2084",
        "policy": "skip",
        "target_pix_fmt": None,
        "target_color_transfer": "smpte2084",
        "roundtripped_to_8bit": False,
    }
    report = _validate(_media(video=src_v), _media(video=out_v), _plan(hdr=hdr))
    codes = {f.code for f in report.failures}
    assert "hdr_transfer_lost" not in codes
    assert "hdr_transfer_residual" not in codes


def test_preserve_with_unspecified_source_transfer_skips_check():
    """If source had no transfer tag, we can't enforce anything."""
    src_v = _v(pix_fmt="yuv420p10le", color_transfer=None)  # high-bit but no tag
    out_v = _v(pix_fmt="yuv420p10le", color_transfer="bt709")
    hdr = {
        "was_10bit": True,
        "was_hdr_transfer": False,
        "source_pix_fmt": "yuv420p10le",
        "source_color_transfer": None,
        "policy": "skip",
        "target_pix_fmt": None,
        "target_color_transfer": None,
        "roundtripped_to_8bit": False,
    }
    report = _validate(_media(video=src_v), _media(video=out_v), _plan(hdr=hdr))
    codes = {f.code for f in report.failures}
    assert "hdr_transfer_lost" not in codes
