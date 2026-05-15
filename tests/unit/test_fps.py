"""Tests for `aep.util.fps`.

Anime sources lean heavily on NTSC fractional rates (24000/1001, 30000/1001).
Float-only math here would silently introduce frame-count drift on long
episodes; we verify the Fraction path stays exact.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from aep.util.fps import (
    derive_target_fps,
    fps_label,
    is_ntsc_drop,
    nominal_fps,
    parse_rational,
    to_num_den,
    total_frames,
)

# ------------------------------------------------------------- parse_rational


def test_parse_rational_fraction_form() -> None:
    assert parse_rational("24000/1001") == Fraction(24000, 1001)


def test_parse_rational_integer_form() -> None:
    assert parse_rational("24/1") == Fraction(24, 1)


def test_parse_rational_decimal_form() -> None:
    # Quantized to 1/100000; should be a fraction close to 23.976.
    r = parse_rational("23.976")
    assert r is not None
    assert abs(float(r) - 23.976) < 1e-5


@pytest.mark.parametrize("bad", ["", None, "0/0", "N/A", "garbage", "1/0"])
def test_parse_rational_rejects(bad: str | None) -> None:
    assert parse_rational(bad) is None


# ------------------------------------------------------------- to_num_den


def test_to_num_den_roundtrips_ntsc() -> None:
    n, d = to_num_den(Fraction(24000, 1001))
    assert (n, d) == (24000, 1001)


# ------------------------------------------------------------- helpers


def test_is_ntsc_drop_recognizes_24000_1001() -> None:
    assert is_ntsc_drop(Fraction(24000, 1001))
    assert is_ntsc_drop(Fraction(30000, 1001))
    assert is_ntsc_drop(Fraction(60000, 1001))
    assert not is_ntsc_drop(Fraction(24, 1))
    assert not is_ntsc_drop(Fraction(25, 1))


def test_nominal_fps_rounds() -> None:
    assert nominal_fps(Fraction(24000, 1001)) == 24
    assert nominal_fps(Fraction(30000, 1001)) == 30
    assert nominal_fps(Fraction(60000, 1001)) == 60


def test_total_frames_rounds_half_up() -> None:
    # 24fps × 60s = 1440 exactly
    assert total_frames(Fraction(24, 1), 60.0) == 1440
    # 23.976 × 60s ≈ 1438.56 → 1439
    assert total_frames(Fraction(24000, 1001), 60.0) == 1439


# ---------------------------------------------------- derive_target_fps


def test_derive_explicit_multiplier_overrides_target_fps() -> None:
    rate, mult, notes = derive_target_fps(
        Fraction(24000, 1001),
        target_fps=999.0,   # ignored when multiplier is set
        multiplier=2,
    )
    assert mult == 2
    assert rate == Fraction(48000, 1001)
    assert notes == []


def test_derive_target_fps_24_to_60_rounds_with_drift_warning() -> None:
    # 60 / (24000/1001) ≈ 2.5025 → rounds to 3 → realized ≈ 71.93fps → ~20% drift → warn.
    # We assert the warning behavior, not the specific multiplier (round() can
    # tie-break either way depending on Python version).
    rate, mult, notes = derive_target_fps(
        Fraction(24000, 1001),
        target_fps=60.0,
        multiplier=None,
    )
    assert mult in (2, 3)
    assert rate == Fraction(24000, 1001) * mult
    assert any("drift" in n for n in notes)


def test_derive_target_fps_24_to_48_clean_no_drift() -> None:
    # 48 / 23.976 ≈ 2.002 → rounds to 2 → realized = 48000/1001 → drift ~0.04% (no warn)
    rate, mult, notes = derive_target_fps(
        Fraction(24000, 1001),
        target_fps=48.0,
        multiplier=None,
    )
    assert mult == 2
    assert rate == Fraction(48000, 1001)
    assert notes == []


def test_derive_target_fps_unknown_source() -> None:
    rate, mult, notes = derive_target_fps(None, target_fps=60.0, multiplier=None)
    assert rate is None and mult == 1
    assert any("source fps unknown" in n for n in notes)

    rate, mult, notes = derive_target_fps(None, target_fps=None, multiplier=3)
    assert rate is None and mult == 3
    assert notes


def test_derive_target_fps_no_interpolation_requested() -> None:
    # target_fps=None and multiplier=None → multiplier=1, output rate = source.
    rate, mult, notes = derive_target_fps(
        Fraction(24, 1), target_fps=None, multiplier=None,
    )
    assert mult == 1
    assert rate == Fraction(24, 1)
    assert notes == []


def test_derive_target_fps_invalid_multiplier_clamped() -> None:
    rate, mult, notes = derive_target_fps(
        Fraction(24, 1), target_fps=None, multiplier=0,
    )
    assert mult == 1
    assert rate == Fraction(24, 1)
    assert notes


# ------------------------------------------------------------- fps_label


def test_fps_label_unknown_and_ntsc() -> None:
    assert fps_label(None) == "unknown"
    assert "ntsc" in fps_label(Fraction(24000, 1001))
