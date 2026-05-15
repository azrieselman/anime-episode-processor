"""FPS math helpers.

Anime sources commonly use NTSC-derived fractional rates (24000/1001 ≈ 23.976,
30000/1001 ≈ 29.97). Round-tripping these through float seconds loses
precision in ways the validate stage will catch (off-by-one on durations).
We pass them around as exact Fractions where possible.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aep.adapters.rife import FrameRun


def parse_rational(text: str | None) -> Fraction | None:
    """Parse strings like ``"24000/1001"``, ``"24/1"``, ``"23.976"``.

    Returns None for empty/None/zero-denominator inputs.
    """
    if not text:
        return None
    text = text.strip()
    if not text or text in {"0/0", "N/A"}:
        return None
    if "/" in text:
        try:
            num, den = text.split("/", 1)
            n, d = int(num), int(den)
            if d == 0:
                return None
            return Fraction(n, d)
        except ValueError:
            return None
    try:
        # Float fallback — quantize to 1/100000 to avoid Fraction blowups.
        return Fraction(float(text)).limit_denominator(100000)
    except ValueError:
        return None


def to_num_den(rate: Fraction) -> tuple[int, int]:
    """Return (num, den) suitable for ffmpeg ``-framerate num/den``."""
    return rate.numerator, rate.denominator


def is_ntsc_drop(rate: Fraction) -> bool:
    """True for NTSC-style 1000/1001 rates (24000/1001, 30000/1001, 60000/1001)."""
    return rate.denominator == 1001 and rate.numerator % 1000 == 0


def nominal_fps(rate: Fraction) -> int:
    """The nearest integer fps. 24000/1001 → 24, 30000/1001 → 30, 60000/1001 → 60."""
    return int(round(float(rate)))


def total_frames(rate: Fraction, duration_s: float) -> int:
    """Conservative total-frame count for a rate × duration. Rounds half-up."""
    return int(math.floor(float(rate) * duration_s + 0.5))


def derive_target_fps(
    source: Fraction | None,
    *,
    target_fps: float | None,
    multiplier: int | None,
) -> tuple[Fraction | None, int | None, list[str]]:
    """Decide the effective output fps and integer multiplier for RIFE.

    Rules:
    * If ``multiplier`` is set explicitly, use it as-is. Output rate = source × multiplier.
    * If ``target_fps`` is set and ``source`` is known, multiplier = round(target/source).
      Output rate = source × multiplier (NOT exactly target_fps; we keep the
      source's fractional structure to avoid drift). The caller is warned if
      the realized fps differs from target_fps by >0.5%.
    * If neither is set, multiplier=1 (no interpolation).
    * Returns (effective_rate, multiplier, warnings).
    """
    notes: list[str] = []
    if source is None:
        if multiplier is not None:
            return None, multiplier, ["source fps unknown; cannot compute output rate"]
        return None, 1, ["source fps unknown; assuming no interpolation"]

    if multiplier is not None:
        if multiplier < 1:
            return source, 1, [f"multiplier {multiplier} < 1 is invalid; using 1"]
        return source * multiplier, multiplier, notes

    if target_fps is None or target_fps <= 0:
        return source, 1, notes

    raw_mult = target_fps / float(source)
    rounded = max(1, int(round(raw_mult)))
    realized = source * rounded
    drift = abs(float(realized) - target_fps) / target_fps
    if drift > 0.005:
        notes.append(
            f"target {target_fps}fps from source {float(source):.3f}fps requires non-integer "
            f"multiplier {raw_mult:.3f}; rounded to {rounded}× → {float(realized):.3f}fps "
            f"(drift {drift * 100:.2f}%)"
        )
    return realized, rounded, notes


def runs_total_input_frames(runs: Iterable["FrameRun"]) -> int:
    return sum(r.length for r in runs)


# Convenience for tests / debugging.
def fps_label(rate: Fraction | None) -> str:
    if rate is None:
        return "unknown"
    if is_ntsc_drop(rate):
        return f"{nominal_fps(rate)}p ntsc ({rate.numerator}/{rate.denominator})"
    return f"{float(rate):.3f}"
