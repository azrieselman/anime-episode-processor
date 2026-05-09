"""Postprocess filter-chain assembly.

Stage 07 (postprocess) is optional. When enabled it applies a small set of
defensive filters between interpolation and final encode:

  * Deband: reduces banding artifacts that NVENC sometimes amplifies on
    smooth gradients (skies, shadows). We use FFmpeg's ``gradfun`` because it's
    faster than the deeper ``deband`` filter and good enough for SDR anime.
  * Deblock: smooths small encoding-block artifacts inherited from the source
    (only useful for compressed sources, hence opt-in). FFmpeg's ``deblock``.
  * Grain add-back: applies a small amount of FFmpeg ``noise`` so the encoder
    has something to chew on, preventing it from producing the over-smoothed
    "plastic" look that aggressive NVENC sometimes yields.

This module is pure: it only assembles the FFmpeg ``-vf`` chain string.
The stage runs FFmpeg with the chain and writes a frame directory or
intermediate file (the stage decides which based on the active pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostprocessChain:
    vf: str | None       # the assembled -vf chain, or None if no-op
    rationale: list[str]
    warnings: list[str]


def build_postprocess_chain(
    *,
    enabled: bool,
    deband: bool,
    deblock: bool,
    grain_addback: int,   # 0..32; 0 = off
) -> PostprocessChain:
    """Build the FFmpeg ``-vf`` chain. Returns ``vf=None`` if no filters apply.

    The order is intentional: deband first so it doesn't smooth out
    intentionally-added grain; deblock second so the frame is "clean" before
    we re-introduce noise; grain last so it's the final layer.
    """
    if not enabled:
        return PostprocessChain(vf=None, rationale=["postprocess disabled"], warnings=[])

    parts: list[str] = []
    rationale: list[str] = []
    warnings: list[str] = []

    if deband:
        # gradfun thresh=1.5 radius=8 is a reasonable middle-ground; lower thresh
        # over-smooths line art, higher does nothing.
        parts.append("gradfun=1.5:8")
        rationale.append("deband: gradfun=1.5:8 (smooths gradient banding)")

    if deblock:
        # deblock with weak alpha/beta; we never want to demolish detail.
        parts.append("deblock=filter=weak:block=4:alpha=0.05:beta=0.05")
        rationale.append("deblock: weak filter, block=4 (cleans source compression artifacts)")

    if grain_addback > 0:
        clamped = max(0, min(32, grain_addback))
        if clamped != grain_addback:
            warnings.append(
                f"grain_addback {grain_addback} clamped to {clamped} (valid range 0..32)"
            )
        # FFmpeg `noise` strength is per-channel; using only c0 (luma) keeps
        # chroma clean while giving the encoder something to grain-encode.
        parts.append(f"noise=c0s={clamped}:c0f=t+u")
        rationale.append(
            f"grain: noise c0s={clamped} luma temporal+uniform (prevents over-smooth NVENC look)"
        )

    if not parts:
        return PostprocessChain(vf=None, rationale=["postprocess enabled but all sub-filters off"], warnings=warnings)

    return PostprocessChain(vf=",".join(parts), rationale=rationale, warnings=warnings)
