"""Tests for `aep.encode.postprocess`.

Pure string-assembly module; tests assert the exact filter graph composition
because downstream cache keys hash this string.
"""

from __future__ import annotations

from aep.encode.postprocess import build_postprocess_chain


def test_disabled_returns_no_vf() -> None:
    chain = build_postprocess_chain(
        enabled=False, deband=True, deblock=True, grain_addback=10,
    )
    assert chain.vf is None
    assert any("disabled" in r for r in chain.rationale)


def test_enabled_with_no_subfilters_is_noop() -> None:
    chain = build_postprocess_chain(
        enabled=True, deband=False, deblock=False, grain_addback=0,
    )
    assert chain.vf is None
    assert any("all sub-filters off" in r for r in chain.rationale)


def test_full_chain_order_is_deband_deblock_grain() -> None:
    chain = build_postprocess_chain(
        enabled=True, deband=True, deblock=True, grain_addback=8,
    )
    assert chain.vf is not None
    parts = chain.vf.split(",")
    assert parts[0].startswith("gradfun")
    assert parts[1].startswith("deblock")
    assert parts[2].startswith("noise=c0s=8")


def test_grain_addback_clamps_above_32() -> None:
    chain = build_postprocess_chain(
        enabled=True, deband=False, deblock=False, grain_addback=99,
    )
    assert chain.vf is not None
    assert "noise=c0s=32" in chain.vf
    assert any("clamped" in w for w in chain.warnings)


def test_grain_only() -> None:
    chain = build_postprocess_chain(
        enabled=True, deband=False, deblock=False, grain_addback=4,
    )
    assert chain.vf == "noise=c0s=4:c0f=t+u"
    assert chain.warnings == []


def test_deband_only_uses_gradfun_default_params() -> None:
    chain = build_postprocess_chain(
        enabled=True, deband=True, deblock=False, grain_addback=0,
    )
    assert chain.vf == "gradfun=1.5:8"
