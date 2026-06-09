"""Validate that all built-in presets parse and conform to the Preset schema."""

from __future__ import annotations

from pathlib import Path

from aep.persist.presets import EncoderCfg, Preset, _load_yaml_file, amf_rc_matches_when


def test_all_builtin_presets_parse() -> None:
    presets_dir = Path(__file__).resolve().parents[2] / "presets"
    yamls = sorted(presets_dir.glob("*.yaml"))
    assert yamls, "expected built-in presets to be present"
    seen_ids: set[str] = set()
    for p in yamls:
        preset: Preset = _load_yaml_file(p)
        assert preset.meta.id, f"preset missing id: {p}"
        assert preset.meta.name, f"preset missing name: {p}"
        assert preset.meta.id not in seen_ids, f"duplicate preset id: {preset.meta.id}"
        seen_ids.add(preset.meta.id)


def test_amf_cqp_coerces_vbaq_off() -> None:
    cfg = EncoderCfg.model_validate(
        {"name": "hevc_amf", "amf_rc": "cqp", "amf_vbaq": True},
    )
    assert cfg.amf_vbaq is False


def test_amf_vbr_latency_allows_vbaq() -> None:
    cfg = EncoderCfg.model_validate(
        {"name": "hevc_amf", "amf_rc": "vbr_latency", "amf_vbaq": True},
    )
    assert cfg.amf_vbaq is True


def test_amf_rc_matches_when_tokens() -> None:
    assert amf_rc_matches_when("cqp", "cqp")
    assert amf_rc_matches_when("cqp", "constqp")
    assert not amf_rc_matches_when("cqp", "vbr_peak")
    assert amf_rc_matches_when("vbr", "vbr_peak")
    assert amf_rc_matches_when("vbr", "vbr_latency")
    assert not amf_rc_matches_when("vbr", "cbr")
    assert amf_rc_matches_when("cbr", "cbr")


def test_required_default_preset_present() -> None:
    presets_dir = Path(__file__).resolve().parents[2] / "presets"
    ids = {_load_yaml_file(p).meta.id for p in presets_dir.glob("*.yaml")}
    for required in ("anime_balanced", "anime_quality", "anime_speed",
                     "mixed_balanced", "low_vram_safe"):
        assert required in ids, f"missing required preset: {required}"
