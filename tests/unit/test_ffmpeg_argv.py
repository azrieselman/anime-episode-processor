"""Tests for ffmpeg extra_args normalization."""

from __future__ import annotations

import yaml

from aep.persist.presets import EncoderCfg
from aep.util.ffmpeg_argv import normalize_ffmpeg_extra_args


def test_strips_posix_style_single_quoted_tokens() -> None:
    assert normalize_ffmpeg_extra_args(["'-preset'", "slow"]) == ["-preset", "slow"]


def test_splits_space_separated_line() -> None:
    text = "-x265-params aq-mode=3:psy-rd=2.0\n-preset slow"
    assert normalize_ffmpeg_extra_args(text) == [
        "-x265-params",
        "aq-mode=3:psy-rd=2.0",
        "-preset",
        "slow",
    ]


def test_coerces_yaml_scalar_string() -> None:
    raw = yaml.safe_load("extra_args: -preset slow")
    assert raw == {"extra_args": "-preset slow"}
    cfg = EncoderCfg.model_validate(raw)
    assert cfg.extra_args == ["-preset", "slow"]


def test_preserves_colon_params_without_extra_quoting() -> None:
    cfg = EncoderCfg.model_validate(
        {"extra_args": ["-x265-params", "aq-mode=3:psy-rd=2.0:rd=4"]},
    )
    assert cfg.extra_args == ["-x265-params", "aq-mode=3:psy-rd=2.0:rd=4"]
