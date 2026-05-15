"""Tests for `JobBroker._derive_output`.

The naming template is locked at three vars for 1.0: {source_stem}, {height},
{fps}. The broker formats the template using preset-resolved values; pipeline
data isn't available yet at output-derivation time.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from aep.jobs.broker import JobBroker
from aep.jobs.models import Job
from aep.persist.presets import InterpolationCfg, Preset, PresetMeta, TargetResolution
from aep.persist.settings import AppSettings


def _job(src: str = "/data/Show.S01E01.mkv") -> Job:
    return Job(source_path=src, output_path=None, preset_id="anime_balanced")


def _preset(
    *,
    container: str = "mkv",
    res_mode: str = "named",
    res_named: str | None = "1080p",
    res_w: int | None = None,
    res_h: int | None = None,
    interp_enabled: bool = False,
    interp_target_fps: float | None = None,
) -> Preset:
    p = Preset(meta=PresetMeta(id="t", name="t"))
    p.container = container  # type: ignore[assignment]
    p.target_resolution = TargetResolution(
        mode=res_mode,  # type: ignore[arg-type]
        named=res_named,  # type: ignore[arg-type]
        width=res_w,
        height=res_h,
    )
    p.interpolation = InterpolationCfg(
        enabled=interp_enabled,
        target_fps=interp_target_fps,
    )
    return p


def _settings(template: str, output_dir: str | None = None) -> AppSettings:
    s = AppSettings()
    s.general.output_naming_template = template
    s.general.output_dir = output_dir
    return s


# ---------------------------------------------------------------- happy paths


def test_default_template_with_named_1080p_and_60fps() -> None:
    job = _job()
    preset = _preset(res_named="1080p", interp_enabled=True, interp_target_fps=60.0)
    settings = _settings("{source_stem}.{height}p.{fps}fps.aep.mkv")
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out == Path("/data/Show.S01E01.1080p.60fps.aep.mkv")


def test_template_with_explicit_resolution() -> None:
    job = _job()
    preset = _preset(res_mode="explicit", res_named=None, res_w=2560, res_h=1440)
    settings = _settings("{source_stem}.{height}p.aep.mkv")
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out.name == "Show.S01E01.1440p.aep.mkv"


def test_height_falls_back_to_src_when_scale_only() -> None:
    job = _job()
    preset = _preset(res_mode="scale_only", res_named=None)
    settings = _settings("{source_stem}.{height}.aep.mkv")
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out.name == "Show.S01E01.src.aep.mkv"


def test_fps_falls_back_to_src_when_interpolation_disabled() -> None:
    job = _job()
    preset = _preset(interp_enabled=False)
    settings = _settings("{source_stem}.{fps}fps.mkv")
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out.name == "Show.S01E01.srcfps.mkv"


def test_non_integer_target_fps_formatted_compactly() -> None:
    job = _job()
    preset = _preset(interp_enabled=True, interp_target_fps=23.976)
    settings = _settings("{source_stem}.{fps}.mkv")
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out.name == "Show.S01E01.23.976.mkv"


# ---------------------------------------------------------------- output_dir


def test_output_dir_overrides_source_parent(tmp_path: Path) -> None:
    job = _job()
    preset = _preset()
    settings = _settings(
        "{source_stem}.aep.mkv",
        output_dir=str(tmp_path),
    )
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out.parent == tmp_path
    assert out.name == "Show.S01E01.aep.mkv"


# ---------------------------------------------------------------- fallbacks


def test_template_with_unknown_var_falls_back_to_legacy_name() -> None:
    job = _job()
    preset = _preset()
    # `{episode}` is not a supported var — formatting raises KeyError, so we
    # fall back to "<stem>.aep.<ext>" instead of refusing the job.
    settings = _settings("{source_stem}.{episode}.mkv")
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out.name == "Show.S01E01.aep.mkv"


def test_blank_template_falls_back_to_legacy_name() -> None:
    job = _job()
    preset = _preset()
    settings = _settings("")
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out.name == "Show.S01E01.aep.mkv"


def test_missing_extension_in_template_gets_container_extension() -> None:
    job = _job()
    preset = _preset(container="mp4")
    # Note: no .ext on the template.
    settings = _settings("{source_stem}_clean")
    with patch("aep.jobs.broker.load_settings", return_value=settings):
        out = JobBroker._derive_output(job, preset)
    assert out.name == "Show.S01E01_clean.mp4"


def test_settings_load_failure_falls_back_to_legacy_name() -> None:
    job = _job()
    preset = _preset()
    with patch("aep.jobs.broker.load_settings", side_effect=RuntimeError("boom")):
        out = JobBroker._derive_output(job, preset)
    assert out.name == "Show.S01E01.aep.mkv"
    # Should still write next to source when settings can't be loaded.
    assert out.parent == Path("/data")
