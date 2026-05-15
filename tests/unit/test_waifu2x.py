"""Tests for the waifu2x-ncnn-vulkan adapter.

Mirrors the patterns in test_ncnn_adapters.py: build a fake on-disk tool dir
under ``tmp_path``, point the adapter at it via ``override_dir``, and assert
on argv shape + validate_combination outputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aep.adapters.waifu2x import Waifu2xAdapter, Waifu2xJob

# ---------------------------------------------------------- fixtures


def _make_fake_tool(tmp_path: Path, model_layout: dict[str, list[str]]) -> Path:
    """Create a fake waifu2x tool dir on disk.

    ``model_layout`` maps subdir-relative-to-bin-dir → list of files to touch.
    Returns the tool dir (use as ``override_dir``).
    """
    tool_dir = tmp_path / "fakebin"
    tool_dir.mkdir()
    (tool_dir / "waifu2x-ncnn-vulkan.exe").write_text("#!/bin/sh\nexit 0\n")
    for subdir, files in model_layout.items():
        d = tool_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_text("fake")
    return tool_dir


# ---------------------------------------------------------- argv shape


def test_waifu2x_argv_includes_required_flags(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path,
        {"models-cunet": ["noise3_scale2.0x_model.bin", "noise3_scale2.0x_model.param"]},
    )
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    in_dir = tmp_path / "in"; in_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir()
    job = Waifu2xJob(
        input_dir=in_dir, output_dir=out_dir,
        model_id="models-cunet", scale=2, denoise=3, tile_size=256,
        gpu_id=0, tta=False, frame_format="png",
    )
    argv = adapter.build_waifu2x_argv(job)
    sargs = [str(a) for a in argv]

    # Binary first.
    assert sargs[0].endswith("waifu2x-ncnn-vulkan.exe")
    # Each required flag-value pair appears.
    assert "-i" in sargs and str(in_dir) in sargs
    assert "-o" in sargs and str(out_dir) in sargs
    assert "-s" in sargs and "2" in sargs
    assert "-n" in sargs and "3" in sargs
    assert "-t" in sargs and "256" in sargs
    assert "-g" in sargs and "0" in sargs
    assert "-m" in sargs
    assert "-f" in sargs and "png" in sargs
    # Model dir resolves to the cunet sibling dir.
    m_idx = sargs.index("-m")
    assert sargs[m_idx + 1].endswith("models-cunet")
    # No TTA when off.
    assert "-x" not in sargs


def test_waifu2x_argv_tile_override_replaces_value(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, {"models-cunet": ["x.bin"]})
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    job = Waifu2xJob(
        input_dir=tmp_path, output_dir=tmp_path,
        model_id="models-cunet", scale=2, denoise=3, tile_size=256,
    )
    argv = adapter.build_waifu2x_argv(job, tile_size_override=128)
    sargs = [str(a) for a in argv]
    t_idx = sargs.index("-t")
    assert sargs[t_idx + 1] == "128"


def test_waifu2x_argv_tta_flag(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, {"models-cunet": ["x.bin"]})
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    job = Waifu2xJob(
        input_dir=tmp_path, output_dir=tmp_path,
        model_id="models-cunet", scale=2, denoise=3, tta=True,
    )
    sargs = [str(a) for a in adapter.build_waifu2x_argv(job)]
    assert "-x" in sargs


def test_waifu2x_argv_denoise_minus_one_passes_literal(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, {"models-cunet": ["x.bin"]})
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    job = Waifu2xJob(
        input_dir=tmp_path, output_dir=tmp_path,
        model_id="models-cunet", scale=2, denoise=-1,
    )
    sargs = [str(a) for a in adapter.build_waifu2x_argv(job)]
    n_idx = sargs.index("-n")
    assert sargs[n_idx + 1] == "-1"


def test_waifu2x_argv_below_floor_raises(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, {"models-cunet": ["x.bin"]})
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    job = Waifu2xJob(input_dir=tmp_path, output_dir=tmp_path)
    with pytest.raises(ValueError, match="below floor"):
        adapter.build_waifu2x_argv(job, tile_size_override=32)


def test_waifu2x_argv_webp_format(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, {"models-cunet": ["x.bin"]})
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    job = Waifu2xJob(
        input_dir=tmp_path, output_dir=tmp_path,
        model_id="models-cunet", scale=2, denoise=3, frame_format="webp",
    )
    sargs = [str(a) for a in adapter.build_waifu2x_argv(job)]
    f_idx = sargs.index("-f")
    assert sargs[f_idx + 1] == "webp"


# ---------------------------------------------------------- model dir resolution


def test_waifu2x_resolves_cunet_model_dir(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, {"models-cunet": ["x.bin"]})
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    resolved = adapter.resolve_model_dir("models-cunet")
    assert resolved == (tool_dir / "models-cunet").resolve()


def test_waifu2x_resolves_upconv_anime_model_dir(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, {"models-upconv_7_anime_style_art_rgb": ["x.bin"]},
    )
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    resolved = adapter.resolve_model_dir("models-upconv_7_anime_style_art_rgb")
    assert resolved.name == "models-upconv_7_anime_style_art_rgb"


def test_waifu2x_missing_model_dir_raises(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, {"models-cunet": ["x.bin"]})
    adapter = Waifu2xAdapter(override_dir=tool_dir)
    from aep.errors import ToolNotFoundError
    with pytest.raises(ToolNotFoundError):
        adapter.resolve_model_dir("models-nonexistent")


# ---------------------------------------------------------- validate_combination


def test_validate_combination_clean_on_cunet_2x_denoise3() -> None:
    assert Waifu2xAdapter.validate_combination("models-cunet", 2, 3) == []


def test_validate_combination_clean_on_cunet_1x_denoise_minus1() -> None:
    # cunet is the only model that supports scale=1 (denoise-only mode).
    assert Waifu2xAdapter.validate_combination("models-cunet", 1, -1) == []


def test_validate_combination_clean_on_upconv_anime() -> None:
    assert Waifu2xAdapter.validate_combination(
        "models-upconv_7_anime_style_art_rgb", 2, 1,
    ) == []


def test_validate_combination_warns_on_unknown_model() -> None:
    warnings = Waifu2xAdapter.validate_combination("models-invented", 2, 3)
    assert any("not in our catalog" in w for w in warnings)


def test_validate_combination_warns_on_4x_scale() -> None:
    # Waifu2x has no native 4x — every model will warn for scale=4.
    warnings = Waifu2xAdapter.validate_combination("models-cunet", 4, 3)
    assert any("scale=4" in w for w in warnings)


def test_validate_combination_warns_on_upconv_1x() -> None:
    # upconv_7 variants only ship scale=2.
    warnings = Waifu2xAdapter.validate_combination(
        "models-upconv_7_anime_style_art_rgb", 1, 3,
    )
    assert any("scale=1" in w for w in warnings)


def test_validate_combination_warns_on_off_grid_denoise() -> None:
    warnings = Waifu2xAdapter.validate_combination("models-cunet", 2, 5)
    assert any("denoise=5" in w for w in warnings)


def test_validate_combination_warns_on_photo_model() -> None:
    warnings = Waifu2xAdapter.validate_combination("models-upconv_7_photo", 2, 3)
    assert any("photo model" in w for w in warnings)
    assert any("models-cunet" in w for w in warnings)
