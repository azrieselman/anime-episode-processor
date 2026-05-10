"""Tests for NCNN-Vulkan adapter base + per-tool argv builders.

We never invoke the real binaries. Instead we build a minimal on-disk fake
(an empty file + a `models/...` dir tree) under ``tmp_path``, point each
adapter at that directory via ``override_dir``, and assert the argv shape.
This validates: flag ordering, model-dir resolution, tile-size override,
TTA flag, frame-format flag, and engine-specific quirks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import aep.adapters.rife as rife_mod
from aep.adapters.anime4kcpp import Anime4kcppAdapter, Anime4kcppJob
from aep.adapters.ncnn_base import (
    SUPPORTED_FRAME_FORMATS,
    NcnnVulkanAdapter,
    count_frames_in_dir,
    expected_frame_filenames,
    load_tile_hint,
    save_tile_hint,
    stderr_indicates_oom,
)
from aep.adapters.realcugan import CuganJob, RealCuganAdapter
from aep.adapters.realesrgan import EsrganJob, RealesrganAdapter
from aep.adapters.rife import RifeAdapter, RifeJob
from aep.util.paths import tools_dir
from aep.util.proc import ProcResult

# ---------------------------------------------------------- fixtures


def _make_fake_tool(tmp_path: Path, bin_name: str, model_layout: dict[str, list[str]]) -> Path:
    """Create a fake tool dir on disk.

    ``model_layout`` maps subdir-relative-to-bin-dir → list of files to touch.
    Returns the tool dir (use as ``override_dir``).
    """
    tool_dir = tmp_path / "fakebin"
    tool_dir.mkdir()
    (tool_dir / bin_name).write_text("#!/bin/sh\nexit 0\n")
    for subdir, files in model_layout.items():
        d = tool_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            (d / f).write_text("fake")
    return tool_dir


# ---------------------------------------------------------- OOM regex


@pytest.mark.parametrize("stderr", [
    "vkAllocateMemory failed",
    "VK_ERROR_OUT_OF_DEVICE_MEMORY",
    "out of device memory",
    "failed to allocate 1024 MiB of memory",
])
def test_stderr_indicates_oom_matches_known_patterns(stderr: str) -> None:
    assert stderr_indicates_oom(stderr)


def test_stderr_indicates_oom_rejects_normal_output() -> None:
    assert not stderr_indicates_oom("processing frame 100/240")
    assert not stderr_indicates_oom("done")
    assert not stderr_indicates_oom("")


def test_ncnn_detect_version_finds_yyyymmdd_in_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Forks omit `version …` lines but frequently echo release tags elsewhere."""
    tool_dir = _make_fake_tool(tmp_path, "realesrgan-ncnn-vulkan.exe", {})

    def fake_capture(
        cmd: list[str | Path], **_: object,
    ) -> ProcResult:
        return ProcResult(
            cmd=[str(x) for x in cmd],
            returncode=-1,
            stdout="",
            stderr="Upstream https://github.com/x/releases/download/20220728/foo.zip\n",
        )

    monkeypatch.setattr("aep.adapters.ncnn_base.run_capture", fake_capture)
    a = RealesrganAdapter(override_dir=tool_dir)
    assert a.version == "20220728"


def test_rife_detect_version_embedded_pin_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "rife-ncnn-vulkan.exe", {"rife-v4.6": ["flownet.bin"]},
    )
    exe = tool_dir / "rife-ncnn-vulkan.exe"
    exe.write_bytes(exe.read_bytes() + b"TAG20250112Z")

    monkeypatch.setattr(
        NcnnVulkanAdapter,
        "_detect_version",
        lambda self: "unknown",
    )
    adapter = RifeAdapter(override_dir=tool_dir)
    assert adapter.version == "20250112"


def test_rife_bundled_slot_reports_manifest_pin_when_unreadable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tools_root = tmp_path / "tools"
    slot = tools_root / "rife-ncnn-vulkan"
    slot.mkdir(parents=True)
    (slot / "rife-ncnn-vulkan.exe").write_bytes(b"MZ")

    monkeypatch.setenv("AEP_TOOLS_DIR", str(tools_root))
    tools_dir.cache_clear()
    monkeypatch.setattr(NcnnVulkanAdapter, "_detect_version", lambda self: "unknown")
    monkeypatch.setattr(rife_mod, "pe_version_resource_strings", lambda _p: [])

    adapter = RifeAdapter()
    try:
        assert adapter.version == "20250112"
    finally:
        monkeypatch.delenv("AEP_TOOLS_DIR", raising=False)
        tools_dir.cache_clear()


# ---------------------------------------------------------- tile-hint persistence


def test_tile_hint_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aep.adapters.ncnn_base.cache_dir", lambda: tmp_path)
    save_tile_hint(
        hardware_fp="hwfp1", tool_id="realcugan-ncnn-vulkan",
        model_id="models-pro", source_height=1080, tile_size=128,
    )
    assert load_tile_hint(
        hardware_fp="hwfp1", tool_id="realcugan-ncnn-vulkan",
        model_id="models-pro", source_height=1080,
    ) == 128


def test_tile_hint_height_buckets_360_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("aep.adapters.ncnn_base.cache_dir", lambda: tmp_path)
    # Bucket = (h // 360) * 360. 1080 and 1439 both fall in bucket 1080.
    save_tile_hint(
        hardware_fp="hwfp1", tool_id="t", model_id="m",
        source_height=1080, tile_size=192,
    )
    assert load_tile_hint(
        hardware_fp="hwfp1", tool_id="t", model_id="m", source_height=1439,
    ) == 192
    # 720 (bucket 720) and 1080 (bucket 1080) live in different buckets.
    assert load_tile_hint(
        hardware_fp="hwfp1", tool_id="t", model_id="m", source_height=720,
    ) is None
    # 1078 is in the 720-bucket (1078 // 360 = 2 → 720), not the 1080-bucket.
    assert load_tile_hint(
        hardware_fp="hwfp1", tool_id="t", model_id="m", source_height=1078,
    ) is None


def test_tile_hint_missing_file_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aep.adapters.ncnn_base.cache_dir", lambda: tmp_path)
    assert load_tile_hint(
        hardware_fp="x", tool_id="y", model_id="z", source_height=1080,
    ) is None


# ---------------------------------------------------------- frame I/O helpers


def test_expected_frame_filenames_is_8digit_zero_padded() -> None:
    names = expected_frame_filenames(3, format="png")
    assert names == ["00000001.png", "00000002.png", "00000003.png"]


def test_expected_frame_filenames_rejects_unknown_format() -> None:
    with pytest.raises(ValueError):
        expected_frame_filenames(1, format="bmp")


def test_count_frames_in_dir_filters_by_format(tmp_path: Path) -> None:
    (tmp_path / "00000001.png").touch()
    (tmp_path / "00000002.png").touch()
    (tmp_path / "00000003.webp").touch()
    (tmp_path / "ignored.txt").touch()
    assert count_frames_in_dir(tmp_path, format="png") == 2
    assert count_frames_in_dir(tmp_path, format="webp") == 1
    # Default counts both supported formats.
    assert count_frames_in_dir(tmp_path) == 3


def test_supported_formats_are_lossless_only() -> None:
    # JPEG is lossy DCT — must NOT appear here even if NCNN binaries accept it.
    assert SUPPORTED_FRAME_FORMATS == ("png", "webp")


# ---------------------------------------------------------- Real-CUGAN argv


def test_realcugan_argv_includes_required_flags(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "realcugan-ncnn-vulkan.exe",
        {"models-pro": ["up2x-conservative.bin", "up2x-conservative.param"]},
    )
    adapter = RealCuganAdapter(override_dir=tool_dir)
    in_dir = tmp_path / "in"; in_dir.mkdir()
    out_dir = tmp_path / "out"; out_dir.mkdir()
    job = CuganJob(
        input_dir=in_dir, output_dir=out_dir,
        model_id="models-pro", scale=2, denoise=3, tile_size=256,
        gpu_id=0, tta=False, frame_format="png",
    )
    argv = adapter.build_cugan_argv(job)
    sargs = [str(a) for a in argv]
    # Binary first.
    assert sargs[0].endswith("realcugan-ncnn-vulkan.exe")
    # Each required flag-value pair appears.
    assert "-i" in sargs and str(in_dir) in sargs
    assert "-o" in sargs and str(out_dir) in sargs
    assert "-s" in sargs and "2" in sargs
    assert "-n" in sargs and "3" in sargs
    assert "-t" in sargs and "256" in sargs
    assert "-g" in sargs and "0" in sargs
    assert "-m" in sargs
    assert "-f" in sargs and "png" in sargs
    # No TTA when off.
    assert "-x" not in sargs


def test_realcugan_argv_tile_override_replaces_value(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "realcugan-ncnn-vulkan.exe", {"models-pro": ["x.bin"]},
    )
    adapter = RealCuganAdapter(override_dir=tool_dir)
    job = CuganJob(
        input_dir=tmp_path, output_dir=tmp_path,
        model_id="models-pro", scale=2, denoise=3, tile_size=256,
    )
    argv = adapter.build_cugan_argv(job, tile_size_override=128)
    sargs = [str(a) for a in argv]
    # The original 256 is gone, replaced with 128.
    t_idx = sargs.index("-t")
    assert sargs[t_idx + 1] == "128"


def test_realcugan_argv_tta_flag(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "realcugan-ncnn-vulkan.exe", {"models-pro": ["x.bin"]},
    )
    adapter = RealCuganAdapter(override_dir=tool_dir)
    job = CuganJob(
        input_dir=tmp_path, output_dir=tmp_path,
        model_id="models-pro", scale=2, denoise=3, tta=True,
    )
    sargs = [str(a) for a in adapter.build_cugan_argv(job)]
    assert "-x" in sargs


def test_realcugan_argv_below_floor_raises(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "realcugan-ncnn-vulkan.exe", {"models-pro": ["x.bin"]},
    )
    adapter = RealCuganAdapter(override_dir=tool_dir)
    job = CuganJob(input_dir=tmp_path, output_dir=tmp_path)
    with pytest.raises(ValueError, match="below floor"):
        adapter.build_cugan_argv(job, tile_size_override=32)


def test_realcugan_validate_combination_warns_off_grid() -> None:
    # (4, denoise=3) is not in _SUPPORTED_PRO and pro lacks native 4x → 2 warnings.
    warnings = RealCuganAdapter.validate_combination("models-pro", 4, 3)
    assert len(warnings) == 2
    assert any("fall back" in w for w in warnings)
    assert any("4x" in w or "x4" in w.lower() for w in warnings)


def test_realcugan_validate_combination_silent_on_supported() -> None:
    assert RealCuganAdapter.validate_combination("models-pro", 2, 3) == []
    assert RealCuganAdapter.validate_combination("models-se", 4, 0) == []


# ---------------------------------------------------------- Real-ESRGAN argv


def test_realesrgan_argv_passes_n_flag(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "realesrgan-ncnn-vulkan.exe",
        {"models": ["realesr-animevideov3.bin", "realesr-animevideov3.param"]},
    )
    adapter = RealesrganAdapter(override_dir=tool_dir)
    job = EsrganJob(
        input_dir=tmp_path, output_dir=tmp_path,
        model_id="realesr-animevideov3", scale=4, tile_size=192,
    )
    sargs = [str(a) for a in adapter.build_esrgan_argv(job)]
    # ESRGAN-specific: -n <model_id> appears.
    assert "-n" in sargs and "realesr-animevideov3" in sargs
    # Scale wired through.
    s_idx = sargs.index("-s")
    assert sargs[s_idx + 1] == "4"


def test_realesrgan_validate_warns_on_stills_model() -> None:
    warnings = RealesrganAdapter.validate_combination("realesrgan-x4plus-anime", 4)
    assert any("stills" in w or "flicker" in w for w in warnings)


def test_realesrgan_validate_warns_on_unknown_model() -> None:
    warnings = RealesrganAdapter.validate_combination("nonexistent-model", 4)
    assert any("not in our catalog" in w for w in warnings)


def test_realesrgan_validate_clean_on_video_default() -> None:
    assert RealesrganAdapter.validate_combination("realesr-animevideov3", 4) == []


# ---------------------------------------------------------- Anime4KCPP argv


def test_anime4kcpp_argv_uses_factor_flag_and_processor(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, "ac_cli.exe", {})
    adapter = Anime4kcppAdapter(override_dir=tool_dir)
    adapter._preferred_processor = "opencl"
    job = Anime4kcppJob(
        input_path=tmp_path / "00000001.png",
        output_path=tmp_path / "out" / "00000001.png",
        model_id="acnet-f8b8-hdn",
        scale=2,
        prefer_cuda=False,
        threads=8,
    )
    sargs = [str(a) for a in adapter.build_anime4kcpp_argv(job)]
    assert sargs[0].endswith("ac_cli.exe")
    assert "-i" in sargs and str(tmp_path / "00000001.png") in sargs
    assert "-o" in sargs and str(tmp_path / "out" / "00000001.png") in sargs
    assert "-m" in sargs and "acnet-f8b8-hdn" in sargs
    assert "-p" in sargs and "opencl" in sargs
    assert "-f" in sargs
    fi = sargs.index("-f")
    assert sargs[fi + 1] in ("2", "2.0")
    assert "-t" in sargs and "8" in sargs
    assert "-z" not in sargs
    assert "-n" not in sargs


def test_anime4kcpp_argv_batch_lists_inputs_and_outputs(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(tmp_path, "ac_cli.exe", {})
    adapter = Anime4kcppAdapter(override_dir=tool_dir)
    argv = adapter.build_anime4kcpp_argv_batch(
        [tmp_path / "a.png", tmp_path / "b.png"],
        [tmp_path / "out" / "a.png", tmp_path / "out" / "b.png"],
        model_id="acnet-f8b8-hdn",
        scale=2,
        processor="cuda",
        gpu_id=0,
        threads=4,
    )
    sargs = [str(a) for a in argv]
    assert sargs[:3] == [str(adapter.path), "-i", str(tmp_path / "a.png")]
    assert str(tmp_path / "b.png") in sargs
    assert "-o" in sargs
    assert "-t" in sargs and "4" in sargs


def test_anime4kcpp_validate_combination_unknown_model_warns() -> None:
    warnings = Anime4kcppAdapter.validate_combination("unknown", 2, 1)
    assert any("not in our catalog" in w for w in warnings)


# ---------------------------------------------------------- RIFE argv


def test_rife_argv_includes_multiplier_via_dash_s(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "rife-ncnn-vulkan.exe",
        {"rife-v4.6": ["flownet.bin", "flownet.param"]},
    )
    adapter = RifeAdapter(override_dir=tool_dir)
    job = RifeJob(
        input_dir=tmp_path, output_dir=tmp_path,
        version="v4.6", multiplier=2, tile_size=0,
    )
    sargs = [str(a) for a in adapter.build_rife_argv(job)]
    s_idx = sargs.index("-s")
    assert sargs[s_idx + 1] == "2"
    # tile_size=0 → no -t flag.
    assert "-t" not in sargs


def test_rife_argv_uses_configured_threads(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "rife-ncnn-vulkan.exe",
        {"rife-v4.6": ["flownet.bin", "flownet.param"]},
    )
    adapter = RifeAdapter(override_dir=tool_dir)
    job = RifeJob(
        input_dir=tmp_path, output_dir=tmp_path,
        version="v4.6", multiplier=2, threads="6:8:6",
    )
    sargs = [str(a) for a in adapter.build_rife_argv(job)]
    j_idx = sargs.index("-j")
    assert sargs[j_idx + 1] == "6:8:6"


def test_rife_argv_uhd_flag(tmp_path: Path) -> None:
    tool_dir = _make_fake_tool(
        tmp_path, "rife-ncnn-vulkan.exe", {"rife-v4.6": ["flownet.bin"]},
    )
    adapter = RifeAdapter(override_dir=tool_dir)
    job = RifeJob(
        input_dir=tmp_path, output_dir=tmp_path,
        version="v4.6", multiplier=2, uhd=True,
    )
    sargs = [str(a) for a in adapter.build_rife_argv(job)]
    assert "-u" in sargs


def test_rife_validate_version_warns_on_lite() -> None:
    warnings = RifeAdapter.validate_version("v4.22-lite")
    assert any("lite" in w for w in warnings)


def test_rife_validate_version_warns_on_unknown() -> None:
    warnings = RifeAdapter.validate_version("v9.99-invented")
    assert any("not in our catalog" in w for w in warnings)


def test_rife_validate_version_clean_on_v4_6() -> None:
    assert RifeAdapter.validate_version("v4.6") == []


