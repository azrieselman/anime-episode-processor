from __future__ import annotations

from pathlib import Path

from aep.adapters.anime4kcpp_legacy import Anime4kcppLegacyAdapter


def test_anime4k_legacy_argv_uses_directories_and_fixed_flags(tmp_path: Path) -> None:
    exe = tmp_path / "Anime4KCPP_CLI.exe"
    exe.write_bytes(b"")
    adapter = Anime4kcppLegacyAdapter(override_dir=tmp_path)
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    argv = adapter.build_anime4k_legacy_argv(
        input_dir=in_dir,
        output_dir=out_dir,
        scale=2,
        threads=8,
        gpgpu="cuda",
        gpu_id=0,
    )
    sargs = [str(a) for a in argv]
    assert sargs[0] == str(exe)
    assert "-i" in sargs and "-o" in sargs
    i_idx = sargs.index("-i")
    o_idx = sargs.index("-o")
    assert Path(sargs[i_idx + 1]) == in_dir.resolve()
    assert Path(sargs[o_idx + 1]) == out_dir.resolve()
    assert "-w" in sargs and "-H" in sargs
    assert sargs[sargs.index("-L") + 1] == "1"
    assert "-z" in sargs and sargs[sargs.index("-z") + 1] == "2.0"
    assert "-t" in sargs and sargs[sargs.index("-t") + 1] == "8"
    assert "-q" in sargs
    assert "-M" in sargs and sargs[sargs.index("-M") + 1] == "cuda"
    assert "-d" in sargs and sargs[sargs.index("-d") + 1] == "0"


def test_anime4k_legacy_argv_appends_platform_id_when_nonzero(tmp_path: Path) -> None:
    exe = tmp_path / "Anime4KCPP_CLI.exe"
    exe.write_bytes(b"")
    adapter = Anime4kcppLegacyAdapter(override_dir=tmp_path)
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    argv = adapter.build_anime4k_legacy_argv(
        input_dir=in_dir,
        output_dir=out_dir,
        scale=2,
        threads=4,
        gpgpu="opencl",
        platform_id=2,
    )
    sargs = [str(a) for a in argv]
    assert sargs[-2] == "-h"
    assert sargs[-1] == "2"


def test_anime4k_legacy_validate_combination_unknown_model_warns() -> None:
    w = Anime4kcppLegacyAdapter.validate_combination("not-a-real-model", 2, 1)
    assert any("not in our catalog" in x for x in w)


def test_anime4k_legacy_validate_combination_scale_range_warns() -> None:
    w = Anime4kcppLegacyAdapter.validate_combination("acnet", 9, 1)
    assert any("1..4" in x for x in w)
