from __future__ import annotations

from pathlib import Path

import pytest

from aep.adapters.anime4kcpp_vs import Anime4kcppVsAdapter, VapourSynthAdapter
from aep.util.paths import tools_dir
from aep.util.proc import ProcResult


def test_anime4kcpp_vs_validate_combination_unknown_model_warns() -> None:
    warnings = Anime4kcppVsAdapter.validate_combination("unknown-model", 2, 1)
    assert any("not in our catalog" in w for w in warnings)


def test_anime4kcpp_vs_validate_combination_scale_range_warns() -> None:
    warnings = Anime4kcppVsAdapter.validate_combination("acnet-f8b8-hdn", 5, 1)
    assert any("within 1..4" in w for w in warnings)


def test_anime4kcpp_vs_plugin_override_missing_raises(tmp_path: Path) -> None:
    adapter = Anime4kcppVsAdapter(
        vspipe_override_dir=tmp_path,
        plugin_override_dir=tmp_path / "missing",
    )
    try:
        adapter._plugin_path()
    except FileNotFoundError:
        return
    raise AssertionError("expected missing plugin override to raise FileNotFoundError")


def test_vapoursynth_runtime_env_includes_tools_on_pythonpath() -> None:
    adapter = VapourSynthAdapter(override_dir=tools_dir() / "vapoursynth")
    env = adapter.runtime_env()
    py_path = env.get("PYTHONPATH", "")
    assert str(tools_dir()) in py_path


def test_vspipe_version_probe_falls_back_to_help_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    (tmp_path / "vspipe.exe").write_bytes(b"#")
    adapter = VapourSynthAdapter(override_dir=tmp_path)
    calls = 0

    def fake_capture(
        cmd: list[str | Path], **_: object,
    ) -> ProcResult:
        nonlocal calls
        calls += 1
        c = [str(x) for x in cmd]
        assert c[1] == "--version" if calls == 1 else c[1] == "-h"
        if calls == 1:
            return ProcResult(
                cmd=c, returncode=1, stdout="", stderr="Failed to create core\n",
            )
        return ProcResult(
            cmd=c,
            returncode=-1,
            stdout="",
            stderr="VSPipe R74 usage:\n vspipe [options]\n",
        )

    monkeypatch.setattr("aep.adapters.anime4kcpp_vs.run_capture", fake_capture)
    assert adapter.version == "R74"
    assert calls == 2


def test_vapoursynth_config_does_not_use_gui_exe_when_frozen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    adapter = VapourSynthAdapter(override_dir=tmp_path)
    calls: list[list[str]] = []

    def fake_capture(cmd: list[str | Path], **_: object) -> ProcResult:
        calls.append([str(x) for x in cmd])
        return ProcResult(cmd=[str(x) for x in cmd], returncode=0, stdout="", stderr="")

    monkeypatch.setattr("aep.adapters.anime4kcpp_vs.run_capture", fake_capture)
    monkeypatch.setattr("aep.adapters.anime4kcpp_vs.sys.frozen", True, raising=False)
    monkeypatch.setattr(
        "aep.adapters.anime4kcpp_vs.sys.executable",
        str(tmp_path / "aep-gui.exe"),
        raising=False,
    )

    VapourSynthAdapter._configured = False
    try:
        adapter._ensure_python_config(tmp_path)
    finally:
        VapourSynthAdapter._configured = False

    assert calls == []
