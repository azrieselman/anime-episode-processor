from pathlib import Path

import pytest

from aep.app import imdisk
from aep.persist.settings import AppSettings


def test_size_gb_to_kb() -> None:
    assert imdisk.size_gb_to_kb(32) == 33_554_432


def test_size_gb_to_kb_rejects_zero() -> None:
    with pytest.raises(ValueError, match="at least 1 GB"):
        imdisk.size_gb_to_kb(0)


def test_is_installed_requires_ramdyn_and_imdisk(tmp_path: Path) -> None:
    assert not imdisk.is_installed(tmp_path)

    (tmp_path / "RamDyn.exe").write_text("", encoding="utf-8")
    assert not imdisk.is_installed(tmp_path)

    (tmp_path / "ImDisk-Dlg.exe").write_text("", encoding="utf-8")
    assert imdisk.is_installed(tmp_path)


def test_drive_letter_normalization() -> None:
    assert imdisk.normalize_drive_letter("x:") == "X"
    assert imdisk.normalize_drive_letter(" r\\") == "R"
    assert imdisk.mountpoint("x") == "X:"
    assert imdisk.root_path("x") == "X:\\"


@pytest.mark.parametrize("letter", ["", "AA", "1", ":"])
def test_drive_letter_normalization_rejects_invalid_values(letter: str) -> None:
    with pytest.raises(ValueError, match="single A-Z"):
        imdisk.normalize_drive_letter(letter)


def test_is_drive_available_uses_get_drive_type(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_get_drive_type(path: str) -> int:
        seen.append(path)
        return imdisk.DRIVE_NO_ROOT_DIR

    monkeypatch.setattr(imdisk.sys, "platform", "win32")
    monkeypatch.setattr(imdisk, "_get_drive_type", fake_get_drive_type)

    assert imdisk.is_drive_available("x") is True
    assert seen == ["X:\\"]


def test_is_drive_available_false_for_existing_drive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(imdisk.sys, "platform", "win32")
    monkeypatch.setattr(imdisk, "_get_drive_type", lambda _path: imdisk.DRIVE_FIXED)

    assert imdisk.is_drive_available("x") is False


def test_build_ramdyn_create_cmd(tmp_path: Path) -> None:
    assert imdisk.build_ramdyn_create_cmd("x", 32, install_dir=tmp_path) == [
        str(tmp_path / "RamDyn.exe"),
        "X:",
        "33554432",
        "-1",
        "0",
        "14",
        "",
    ]


def test_build_ramdyn_launch_cmd(tmp_path: Path) -> None:
    assert imdisk.build_ramdyn_launch_cmd("x", 32, install_dir=tmp_path) == [
        "cmd",
        "/c",
        "start",
        "/b",
        "",
        str(tmp_path / "RamDyn.exe"),
        "X:",
        "33554432",
        "-1",
        "0",
        "14",
        "",
    ]


def test_build_format_cmd() -> None:
    assert imdisk.build_format_cmd("x") == [
        "powershell",
        "-NoProfile",
        "-Command",
        'format X: /fs:ntfs /v:"AEP" /q /y',
    ]


def test_build_remove_cmd() -> None:
    assert imdisk.build_remove_cmd("x") == [
        r"C:\Windows\System32\imdisk.exe",
        "-D",
        "-m",
        "X:",
    ]


def test_build_install_cmd_passes_sync_arg(tmp_path: Path) -> None:
    install_bat = tmp_path / "install.bat"
    assert imdisk.build_install_cmd(install_bat) == [
        "cmd",
        "/c",
        str(install_bat),
        "7",
    ]


def test_apply_ramdisk_path_sets_drive_root() -> None:
    settings = AppSettings()

    updated = imdisk.apply_ramdisk_path(settings, "x")

    assert updated.paths.ramdisk_path == "X:\\"
    assert settings.paths.ramdisk_path is None


def test_apply_ramdisk_path_clears_drive_root() -> None:
    settings = AppSettings()
    settings.paths.ramdisk_path = "X:\\"

    updated = imdisk.apply_ramdisk_path(settings, None)

    assert updated.paths.ramdisk_path is None
