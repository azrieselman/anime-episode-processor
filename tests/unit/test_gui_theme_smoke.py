"""Smoke tests for bundled GUI assets (no Qt GUI instantiation)."""


def test_bundled_app_ico_readable() -> None:
    """Windows shell expects a valid `.ico`; ensure package data is present."""
    from importlib import resources as ir

    root = ir.files("aep.gui.resources")
    ico = root.joinpath("app.ico")
    assert ico.is_file()
    data = ico.read_bytes()
    # ICONDIR: reserved=0, type=1 (icon), count>=1
    assert len(data) >= 22
    assert data[2:4] == b"\x01\x00"


def test_bundled_sidebar_nav_icons_present() -> None:
    """Main-window sidebar icons ship under ``gui/resources/sidebar``."""
    from importlib import resources as ir

    side = ir.files("aep.gui.resources").joinpath("sidebar")
    stems = (
        "queue.ico",
        "job config.ico",
        "stream inspector.ico",
        "preset designer.ico",
        "logs.ico",
        "settings.ico",
    )
    for filename in stems:
        assert side.joinpath(filename).is_file(), filename
