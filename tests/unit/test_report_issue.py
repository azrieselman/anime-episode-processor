"""Tests for ``Help → Report Issue`` body builder.

The action constructs a markdown body by introspecting the runtime: the
contract we lock in here is "version + Python + OS + GPU are present, and
the format is markdown the user can review before submitting." We don't
assert the URL itself because that's tested implicitly by the GUI smoke
imports — what matters for bug-report quality is the body content.
"""

from __future__ import annotations

from aep.gui.app_window import MainWindow
from aep.version import __version__


def test_issue_body_contains_required_environment_keys() -> None:
    body = MainWindow._build_issue_body()
    assert "## Environment" in body
    assert f"AEP version: `{__version__}`" in body
    assert "OS:" in body
    assert "Python:" in body
    assert "GPU:" in body


def test_issue_body_contains_repro_template() -> None:
    """We pre-fill the markdown skeleton so users don't ship blank issues."""
    body = MainWindow._build_issue_body()
    assert "## What happened" in body
    assert "## Steps to reproduce" in body
    assert "## Logs" in body


def test_issue_body_does_not_leak_user_paths() -> None:
    """Body must not accidentally include the running user's home directory."""
    import os
    body = MainWindow._build_issue_body()
    home = os.path.expanduser("~")
    if home and home != "~":
        assert home not in body, (
            "issue body unexpectedly contains the user's home dir; "
            "leaking absolute paths is a privacy regression"
        )
