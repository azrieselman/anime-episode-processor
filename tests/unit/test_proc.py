"""Tests for aep.util.proc helpers."""

from __future__ import annotations

import logging

import pytest


def test_run_capture_exec_log_summary(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import subprocess

    from aep.util.proc import run_capture

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Completed())

    with caplog.at_level(logging.INFO, logger="aep.util.proc"):
        run_capture(["tool.exe", "-x", "y"], exec_log_summary="tool.exe -x …")

    messages = [r.message for r in caplog.records if r.name == "aep.util.proc"]
    assert any(m == "exec: tool.exe -x …" for m in messages)
    assert not any("-x y" in m and "…" not in m for m in messages)


def test_run_capture_default_logs_full_cmd(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    import subprocess

    from aep.util.proc import run_capture

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: _Completed())

    with caplog.at_level(logging.INFO, logger="aep.util.proc"):
        run_capture(["tool.exe", "-x", "spaces here"])

    messages = [r.message for r in caplog.records if r.name == "aep.util.proc"]
    assert any("exec:" in m and "spaces here" in m for m in messages)
