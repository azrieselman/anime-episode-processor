"""Tests for per-job preset overrides.

Three things to lock down:
  1. The broker's `_deep_merge` is genuinely recursive and doesn't clobber
     unrelated subtrees on a partial override.
  2. `JobService.set_preset_overrides` persists on QUEUED jobs and refuses
     others (so no silent drops once the broker has loaded the preset).
  3. A round-trip through the DB preserves the dict.
"""

from __future__ import annotations

import pytest

from aep.jobs.broker import _deep_merge
from aep.jobs.models import Job, JobState
from aep.jobs.queue import get_job, insert_job
from aep.persist.db import init_db


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("AEP_RUNTIME_DIR", str(tmp_path))
    init_db()
    yield


def test_deep_merge_replaces_only_overridden_leaves():
    base = {
        "upscaler": {"scale": 2, "denoise": 3, "model": "models-pro"},
        "encoder": {"name": "hevc_nvenc", "nvenc_cq": 20},
    }
    overrides = {"upscaler": {"scale": 4}, "encoder": {"name": "av1_nvenc"}}
    merged = _deep_merge(base, overrides)
    # Override applied
    assert merged["upscaler"]["scale"] == 4
    assert merged["encoder"]["name"] == "av1_nvenc"
    # Untouched siblings preserved
    assert merged["upscaler"]["denoise"] == 3
    assert merged["upscaler"]["model"] == "models-pro"
    assert merged["encoder"]["nvenc_cq"] == 20
    # Original is not mutated (defensive copy)
    assert base["upscaler"]["scale"] == 2


def test_deep_merge_replaces_lists_wholesale():
    base = {"encoder": {"extra_args": ["-x", "1", "-y", "2"]}}
    overrides = {"encoder": {"extra_args": ["-z", "9"]}}
    merged = _deep_merge(base, overrides)
    assert merged["encoder"]["extra_args"] == ["-z", "9"]


def test_set_preset_overrides_persists_for_queued_job():
    from aep.app.services import AppServices
    services = AppServices()
    job = Job(source_path="/tmp/in.mkv", preset_id="anime_balanced")
    insert_job(job)

    overrides = {"upscaler": {"scale": 4}, "encoder": {"name": "libx265"}}
    result = services.jobs.set_preset_overrides(job.id, overrides)
    assert result is not None
    assert result.preset_overrides == overrides

    reloaded = get_job(job.id)
    assert reloaded is not None
    assert reloaded.preset_overrides == overrides


def test_set_preset_overrides_refuses_running_job():
    from aep.app.services import AppServices
    services = AppServices()
    job = Job(
        source_path="/tmp/in.mkv",
        preset_id="anime_balanced",
        state=JobState.RUNNING,
    )
    insert_job(job)
    result = services.jobs.set_preset_overrides(job.id, {"upscaler": {"scale": 4}})
    assert result is not None
    # Returned job reflects stored state; overrides remain unset.
    reloaded = get_job(job.id)
    assert reloaded is not None
    assert reloaded.preset_overrides is None


def test_set_preset_overrides_empty_dict_clears_to_none():
    from aep.app.services import AppServices
    services = AppServices()
    job = Job(source_path="/tmp/in.mkv", preset_id="anime_balanced",
              preset_overrides={"upscaler": {"scale": 4}})
    insert_job(job)
    services.jobs.set_preset_overrides(job.id, {})
    reloaded = get_job(job.id)
    assert reloaded is not None
    assert reloaded.preset_overrides is None
