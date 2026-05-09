"""End-to-end pipeline smoke test.

Synthesizes a tiny solid-color clip with ``ffmpeg`` and runs it through the
full broker → pipeline → mux → validate path with a minimal source-mode
preset (no upscaler, no interpolation, no postprocess, libx264 software
encoder, MP4 container so we don't need MKVToolNix). The goal is to guard
against silent stage-wiring regressions in beta-2 and beyond — each shipped
release should be able to take any well-formed input file and produce a
COMPLETED job, even on machines without GPUs or NCNN tooling.

The test is automatically skipped when ``ffmpeg`` / ``ffprobe`` aren't on
PATH (or in the bundled tools dir), which is the case for many CI runners.
The matching CI workflow installs them explicitly so this still gets
exercised before each release tag.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from aep.jobs.broker import JobBroker
from aep.jobs.models import JobState
from aep.jobs.queue import get_job
from aep.persist.db import init_db
from aep.persist.presets import (
    BatchingCfg,
    DecodeCfg,
    EncoderCfg,
    InterpolationCfg,
    PostprocessCfg,
    Preset,
    PresetMeta,
    StreamMappingCfg,
    TargetResolution,
    UpscalerCfg,
    save_user_preset,
)


def _which_or_skip(*names: str) -> None:
    missing = [n for n in names if shutil.which(n) is None]
    if missing:
        pytest.skip(f"required tool(s) not on PATH: {', '.join(missing)}")


def _make_test_clip(out_path: Path) -> None:
    """Create a 1s/24fps/64x64 testsrc MP4 with libx264.

    Using the smallest viable set of ffmpeg options so the synthesis itself
    can't fail on minimal CI builds. ``yuv420p`` keeps the file decodable by
    every consumer; ``-preset ultrafast`` keeps the smoke test fast.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=1:size=64x64:rate=24",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, timeout=60)
    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no output at {out_path}")


def _build_smoke_preset(preset_id: str = "smoke_e2e") -> Preset:
    """Minimal preset: source-mode encode through libx264 → mp4.

    All frame stages disabled (upscale / interpolation / postprocess) and
    batching off so the pipeline takes the source-mode short-circuit through
    s04 (skip), s05/06/07 (skip), s08 (re-encode source video stream),
    s09 (ffmpeg mux — required because mp4 always uses ffmpeg), and
    s10 (validate via ffprobe).
    """
    return Preset(
        meta=PresetMeta(
            id=preset_id,
            name="Smoke E2E",
            description="Internal smoke-test preset; not for end users.",
            builtin=False,
        ),
        container="mp4",
        target_resolution=TargetResolution(mode="scale_only"),
        upscaler=UpscalerCfg(enabled=False),
        interpolation=InterpolationCfg(enabled=False, target_fps=None),
        postprocess=PostprocessCfg(enabled=False),
        batching=BatchingCfg(enabled=False),
        decode=DecodeCfg(hwaccel="off"),
        encoder=EncoderCfg(
            name="libx264",
            x_crf=28,
            x_preset="ultrafast",
            x_tune=None,
        ),
        streams=StreamMappingCfg(
            copy_audio=False,
            copy_subtitles=False,
            copy_chapters=False,
            copy_attachments=False,
            burn_in_subtitles=False,
        ),
    )


def _wait_for_terminal(job_id: str, *, timeout_s: float) -> JobState:
    """Poll the DB until the job hits a terminal state or the timeout fires.

    We poll the DB rather than subscribing to broker events because the
    broker writes the final state transition before the worker thread
    publishes its sentinel — polling avoids an inherent race where the
    subscriber sees ``RUNNING`` last on a fast machine.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = get_job(job_id)
        if job is not None and job.is_terminal():
            return job.state
        time.sleep(0.1)
    final = get_job(job_id)
    raise AssertionError(
        f"job {job_id} did not reach a terminal state within {timeout_s}s; "
        f"last seen state={final.state.value if final else 'missing'} "
        f"current_stage={final.current_stage if final else 'n/a'} "
        f"error={final.error if final else 'n/a'}",
    )


@pytest.fixture(autouse=True)
def _db(tmp_runtime: Path) -> None:
    init_db()


def test_pipeline_smoke_e2e_completes(tmp_runtime: Path, tmp_path: Path) -> None:
    """A trivial source-mode job runs to COMPLETED end-to-end.

    Skipped when ffmpeg/ffprobe aren't available; the CI release workflow
    installs them so this guard runs at least once per release candidate.
    """
    _which_or_skip("ffmpeg", "ffprobe")

    source = tmp_path / "smoke_input.mp4"
    output = tmp_path / "smoke_output.mp4"
    _make_test_clip(source)

    preset = _build_smoke_preset()
    save_user_preset(preset)

    broker = JobBroker()
    broker.start()
    try:
        broker.start_queue()
        job = broker.enqueue(source, preset.meta.id, output_path=output)

        # Generous timeout: software libx264 ultrafast on a 1-second 64x64
        # clip is sub-second on any modern machine, but CI runners can be
        # slow on cold start (Python imports, ffmpeg first invocation).
        final_state = _wait_for_terminal(job.id, timeout_s=120.0)
    finally:
        broker.stop(timeout=10.0)
        # Background worker threads can briefly outlive .stop(); give them a
        # moment to drain before pytest tears the runtime dir out from under
        # them. Not strictly required for correctness, but it keeps the test
        # log clean of "logging on closed handler" warnings on Windows.
        for _ in range(20):
            if threading.active_count() <= 2:
                break
            time.sleep(0.05)

    assert final_state == JobState.COMPLETED, (
        f"expected COMPLETED, got {final_state.value}"
    )
    assert output.is_file(), f"output file missing: {output}"
    assert output.stat().st_size > 0, f"output file is empty: {output}"
