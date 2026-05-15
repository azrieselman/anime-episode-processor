"""Pipeline context — the mutable state passed across stages.

Design choices:

* Single context object, not many positional args; stages explicitly declare which fields
  they read and write (in their docstring), so reasoning stays local even though state
  is shared.
* The context holds *paths and small structures*, never huge frame buffers — those live
  on disk or in pipes.
* Stage results are recorded into `stage_results` so later stages and validators can ask
  "did upscale produce N frames?" without re-reading manifests.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aep.errors import PipelineError
from aep.media.models import MediaInfo

log = logging.getLogger(__name__)


# Stages that produce frame sequences and benefit most from a ramdisk: their
# output is many small PNG/WebP files that the next stage immediately re-reads.
# Encode (08) and mux (09) read these once and write a single video file, so
# they do not benefit from ramdisk. Validation (10) only re-probes the final
# output. Probe/plan/sample stages don't write frames.
_RAMDISK_STAGES: frozenset[str] = frozenset({
    "04_decode_serve",
    "05_upscale",
    "06_interpolate",
    "07_postprocess",
})

# M6.5: stages that run per-batch when batched mode is active. Includes 08
# (encode) because each batch encodes to its own segment file before the
# concat-mux pass; the segment is copied out of the RAM-disk into
# <workdir>/batch_segments/segment_<NN>.mkv before cleanup_batch_dir runs.
_BATCHED_STAGES: frozenset[str] = frozenset({
    "04_decode_serve",
    "05_upscale",
    "06_interpolate",
    "07_postprocess",
    "08_encode",
})

@dataclass
class PipelineContext:
    job_id: str
    source_path: Path
    workdir: Path                 # <runtime>/jobs/<job_id>/
    output_path: Path             # final destination
    preset_id: str
    preset_data: dict[str, Any]
    plan: dict[str, Any] = field(default_factory=dict)         # frozen JobPlan dict
    media_info: MediaInfo | None = None
    scene_cuts: list[int] = field(default_factory=list)
    stage_results: dict[str, StageResult] = field(default_factory=dict)  # noqa: F821
    cancel_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    extras: dict[str, Any] = field(default_factory=dict)
    # Optional ramdisk root. When set and free space ≥ planner frame estimate,
    # frame-heavy stages (04-07) route their working frames here. Stage layout under the
    # ramdisk mirrors the workdir layout: <ramdisk>/<job_id>/<stage_name>/.
    ramdisk_path: Path | None = None
    # Estimated total bytes of frame data the pipeline will produce; used as the
    # free-space guard. Populated by stage 01 once geometry/frame count is known.
    # 0 means "unknown" — in that case we trust the user and use the ramdisk if
    # it's writable.
    ramdisk_estimate_bytes: int = 0
    # M6.5: per-batch encoded video segment paths, in batch-index order. The
    # mux stage concatenates these via ffmpeg's concat demuxer. Empty when
    # batching is disabled — the encode stage writes a single full-length
    # segment as before. Owned by the runner; stages should append in order.
    encoded_segments: list[Path] = field(default_factory=list)
    # M6.5: when set by the runner during batched mode, frame-producing stages
    # (and 08 encode) transparently route onto the per-batch RAM-disk dir via
    # `stage_dir()`. None means "single-pass mode" — pre-existing behavior.
    # Stages do NOT read this directly; they call stage_dir() as before and the
    # context handles routing. Set/cleared only by PipelineRunner around the
    # per-batch loop.
    _active_batch_idx: int | None = None
    # Cached frame manifests keyed by "<abs_dir>|<format|*>" to avoid repeated
    # full directory scans across stages.
    frame_manifests: dict[str, dict[str, int]] = field(default_factory=dict)
    # Lightweight accounting for how often we had to scan frame directories.
    frame_manifest_stats: dict[str, int] = field(default_factory=lambda: {"scans": 0, "cache_hits": 0})

    def stage_dir(self, stage_name: str) -> Path:
        """Return the working directory for a stage.

        Routing rules, in priority order:
          1. If batched mode is active (`_active_batch_idx is not None`) AND
             this stage runs per-batch (decode/upscale/interp/postprocess/
             encode), route to `<ramdisk>/<job_id>/batch_<NN>/<stage>/` via
             `batch_dir()`. Hard-fails when ramdisk_path is unset.
          2. Else if a ramdisk is configured AND this is a frame-producing
             stage (04-07) AND it has enough headroom, route to
             `<ramdisk>/<job_id>/<stage>/`.
          3. Otherwise, fall back to the regular `<workdir>/<stage>/`.
        """
        if (
            self._active_batch_idx is not None
            and stage_name in _BATCHED_STAGES
        ):
            return self.batch_dir(self._active_batch_idx, stage_name)
        if (
            self.ramdisk_path is not None
            and stage_name in _RAMDISK_STAGES
            and _ramdisk_usable(self.ramdisk_path, self.ramdisk_estimate_bytes)
        ):
            d = self.ramdisk_path / self.job_id / stage_name
        else:
            d = self.workdir / stage_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ----- M6.5 batched pipeline ----------------------------------------

    def batch_dir(self, batch_idx: int, stage_name: str) -> Path:
        """Return a RAM-disk-rooted working directory for one batch + stage.

        Layout:  <ramdisk>/<job_id>/batch_<NN>/<stage_name>/

        The batched pipeline (stages 04-08) writes ALL intermediates here —
        never to the regular workdir. Encoded segments land back under
        workdir/batch_segments/ once a batch finishes; the per-batch dir is
        deleted by `cleanup_batch_dir()` after that copy succeeds.

        Hard-fails when ramdisk_path is None. M6.5 design choice (per user
        confirmation): no fallback directory — if the user enabled batching
        without configuring a RAM-disk, that's a misconfiguration we surface
        early rather than silently degrade onto a slow SSD.
        """
        if self.ramdisk_path is None:
            raise PipelineError(
                "batched pipeline requires settings.paths.ramdisk_path; "
                "set it to a writable RAM-disk mount point or disable batching "
                "in the preset (batching.enabled=False).",
                context={"batch_idx": batch_idx, "stage": stage_name},
            )
        d = self.ramdisk_path / self.job_id / f"batch_{batch_idx:02d}" / stage_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _batch_dir_usage_bytes(self, batch_idx: int) -> int:
        """Sum file sizes under ``<ramdisk>/<job_id>/batch_<NN>/`` (0 if missing).

        Used by the RAM-disk gate when resuming a paused batch: space occupied by
        this job's in-progress intermediates still counts toward the batch budget
        — it must not be treated as missing from ``disk_usage().free``.
        """
        if self.ramdisk_path is None:
            return 0
        root = self.ramdisk_path / self.job_id / f"batch_{batch_idx:02d}"
        if not root.is_dir():
            return 0
        total = 0
        try:
            for p in root.rglob("*"):
                try:
                    if p.is_file():
                        total += p.stat().st_size
                except OSError:
                    continue
        except OSError as exc:
            log.warning(
                "batch %02d: could not measure RAM-disk footprint under %s: %s",
                batch_idx,
                root,
                exc,
            )
        return total

    def assert_ramdisk_has_room_for(self, batch) -> None:
        """Pre-batch gate: hard-fail if the RAM-disk lacks room for this batch.

        Required effective free bytes = ``batch.est_bytes`` (the planner already
        uses a conservative per-frame byte budget). Raises PipelineError when:
          * ramdisk_path is unset (caller should have caught this earlier);
          * disk_usage() fails (path missing / unmounted);
          * effective space is below the requirement.

        Effective space is ``disk_usage().free`` plus bytes already stored under
        this job's ``batch_<NN>/`` tree. After a pause mid-batch, frames and
        partial encode outputs remain on the RAM-disk; they reduce reported free
        space but are still available for the same batch, so they are counted
        here.

        We re-check on every batch (not just the first) because earlier
        batches' cleanup must succeed before the next one starts — a leaked
        batch dir would only be visible here.
        """
        from aep.pipeline.batches import BatchSpec  # local import: avoid cycle
        if not isinstance(batch, BatchSpec):  # pragma: no cover — type guard
            raise TypeError(f"expected BatchSpec, got {type(batch).__name__}")
        if self.ramdisk_path is None:
            raise PipelineError(
                "batched pipeline requires settings.paths.ramdisk_path",
                context={"batch_idx": batch.index},
            )
        # If the planner couldn't compute a byte estimate (unknown geometry),
        # we'd rather skip the gate than block the job; the per-stage write
        # will ENOSPC if it really overruns.
        if batch.est_bytes <= 0:
            log.info(
                "batch %02d: skipping RAM-disk gate (no byte estimate)",
                batch.index,
            )
            return
        try:
            usage = shutil.disk_usage(self.ramdisk_path)
        except OSError as exc:
            raise PipelineError(
                f"RAM-disk path is unreadable: {self.ramdisk_path}",
                context={"batch_idx": batch.index, "reason": str(exc)},
            ) from exc
        required = int(batch.est_bytes)
        existing_batch_bytes = self._batch_dir_usage_bytes(batch.index)
        effective_free = usage.free + existing_batch_bytes
        if effective_free < required:
            need_mb = required // (1024 * 1024)
            have_mb = effective_free // (1024 * 1024)
            detail = (
                f" ({usage.free // (1024 * 1024)} MiB reported free + "
                f"{existing_batch_bytes // (1024 * 1024)} MiB in this job's batch folder)"
                if existing_batch_bytes > 0
                else ""
            )
            raise PipelineError(
                f"RAM-disk insufficient for batch {batch.index:02d}: "
                f"need {need_mb} MiB, have {have_mb} MiB effective free{detail}. "
                f"Increase RAM-disk size or reduce preset's batching.chunk_seconds.",
                context={
                    "batch_idx": batch.index,
                    "required_bytes": required,
                    "free_bytes": usage.free,
                    "existing_batch_bytes": existing_batch_bytes,
                    "effective_free_bytes": effective_free,
                    "ramdisk_path": str(self.ramdisk_path),
                },
            )
        log.debug(
            "batch %02d RAM-disk gate ok: need %d MiB, have %d MiB effective free "
            "(%d MiB reported free, %d MiB in batch folder)",
            batch.index,
            required // (1024 * 1024),
            effective_free // (1024 * 1024),
            usage.free // (1024 * 1024),
            existing_batch_bytes // (1024 * 1024),
        )

    def cleanup_batch_dir(self, batch_idx: int) -> None:
        """Recursively delete a batch's RAM-disk dir, if present.

        Called after the batch's encoded segment is durably copied out of the
        RAM-disk. Best-effort: a delete failure is logged but does NOT raise,
        because the worst case is that the next batch's gate catches the
        leak and surfaces a clear error.
        """
        if self.ramdisk_path is None:
            return
        d = self.ramdisk_path / self.job_id / f"batch_{batch_idx:02d}"
        if not d.exists():
            return
        try:
            shutil.rmtree(d)
            log.debug("batch %02d: cleaned up %s", batch_idx, d)
        except OSError as exc:
            log.warning(
                "batch %02d: failed to clean up %s: %s",
                batch_idx, d, exc,
            )

    @property
    def is_cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancel(self) -> None:
        from aep.errors import CancelledError
        if self.cancel_event.is_set():
            raise CancelledError("job cancelled")

    def get_frame_manifest(self, dir_path: Path, *, format: str | None = None) -> dict[str, int]:
        """Return cached frame stats for a directory.

        Keys:
          - count: matching frame file count
          - bytes: total bytes of matching files
        """
        if not dir_path.is_dir():
            return {"count": 0, "bytes": 0}
        key = f"{dir_path.resolve()}|{format or '*'}"
        mtime_ns = dir_path.stat().st_mtime_ns
        cached = self.frame_manifests.get(key)
        if cached and cached.get("mtime_ns") == mtime_ns:
            self.frame_manifest_stats["cache_hits"] = int(self.frame_manifest_stats.get("cache_hits", 0)) + 1
            return {"count": int(cached.get("count", 0)), "bytes": int(cached.get("bytes", 0))}
        suffixes = {f".{format.lower()}"} if format else {".png", ".webp"}
        count = 0
        total_bytes = 0
        for p in dir_path.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in suffixes:
                continue
            count += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                continue
        self.frame_manifest_stats["scans"] = int(self.frame_manifest_stats.get("scans", 0)) + 1
        self.frame_manifests[key] = {
            "mtime_ns": mtime_ns,
            "count": count,
            "bytes": total_bytes,
        }
        return {"count": count, "bytes": total_bytes}


def _ramdisk_usable(ramdisk_path: Path, estimate_bytes: int) -> bool:
    """Decide whether to route a stage onto the ramdisk.

    Returns False (and logs a single warning) when the path is missing,
    not writable, or doesn't have enough free space. Callers fall back to
    workdir on False so the pipeline continues without ramdisk benefits.
    """
    try:
        ramdisk_path.mkdir(parents=True, exist_ok=True)
        if not ramdisk_path.is_dir():
            log.warning("ramdisk_path %s is not a directory; ignoring", ramdisk_path)
            return False
        usage = shutil.disk_usage(ramdisk_path)
    except OSError as exc:
        log.warning("ramdisk_path %s unusable (%s); falling back to workdir", ramdisk_path, exc)
        return False

    # If we have an estimate, require that much free space; otherwise trust the path.
    if estimate_bytes > 0:
        required = int(estimate_bytes)
        if usage.free < required:
            log.warning(
                "ramdisk_path %s has %d bytes free, needs %d (planner estimate); "
                "falling back to workdir",
                ramdisk_path, usage.free, required,
            )
            return False
    return True
