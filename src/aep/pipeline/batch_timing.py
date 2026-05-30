"""Batch PTS windows → exact frame counts, overlap, and trim helpers.

Batched decode uses ffmpeg ``-t`` with planner ``end_pts`` treated as exclusive.
ffmpeg often emits one extra frame per chunk; we trim to the planned count.
Later batches decode one overlap source frame before the content window so RIFE
has temporal context at joins; those outputs are dropped after interpolation.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from aep.pipeline.context import PipelineContext
from aep.util.fps import parse_rational, total_frames

log = logging.getLogger(__name__)

# One source frame of context before each batch (except the first).
RIFE_BATCH_OVERLAP_SOURCE_FRAMES: int = 1


@dataclass(frozen=True)
class BatchFramePlan:
    """Frame accounting for the active batch's decode → RIFE → encode chain."""

    start_pts: float
    end_pts: float
    source_fps: Fraction
    overlap_source_frames: int
    content_frame_offset: int  # 0-based global index of first *content* frame
    expected_content_frames: int
    rife_input_base: int  # global index of RIFE input frame 1
    rife_input_count: int
    rife_output_skip: int  # drop this many 1-based RIFE outputs after inference
    expected_output_frames: int  # after skip (content × multiplier)

    @property
    def decode_start_pts(self) -> float:
        if self.overlap_source_frames <= 0:
            return self.start_pts
        return max(
            0.0,
            self.start_pts - float(self.overlap_source_frames) / float(self.source_fps),
        )

    @property
    def expected_decode_frames(self) -> int:
        return self.overlap_source_frames + self.expected_content_frames


def output_fps_fraction_from_plan(ctx: PipelineContext) -> Fraction | None:
    raw = (ctx.plan or {}).get("output_fps") or ""
    if not raw or "/" not in str(raw):
        return None
    return parse_rational(str(raw))


def source_fps_from_context(ctx: PipelineContext) -> Fraction | None:
    media = ctx.media_info
    primary = media.primary_video if media is not None else None
    if primary is None:
        return None
    return parse_rational(primary.avg_frame_rate) or parse_rational(primary.r_frame_rate)


def source_frame_count_from_context(ctx: PipelineContext) -> int | None:
    """Best-effort total source frame count (nb_frames probe, else duration × fps)."""
    media = ctx.media_info
    if media is None:
        return None
    primary = media.primary_video
    if primary is not None and primary.nb_frames is not None and primary.nb_frames > 0:
        return int(primary.nb_frames)
    fps = source_fps_from_context(ctx)
    dur = media.fmt.duration_s if media.fmt is not None else None
    if fps is not None and fps > 0 and dur is not None and dur > 0:
        return total_frames(fps, float(dur))
    return None


def content_frame_offset_for_pts(start_pts: float, source_fps: Fraction) -> int:
    """0-based source frame index for the first frame at or after ``start_pts``."""
    if start_pts <= 0.0:
        return 0
    return total_frames(source_fps, start_pts)


def expected_content_frames(
    *,
    start_pts: float,
    end_pts: float,
    source_fps: Fraction,
    source_total_frames: int | None = None,
) -> int:
    """Frames with PTS in ``[start_pts, end_pts)`` (``end_pts`` exclusive).

    Uses cumulative frame indices (not ``duration × fps`` alone) so batch windows
    line up with ``content_frame_offset``. When ``source_total_frames`` is known,
    caps the result so a batch cannot claim frames past the probed stream end.
    """
    if end_pts <= start_pts:
        return 0
    start_idx = content_frame_offset_for_pts(start_pts, source_fps)
    end_idx = total_frames(source_fps, end_pts)
    count = max(0, end_idx - start_idx)
    if source_total_frames is not None:
        count = min(count, max(0, source_total_frames - start_idx))
    return count


def count_numeric_frames_in_dir(dir_path: Path, *, frame_format: str) -> int:
    """Count ``NNNN.ext`` files (same rule as :func:`assert_frame_dir_count`)."""
    ext = f".{frame_format.lower()}"
    return sum(
        1
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() == ext and p.stem.isdigit()
    )


def decode_time_pad_s(source_fps: Fraction, *, frame_periods: int = 2) -> float:
    """Extra seconds on ffmpeg ``-t`` so batched decodes do not stop early vs CFR math."""
    if source_fps <= 0:
        return 0.0
    return float(frame_periods) / float(source_fps)


def batch_frame_plan_for_actual_decode(
    plan: BatchFramePlan,
    actual_decode_frames: int,
) -> BatchFramePlan:
    """Rebuild batch accounting from the frames ffmpeg actually wrote."""
    overlap = plan.overlap_source_frames
    actual_content = max(0, actual_decode_frames - overlap)
    mult = 1
    if plan.expected_content_frames > 0:
        mult = plan.expected_output_frames // plan.expected_content_frames
    return BatchFramePlan(
        start_pts=plan.start_pts,
        end_pts=plan.end_pts,
        source_fps=plan.source_fps,
        overlap_source_frames=overlap,
        content_frame_offset=plan.content_frame_offset,
        expected_content_frames=actual_content,
        rife_input_base=plan.rife_input_base,
        rife_input_count=overlap + actual_content,
        rife_output_skip=overlap * mult,
        expected_output_frames=actual_content * mult,
    )


def reconcile_batch_decode_outputs(
    ctx: PipelineContext,
    *,
    out_dir: Path,
    frame_format: str,
    plan: BatchFramePlan,
) -> tuple[int, int, BatchFramePlan, bool]:
    """Trim excess frames or lower the batch plan to match a short decode.

    Returns ``(final_frame_count, trim_removed, plan, reconciled_shortfall)``.
    """
    from aep.errors import StageError

    actual = count_numeric_frames_in_dir(out_dir, frame_format=frame_format)
    expected = plan.expected_decode_frames
    trim_removed = 0
    updated = plan
    reconciled_shortfall = False

    if actual > expected:
        trim_removed = trim_frames_dir(
            out_dir,
            frame_format=frame_format,
            keep_count=expected,
        )
        actual = expected
    elif actual < expected:
        log.warning(
            "batch decode shortfall: expected %s %s frames, got %s "
            "(batch idx=%s window [%.3f, %.3f)s); reconciling plan to decode output",
            expected,
            frame_format,
            actual,
            ctx._active_batch_idx,
            plan.start_pts,
            plan.end_pts,
        )
        updated = batch_frame_plan_for_actual_decode(plan, actual)
        merge_batch_frame_plan_into_decode(ctx, updated)
        reconciled_shortfall = True

    cache_key = f"{out_dir.resolve()}|{frame_format}"
    ctx.frame_manifests.pop(cache_key, None)

    if actual != updated.expected_decode_frames:
        raise StageError(
            f"decode_serve: expected {updated.expected_decode_frames} {frame_format} "
            f"frames after reconcile, found {actual}",
            context={
                "dir": str(out_dir),
                "pts_window": [plan.start_pts, plan.end_pts],
                "batch_index": ctx._active_batch_idx,
            },
        )
    return actual, trim_removed, updated, reconciled_shortfall


def overlap_source_frames_for_batch(ctx: PipelineContext) -> int:
    if ctx._active_batch_idx is None or ctx._active_batch_idx <= 0:
        return 0
    return RIFE_BATCH_OVERLAP_SOURCE_FRAMES


def _pts_window_matches_stored(
    decode: dict,
    start_pts: float,
    end_pts: float,
) -> bool:
    stored = decode.get("batch_pts_window")
    if not isinstance(stored, (list, tuple)) or len(stored) != 2:
        return False
    try:
        s0, s1 = float(stored[0]), float(stored[1])
    except (TypeError, ValueError):
        return False
    return round(s0, 6) == round(start_pts, 6) and round(s1, 6) == round(end_pts, 6)


def batch_frame_plan_from_stored_decode(
    ctx: PipelineContext,
    *,
    start_pts: float,
    end_pts: float,
    source_fps: Fraction,
) -> BatchFramePlan | None:
    """Reuse per-batch counts written by decode (including post-reconcile adjustments)."""
    decode = ctx.plan.get("decode") or {}
    if not _pts_window_matches_stored(decode, start_pts, end_pts):
        return None
    keys = (
        "batch_content_frame_offset",
        "batch_overlap_source_frames",
        "batch_expected_content_frames",
        "batch_rife_input_base",
        "batch_rife_input_count",
        "batch_rife_output_skip",
        "batch_expected_output_frames",
    )
    if not all(k in decode for k in keys):
        return None
    try:
        content_offset = int(decode["batch_content_frame_offset"])
        overlap = int(decode["batch_overlap_source_frames"])
        content_frames = int(decode["batch_expected_content_frames"])
        rife_base = int(decode["batch_rife_input_base"])
        rife_in = int(decode["batch_rife_input_count"])
        rife_skip = int(decode["batch_rife_output_skip"])
        expected_out = int(decode["batch_expected_output_frames"])
    except (TypeError, ValueError):
        return None
    if rife_in <= 0 or content_frames < 0:
        return None
    return BatchFramePlan(
        start_pts=start_pts,
        end_pts=end_pts,
        source_fps=source_fps,
        overlap_source_frames=overlap,
        content_frame_offset=content_offset,
        expected_content_frames=content_frames,
        rife_input_base=rife_base,
        rife_input_count=rife_in,
        rife_output_skip=rife_skip,
        expected_output_frames=expected_out,
    )


def resolve_batch_frame_plan(ctx: PipelineContext) -> BatchFramePlan | None:
    """Build a plan when ``decode.pts_window`` is set; else ``None`` (full source)."""
    decode = ctx.plan.get("decode") or {}
    pts_window = decode.get("pts_window")
    if not pts_window:
        return None
    try:
        start_pts = float(pts_window[0])
        end_pts = float(pts_window[1])
    except (TypeError, ValueError, IndexError):
        return None
    source_fps = source_fps_from_context(ctx)
    if source_fps is None or source_fps <= 0:
        return None

    stored = batch_frame_plan_from_stored_decode(
        ctx,
        start_pts=start_pts,
        end_pts=end_pts,
        source_fps=source_fps,
    )
    if stored is not None:
        return stored

    overlap = overlap_source_frames_for_batch(ctx)
    content_offset = content_frame_offset_for_pts(start_pts, source_fps)
    content_frames = expected_content_frames(
        start_pts=start_pts,
        end_pts=end_pts,
        source_fps=source_fps,
        source_total_frames=source_frame_count_from_context(ctx),
    )
    rife_base = content_offset - overlap
    rife_in = overlap + content_frames

    mult = 1
    interp = ctx.plan.get("interpolate") or {}
    if isinstance(interp.get("multiplier"), int):
        mult = max(1, int(interp["multiplier"]))
    elif bool(interp.get("active")):
        mult = max(1, int(interp.get("multiplier") or 1))

    rife_skip = overlap * mult
    return BatchFramePlan(
        start_pts=start_pts,
        end_pts=end_pts,
        source_fps=source_fps,
        overlap_source_frames=overlap,
        content_frame_offset=content_offset,
        expected_content_frames=content_frames,
        rife_input_base=rife_base,
        rife_input_count=rife_in,
        rife_output_skip=rife_skip,
        expected_output_frames=content_frames * mult,
    )


def merge_batch_frame_plan_into_decode(ctx: PipelineContext, plan: BatchFramePlan) -> None:
    decode = ctx.plan.setdefault("decode", {})
    decode["batch_pts_window"] = [plan.start_pts, plan.end_pts]
    decode["batch_content_frame_offset"] = plan.content_frame_offset
    decode["batch_overlap_source_frames"] = plan.overlap_source_frames
    decode["batch_expected_content_frames"] = plan.expected_content_frames
    decode["batch_expected_decode_frames"] = plan.expected_decode_frames
    decode["batch_rife_input_base"] = plan.rife_input_base
    decode["batch_rife_input_count"] = plan.rife_input_count
    decode["batch_rife_output_skip"] = plan.rife_output_skip
    decode["batch_expected_output_frames"] = plan.expected_output_frames


def trim_frames_dir(
    dir_path: Path,
    *,
    frame_format: str,
    keep_count: int,
    digits: int = 8,
) -> int:
    """Delete numbered frames above ``keep_count``. Returns number removed."""
    if keep_count < 0:
        raise ValueError(f"keep_count must be >= 0, got {keep_count!r}")
    ext = frame_format.lower()
    removed = 0
    for p in sorted(dir_path.iterdir()):
        if not p.is_file() or p.suffix.lower() != f".{ext}":
            continue
        stem = p.stem
        if not stem.isdigit():
            continue
        idx = int(stem)
        if idx > keep_count:
            p.unlink(missing_ok=True)
            removed += 1
    return removed


def drop_rife_output_prefix(
    out_dir: Path,
    *,
    frame_format: str,
    drop_count: int,
    digits: int = 8,
) -> None:
    """Remove the first ``drop_count`` RIFE outputs and renumber the rest from 1."""
    if drop_count <= 0:
        return
    ext = frame_format.lower()

    def frame_path(n: int) -> Path:
        return out_dir / f"{n:0{digits}d}.{ext}"

    manifest = sorted(
        int(p.stem)
        for p in out_dir.iterdir()
        if p.is_file() and p.suffix.lower() == f".{ext}" and p.stem.isdigit()
    )
    if not manifest:
        return
    total = len(manifest)
    if drop_count >= total:
        raise ValueError(
            f"drop_rife_output_prefix: drop_count={drop_count} >= frame count {total}",
        )
    for i in range(1, drop_count + 1):
        frame_path(i).unlink(missing_ok=True)
    # Renumber remaining frames to close the gap.
    tmp_pairs: list[tuple[Path, int]] = []
    for old_idx in range(drop_count + 1, total + 1):
        src = frame_path(old_idx)
        if not src.is_file():
            raise FileNotFoundError(f"missing RIFE output frame {old_idx}: {src}")
        tmp = out_dir / f".rife_drop_tmp_{old_idx:0{digits}d}.{ext}"
        shutil.move(str(src), str(tmp))
        tmp_pairs.append((tmp, old_idx - drop_count))
    for tmp, new_idx in tmp_pairs:
        shutil.move(str(tmp), str(frame_path(new_idx)))


def expected_segment_duration_s(
    plan: BatchFramePlan,
    *,
    output_fps: Fraction,
) -> float:
    """Wall-clock seconds for an encoded batch segment at ``output_fps``."""
    if output_fps <= 0:
        return 0.0
    return float(plan.expected_output_frames) / float(output_fps)


def validate_encoded_segment_duration(
    segment: Path,
    *,
    expected_duration_s: float,
    tolerance_s: float = 0.25,
    ffprobe: object | None = None,
) -> None:
    """Fail fast when a per-batch encoded segment drifts from the planned timeline."""
    from aep.adapters.ffprobe import FFProbeAdapter
    from aep.errors import PipelineError
    from aep.media.ffprobe import FfprobeAnalyzer

    if expected_duration_s <= 0:
        return
    adapter = ffprobe if ffprobe is not None else FFProbeAdapter()
    info = FfprobeAnalyzer(adapter).analyze(segment)
    got = info.fmt.duration_s
    if got is None:
        raise PipelineError(
            "batch segment duration unknown after encode",
            context={"segment": str(segment)},
        )
    if abs(got - expected_duration_s) > tolerance_s:
        raise PipelineError(
            "batch segment duration mismatch",
            context={
                "segment": str(segment),
                "expected_s": f"{expected_duration_s:.3f}",
                "got_s": f"{got:.3f}",
                "tolerance_s": tolerance_s,
            },
        )


def assert_frame_dir_count(
    dir_path: Path,
    *,
    frame_format: str,
    expected: int,
    label: str,
) -> None:
    ext = f".{frame_format.lower()}"
    count = sum(
        1
        for p in dir_path.iterdir()
        if p.is_file() and p.suffix.lower() == ext and p.stem.isdigit()
    )
    if count != expected:
        from aep.errors import StageError

        raise StageError(
            f"{label}: expected {expected} {frame_format} frames, found {count}",
            context={"dir": str(dir_path)},
        )
