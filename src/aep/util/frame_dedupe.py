"""Perceptual duplicate-frame helpers (ffmpeg select scene scores + compact/expand).

Used by decode (compact) and upscale/interpolate (expand) so NCNN stages see
contiguous numbered frames while the final encode timeline matches the full
decoded frame count.

Scene scores come from the **select** filter's ``scene`` value (see
``FFmpegAdapter.build_scene_score_scan``): ``select`` stores ``lavfi.scene_score``
on each frame's metadata; we read it via **metadata=print** (``showinfo`` does not
dump that metadata). Not the optional ``scenedetect`` filter, so minimal FFmpeg
builds still work.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from aep.pipeline.context import PipelineContext
from aep.util.fps import parse_rational

log = logging.getLogger(__name__)

DEDUPE_MAP_NAME = "dedupe_map.json"

# Written next to ``frames/`` during scene scan; basename-only for ``-vf file=``.
SCENE_SCORE_META_BASENAME = "aep_frame_dedupe_scene_scores.txt"

# Prefer lavfi metadata from scenedetect; generic `scene:` is last — it can appear on
# unrelated lines and must not drive one-score-per-line ordering (that misaligns vs `n:`).
_SCENE_SCORE_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"lavfi\.scene_score[=:]\s*([0-9eE+.-]+)", re.IGNORECASE),
    re.compile(r"scene_score[=:]\s*([0-9eE+.-]+)", re.IGNORECASE),
    re.compile(r"\bscene:([0-9eE+.-]+)\b", re.IGNORECASE),
)

_FRAME_N_RE = re.compile(r"\bn:\s*(\d+)\b", re.IGNORECASE)

# metadata=print sidecar: "frame:N ..." then "key=value" lines until the next frame.
_METADATA_FRAME_LINE_RE = re.compile(r"^frame:\s*(\d+)", re.IGNORECASE)

# Skip when score < threshold; scores are often ~1e-8..1e-2. Below this, float noise /
# bogus parses would mark every frame a duplicate — clamp at use sites.
DEDUPE_THRESHOLD_EPS = 1e-12


def decode_batch_frame_offset(ctx: PipelineContext) -> int:
    """Source-frame index of this batch's first decoded frame (0 if unbatched)."""
    decode = ctx.plan.get("decode", {}) or {}
    pts_window = decode.get("pts_window")
    if not pts_window:
        return 0
    try:
        start_pts = float(pts_window[0])
    except (TypeError, ValueError, IndexError):
        log.warning("frame_dedupe: malformed pts_window %r; offset=0", pts_window)
        return 0
    if start_pts <= 0.0:
        return 0
    media = ctx.media_info
    primary = media.primary_video if media is not None else None
    if primary is None:
        return 0
    fps = parse_rational(primary.avg_frame_rate) or parse_rational(primary.r_frame_rate)
    if fps is None or fps <= 0:
        log.warning("frame_dedupe: source fps unknown; offset=0")
        return 0
    return int(round(start_pts * float(fps)))


def parse_metadata_print_scene_scores(text: str) -> dict[int, float]:
    """Map frame index ``N`` from ``metadata=print`` output -> ``lavfi.scene_score``.

    ``N`` matches the 0-based sequence of decoded frames in the scanned window
    (same ordering as ``select`` -> ``metadata`` in ``build_scene_score_scan``).
    """
    by_n: dict[int, float] = {}
    current_n: int | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        fm = _METADATA_FRAME_LINE_RE.match(line)
        if fm:
            try:
                current_n = int(fm.group(1))
            except ValueError:
                current_n = None
            continue
        if current_n is None:
            continue
        val: float | None = None
        for rx in _SCENE_SCORE_RES:
            m = rx.search(line)
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    val = None
                break
        if val is not None:
            by_n[current_n] = val
    return by_n


def load_scene_score_scan_results(*, meta_path: Path, stderr: str) -> dict[int, float]:
    """Prefer ``metadata=print`` sidecar next to stderr ``showinfo`` fallback (legacy)."""
    if meta_path.is_file():
        try:
            parsed = parse_metadata_print_scene_scores(
                meta_path.read_text(encoding="utf-8", errors="replace"),
            )
        except OSError as exc:
            log.warning("frame_dedupe: could not read scene meta file %s: %s", meta_path, exc)
            parsed = {}
        if parsed:
            return parsed
    return parse_showinfo_scene_scores(stderr)


def parse_showinfo_scene_scores(stderr: str) -> dict[int, float]:
    """Map showinfo frame index ``n`` (0-based) -> scene score for that output frame.

    ffmpeg may emit multiple ``showinfo`` lines per frame or extra diagnostic lines;
    indexing only by explicit ``n:`` avoids misalignment where the first ``full_count``
    arbitrary matches were nearly all zeros — which made tiny thresholds skip every frame.
    Later lines for the same ``n`` overwrite earlier ones.
    """
    by_n: dict[int, float] = {}
    for line in stderr.splitlines():
        low = line.lower()
        if "showinfo" not in low:
            continue
        nm = _FRAME_N_RE.search(line)
        if not nm:
            continue
        try:
            n = int(nm.group(1))
        except ValueError:
            continue
        val: float | None = None
        for rx in _SCENE_SCORE_RES:
            m = rx.search(line)
            if m:
                try:
                    val = float(m.group(1))
                except ValueError:
                    val = None
                break
        if val is not None:
            by_n[n] = val
    return by_n


def scene_cut_protect_indices(
    scene_cuts_global: list[int],
    *,
    batch_offset: int,
    full_count: int,
) -> set[int]:
    """1-based frame indices never skipped when protect_scene_cuts is enabled."""
    protected: set[int] = set()
    if full_count <= 0:
        return protected
    from aep.adapters.rife import local_cuts_from_global

    local_cuts = local_cuts_from_global(
        scene_cuts_global,
        batch_offset=batch_offset,
        in_count=full_count,
    )
    for c in local_cuts:
        for d in (-1, 0, 1):
            j = c + d
            if 1 <= j <= full_count:
                protected.add(j)
    return protected


def _scores_by_showinfo_n(
    scores: dict[int, float] | list[float] | tuple[float, ...],
) -> dict[int, float]:
    if isinstance(scores, dict):
        return dict(scores)
    return {i: float(v) for i, v in enumerate(scores)}


def skip_indices_from_scores(
    scores: dict[int, float] | list[float] | tuple[float, ...],
    *,
    full_count: int,
    threshold: float,
    protected: set[int],
) -> set[int]:
    """1-based indices to remove from decode before the first NCNN stage.

    ``scores`` is either a map ``showinfo n`` (0-based) -> score, or a dense list where
    index ``i`` is the score for the same ``n`` (legacy tests).

    For decoded frame ``k`` (1-based, k>=2), we compare using ``n = k - 1`` (the showinfo
    index for that frame). Missing scores default to **not** skipping (conservative).
    """
    skip: set[int] = set()
    if full_count <= 1:
        return skip
    by_n = _scores_by_showinfo_n(scores)
    thr = max(float(threshold), DEDUPE_THRESHOLD_EPS)
    for k in range(2, full_count + 1):
        if k in protected:
            continue
        n = k - 1
        if n not in by_n:
            continue
        s = by_n[n]
        if s < thr:
            skip.add(k)
    return skip


def _frame_path(out_dir: Path, index: int, fmt: str, *, digits: int = 8) -> Path:
    return out_dir / f"{index:0{digits}d}.{fmt.lower()}"


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        if src.resolve() == dst.resolve():
            return
    except OSError:
        pass
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def compact_decode_directory(
    *,
    frames_dir: Path,
    full_count: int,
    skip: set[int],
    frame_format: str,
    digits: int = 8,
) -> tuple[list[int], Path]:
    """Move skipped frames aside and renumber kept frames to 1..L'.

    Returns (kept_order, deduped_subdir).
    """
    ext = frame_format.lower()
    kept_order = [i for i in range(1, full_count + 1) if i not in skip]
    if not kept_order:
        raise ValueError("compact_decode_directory: no frames kept")
    deduped = frames_dir.parent / "deduped_frames"
    if deduped.exists():
        shutil.rmtree(deduped)
    deduped.mkdir(parents=True, exist_ok=True)
    for k in sorted(skip):
        src = _frame_path(frames_dir, k, ext, digits=digits)
        if src.is_file():
            shutil.move(str(src), str(deduped / src.name))
    # Two-phase renumber to avoid collisions.
    tmp_pairs: list[tuple[Path, int]] = []
    for new_idx, old_idx in enumerate(kept_order, start=1):
        src = _frame_path(frames_dir, old_idx, ext, digits=digits)
        if not src.is_file():
            raise FileNotFoundError(f"missing frame file for index {old_idx}: {src}")
        tmp = frames_dir / f".rfd_tmp_{new_idx:0{digits}d}.{ext}"
        shutil.move(str(src), str(tmp))
        tmp_pairs.append((tmp, new_idx))
    for tmp, new_idx in tmp_pairs:
        dst = _frame_path(frames_dir, new_idx, ext, digits=digits)
        shutil.move(str(tmp), str(dst))
    return kept_order, deduped


def write_dedupe_map(stage_dir: Path, doc: dict[str, Any]) -> Path:
    path = stage_dir / DEDUPE_MAP_NAME
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def load_dedupe_map(stage_dir: Path) -> dict[str, Any] | None:
    path = stage_dir / DEDUPE_MAP_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("load_dedupe_map failed for %s: %s", path, exc)
        return None


def local_cuts_compact_from_full(
    local_cuts_full: list[int],
    kept_order: list[int],
    *,
    l_prime: int,
) -> list[int]:
    """Remap 1-based full-batch cut indices to compact input indices for RIFE."""
    if not kept_order or l_prime <= 0:
        return []
    out: list[int] = []
    seen: set[int] = set()
    for c in sorted(local_cuts_full):
        i_next = None
        for i, full_idx in enumerate(kept_order):
            if full_idx >= c:
                i_next = i
                break
        if i_next is None:
            continue
        local = i_next + 1
        if 2 <= local <= l_prime and local not in seen:
            seen.add(local)
            out.append(local)
    return sorted(out)


def expand_rife_output_dir(
    *,
    compact_rife_dir: Path,
    dest_dir: Path,
    kept_order: list[int],
    full_count: int,
    multiplier: int,
    frame_format: str,
    digits: int = 8,
) -> None:
    """Materialize N×M frames from L'×M compact RIFE output using kept_order."""
    ext = frame_format.lower()
    if full_count <= 0 or multiplier <= 0:
        raise ValueError("expand_rife_output_dir: invalid dimensions")
    kept_set = set(kept_order)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def rank_one_based(s: int) -> int:
        k = s
        while k >= 1:
            if k in kept_set:
                return kept_order.index(k) + 1
            k -= 1
        return 1

    for s in range(1, full_count + 1):
        p = rank_one_based(s)
        for m in range(multiplier):
            src_i = (p - 1) * multiplier + 1 + m
            dst_i = (s - 1) * multiplier + 1 + m
            src = _frame_path(compact_rife_dir, src_i, ext, digits=digits)
            dst = _frame_path(dest_dir, dst_i, ext, digits=digits)
            if not src.is_file():
                raise FileNotFoundError(f"expand RIFE: missing {src}")
            _link_or_copy(src, dst)


def expand_upscale_output_dir(
    *,
    compact_up_dir: Path,
    dest_dir: Path,
    kept_order: list[int],
    full_count: int,
    frame_format: str,
    digits: int = 8,
) -> None:
    """Materialize N upscaled frames from L' compact outputs."""
    ext = frame_format.lower()
    if full_count <= 0:
        raise ValueError("expand_upscale_output_dir: invalid full_count")
    kept_set = set(kept_order)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def rank_one_based(s: int) -> int:
        k = s
        while k >= 1:
            if k in kept_set:
                return kept_order.index(k) + 1
            k -= 1
        return 1

    for s in range(1, full_count + 1):
        p = rank_one_based(s)
        src = _frame_path(compact_up_dir, p, ext, digits=digits)
        dst = _frame_path(dest_dir, s, ext, digits=digits)
        if not src.is_file():
            raise FileNotFoundError(f"expand upscale: missing {src}")
        _link_or_copy(src, dst)


def merge_dedupe_state_into_plan(ctx: PipelineContext, doc: dict[str, Any]) -> None:
    ctx.plan.setdefault("frame_dedupe", {})
    base = ctx.plan["frame_dedupe"]
    if not isinstance(base, dict):
        base = {}
        ctx.plan["frame_dedupe"] = base
    for k, v in doc.items():
        base[k] = v
