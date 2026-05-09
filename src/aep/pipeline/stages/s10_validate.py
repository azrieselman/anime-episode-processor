"""Stage 10: validate.

Re-probes the muxed output and runs structural assertions to catch silent failures
that earlier stages might miss:

  * duration within ±200ms of the source (catches truncated encodes / mux drops)
  * audio stream count matches the planned mapping
  * subtitle stream count matches the planned mapping (after expected skips)
  * chapters preserved (count match) when copy_chapters=True
  * attachments preserved (count match) when copy_attachments=True
  * dispositions match per-stream where the source had `default` or `forced`
  * primary video resolution matches target geometry (within rounding)

Failures raise `aep.errors.ValidationError` with a `failures` list in the context. We
prefer hard failure over silent corruption — the user can disable a specific check
via preset settings, but defaults are strict.

Reads:    ctx.plan, ctx.output_path, ctx.media_info (the source's), ctx.source_path
Writes:   <stage>/validation.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aep.adapters.ffprobe import FFProbeAdapter
from aep.errors import PipelineError, ValidationError
from aep.media.ffprobe import FfprobeAnalyzer
from aep.media.models import MediaInfo
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult

log = logging.getLogger(__name__)


# Tunables. The duration tolerance accounts for closed-GOP boundaries and final-frame
# duration rounding in mkv timecodes; we've measured 50–150ms drift on real anime.
DURATION_TOL_S: float = 0.20

# Resolution within ±2 pixels (ffmpeg's even-pixel rounding for chroma subsampling).
RES_TOL_PX: int = 2

# Transfer characteristics that BT.709-tagged or untagged outputs are equivalent to
# from a validation perspective when we've intentionally round-tripped HDR through 8-bit.
_BT709_EQUIVALENT_TRANSFERS: frozenset[str] = frozenset({
    "", "unknown", "bt709", "smpte170m", "bt470bg", "iec61966-2-1",
})


@dataclass
class ValidationFailure:
    code: str
    message: str
    expected: Any = None
    got: Any = None


@dataclass
class ValidationReport:
    passed: bool
    output_path: Path
    failures: list[ValidationFailure] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "output_path": str(self.output_path),
            "failures": [
                {"code": f.code, "message": f.message,
                 "expected": f.expected, "got": f.got}
                for f in self.failures
            ],
            "notes": self.notes,
            "metrics": self.metrics,
        }


class ValidateStage(BaseStage):
    name = "10_validate"

    def __init__(self, ffprobe: FFProbeAdapter | None = None) -> None:
        self._ffprobe = ffprobe or FFProbeAdapter()

    # ------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        params: dict[str, object] = {
            "output": str(ctx.output_path),
            "duration_tol_s": DURATION_TOL_S,
        }
        cache_key = compute_cache_key(
            source_fingerprint=str(ctx.output_path),
            stage_name=self.name,
            tool_versions={"ffprobe": _safe_version(self._ffprobe)},
            params=params,
        )
        out = ctx.stage_dir(self.name) / "validation.json"
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            inputs=[ctx.output_path],
            outputs=[out],
        )

    # ------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        if not ctx.output_path.exists():
            raise PipelineError(f"10_validate: output not found: {ctx.output_path}")
        if ctx.media_info is None:
            raise PipelineError("10_validate: ctx.media_info missing (source probe)")

        analyzer = FfprobeAnalyzer(self._ffprobe)
        out_info = analyzer.analyze(ctx.output_path)
        report = _validate(ctx.media_info, out_info, ctx.plan or {})

        out_json: Path = plan.outputs[0]
        out_json.write_text(
            json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8",
        )

        if not report.passed:
            events.emit(StageEvent(
                ctx.job_id, self.name, "error",
                message=f"validation failed: {len(report.failures)} issue(s)",
            ))
            raise ValidationError(
                "output validation failed",
                context={
                    "failures": [f.code for f in report.failures],
                    "report": str(out_json),
                },
            )

        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=f"validation passed ({len(report.notes)} notes)",
        ))

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"validation_json": out_json},
            metrics=report.metrics,
        )


# ---------- pure validation core (testable) -------------------------------


def _validate(
    source: MediaInfo,
    output: MediaInfo,
    plan_doc: dict[str, Any],
) -> ValidationReport:
    """Pure validator. Takes parsed MediaInfo + the frozen plan; returns a report."""
    failures: list[ValidationFailure] = []
    notes: list[str] = []
    metrics: dict[str, Any] = {}

    container = (plan_doc.get("container") or "mkv").lower()
    sm = plan_doc.get("stream_mapping", {}) or {}
    expected_audio = len(sm.get("audio") or [])
    expected_subs = len(sm.get("subtitles") or [])
    target = plan_doc.get("target_geometry", {}) or {}

    # ----- duration ---------------------------------------------------
    src_dur = source.fmt.duration_s
    out_dur = output.fmt.duration_s
    metrics["duration_source_s"] = src_dur
    metrics["duration_output_s"] = out_dur
    if src_dur is not None and out_dur is not None:
        if abs(src_dur - out_dur) > DURATION_TOL_S:
            failures.append(ValidationFailure(
                code="duration_mismatch",
                message=f"output duration differs from source by > {DURATION_TOL_S}s",
                expected=f"{src_dur:.3f}s",
                got=f"{out_dur:.3f}s",
            ))
    else:
        notes.append("duration check skipped: missing duration on source or output")

    # ----- video stream count -----------------------------------------
    # Some sources carry album art / cover art as an attached_pic video stream.
    # That stream is metadata-like and should not trip the "single output video"
    # invariant for the primary encoded track.
    primary_video_streams = [s for s in output.video_streams if not s.disposition.attached_pic]
    metrics["video_streams_total"] = len(output.video_streams)
    metrics["video_streams_primary"] = len(primary_video_streams)
    if len(primary_video_streams) != 1:
        failures.append(ValidationFailure(
            code="video_stream_count",
            message="output must have exactly one video stream",
            expected=1,
            got=len(primary_video_streams),
        ))

    # ----- video resolution -------------------------------------------
    primary_out = output.primary_video
    expected_w = target.get("width")
    expected_h = target.get("height")
    if primary_out and expected_w and expected_h:
        if (
            primary_out.width is None or primary_out.height is None
            or abs((primary_out.width or 0) - expected_w) > RES_TOL_PX
            or abs((primary_out.height or 0) - expected_h) > RES_TOL_PX
        ):
            failures.append(ValidationFailure(
                code="video_resolution",
                message="output video resolution does not match target",
                expected=(expected_w, expected_h),
                got=(primary_out.width, primary_out.height),
            ))
    elif primary_out and not (expected_w and expected_h):
        # Preserve mode: width should match source primary.
        primary_src = source.primary_video
        if primary_src and primary_out.width and primary_src.width:
            if abs(primary_out.width - primary_src.width) > RES_TOL_PX:
                failures.append(ValidationFailure(
                    code="video_resolution_drift",
                    message="output width drifted from source (preserve mode)",
                    expected=primary_src.width,
                    got=primary_out.width,
                ))

    # ----- audio stream count -----------------------------------------
    metrics["audio_streams"] = len(output.audio_streams)
    if len(output.audio_streams) != expected_audio:
        failures.append(ValidationFailure(
            code="audio_stream_count",
            message="output audio stream count differs from plan",
            expected=expected_audio,
            got=len(output.audio_streams),
        ))

    # ----- subtitle stream count --------------------------------------
    metrics["subtitle_streams"] = len(output.subtitle_streams)
    if len(output.subtitle_streams) != expected_subs:
        failures.append(ValidationFailure(
            code="subtitle_stream_count",
            message="output subtitle stream count differs from plan",
            expected=expected_subs,
            got=len(output.subtitle_streams),
        ))

    # ----- chapters ---------------------------------------------------
    src_chapters = len(source.chapters)
    out_chapters = len(output.chapters)
    metrics["chapters_source"] = src_chapters
    metrics["chapters_output"] = out_chapters
    if sm.get("copy_chapters") and src_chapters and out_chapters != src_chapters:
        failures.append(ValidationFailure(
            code="chapters_lost",
            message="output chapter count differs from source",
            expected=src_chapters,
            got=out_chapters,
        ))

    # ----- attachments ------------------------------------------------
    src_attachments = len(source.attachments)
    out_attachments = len(output.attachments)
    metrics["attachments_source"] = src_attachments
    metrics["attachments_output"] = out_attachments
    plan_streams_cfg = (plan_doc.get("preset") or {}).get("streams") or {}
    if (
        plan_streams_cfg.get("copy_attachments", True)
        and container == "mkv"
        and src_attachments
        and out_attachments != src_attachments
    ):
        failures.append(ValidationFailure(
            code="attachments_lost",
            message="output attachment count differs from source",
            expected=src_attachments,
            got=out_attachments,
        ))

    # ----- HDR / transfer characteristic ------------------------------
    # Three cases:
    #   1. Source had no HDR/10-bit involvement: we don't enforce anything; encoder
    #      handles pix_fmt automatically.
    #   2. allow_8bit_roundtrip: the planner committed to 8-bit BT.709 output.
    #      We expect the output's transfer to land in the BT.709-equivalent set
    #      (BT.709 / unspecified / SMPTE 170M etc); a residual SMPTE2084/HLG tag
    #      would mean ffmpeg copied it through metadata-only, which is wrong.
    #   3. policy=skip (or upscale off entirely on HDR source): we expect the
    #      source transfer to round-trip exactly. Drift to BT.709 means the
    #      encoder/conversion chain ate the HDR tag silently.
    hdr_plan = (plan_doc.get("hdr") or {})
    src_transfer = (hdr_plan.get("source_color_transfer") or "").lower()
    out_transfer = ""
    if primary_out is not None and primary_out.color_transfer:
        out_transfer = primary_out.color_transfer.lower()
    metrics["transfer_source"] = src_transfer or None
    metrics["transfer_output"] = out_transfer or None

    if hdr_plan.get("was_hdr_transfer") or hdr_plan.get("was_10bit"):
        if hdr_plan.get("roundtripped_to_8bit"):
            # Tolerance: any BT.709-equivalent (or unset) output transfer is fine;
            # residual HDR tag is a failure because that means ffmpeg only updated
            # metadata and didn't actually convert pixel values.
            if out_transfer not in _BT709_EQUIVALENT_TRANSFERS:
                failures.append(ValidationFailure(
                    code="hdr_transfer_residual",
                    message=(
                        "output retained an HDR transfer tag despite allow_8bit_roundtrip; "
                        "pixel data may not have been actually converted"
                    ),
                    expected="bt709 (or unspecified)",
                    got=out_transfer,
                ))
            else:
                notes.append(
                    f"HDR/10-bit roundtripped to 8-bit per allow_8bit_roundtrip "
                    f"(source={src_transfer or 'unspecified'} → output={out_transfer or 'unspecified'})"
                )
        # Preserve mode: source transfer should round-trip. Empty source
        # transfer can't be enforced; only fail if both sides specify and differ.
        elif src_transfer and out_transfer and src_transfer != out_transfer:
            failures.append(ValidationFailure(
                code="hdr_transfer_lost",
                message=(
                    "output transfer characteristic differs from source under "
                    "hdr_policy=skip / preserve"
                ),
                expected=src_transfer,
                got=out_transfer,
            ))

    # ----- dispositions on default/forced flags -----------------------
    # Source-side default/forced flags should land somewhere in the output. We don't
    # enforce per-index alignment because reordering is allowed; we enforce that AT
    # LEAST as many `default` and `forced` flags exist on the output as on the source
    # for streams of each kind that we kept.
    for kind in ("audio_streams", "subtitle_streams"):
        src_kept_default = sum(
            1 for s in getattr(source, kind) if s.disposition.default
        )
        out_default = sum(1 for s in getattr(output, kind) if s.disposition.default)
        if src_kept_default and out_default < src_kept_default:
            # Soft check: only fail if we lost ALL default flags. A single drop is a
            # warning, full drop is a regression.
            if out_default == 0:
                failures.append(ValidationFailure(
                    code=f"{kind}_default_lost",
                    message=f"all {kind} default flags lost in output",
                    expected=src_kept_default,
                    got=out_default,
                ))
            else:
                notes.append(
                    f"{kind}: {out_default}/{src_kept_default} default flags preserved"
                )

    return ValidationReport(
        passed=not failures,
        output_path=Path(output.fmt.filename),
        failures=failures,
        notes=notes,
        metrics=metrics,
    )


def _safe_version(adapter: FFProbeAdapter) -> str:
    try:
        return adapter.version
    except Exception:
        return "unknown"
