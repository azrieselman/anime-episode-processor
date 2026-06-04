"""Stage 00: probe.

Reads MediaInfo via ffprobe, writes probe.json into the stage dir, and stores the result
on the context for later stages.

Reads:    ctx.source_path
Writes:   ctx.media_info, <stage>/probe.json
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aep.adapters.ffprobe import FFProbeAdapter
from aep.media.extent import enrich_media_decodable_extent
from aep.media.ffprobe import FfprobeAnalyzer
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.hashing import fast_file_fingerprint

log = logging.getLogger(__name__)


class ProbeStage(BaseStage):
    name = "00_probe"

    def __init__(self, ffprobe: FFProbeAdapter | None = None) -> None:
        self._ffprobe = ffprobe or FFProbeAdapter()

    def plan(self, ctx: PipelineContext) -> StagePlan:
        fingerprint = fast_file_fingerprint(ctx.source_path)
        try:
            ffprobe_version = self._ffprobe.version
        except Exception:
            ffprobe_version = "unknown"

        params: dict[str, object] = {"source_fingerprint": fingerprint}
        cache_key = compute_cache_key(
            source_fingerprint=fingerprint,
            stage_name=self.name,
            tool_versions={"ffprobe": ffprobe_version},
            params=params,
        )
        out = ctx.stage_dir(self.name) / "probe.json"
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            inputs=[ctx.source_path],
            outputs=[out],
        )

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        analyzer = FfprobeAnalyzer(self._ffprobe)
        info = analyzer.analyze(ctx.source_path)
        info = enrich_media_decodable_extent(info, ctx.source_path, ffprobe=self._ffprobe)
        ctx.media_info = info

        out_path: Path = plan.outputs[0]
        out_path.write_text(info.model_dump_json(indent=2), encoding="utf-8")

        events.emit(StageEvent(
            ctx.job_id, self.name, "log",
            message=(
                f"streams: video={len(info.video_streams)} audio={len(info.audio_streams)} "
                f"subs={len(info.subtitle_streams)} attachments={len(info.attachments)} "
                f"chapters={len(info.chapters)} mkv={info.is_matroska}"
            ),
        ))

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"probe_json": out_path},
            metrics={
                "video_streams": len(info.video_streams),
                "audio_streams": len(info.audio_streams),
                "subtitle_streams": len(info.subtitle_streams),
                "attachments": len(info.attachments),
                "chapters": len(info.chapters),
                "is_matroska": info.is_matroska,
                "duration_s": info.fmt.duration_s,
                "decodable_end_s": (
                    info.primary_video.decodable_end_s
                    if info.primary_video is not None
                    else None
                ),
            },
        )
