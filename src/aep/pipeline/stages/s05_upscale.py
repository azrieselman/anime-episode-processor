"""Stage 05: upscale.

Runs the configured upscaler (Real-CUGAN, Real-ESRGAN, waifu2x-ncnn-vulkan,
or Anime4KCPP) against the frames produced by stage 04.

Behavior:
* When the plan disables upscaling (``upscale.active=False``), we no-op and
  the next stage falls back to stage 04's frames.
* Otherwise we invoke the adapter with OOM-aware tile fallback. The starting
  tile size comes from the plan; the adapter halves it on Vulkan OOM and
  persists a hint for future jobs on the same hardware.

Reads:    ctx.plan, upstream frames (``decode`` or ``interpolate`` per plan)
Writes:   ctx.plan["upscale"] = {count, dir, tile_size_used, scale}, <stage>/frames/
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from aep.adapters.anime4kcpp import Anime4kcppAdapter
from aep.adapters.anime4kcpp_vs import Anime4kcppVsAdapter
from aep.adapters.ncnn_base import (
    NcnnRunResult,
    NcnnVulkanAdapter,
    empty_dir,
)
from aep.adapters.realcugan import CuganJob, RealCuganAdapter
from aep.adapters.realesrgan import EsrganJob, RealesrganAdapter
from aep.adapters.waifu2x import Waifu2xAdapter, Waifu2xJob
from aep.bench.hardware import HardwareProfile
from aep.errors import CancelledError, PausedError, PipelineError, StageError
from aep.persist.settings import load_settings
from aep.pipeline.cache import compute_cache_key
from aep.pipeline.context import PipelineContext
from aep.pipeline.events import EventSink, StageEvent
from aep.pipeline.stage import BaseStage, StagePlan, StageResult
from aep.util.proc import ProcInterrupted

log = logging.getLogger(__name__)


class UpscaleStage(BaseStage):
    name = "05_upscale"

    def __init__(
        self,
        *,
        cugan: RealCuganAdapter | None = None,
        esrgan: RealesrganAdapter | None = None,
        waifu2x: Waifu2xAdapter | None = None,
        anime4kcpp: Anime4kcppAdapter | None = None,
        anime4kcpp_vs: Anime4kcppVsAdapter | None = None,
    ) -> None:
        # Lazy-construct on first use so importing this module doesn't probe
        # for binaries that may not be installed in dev.
        self._cugan = cugan
        self._esrgan = esrgan
        self._waifu2x = waifu2x
        self._anime4kcpp = anime4kcpp
        self._anime4kcpp_vs = anime4kcpp_vs

    # --------------------------------------------------------------- plan

    def plan(self, ctx: PipelineContext) -> StagePlan:
        if not ctx.plan:
            raise PipelineError(f"{self.name} requires 01_plan to have populated ctx.plan")
        upscale_cfg = ctx.plan.get("upscale", {}) or {}
        active = bool(upscale_cfg.get("active", False))
        engine = str(upscale_cfg.get("engine", "none"))
        in_src = str(upscale_cfg.get("input_source", "decode"))
        # Read chunked-execution thresholds from app settings. Failing to load
        # settings is non-fatal here — we silently fall back to single-shot.
        try:
            hw_settings = load_settings().hardware
            chunk_threshold: int | None = int(hw_settings.ncnn_chunk_threshold)
            chunk_size: int | None = int(hw_settings.ncnn_chunk_size)
        except Exception as exc:
            log.debug("settings load failed at plan time; chunked path disabled: %s", exc)
            chunk_threshold = None
            chunk_size = None
        upstream_path_str = str(
            (ctx.plan.get(in_src, {}) or {}).get("dir")
            or (ctx.plan.get("decode", {}) or {}).get("dir")
            or "",
        )
        params: dict[str, object] = {
            "active": active,
            "engine": engine,
            "model": upscale_cfg.get("model"),
            "scale": upscale_cfg.get("scale"),
            "denoise": upscale_cfg.get("denoise"),
            "tile_size": upscale_cfg.get("tile_size"),
            "tta": upscale_cfg.get("tta", False),
            "frame_format": upscale_cfg.get("frame_format", "png"),
            "ncnn_chunk_threshold": chunk_threshold,
            "ncnn_chunk_size": chunk_size,
            "input_source": in_src,
            # So cache entries cannot be reused after a different upstream frame path.
            "upstream_frames_dir": upstream_path_str if active else "",
        }
        # Tool-version inputs differ by engine; collapse to a single string.
        tool_version = "skipped"
        if active:
            try:
                tool_version = self._adapter_for(engine).version
            except Exception as exc:
                log.debug("could not detect upscaler version at plan time: %s", exc)
                tool_version = "unknown"
        cache_key = compute_cache_key(
            source_fingerprint=str(ctx.source_path),
            stage_name=self.name,
            tool_versions={engine: tool_version},
            params=params,
        )
        out_dir = ctx.stage_dir(self.name) / "frames"
        in_dir = Path(upstream_path_str) if active and upstream_path_str else None
        return StagePlan(
            stage_name=self.name,
            cache_key=cache_key,
            params=params,
            inputs=[in_dir] if in_dir else [],
            outputs=[out_dir],
        )

    # --------------------------------------------------------------- run

    def run(self, ctx: PipelineContext, plan: StagePlan, events: EventSink) -> StageResult:
        t0 = time.monotonic()
        active = bool(plan.params.get("active", False))
        out_dir: Path = plan.outputs[0]

        if not active:
            events.emit(StageEvent(ctx.job_id, self.name, "log",
                                   message="upscale skipped (active=False)"))
            return StageResult(stage_name=self.name, success=True,
                               duration_s=time.monotonic() - t0,
                               metrics={"skipped": True})

        engine = str(plan.params.get("engine", "none"))
        in_source = str(plan.params.get("input_source", "decode"))
        in_dir_str = ctx.plan.get(in_source, {}).get("dir") or ctx.plan.get("decode", {}).get("dir")
        if not in_dir_str:
            raise StageError(
                f"{self.name}: cannot resolve input dir from plan ({in_source})"
            )
        in_dir = Path(in_dir_str)
        if not in_dir.is_dir():
            raise StageError(
                f"{self.name}: upstream frames dir missing: {in_dir}"
            )
        frame_format = str(plan.params.get("frame_format", "png"))
        in_manifest = ctx.get_frame_manifest(in_dir, format=frame_format)
        in_count = in_manifest["count"]
        if in_count == 0:
            raise StageError(
                f"{self.name}: no input frames in {in_dir}"
            )
        empty_dir(out_dir)

        hw: HardwareProfile | None = ctx.extras.get("hardware_profile")
        hardware_fp = hw.fingerprint() if hw else "no-hw"

        primary = ctx.media_info.primary_video if ctx.media_info else None
        source_height = primary.height if primary and primary.height else None

        tile_size = int(plan.params.get("tile_size") or 256)
        scale = int(plan.params.get("scale") or 2)
        denoise = plan.params.get("denoise")
        denoise_int = int(denoise) if isinstance(denoise, int) else None
        tta = bool(plan.params.get("tta", False))
        model = str(plan.params.get("model") or "")
        prefer_cuda = True
        anime4k_threads = 4
        try:
            hw_settings = load_settings().hardware
            prefer_cuda = bool(hw_settings.anime4k_prefer_cuda)
            anime4k_threads = int(hw_settings.anime4k_threads)
        except Exception:
            prefer_cuda = True
            anime4k_threads = 4

        events.emit(StageEvent(
            ctx.job_id, self.name, "started",
            message=f"upscale {in_count} frames via {engine} model={model} scale={scale} tile={tile_size}",
        ))

        # Each engine builds an argv_factory taking (in_dir, out_dir, tile) so
        # we can dispatch it through either the single-shot OOM path or the
        # chunked path with the same shape.
        adapter: NcnnVulkanAdapter
        resolved_model_id: str
        anime_run_result: NcnnRunResult | None = None
        if engine == "realcugan-ncnn-vulkan":
            adapter_typed = self._adapter_for(engine)
            assert isinstance(adapter_typed, RealCuganAdapter)
            adapter = adapter_typed
            resolved_model_id = model or "models-pro"
            denoise_eff = denoise_int if denoise_int is not None else 3

            def argv_factory(
                ci: Path, co: Path, t: int,
                _ad: RealCuganAdapter = adapter_typed,
                _mid: str = resolved_model_id,
                _scale: int = scale, _den: int = denoise_eff,
                _tta: bool = tta, _fmt: str = frame_format,
            ) -> list[str | Path]:
                return _ad.build_cugan_argv(
                    CuganJob(
                        input_dir=ci, output_dir=co, model_id=_mid,
                        scale=_scale, denoise=_den, tile_size=t,
                        tta=_tta, frame_format=_fmt,
                    ),
                )
        elif engine == "realesrgan-ncnn-vulkan":
            adapter_typed = self._adapter_for(engine)
            assert isinstance(adapter_typed, RealesrganAdapter)
            adapter = adapter_typed
            resolved_model_id = model or "realesr-animevideov3"

            def argv_factory(
                ci: Path, co: Path, t: int,
                _ad: RealesrganAdapter = adapter_typed,
                _mid: str = resolved_model_id,
                _scale: int = scale, _tta: bool = tta, _fmt: str = frame_format,
            ) -> list[str | Path]:
                return _ad.build_esrgan_argv(
                    EsrganJob(
                        input_dir=ci, output_dir=co, model_id=_mid,
                        scale=_scale, tile_size=t, tta=_tta, frame_format=_fmt,
                    ),
                )
        elif engine == "waifu2x-ncnn-vulkan":
            adapter_typed = self._adapter_for(engine)
            assert isinstance(adapter_typed, Waifu2xAdapter)
            adapter = adapter_typed
            resolved_model_id = model or "models-cunet"
            denoise_eff = denoise_int if denoise_int is not None else 3

            def argv_factory(
                ci: Path, co: Path, t: int,
                _ad: Waifu2xAdapter = adapter_typed,
                _mid: str = resolved_model_id,
                _scale: int = scale, _den: int = denoise_eff,
                _tta: bool = tta, _fmt: str = frame_format,
            ) -> list[str | Path]:
                return _ad.build_waifu2x_argv(
                    Waifu2xJob(
                        input_dir=ci, output_dir=co, model_id=_mid,
                        scale=_scale, denoise=_den, tile_size=t,
                        tta=_tta, frame_format=_fmt,
                    ),
                )
        elif engine == "anime4kcpp":
            adapter_typed = self._adapter_for(engine)
            if not hasattr(adapter_typed, "run_frame_sequence"):
                raise StageError(
                    f"{self.name}: anime4kcpp adapter missing run_frame_sequence()"
                )
            adapter = adapter_typed  # type: ignore[assignment]
            resolved_model_id = model or "acnet-hdn-gan"
            anime_run_result = adapter_typed.run_frame_sequence(
                input_dir=in_dir,
                output_dir=out_dir,
                model_id=resolved_model_id,
                scale=scale,
                prefer_cuda=prefer_cuda,
                frame_format=frame_format,
                threads=anime4k_threads,
                on_progress=lambda line: _maybe_emit_progress(ctx, events, self.name, line),
            )
        elif engine == "anime4kcpp-vs":
            adapter_typed = self._adapter_for(engine)
            if not hasattr(adapter_typed, "run_frame_sequence"):
                raise StageError(
                    f"{self.name}: anime4kcpp-vs adapter missing run_frame_sequence()"
                )
            adapter = adapter_typed  # type: ignore[assignment]
            resolved_model_id = model or "acnet-hdn0"
            anime_run_result = adapter_typed.run_frame_sequence(
                input_dir=in_dir,
                output_dir=out_dir,
                model_id=resolved_model_id,
                scale=scale,
                prefer_cuda=prefer_cuda,
                frame_format=frame_format,
                on_progress=lambda line: _maybe_emit_progress(ctx, events, self.name, line),
            )
        else:
            raise StageError(f"{self.name}: unknown upscaler engine {engine!r}")

        # Choose single-shot vs chunked based on the configured threshold. The
        # chunked path is materially slower per frame on small inputs (per-chunk
        # process spawn + frame materialization overhead), so we only use it
        # for runs where the cumulative GPU-state issues actually appear.
        try:
            if anime_run_result is not None:
                run_result = anime_run_result
            else:
                chunk_threshold = int(plan.params.get("ncnn_chunk_threshold") or 0)
                chunk_size = int(plan.params.get("ncnn_chunk_size") or 500)
                if chunk_threshold > 0 and in_count >= chunk_threshold:
                    events.emit(StageEvent(
                        ctx.job_id, self.name, "log",
                        message=(
                            f"chunked NCNN dispatch: {in_count} ≥ threshold {chunk_threshold}, "
                            f"chunk_size={chunk_size}"
                        ),
                    ))
                    run_result = adapter.run_chunked(
                        input_dir=in_dir,
                        output_dir=out_dir,
                        chunk_size=chunk_size,
                        argv_factory=argv_factory,
                        initial_tile_size=tile_size,
                        hardware_fp=hardware_fp,
                        model_id=resolved_model_id,
                        source_height=source_height,
                        frame_format=frame_format,
                        on_progress=lambda line: _maybe_emit_progress(ctx, events, self.name, line),
                        should_interrupt=lambda: "cancel" if ctx.cancel_event.is_set() else (
                            "pause" if ctx.pause_event.is_set() else None
                        ),
                    )
                else:
                    run_result = adapter.run_with_oom_fallback(
                        argv_factory=lambda t: argv_factory(in_dir, out_dir, t),
                        initial_tile_size=tile_size,
                        hardware_fp=hardware_fp,
                        model_id=resolved_model_id,
                        source_height=source_height,
                        on_progress=lambda line: _maybe_emit_progress(ctx, events, self.name, line),
                        should_interrupt=lambda: "cancel" if ctx.cancel_event.is_set() else (
                            "pause" if ctx.pause_event.is_set() else None
                        ),
                    )
        except ProcInterrupted as exc:
            if exc.reason == "cancel":
                raise CancelledError("cancelled during upscale") from exc
            ctx.extras["pause_checkpoint"] = {"stage": self.name}
            raise PausedError("paused during upscale") from exc
        if ctx.pause_event.is_set():
            ctx.extras["pause_checkpoint"] = {"stage": self.name, "frames_done": 0}
            raise PausedError("paused during upscale")
        if ctx.cancel_event.is_set():
            raise CancelledError("cancelled during upscale")

        for w in run_result.warnings:
            events.emit(StageEvent(ctx.job_id, self.name, "warning", message=w))
        for r in run_result.rationale:
            events.emit(StageEvent(ctx.job_id, self.name, "log", message=r))

        out_manifest = ctx.get_frame_manifest(out_dir, format=frame_format)
        out_count = out_manifest["count"]
        if out_count != in_count:
            raise StageError(
                f"{self.name}: produced {out_count} frames, expected {in_count}",
                context={"in_count": in_count, "out_count": out_count},
            )

        ctx.plan.setdefault("upscale", {})
        ctx.plan["upscale"]["count"] = out_count
        ctx.plan["upscale"]["dir"] = str(out_dir)
        ctx.plan["upscale"]["tile_size_used"] = run_result.tile_size_used
        ctx.plan["upscale"]["attempts"] = run_result.attempts

        return StageResult(
            stage_name=self.name,
            success=True,
            duration_s=time.monotonic() - t0,
            artifacts={"frames_dir": out_dir},
            metrics={
                "engine": engine,
                "frames": out_count,
                "tile_size_used": run_result.tile_size_used,
                "attempts": run_result.attempts,
                "scale": scale,
                "input_bytes": in_manifest["bytes"],
                "output_bytes": out_manifest["bytes"],
            },
        )

    # --------------------------------------------------------------- helpers

    def _adapter_for(self, engine: str):
        if engine == "realcugan-ncnn-vulkan":
            if self._cugan is None:
                self._cugan = RealCuganAdapter()
            return self._cugan
        if engine == "realesrgan-ncnn-vulkan":
            if self._esrgan is None:
                self._esrgan = RealesrganAdapter()
            return self._esrgan
        if engine == "waifu2x-ncnn-vulkan":
            if self._waifu2x is None:
                self._waifu2x = Waifu2xAdapter()
            return self._waifu2x
        if engine == "anime4kcpp":
            if self._anime4kcpp is None:
                self._anime4kcpp = Anime4kcppAdapter()
            return self._anime4kcpp
        if engine == "anime4kcpp-vs":
            if self._anime4kcpp_vs is None:
                settings = load_settings()
                self._anime4kcpp_vs = Anime4kcppVsAdapter(
                    vspipe_override_dir=settings.paths.vapoursynth_dir,
                    plugin_override_dir=settings.paths.anime4kcpp_vs_filter_dir,
                )
            return self._anime4kcpp_vs
        raise StageError(f"unknown upscaler engine: {engine!r}")


def _maybe_emit_progress(ctx: PipelineContext, events: EventSink, stage: str, line: str) -> None:
    """NCNN binaries occasionally emit per-frame "x/y" progress lines on stderr.
    We emit them as ``log`` events so the GUI can surface them.
    """
    s = line.strip()
    if not s:
        return
    if "/" in s and any(ch.isdigit() for ch in s):
        events.emit(StageEvent(ctx.job_id, stage, "log", message=s))
