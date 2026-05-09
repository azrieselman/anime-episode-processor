# Changelog

All notable changes to Anime Episode Processor (AEP) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0 milestone tags (`1.0.0rc1`, `1.0.0rc2`) were internal — `1.0.0-beta1` is
the first publicly distributed build.

## [Unreleased]

Development has started for `1.0.0-beta2`.

### Changed
- Bumped project version to `1.0.0b2.dev0` to mark post-`1.0.0-beta1` development.

## [1.0.0-beta1] — 2026-05-08

First public beta. Source release on GitHub plus a Windows installer with
first-run tools fetcher.

### Added
- **GUI shell** (PySide6): Queue, Job Config, Stream Inspector, Preset Designer,
  Logs, Settings, and a Verify Tools dialog.
- **CLI**: `aep-cli` (probe / list-presets / enqueue / list-jobs) and
  `aep-worker` (synchronous single-job runner stub).
- **Pipeline** (M1–M6.5): probe → plan → scene-detect → batched
  decode/upscale/interpolate/postprocess/encode → mux → validate. Stages are
  resumable, deterministic, and content-addressed via the cache layer.
- **RAM-disk routing**: frame-heavy stages (decode, upscale, interpolate,
  postprocess) and per-batch encode segments transparently route to a
  configured RAM-disk when there's enough headroom.
- **Batched pipeline** (M6.5): long videos are split into time-bounded chunks;
  each chunk runs decode→encode end-to-end on the RAM-disk and is concat-muxed
  at the end.
- **Encoders**: `hevc_nvenc`, `h264_nvenc`, `av1_nvenc`, `libx264`, `libx265`.
- **NCNN-Vulkan upscalers**: Real-CUGAN, Real-ESRGAN, waifu2x.
- **Frame interpolation**: RIFE (ncnn-vulkan) with scene-cut handling that
  replaces morphed frames at hard cuts with hardlinks to the boundary frame.
- **Stream preservation**: audio, subtitles, chapters, attachments, fonts,
  global metadata, per-stream metadata, and dispositions are preserved by
  default through MKVToolNix.
- **First-run tools fetcher**: GUI dialog downloads and SHA256-verifies the
  pinned third-party binaries (~2 GB) into `%LOCALAPPDATA%\AEP\tools\`.
- **Settings**: `general.confirm_overwrite` now prompts before re-queueing a
  source when the resolved output path already exists.

### Changed
- `av1_nvenc` warning text clarified — Ada-class GPUs only; no auto-fallback
  to `hevc_nvenc` (switch the preset's encoder manually).
- `02_sample_bench` is now a documented placeholder slot (kept in the stage
  list to keep cache-key topology stable for future autotune work).

### Removed
- `presets/smart_auto.yaml` (relied on the unimplemented `02_sample_bench`
  benchmarking stage).
- All session-tagged `_debug_log` instrumentation that an earlier agent run
  left in the broker, runner, context, and interpolate stage.

### Known limitations (not blockers for beta-1)
- `02_sample_bench` real implementation (smart auto-tuning) — post-beta.
- `av1_nvenc` → `hevc_nvenc` automatic fallback on encode failure.
- macOS / Linux ports.
- Out-of-process worker broker (the service layer is already abstracted for
  the eventual move; today everything runs in-process).
- Code-signed Windows installer.

[1.0.0-beta1]: https://github.com/azrieselman/anime-episode-processor/releases/tag/v1.0.0-beta1
[Unreleased]: https://github.com/azrieselman/anime-episode-processor/compare/v1.0.0-beta1...HEAD
