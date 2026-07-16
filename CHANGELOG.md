# Changelog

All notable changes to Anime Episode Processor (AEP) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Pre-1.0 milestone tags (`1.0.0rc1`, `1.0.0rc2`) were internal — `1.0.0-beta1` is
the first publicly distributed build.

## [Unreleased]

## [1.0.0-beta4] — 2026-07-15

Fourth public beta. Broader GPU encoder support (QSV / AMF / D3D12 / Vulkan),
FFmpeg full build 8.1.1, Benchmark and RamDisk GUI surfaces, tighter batch frame
accounting, and a refreshed dark theme.

### Added

- **Hardware encoders**: Intel QSV, AMD AMF, and D3D12 / Vulkan encoder families
  with parity-style preset fields and recommender coverage alongside NVENC / libx26x.
- **Benchmark** sidebar view: segment-level pipeline runs with optional VMAF
  (when `libvmaf` is present in the pinned FFmpeg build).
- **RamDisk** sidebar view and ImDisk helpers for configuring / managing a
  Windows RAM disk used by frame-heavy stages.
- **Batch timing**: frame-plan utilities for overlap, trim, and expected-count
  reconciliation across decode / interpolate batches (more reliable duration and
  frame accounting when container metadata overstates length).
- **FFProbe**: video packet timeline / `decodable_end_s` for accurate keyframe and
  duration planning.
- **RIFE robustness**: detect Vulkan `vkQueueSubmit failed` GPU faults and retry
  interpolation up to three times.
- **GUI**: dark theme refresh, wheel-guard for spinboxes, simplified sidebar
  (Job Config removed; queue-centric layout).
- **FFmpeg argv**: safer tokenization / auto-quoting for custom encoder args.

### Changed

- **FFmpeg pin**: gyan.dev **8.1.1 full** build (was essentials) for broader
  hardware-encoder and filter coverage including VMAF where available.
- **Default `anime_balanced`**: Anime4K ARTCNN 2x + RIFE v4.25; encoder fields
  include NVENC / QSV / AMF / D3D12 / Vulkan knobs.
- **Presets**: dropped unused built-ins (`anime_quality`, `anime_speed`,
  `low_vram_safe`, `mixed_balanced`, `waifu2x_anime`); designer expanded for the
  new encoder options.
- **Preset designer / broker**: richer encoder configuration and concurrency /
  job-lifecycle wiring improvements.

### Removed

- **Anime4KCPP-legacy (2.5.x)** adapter and related tests — Anime4KCPP 3.x is the
  sole Anime4K path.

### Known limitations

- `02_sample_bench` pipeline-stage smart auto-tuning — post-beta (Benchmark tab
  is a manual segment tool, not auto-preset selection).
- `av1_nvenc` → `hevc_nvenc` automatic fallback on encode failure.
- macOS / Linux ports.
- Out-of-process worker broker (the service layer is already abstracted for
  the eventual move; today everything runs in-process).
- Code-signed Windows installer.

## [1.0.0-beta3] — 2026-05-16

Third public beta. Anime4K-legacy upscaling, faster decode/scene-detect paths,
and decode hwaccel behavior that respects preset pins vs global Settings.

### Added

- **Anime4K-legacy (2.5.x)**: directory-batch `Anime4KCPP_CLI` upscaler engine with
  CUDA-first / OpenCL-fallback GPU selection; pinned in the tools manifest.
- **Interpolation**: `ffmpeg_scdet_scale_width` preset field (default 320) to
  downscale before FFmpeg `scdet` for faster scene-cut detection; `0` keeps
  full-resolution analysis.
- **Decode**: `png_intermediate_codec` plan wiring (e.g. MJPEG intermediates for
  faster scratch I/O vs libpng).

### Changed

- **Default `anime_balanced` preset**: ARNET F8B8 HDN model, MJPEG decode
  intermediates, frame dedupe off, NVENC temporal AQ off; scene detect via
  FFmpeg `scdet`.
- **Anime4KCPP 3.2**: repinned CLI bundle URL/SHA256; adapter improvements for
  long command lines and batch upscale paths.
- **Broker / plan**: global Settings decode hwaccel applies only when the merged
  preset still has `hwaccel: auto`; preset-pinned values (e.g. `cuda`) win.
- **Decode stage**: hardware-accel fallback chain and related robustness tweaks.

### Known limitations (unchanged from beta-2)

- `02_sample_bench` real implementation (smart auto-tuning) — post-beta.
- `av1_nvenc` → `hevc_nvenc` automatic fallback on encode failure.
- macOS / Linux ports.
- Out-of-process worker broker (the service layer is already abstracted for
  the eventual move; today everything runs in-process).
- Code-signed Windows installer.

## [1.0.0-beta2] — 2026-05-11

Second public beta. Focus on queue UX, tooling verification, decode/encode
robustness, duplicate-frame handling in the frame pipeline, and batched-job
resume correctness.

### Added

- **Perceptual duplicate-frame path**: decode can compact near-duplicate frames
  for downstream NCNN stages; upscale/interpolate expand back so the encoded
  timeline matches the full decoded frame count (FFmpeg `select` scene scores +
  metadata, no hard dependency on the optional `scenedetect` lavfi filter).
- **Help → Check for Updates**: queries GitHub releases (including prereleases)
  and compares versions with the installed build.
- **Verify Tools** dialog and related first-run tooling checks.
- **Settings**: toggle to show or hide the queue **Job ID** column.
- **Planning**: automatic batch chunk sizing from free scratch-disk and RAM-disk
  headroom (when configured).

### Changed

- **Queue**: job dispatch order matches the queue table’s filename sort.
- **Queue**: pausing waits for the worker to halt cooperatively before the UI
  proceeds.
- **GUI**: refreshed application window icon.
- **Decode / encode**: more robust FFmpeg invocation, hardware-accel probing,
  and fallback behavior; scene-detect and encode-stage tweaks for reliability.
- **Anime4KCPP**: optional truncated argv/exec logging for huge command lines.

### Fixed

- **Batched pipeline**: rehydrate per-segment metadata on resume so the validate
  stage no longer fails after an interrupted batched run.

### Known limitations (unchanged from beta-1)

- `02_sample_bench` real implementation (smart auto-tuning) — post-beta.
- `av1_nvenc` → `hevc_nvenc` automatic fallback on encode failure.
- macOS / Linux ports.
- Out-of-process worker broker (the service layer is already abstracted for
  the eventual move; today everything runs in-process).
- Code-signed Windows installer.

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
[1.0.0-beta2]: https://github.com/azrieselman/anime-episode-processor/releases/tag/v1.0.0-beta2
[1.0.0-beta3]: https://github.com/azrieselman/anime-episode-processor/releases/tag/v1.0.0-beta3
[1.0.0-beta4]: https://github.com/azrieselman/anime-episode-processor/releases/tag/v1.0.0-beta4
[Unreleased]: https://github.com/azrieselman/anime-episode-processor/compare/v1.0.0-beta4...HEAD
