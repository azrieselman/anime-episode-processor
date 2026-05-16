# Anime Episode Processor (AEP)

A local Windows desktop application for upscaling, frame-interpolating, and re-encoding
anime episodes with first-class preservation of subtitles, chapters, attachments, fonts,
and metadata.

- 100% local. No cloud inference, no telemetry.
- Optimized for NVIDIA RTX 3000-series and newer (CUDA / NCNN-Vulkan) with support for all GPUs as well as CPU for video processing and inference.
- Anime-tuned defaults (Real-CUGAN, Real-ESRGAN anime, RIFE).
- MKV-first, with mkvmerge/mkvpropedit for advanced Matroska preservation.
- Stage-based, resumable, deterministic pipeline (probe → plan → scene-detect → batched
  decode/upscale/interpolate/postprocess/encode → mux → validate).

Current release: **1.0.0-beta3**. See [`CHANGELOG.md`](CHANGELOG.md) for what's in the
box and known limitations.
The `main` branch tracks post-release development after **1.0.0-beta3**.

## Install (end users)

Download the latest `AEP-Setup-1.0.0-beta3.exe` from the
[GitHub Releases](https://github.com/azrieselman/anime-episode-processor/releases) page and run
it. The installer is small (~75 MB); on first launch the GUI offers to download and
verify the pinned third-party binaries (FFmpeg, MKVToolNix, NCNN-Vulkan tools — about
2 GB total) into `%LOCALAPPDATA%\AEP\tools\`.

## Dev setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts/fetch_tools.py    # downloads pinned ffmpeg/mkvtoolnix/etc. (~2 GB)
aep-gui
```

To build the installer locally, see [`BUILDING.md`](BUILDING.md).

## License

GPL-3.0-or-later. See [`LICENSE`](LICENSE).

Bundled binaries (FFmpeg, MKVToolNix, NCNN-Vulkan tools, RIFE/CUGAN/ESRGAN models) ship
under their own licenses; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
