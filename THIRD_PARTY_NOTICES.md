# Third-Party Notices

Anime Episode Processor (AEP) is licensed under
[GPL-3.0-or-later](LICENSE). The application orchestrates several third-party
binaries that are **not** redistributed inside the source repository or the
Windows installer. On first launch (or via `python scripts/fetch_tools.py`) the
app downloads and SHA256-verifies these binaries from each vendor's official
release URL into `%LOCALAPPDATA%\AEP\tools\` (or `<repo>/tools/` for source
installs).

Each tool retains its own license. Per-tool license texts are installed
alongside each binary under `tools/<subdir>/` after the first-run fetch (when
the upstream archive ships one).

| Tool | Pinned version | Upstream | License |
| --- | --- | --- | --- |
| FFmpeg / FFprobe (gyan.dev full build) | n8.1.1 | <https://www.gyan.dev/ffmpeg/builds/> · <https://github.com/GyanD/codexffmpeg> | GPL-3.0-only (full build) |
| MKVToolNix (mkvmerge, mkvpropedit, mkvinfo) | 85.0 | <https://mkvtoolnix.download/> | GPL-2.0-or-later |
| Real-CUGAN (ncnn-vulkan) | 20220728 | <https://github.com/nihui/realcugan-ncnn-vulkan> | MIT |
| Real-ESRGAN (ncnn-vulkan) | v0.2.0 | <https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan> | BSD-3-Clause |
| RIFE (ncnn-vulkan, TNTwise build) | 20250112 | <https://github.com/TNTwise/rife-ncnn-vulkan> | MIT |
| waifu2x (ncnn-vulkan) | 20220728 | <https://github.com/nihui/waifu2x-ncnn-vulkan> | MIT |
| Anime4KCPP CLI | 3.2.0/2.5.0 | <https://github.com/TianZerL/Anime4KCPP> | MIT |
| Anime4KCPP VapourSynth filter | 3.2.0 | <https://github.com/TianZerL/Anime4KCPP> | MIT |
| VapourSynth (portable wheel) | R74 | <https://github.com/vapoursynth/vapoursynth> | LGPL-2.1-or-later |
| FFMS2 (VapourSynth source filter) | 2.40 | <https://github.com/FFMS/ffms2> | GPL-2.0-or-later |

## Source / build instructions

Per GPL-3.0 §6 and the equivalent obligations of GPL-2.0-or-later for the
MKVToolNix and FFMS2 binaries above:

* All binaries above are obtained from their upstream maintainers' immutable
  release URLs. AEP does **not** modify, recompile, or relink them.
* Complete corresponding source code for each binary is available at the
  upstream URL listed in the table. Each project ships its own build
  instructions (typically `BUILD.md` or `README.md` at the repository root).

## Models

NCNN-Vulkan upscalers and the RIFE interpolator ship pretrained models inside
their own release archives. Those models are licensed by their respective
authors:

* Real-CUGAN models — released by the upstream project under the same MIT
  license terms as the binary.
* Real-ESRGAN anime models — released by Xintao Wang under BSD-3-Clause; see
  the upstream repo for per-model attribution.
* RIFE models — released by their respective authors; see the
  TNTwise/rife-ncnn-vulkan README for model-specific attribution.
* waifu2x models — released under MIT (CUNet) and other open licenses; see
  the upstream waifu2x repo for per-model attribution.

## FFmpeg note

The pinned gyan.dev "full" FFmpeg build is GPL-licensed and includes both
hardware encoder families (NVENC/QSV/AMF/D3D12/Vulkan depending on runtime
driver support) and broad software codec/filter support. AEP does not link
against FFmpeg's libraries; it spawns the ffmpeg/ffprobe executables as
subprocesses.
