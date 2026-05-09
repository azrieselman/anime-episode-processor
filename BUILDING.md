# Building Anime Episode Processor

This document covers building from source — the developer flow and the
Windows installer flow. End users should grab the pre-built installer from
the [GitHub Releases page](https://github.com/azrieselman/anime-episode-processor/releases)
instead.

## Prerequisites

* Windows 10 1809+ or Windows 11 (x64).
* Python 3.11 or 3.12 from [python.org](https://www.python.org/downloads/).
* Git for Windows.
* About 4 GB free disk space (for the venv, the third-party tools, and the
  installer build artifacts).

For installer builds:

* [Inno Setup 6.x](https://jrsoftware.org/isinfo.php) — installs `ISCC.exe`
  to `C:\Program Files (x86)\Inno Setup 6\`. The build script auto-detects
  it from PATH or that location.
* [7-Zip](https://www.7-zip.org/) — required at first-run by the tools
  fetcher (some pinned archives are `.7z`). Most dev machines have it.

## Developer flow

```powershell
git clone https://github.com/azrieselman/anime-episode-processor.git
cd aep
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -e ".[dev,package]"
python scripts/fetch_tools.py    # ~2 GB of pinned binaries
aep-gui
```

`fetch_tools.py` writes everything under `<repo>/tools/`. If you'd rather
mirror the installed-build layout (`%LOCALAPPDATA%\AEP\tools\`), set
`AEP_TOOLS_DIR` before launching:

```powershell
$env:AEP_TOOLS_DIR = "$env:LOCALAPPDATA\AEP\tools"
python scripts/fetch_tools.py
aep-gui
```

### Running the test suite

```powershell
pytest tests/unit -q                # 349 unit tests, all should pass
ruff check src tests                # the beta-1 baseline must stay clean
mypy                                 # informational; --strict is post-beta cleanup
```

## Installer flow

The installer is a two-step build: PyInstaller produces a one-folder bundle
of the Python runtime + Qt + the AEP code, then Inno Setup wraps it into a
single `.exe` installer.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -e ".[package]"          # ensures pyinstaller is in the venv
python scripts/build_installer.py    # runs PyInstaller, then ISCC
```

Output:

* `dist\aep-gui\` — the one-folder bundle (run `aep-gui.exe` directly to
  verify it works without re-installing).
* `dist\AEP-Setup-1.0.0-beta1.exe` — the Inno Setup installer.

### Re-running just the installer step

If you've already produced `dist\aep-gui\` and only changed
`packaging\installer.iss`, you can skip the slow PyInstaller step:

```powershell
python scripts/build_installer.py --skip-pyinstaller
```

### Custom ISCC path

If `ISCC.exe` lives somewhere unusual:

```powershell
python scripts/build_installer.py --iscc "D:\Tools\Inno Setup\ISCC.exe"
```

## Notes on what gets bundled

PyInstaller (`packaging/aep.spec`) ships:

* the Python interpreter, PySide6 / Qt, and every wheel from `requirements`;
* the entire `presets/` and `pipelines/` directories (read at runtime);
* `src/aep/gui/resources/**` (icons);
* `scripts/_tool_manifest.py` and `scripts/fetch_tools.py` so the in-app
  first-run dialog can read the same pin manifest the dev script uses;
* `LICENSE`, `THIRD_PARTY_NOTICES.md`, `CHANGELOG.md`.

It explicitly does **not** ship FFmpeg, MKVToolNix, the NCNN-Vulkan
binaries, or any model weights. Those are downloaded into
`%LOCALAPPDATA%\AEP\tools\` on first launch by the GUI's first-run dialog
(or by running `python scripts/fetch_tools.py` manually).

## Code signing (post-beta)

Beta-1 ships unsigned. End users will see a SmartScreen prompt the first
time they run the installer; "More info → Run anyway" works fine. Code
signing is tracked as post-beta cleanup — it requires a paid certificate
and a CI signing pipeline that's beyond the scope of beta-1.

## Troubleshooting

**`ModuleNotFoundError: No module named '_tool_manifest'` in the frozen exe**
— PyInstaller missed the manifest module. Confirm
`scripts/_tool_manifest.py` exists and `aep.spec` has it in `datas`.
Re-run `python scripts/build_installer.py --clean` to force a fresh build.

**Inno Setup fails with `File not found: dist\aep-gui\*`** — PyInstaller
produced no output. Re-run with `python scripts/build_installer.py` (without
`--skip-pyinstaller`) and watch its output for errors. A common cause is a
stale `dist\build\` directory; the `--clean` flag clears it.

**First-run dialog says "7z extraction requires the `7z` CLI on PATH"** —
install [7-Zip](https://www.7-zip.org/) and ensure it's on PATH for the
user account running AEP. (`mkvtoolnix.7z` and `Anime4KCPP-CLI*.7z` need
it; the other archives are .zip.)
