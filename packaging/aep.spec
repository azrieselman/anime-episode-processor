# PyInstaller spec for the Anime Episode Processor GUI.
#
# Build:    pyinstaller packaging/aep.spec --clean
# Output:   dist/aep-gui/                 (one-folder bundle)
#           dist/aep-gui/aep-gui.exe      (entry point)
#
# Design notes:
#   * One-folder build (not one-file) — keeps DLL load times fast,
#     simplifies code-signing, and matches Inno Setup's expectation that
#     the install image is a directory tree.
#   * No third-party tools are bundled. The first-run dialog downloads them
#     into %LOCALAPPDATA%\AEP\tools\ on demand.
#   * Hidden imports cover dynamically-loaded adapter modules and the
#     scripts/_tool_manifest helper that aep.app.tools_fetcher imports at
#     runtime via a sys.path shim.

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

# `packaging/aep.spec` is invoked from the repo root, so SPECPATH is the
# packaging dir; we resolve the project root one level up.
PROJECT_ROOT = Path(SPECPATH).parent.resolve()
SRC_ROOT = PROJECT_ROOT / "src"

block_cipher = None

# ----- runtime data files ----------------------------------------------------

datas = [
    (str(PROJECT_ROOT / "presets"), "presets"),
    (str(PROJECT_ROOT / "pipelines"), "pipelines"),
    (str(SRC_ROOT / "aep" / "gui" / "resources"), "aep/gui/resources"),
    # Tools fetcher imports the manifest from scripts/ at runtime.
    (str(PROJECT_ROOT / "scripts" / "_tool_manifest.py"), "scripts"),
    (str(PROJECT_ROOT / "scripts" / "fetch_tools.py"), "scripts"),
    (str(PROJECT_ROOT / "LICENSE"), "."),
    (str(PROJECT_ROOT / "THIRD_PARTY_NOTICES.md"), "."),
    (str(PROJECT_ROOT / "CHANGELOG.md"), "."),
]

# ----- hidden imports --------------------------------------------------------
# verification.DEFAULT_ADAPTERS instantiates these adapter classes by import,
# so PyInstaller's static analysis sees them; but a couple of submodules are
# referenced via string keys (PINNED_VERSIONS, tool_id) so we belt-and-brace
# by collecting the whole adapter package.

hiddenimports = collect_submodules("aep.adapters") + [
    "aep.adapters.anime4kcpp",
    "aep.adapters.anime4kcpp_vs",
    "aep.adapters.ffmpeg",
    "aep.adapters.ffprobe",
    "aep.adapters.mkvtoolnix",
    "aep.adapters.ncnn_base",
    "aep.adapters.realcugan",
    "aep.adapters.realesrgan",
    "aep.adapters.rife",
    "aep.adapters.verification",
    "aep.adapters.waifu2x",
    "aep.adapters.windows_gpu",
    # Pipeline stages are also discovered indirectly.
    "aep.pipeline.stages.s00_probe",
    "aep.pipeline.stages.s01_plan",
    "aep.pipeline.stages.s02_sample_bench",
    "aep.pipeline.stages.s03_scene_detect",
    "aep.pipeline.stages.s04_decode_serve",
    "aep.pipeline.stages.s05_upscale",
    "aep.pipeline.stages.s06_interpolate",
    "aep.pipeline.stages.s07_postprocess",
    "aep.pipeline.stages.s08_encode",
    "aep.pipeline.stages.s09_mux",
    "aep.pipeline.stages.s10_validate",
    "aep.pipeline.stages.placeholder",
    # GUI views — discovered via app_window.py imports, but listing them is
    # cheap insurance.
    "aep.gui.views.queue_view",
    "aep.gui.views.job_config_view",
    "aep.gui.views.stream_inspector_view",
    "aep.gui.views.logs_view",
    "aep.gui.views.settings_view",
    "aep.gui.preset_design",
    "aep.gui.widgets.first_run_dialog",
    "aep.gui.widgets.verify_tools_dialog",
    "aep.gui.widgets.drop_area",
]

# ----- analysis --------------------------------------------------------------

a = Analysis(
    [str(SRC_ROOT / "aep" / "gui" / "main.py")],
    pathex=[str(SRC_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # We never invoke pytest at runtime; let PyInstaller skip it to keep
        # the bundle size sane. mypy/ruff are dev-only too.
        "pytest",
        "mypy",
        "ruff",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ----- exe -------------------------------------------------------------------

icon_path = SRC_ROOT / "aep" / "gui" / "resources" / "app.ico"
icon_arg = str(icon_path) if icon_path.is_file() else None

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="aep-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,         # UPX trips Windows Defender heuristics; not worth it.
    console=False,     # GUI app — no console window on launch.
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_arg,
)

# ----- collect ---------------------------------------------------------------

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="aep-gui",
)
