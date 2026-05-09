"""Build the AEP Windows installer end-to-end.

Pipeline:
  1. Run PyInstaller against ``packaging/aep.spec`` to produce
     ``dist/aep-gui/`` (one-folder bundle with the GUI exe + Qt + Python).
  2. Run Inno Setup's ``iscc`` against ``packaging/installer.iss`` to wrap the
     bundle into ``dist/AEP-Setup-<version>.exe``.

This script is the single source of truth for "how do I build the installer
locally". CI is **not** wired to invoke it — installer artifacts are produced
on a maintainer's signed Windows box and uploaded to the GitHub Release page
manually for beta-1 (codesigning is post-beta cleanup).

Usage:
    python scripts/build_installer.py
    python scripts/build_installer.py --skip-pyinstaller   # iss-only rebuild
    python scripts/build_installer.py --iscc "C:/Program Files (x86)/Inno Setup 6/ISCC.exe"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPEC = ROOT / "packaging" / "aep.spec"
ISS = ROOT / "packaging" / "installer.iss"
DIST = ROOT / "dist"


def _resolve_iscc(explicit: str | None) -> Path:
    """Find Inno Setup's ``ISCC`` compiler. Try (in order): explicit arg, PATH,
    common install locations under Program Files."""
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise FileNotFoundError(f"--iscc points at a missing file: {p}")
    onpath = shutil.which("iscc") or shutil.which("ISCC")
    if onpath:
        return Path(onpath)
    program_files_candidates = [
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        os.environ.get("ProgramFiles", r"C:\Program Files"),
    ]
    for base in program_files_candidates:
        for inno_dir in ("Inno Setup 6", "Inno Setup 5"):
            candidate = Path(base) / inno_dir / "ISCC.exe"
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        "ISCC.exe not found. Install Inno Setup 6 from https://jrsoftware.org/isinfo.php "
        "or pass --iscc <path>."
    )


def _run(cmd: list[str], *, cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}\n  (cwd: {cwd})")
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-pyinstaller",
        action="store_true",
        help="Skip PyInstaller (assumes dist/aep-gui/ is already populated).",
    )
    parser.add_argument(
        "--skip-iscc",
        action="store_true",
        help="Skip Inno Setup (only run PyInstaller).",
    )
    parser.add_argument(
        "--iscc",
        default=None,
        help="Path to ISCC.exe; auto-detected if omitted.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Pass --clean to PyInstaller (default: yes).",
    )
    args = parser.parse_args()

    if not SPEC.is_file():
        print(f"missing PyInstaller spec: {SPEC}", file=sys.stderr)
        return 2
    if not ISS.is_file():
        print(f"missing Inno Setup script: {ISS}", file=sys.stderr)
        return 2

    DIST.mkdir(parents=True, exist_ok=True)

    if not args.skip_pyinstaller:
        pyinstaller_cmd: list[str] = [
            sys.executable, "-m", "PyInstaller",
            str(SPEC),
            "--workpath", str(DIST / "build"),
            "--distpath", str(DIST),
        ]
        if args.clean:
            pyinstaller_cmd.append("--clean")
        _run(pyinstaller_cmd, cwd=ROOT)

    if not args.skip_iscc:
        iscc = _resolve_iscc(args.iscc)
        print(f"using ISCC: {iscc}")
        # ISCC writes to the OutputDir specified inside the .iss (../dist).
        _run([str(iscc), str(ISS)], cwd=ROOT / "packaging")

    print("\nbuild complete. Artifacts:")
    if (DIST / "aep-gui").is_dir():
        print(f"  one-folder bundle: {DIST / 'aep-gui'}")
    for installer in DIST.glob("AEP-Setup-*.exe"):
        print(f"  installer: {installer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
