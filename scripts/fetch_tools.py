"""Download and install pinned external tools.

Usage:
    python scripts/fetch_tools.py [--force] [--tool ffmpeg]

What it does (per pin in `_tool_manifest.ALL_PINS`):
  1. Skips if the destination dir already contains the expected primary binary
     (unless --force).
  2. Downloads the archive to a temp file under tools/_downloads/.
  3. Verifies sha256 against the pinned value (refuses to install on mismatch
     or if the pin is still the placeholder TBD value).
  4. Extracts requested files into tools/<subdir>/.

The script is intentionally pure-stdlib for ZIPs (no extra deps for the common case).
7z archives require the system `7z` CLI on PATH; this is the case for any developer
machine that ships MKVToolNix dev tools, and the fetch step is a one-time install.

This module never runs at app runtime — only via the dev/CI install flow.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

# Allow `python scripts/fetch_tools.py` from a checkout.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from _tool_manifest import ALL_PINS, SHA256_TBD, ToolPin  # noqa: E402

log = logging.getLogger("fetch_tools")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)


def tools_root() -> Path:
    return ROOT / "tools"


def download_dir() -> Path:
    d = tools_root() / "_downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_download_dir() -> None:
    """Remove tools/_downloads when it no longer contains files."""
    d = tools_root() / "_downloads"
    if not d.exists():
        return
    try:
        next(d.iterdir())
    except StopIteration:
        d.rmdir()
    except OSError:
        # Best-effort cleanup only.
        return


def download(url: str, dest: Path) -> None:
    log.info("downloading %s", url)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as out:
        shutil.copyfileobj(resp, out)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_archive(archive: Path, fmt: str, dest: Path) -> Path:
    """Extract archive into a temp dir under `dest` and return that dir.

    We always extract to a sandbox first, then copy requested files out — this is
    safer than extracting straight onto a user's tools/ dir.
    """
    sandbox = Path(tempfile.mkdtemp(prefix="aep-extract-", dir=download_dir()))
    if fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(sandbox)
    elif fmt == "7z":
        # Requires 7-Zip's `7z` (or `7zz` on Linux) on PATH.
        cli = shutil.which("7z") or shutil.which("7zz")
        if not cli:
            raise RuntimeError(
                "7z extraction requires the `7z` CLI on PATH (install MKVToolNix's "
                "shipped 7-Zip, or `apt install p7zip-full` / `brew install p7zip`)."
            )
        subprocess.run([cli, "x", "-y", f"-o{sandbox}", str(archive)], check=True)
    else:
        raise ValueError(f"unsupported archive format: {fmt}")
    return sandbox


def install_files(pin: ToolPin, sandbox: Path) -> None:
    install_root = tools_root() / pin.subdir
    install_root.mkdir(parents=True, exist_ok=True)

    for src_pattern, dest_rel in pin.files:
        if src_pattern.endswith("/*"):
            src_dir = sandbox / src_pattern[:-2]
            if not src_dir.is_dir():
                raise FileNotFoundError(f"expected dir not in archive: {src_pattern}")
            target = install_root if dest_rel == "." else (install_root / dest_rel)
            target.mkdir(parents=True, exist_ok=True)
            for item in src_dir.rglob("*"):
                if item.is_file():
                    rel = item.relative_to(src_dir)
                    out = target / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, out)
        else:
            src = sandbox / src_pattern
            if not src.is_file():
                raise FileNotFoundError(f"expected file not in archive: {src_pattern}")
            dest = install_root / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    log.info("installed pin %s -> %s", pin.tool_id, install_root)


def fetch_one(pin: ToolPin, *, force: bool, allow_unpinned: bool = False) -> None:
    install_root = tools_root() / pin.subdir

    # Skip if the primary file (first entry) is already present and not forcing.
    primary_file = pin.files[0][1]
    primary_path = install_root / primary_file
    if not force and primary_path.exists() and primary_path.stat().st_size > 0:
        log.info("[skip] %s already installed at %s", pin.tool_id, primary_path)
        return

    placeholder = pin.archive_sha256 == SHA256_TBD
    if placeholder and not allow_unpinned:
        log.error(
            "[fail] %s pin has placeholder sha256; refusing to install. "
            "Edit scripts/_tool_manifest.py with the real checksum, or pass "
            "--allow-unpinned for development bootstrap (not for releases).",
            pin.tool_id,
        )
        raise SystemExit(2)

    archive = download_dir() / Path(pin.archive_url).name
    if force or not archive.exists():
        download(pin.archive_url, archive)

    actual = sha256_of(archive)
    if placeholder and allow_unpinned:
        log.warning(
            "[unpinned] %s installed without a pinned sha256. "
            "Computed sha256=%s — paste this into scripts/_tool_manifest.py before "
            "shipping a release.",
            pin.tool_id, actual,
        )
    elif actual != pin.archive_sha256:
        log.error(
            "[fail] %s archive sha256 mismatch: got %s, expected %s",
            pin.tool_id, actual, pin.archive_sha256,
        )
        raise SystemExit(3)

    sandbox = extract_archive(archive, pin.archive_format, install_root)
    try:
        install_files(pin, sandbox)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
    # Keep downloads ephemeral once a tool is successfully installed.
    archive.unlink(missing_ok=True)
    cleanup_download_dir()


def compute_hashes(pins: tuple[ToolPin, ...]) -> dict[str, str]:
    """Download each archive and report its sha256 without installing.

    Used by `--update-hashes` to bootstrap the manifest after a version bump.
    Prints one line per tool in `tool_id  sha256` form so you can paste the
    values into _tool_manifest.py manually — we deliberately don't auto-edit
    the file because pin updates should be a reviewed commit.
    """
    out: dict[str, str] = {}
    for pin in pins:
        archive = download_dir() / Path(pin.archive_url).name
        if not archive.exists():
            download(pin.archive_url, archive)
        out[pin.tool_id] = sha256_of(archive)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download/install even if present")
    parser.add_argument("--tool", help="only fetch this tool_id")
    parser.add_argument(
        "--allow-unpinned",
        action="store_true",
        help="install pins with a placeholder sha256 (DEV BOOTSTRAP ONLY; never ship like this)",
    )
    parser.add_argument(
        "--update-hashes",
        action="store_true",
        help="download and print sha256 for each pin without installing; use to refresh the manifest",
    )
    args = parser.parse_args()

    pins = ALL_PINS
    if args.tool:
        pins = tuple(p for p in pins if p.tool_id == args.tool)
        if not pins:
            log.error("unknown tool_id: %s", args.tool)
            return 1

    if args.update_hashes:
        hashes = compute_hashes(pins)
        # Output is intentionally machine-greppable.
        for tool_id, sha in hashes.items():
            print(f"{tool_id}  {sha}")
        return 0

    for pin in pins:
        fetch_one(pin, force=args.force, allow_unpinned=args.allow_unpinned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
