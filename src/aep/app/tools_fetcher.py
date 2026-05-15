"""Runtime tools fetcher used by the GUI's first-run dialog.

This module mirrors the CLI logic in ``scripts/fetch_tools.py`` but is
designed for embedded use: it accepts progress and cancel callbacks, installs
into a caller-supplied tools root (defaulting to :func:`aep.util.paths.tools_dir`),
and never calls ``sys.exit`` on failure — it raises typed exceptions instead.

We intentionally re-implement download/extract/install rather than importing
``scripts.fetch_tools`` because:

* the script lives outside the package (``scripts/`` is not on ``sys.path``
  in installed builds), and importing it would couple runtime to a
  development-time module layout;
* the script targets the in-repo ``tools/`` dir, while installer builds need
  ``%LOCALAPPDATA%/AEP/tools/``;
* progress and cancel hooks need to wrap the download loop, which the CLI
  script doesn't expose.

The pin manifest itself (``ALL_PINS``) is the single source of truth and is
loaded at import time from ``scripts/_tool_manifest.py`` via the same
``sys.path`` shim the CLI uses.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aep.util.paths import tools_dir

# Make the manifest module importable from runtime code. The CLI script does
# the same thing (``sys.path.insert(0, str(HERE))``); we follow suit so the
# manifest stays the single source of truth.
#
# PyInstaller bundles ``scripts/_tool_manifest.py`` under ``_MEIPASS/scripts``
# (see packaging/aep.spec ``datas``). ``__file__`` there is
# ``.../_internal/aep/app/tools_fetcher.py``, so ``parents[3]/scripts`` would
# resolve beside ``_internal`` and miss the copy — frozen builds must use
# ``_MEIPASS`` explicitly.
_scripts_dir: Path | None = None
if getattr(sys, "frozen", False):
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cand = Path(meipass) / "scripts"
        if cand.is_dir():
            _scripts_dir = cand
if _scripts_dir is None:
    _scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
if _scripts_dir.is_dir():
    p = str(_scripts_dir)
    if p not in sys.path:
        sys.path.insert(0, p)

from _tool_manifest import ALL_PINS, SHA256_TBD, ToolPin  # noqa: E402

log = logging.getLogger(__name__)


# Re-export so callers can introspect the install plan without poking at the
# private manifest module.
__all__ = [
    "ALL_PINS",
    "FetchCancelled",
    "FetchError",
    "FetchProgress",
    "ToolPin",
    "fetch_all",
    "fetch_one",
    "is_installed",
    "missing_pins",
]


class FetchCancelled(Exception):
    """Raised when ``cancel_check()`` returns True during a fetch."""


class FetchError(RuntimeError):
    """Hash mismatch, download failure, or post-extract install failure."""


@dataclass(frozen=True)
class FetchProgress:
    """Snapshot of download/install progress for a single tool.

    The dialog renders one row per pin and updates it on every callback.
    ``stage`` lets the UI distinguish download progress from extraction (which
    has no good byte-level meter) so it can switch the bar to indeterminate.
    """
    tool_id: str
    pin_index: int             # 0-based index in the fetch list
    pin_total: int             # total tools being fetched in this batch
    stage: str                 # "starting" | "downloading" | "verifying" | "extracting" | "installing" | "done" | "skipped" | "failed"
    bytes_downloaded: int = 0
    bytes_total: int | None = None
    message: str = ""


ProgressCallback = Callable[[FetchProgress], None]
CancelCheck = Callable[[], bool]


# ---------------------------------------------------------- internal helpers


def _emit(cb: ProgressCallback | None, progress: FetchProgress) -> None:
    if cb is None:
        return
    try:
        cb(progress)
    except Exception:
        log.exception("tools_fetcher progress callback raised; continuing")


def _check_cancel(cancel: CancelCheck | None) -> None:
    if cancel is None:
        return
    try:
        cancelled = cancel()
    except Exception:
        log.exception("tools_fetcher cancel callback raised; ignoring")
        return
    if cancelled:
        raise FetchCancelled("tools fetch cancelled by caller")


def _download_dir(install_root: Path) -> Path:
    d = install_root / "_downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_installed(pin: ToolPin, install_root: Path | None = None) -> bool:
    """Return True if the pin's primary file is present in the install root."""
    root = (install_root or tools_dir()) / pin.subdir
    primary = root / pin.files[0][1]
    return primary.is_file() and primary.stat().st_size > 0


def missing_pins(install_root: Path | None = None) -> list[ToolPin]:
    """Return pins from ``ALL_PINS`` that are not yet installed."""
    return [p for p in ALL_PINS if not is_installed(p, install_root)]


# ----------------------------------------------------------- download / hash


def _download_to(
    url: str,
    dest: Path,
    *,
    progress_cb: ProgressCallback | None,
    cancel: CancelCheck | None,
    pin: ToolPin,
    pin_index: int,
    pin_total: int,
) -> None:
    """Stream ``url`` into ``dest``, calling ``progress_cb`` per chunk.

    Uses urllib so we don't pull in `requests`. Caller is responsible for
    sha256 verification afterward.
    """
    log.info("tools_fetcher: downloading %s", url)
    chunk = 1024 * 256  # 256 KiB; balances callback overhead vs. responsiveness
    with urllib.request.urlopen(url) as resp:
        total_str = resp.headers.get("Content-Length")
        try:
            total = int(total_str) if total_str else None
        except (TypeError, ValueError):
            total = None
        with dest.open("wb") as out:
            written = 0
            while True:
                _check_cancel(cancel)
                buf = resp.read(chunk)
                if not buf:
                    break
                out.write(buf)
                written += len(buf)
                _emit(progress_cb, FetchProgress(
                    tool_id=pin.tool_id,
                    pin_index=pin_index,
                    pin_total=pin_total,
                    stage="downloading",
                    bytes_downloaded=written,
                    bytes_total=total,
                ))


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------- extract / install


def _extract_archive(archive: Path, fmt: str, install_root: Path) -> Path:
    """Extract ``archive`` into a temp sandbox and return the sandbox dir."""
    sandbox = Path(tempfile.mkdtemp(prefix="aep-extract-", dir=_download_dir(install_root)))
    if fmt == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(sandbox)
    elif fmt == "7z":
        cli = shutil.which("7z") or shutil.which("7zz")
        if not cli:
            raise FetchError(
                "7z extraction requires the `7z` CLI on PATH; install 7-Zip "
                "(https://www.7-zip.org/) and re-run the first-run fetch."
            )
        try:
            subprocess.run(
                [cli, "x", "-y", f"-o{sandbox}", str(archive)],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            raise FetchError(
                f"7z extraction failed for {archive.name}: "
                f"{exc.stderr.decode('utf-8', 'replace')[:500]}"
            ) from exc
    else:
        raise FetchError(f"unsupported archive format: {fmt}")
    return sandbox


def _install_files(pin: ToolPin, sandbox: Path, install_root: Path) -> None:
    target_root = install_root / pin.subdir
    target_root.mkdir(parents=True, exist_ok=True)

    for src_pattern, dest_rel in pin.files:
        if src_pattern.endswith("/*"):
            src_dir = sandbox / src_pattern[:-2]
            if not src_dir.is_dir():
                raise FetchError(
                    f"{pin.tool_id}: expected dir not in archive: {src_pattern}"
                )
            target = target_root if dest_rel == "." else (target_root / dest_rel)
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
                raise FetchError(
                    f"{pin.tool_id}: expected file not in archive: {src_pattern}"
                )
            dest = target_root / dest_rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)


# ----------------------------------------------------------------- public API


def fetch_one(
    pin: ToolPin,
    *,
    install_root: Path | None = None,
    force: bool = False,
    progress_cb: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
    pin_index: int = 0,
    pin_total: int = 1,
) -> bool:
    """Download, verify, extract, and install a single pinned tool.

    Returns ``True`` if work was done, ``False`` if the pin was already
    installed and ``force=False``. Raises :class:`FetchError` for any
    download/verify/install failure and :class:`FetchCancelled` when the
    caller's cancel hook signals.
    """
    root = install_root or tools_dir()
    target_root = root / pin.subdir

    if not force and is_installed(pin, root):
        _emit(progress_cb, FetchProgress(
            tool_id=pin.tool_id, pin_index=pin_index, pin_total=pin_total,
            stage="skipped", message="already installed",
        ))
        return False

    if pin.archive_sha256 == SHA256_TBD:
        raise FetchError(
            f"{pin.tool_id}: pin has placeholder sha256 ({SHA256_TBD}); "
            "this is a developer-only state. Update scripts/_tool_manifest.py "
            "before shipping."
        )

    target_root.mkdir(parents=True, exist_ok=True)
    _emit(progress_cb, FetchProgress(
        tool_id=pin.tool_id, pin_index=pin_index, pin_total=pin_total,
        stage="starting", message=pin.archive_url,
    ))

    archive = _download_dir(root) / Path(pin.archive_url).name
    if force or not archive.exists():
        _check_cancel(cancel)
        _download_to(
            pin.archive_url, archive,
            progress_cb=progress_cb, cancel=cancel,
            pin=pin, pin_index=pin_index, pin_total=pin_total,
        )

    _check_cancel(cancel)
    _emit(progress_cb, FetchProgress(
        tool_id=pin.tool_id, pin_index=pin_index, pin_total=pin_total,
        stage="verifying", message="checking sha256",
    ))
    actual = _sha256_of(archive)
    if actual != pin.archive_sha256:
        # Remove the corrupt archive so a retry doesn't re-use it.
        try:
            archive.unlink()
        except OSError:
            pass
        raise FetchError(
            f"{pin.tool_id}: archive sha256 mismatch — got {actual}, "
            f"expected {pin.archive_sha256}. The download may have been "
            "tampered with or truncated; check your network and retry."
        )

    _check_cancel(cancel)
    _emit(progress_cb, FetchProgress(
        tool_id=pin.tool_id, pin_index=pin_index, pin_total=pin_total,
        stage="extracting", message=pin.archive_format,
    ))
    sandbox = _extract_archive(archive, pin.archive_format, root)
    try:
        _check_cancel(cancel)
        _emit(progress_cb, FetchProgress(
            tool_id=pin.tool_id, pin_index=pin_index, pin_total=pin_total,
            stage="installing", message=str(target_root),
        ))
        _install_files(pin, sandbox, root)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

    _emit(progress_cb, FetchProgress(
        tool_id=pin.tool_id, pin_index=pin_index, pin_total=pin_total,
        stage="done", message="installed",
    ))
    return True


def fetch_all(
    pins: list[ToolPin] | tuple[ToolPin, ...] | None = None,
    *,
    install_root: Path | None = None,
    force: bool = False,
    progress_cb: ProgressCallback | None = None,
    cancel: CancelCheck | None = None,
) -> list[ToolPin]:
    """Fetch every pin in ``pins`` (default: ``ALL_PINS``).

    Returns the list of pins that were actually installed (skipped pins are
    not included). Re-raises :class:`FetchCancelled` so the caller can
    surface a clean cancel state to the user.
    """
    todo = list(pins) if pins is not None else list(ALL_PINS)
    installed: list[ToolPin] = []
    total = len(todo)
    for idx, pin in enumerate(todo):
        try:
            did_work = fetch_one(
                pin,
                install_root=install_root,
                force=force,
                progress_cb=progress_cb,
                cancel=cancel,
                pin_index=idx,
                pin_total=total,
            )
        except FetchCancelled:
            raise
        except FetchError as exc:
            _emit(progress_cb, FetchProgress(
                tool_id=pin.tool_id, pin_index=idx, pin_total=total,
                stage="failed", message=str(exc),
            ))
            raise
        if did_work:
            installed.append(pin)
    return installed
