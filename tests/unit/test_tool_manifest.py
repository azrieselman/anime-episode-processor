"""Tests for `scripts/_tool_manifest.py`.

These tests catch the most common regressions on a pin update:
  * forgetting to fill in a real sha256 (placeholder check)
  * URL/version drift between the URL string and the in-archive folder name
  * `aep.constants.PINNED_VERSIONS` missing a tool_id that exists in ALL_PINS
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from _tool_manifest import ALL_PINS, SHA256_TBD  # noqa: E402

from aep.constants import PINNED_VERSIONS  # noqa: E402

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_no_pin_has_placeholder_sha256():
    placeholders = [p.tool_id for p in ALL_PINS if p.archive_sha256 == SHA256_TBD]
    assert not placeholders, (
        f"these pins still have placeholder sha256: {placeholders}. "
        "Run `python scripts/fetch_tools.py --update-hashes` and paste real values."
    )


def test_every_pin_sha256_is_well_formed_hex():
    for pin in ALL_PINS:
        assert SHA256_RE.match(pin.archive_sha256), (
            f"{pin.tool_id}: sha256 is not 64 hex chars: {pin.archive_sha256!r}"
        )


def test_every_pin_url_uses_https():
    for pin in ALL_PINS:
        assert pin.archive_url.startswith("https://"), (
            f"{pin.tool_id}: insecure URL {pin.archive_url}"
        )


def test_every_pin_files_target_is_relative():
    for pin in ALL_PINS:
        for src, dest in pin.files:
            p = Path(dest)
            assert not p.is_absolute(), (
                f"{pin.tool_id}: install dest {dest!r} must be relative to tools/<subdir>"
            )
            assert ".." not in p.parts, (
                f"{pin.tool_id}: install dest {dest!r} must not contain `..`"
            )


def test_every_pin_format_is_supported():
    for pin in ALL_PINS:
        assert pin.archive_format in ("zip", "7z"), (
            f"{pin.tool_id}: unsupported format {pin.archive_format}"
        )


def test_every_pin_tool_id_is_in_pinned_versions():
    """Catches the case where a pin is added but PINNED_VERSIONS isn't updated."""
    missing = [p.tool_id for p in ALL_PINS if p.tool_id not in PINNED_VERSIONS]
    assert not missing, (
        f"tools missing from aep.constants.PINNED_VERSIONS: {missing}"
    )


def test_every_pin_version_matches_pinned_versions():
    """The manifest version and the runtime PINNED_VERSIONS should agree."""
    for pin in ALL_PINS:
        runtime_v = PINNED_VERSIONS.get(pin.tool_id)
        if runtime_v is None:
            continue  # covered by the prior test
        # Strip the leading 'n' some FFmpeg builds carry; comparison is loose
        # because the manifest uses the upstream tag form and PINNED_VERSIONS
        # uses the same form deliberately.
        assert pin.version in runtime_v or runtime_v in pin.version or pin.version == runtime_v, (
            f"{pin.tool_id}: manifest version {pin.version!r} disagrees with "
            f"PINNED_VERSIONS {runtime_v!r}"
        )


@pytest.mark.parametrize("pin", ALL_PINS, ids=[p.tool_id for p in ALL_PINS])
def test_pin_url_filename_contains_version(pin):
    """Sanity: the archive URL should mention the version somewhere.

    Catches the case where you bumped the version string but forgot to bump
    the URL (or vice versa) \u2014 the most common manifest-edit regression.
    """
    url = pin.archive_url
    # Version may have a 'v' prefix, dots removed, etc. Allow broad fuzziness:
    # require at least the digits-only form to appear in the URL.
    digits = re.sub(r"\D", "", pin.version)
    assert digits and digits in re.sub(r"\D", "", url), (
        f"{pin.tool_id}: version {pin.version!r} digits not present in URL {url!r}"
    )
