"""GitHub Releases API client for Help → Check for Updates.

Uses ``urllib`` only (no requests). Parses recent releases — including prereleases —
because GitHub's ``/releases/latest`` endpoint omits prerelease tags entirely.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from packaging.version import InvalidVersion, Version

log = logging.getLogger(__name__)

API_RELEASES_PAGE = (
    "https://api.github.com/repos/azrieselman/anime-episode-processor/"
    "releases?per_page=20"
)


@dataclass(frozen=True)
class UpdateCheckResult:
    """Outcome of querying GitHub for the newest semver-like release."""

    error: str | None
    """Populated when the HTTP call failed or the response was unusable."""

    latest_version: Version | None
    """Highest successfully parsed ``Version``, or ``None`` on error."""

    latest_tag_name: str | None
    """Raw ``tag_name`` from GitHub for the chosen release."""

    release_url: str | None
    """``html_url`` of the newest release."""

    installed_version: Version
    installed_raw: str

    is_update_available: bool
    """``True`` when ``latest_version > installed_version``."""

    def user_message_summary(self) -> str:
        if self.error:
            return self.error
        if self.latest_version is None or self.latest_tag_name is None:
            return "No parseable releases were found on GitHub."
        if self.is_update_available:
            return (
                f"A newer release is available: {self.latest_tag_name} "
                f"(you have {self.installed_raw})."
            )
        return f"You are up to date (latest: {self.latest_tag_name})."


def normalize_tag(tag: str) -> str:
    s = tag.strip()
    return s[1:] if s.startswith(("v", "V")) else s


def try_parse_release_version(tag_name: str) -> Version | None:
    raw = normalize_tag(tag_name)
    try:
        return Version(raw)
    except InvalidVersion:
        log.debug("skip release tag_name=%r: not PEP 440", tag_name)
        return None


def check_for_updates(installed_version_raw: str) -> UpdateCheckResult:
    """GET recent releases from GitHub and compare the newest semver to *installed*."""

    try:
        installed = Version(normalize_tag(installed_version_raw))
    except InvalidVersion as exc:
        return UpdateCheckResult(
            error=f"Invalid installed version string: {installed_version_raw!r} ({exc})",
            latest_version=None,
            latest_tag_name=None,
            release_url=None,
            installed_version=Version("0"),
            installed_raw=installed_version_raw,
            is_update_available=False,
        )

    ua = f"AnimeEpisodeProcessor/{installed_version_raw}"
    req = urllib.request.Request(
        API_RELEASES_PAGE,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": ua,
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        log.warning("update check HTTP error: %s", exc)
        return UpdateCheckResult(
            error=f"GitHub returned HTTP {exc.code}. Try again later or check Releases in your browser.",
            latest_version=None,
            latest_tag_name=None,
            release_url=None,
            installed_version=installed,
            installed_raw=installed_version_raw,
            is_update_available=False,
        )
    except (TimeoutError, OSError, urllib.error.URLError) as exc:
        log.warning("update check network failure: %s", exc)
        return UpdateCheckResult(
            error=f"Could not reach GitHub ({exc}). Check your connection.",
            latest_version=None,
            latest_tag_name=None,
            release_url=None,
            installed_version=installed,
            installed_raw=installed_version_raw,
            is_update_available=False,
        )

    try:
        releases = json.loads(payload)
    except json.JSONDecodeError as exc:
        return UpdateCheckResult(
            error=f"Unexpected GitHub response (not JSON): {exc}",
            latest_version=None,
            latest_tag_name=None,
            release_url=None,
            installed_version=installed,
            installed_raw=installed_version_raw,
            is_update_available=False,
        )

    if not isinstance(releases, list):
        return UpdateCheckResult(
            error="Unexpected GitHub response shape.",
            latest_version=None,
            latest_tag_name=None,
            release_url=None,
            installed_version=installed,
            installed_raw=installed_version_raw,
            is_update_available=False,
        )

    best_ver: Version | None = None
    best_tag: str | None = None
    best_url: str | None = None

    for rel in releases:
        if not isinstance(rel, dict):
            continue
        if rel.get("draft"):
            continue
        tag_name = rel.get("tag_name")
        if not isinstance(tag_name, str):
            continue
        ver = try_parse_release_version(tag_name)
        if ver is None:
            continue
        if best_ver is None or ver > best_ver:
            best_ver = ver
            best_tag = tag_name
            url_raw = rel.get("html_url")
            best_url = url_raw if isinstance(url_raw, str) else None

    if best_ver is None:
        return UpdateCheckResult(
            error=None,
            latest_version=None,
            latest_tag_name=None,
            release_url=None,
            installed_version=installed,
            installed_raw=installed_version_raw,
            is_update_available=False,
        )

    return UpdateCheckResult(
        error=None,
        latest_version=best_ver,
        latest_tag_name=best_tag,
        release_url=best_url,
        installed_version=installed,
        installed_raw=installed_version_raw,
        is_update_available=best_ver > installed,
    )


__all__ = [
    "API_RELEASES_PAGE",
    "UpdateCheckResult",
    "check_for_updates",
    "normalize_tag",
    "try_parse_release_version",
]
