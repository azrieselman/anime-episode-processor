"""Tests for :mod:`aep.app.update_checker`."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aep.app import update_checker


def test_normalize_tag() -> None:
    assert update_checker.normalize_tag("v1.0.0b3") == "1.0.0b3"
    assert update_checker.normalize_tag("V2.0.0") == "2.0.0"
    assert update_checker.normalize_tag("1.0.0") == "1.0.0"


def test_try_parse_release_version_skips_garbage() -> None:
    assert update_checker.try_parse_release_version("not-a-version") is None
    v = update_checker.try_parse_release_version("v1.0.0b3")
    assert v is not None and str(v) == "1.0.0b3"


@pytest.fixture
def sample_releases_payload() -> bytes:
    rows = [
        {
            "tag_name": "v1.0.0-beta1",
            "draft": False,
            "html_url": "https://example.com/r1",
        },
        {
            "tag_name": "not-a-pep440-tag",
            "draft": False,
            "html_url": "https://example.com/bad",
        },
        {
            "tag_name": "v1.0.0b2",
            "draft": False,
            "prerelease": True,
            "html_url": "https://example.com/r-old",
        },
        {
            "tag_name": "v1.0.0b3",
            "draft": False,
            "prerelease": True,
            "html_url": "https://example.com/r-new",
        },
    ]
    return json.dumps(rows).encode("utf-8")


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _FakeUrlResponse:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> _FakeBody:
        return _FakeBody(self._data)

    def __exit__(self, *_args) -> None:
        pass


def test_check_for_updates_picks_newest_prerelease(sample_releases_payload: bytes) -> None:
    with patch.object(
        update_checker.urllib.request,
        "urlopen",
        return_value=_FakeUrlResponse(sample_releases_payload),
    ):
        result = update_checker.check_for_updates("1.0.0b1")

    assert result.error is None
    assert result.latest_tag_name == "v1.0.0b3"
    assert result.release_url == "https://example.com/r-new"
    assert result.is_update_available is True


def test_check_for_updates_same_version_not_newer(sample_releases_payload: bytes) -> None:
    with patch.object(
        update_checker.urllib.request,
        "urlopen",
        return_value=_FakeUrlResponse(sample_releases_payload),
    ):
        result = update_checker.check_for_updates("1.0.0b3")

    assert result.is_update_available is False


def test_check_for_updates_skips_drafts(sample_releases_payload: bytes) -> None:
    rows = json.loads(sample_releases_payload.decode("utf-8"))
    rows.insert(
        0,
        {
            "tag_name": "v99.0.0",
            "draft": True,
            "html_url": "https://example.com/draft",
        },
    )
    payload = json.dumps(rows).encode("utf-8")

    with patch.object(
        update_checker.urllib.request,
        "urlopen",
        return_value=_FakeUrlResponse(payload),
    ):
        result = update_checker.check_for_updates("1.0.0b1")

    assert result.latest_tag_name == "v1.0.0b3"
