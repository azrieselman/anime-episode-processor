"""Tool-verification library.

Single source of truth for the question "are all required external tools
present and at the right version?". Three call sites depend on this:

  * `scripts/verify_tools.py` -- CI / installer post-step
  * `aep.gui.widgets.verify_tools_dialog.VerifyToolsDialog` -- interactive
    Tools menu entry
  * `AppWindow` first-launch hook -- runs once at GUI startup; if anything
    is missing or mismatched, opens the dialog so the user sees what's wrong

Keeping the logic in one place means the three surfaces can't drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from aep.adapters.anime4kcpp import Anime4kcppAdapter
from aep.adapters.anime4kcpp_vs import Ffms2VapourSynthAdapter, VapourSynthAdapter
from aep.adapters.base import ToolAdapter
from aep.adapters.ffmpeg import FFmpegAdapter
from aep.adapters.ffprobe import FFProbeAdapter
from aep.adapters.mkvtoolnix import MkvinfoAdapter, MkvmergeAdapter, MkvpropeditAdapter
from aep.adapters.rife import RifeAdapter
from aep.constants import PINNED_VERSIONS

log = logging.getLogger(__name__)


# Status values are kept as plain strings (not Enum) because they end up in
# Qt cells, log lines, and CI output unchanged -- a string is the simplest
# common denominator across those consumers.
StatusValue = str  # "ok" | "mismatch" | "missing" | "version_unknown"


@dataclass
class ToolStatus:
    tool_id: str
    path: str
    version: str
    expected: str
    status: StatusValue
    note: str = ""


# Adapters checked by all three consumers. Order is the display order in the
# dialog; CI doesn't care.
DEFAULT_ADAPTERS: list[type[ToolAdapter]] = [
    FFmpegAdapter,
    FFProbeAdapter,
    MkvmergeAdapter,
    MkvpropeditAdapter,
    MkvinfoAdapter,
    Anime4kcppAdapter,
    VapourSynthAdapter,
    Ffms2VapourSynthAdapter,
    RifeAdapter,
]


def check_adapter(adapter: ToolAdapter) -> ToolStatus:
    """Resolve, version-probe, and pin-compare a single adapter.

    Never raises -- turns every exception into a ToolStatus row so callers
    can render a complete table even when half the tools are missing.
    """
    expected = PINNED_VERSIONS.get(adapter.tool_id, "")
    try:
        path = str(adapter.path)
    except Exception as exc:
        return ToolStatus(
            tool_id=adapter.tool_id,
            path="(not found)",
            version="",
            expected=expected,
            status="missing",
            note=str(exc),
        )
    try:
        version = adapter.version
    except Exception as exc:
        return ToolStatus(
            tool_id=adapter.tool_id,
            path=path,
            version="(probe failed)",
            expected=expected,
            status="version_unknown",
            note=str(exc),
        )
    status: StatusValue = "ok"
    # Pinned versions for ffmpeg use a leading "n" ("n7.0.2") that ffmpeg's
    # `-version` output drops, so strip it before prefix-matching.
    if expected and not version.startswith(expected.lstrip("n")):
        status = "mismatch"
    return ToolStatus(
        tool_id=adapter.tool_id,
        path=path,
        version=version,
        expected=expected,
        status=status,
    )


def check_all() -> list[ToolStatus]:
    """Run check_adapter() over every default adapter."""
    out: list[ToolStatus] = []
    for cls in DEFAULT_ADAPTERS:
        try:
            out.append(check_adapter(cls()))
        except Exception as exc:
            out.append(ToolStatus(
                tool_id=cls.tool_id,
                path="(error)",
                version="",
                expected=PINNED_VERSIONS.get(cls.tool_id, ""),
                status="missing",
                note=str(exc),
            ))
    return out


def has_blocking_issues(statuses: list[ToolStatus]) -> bool:
    """True if any status would prevent the pipeline from running at all.

    A version_unknown is *not* blocking -- we couldn't probe the version but
    the binary was found, so the pipeline can attempt to run. A version
    mismatch is also not blocking; we surface it for the user but trust
    them to ignore it if they know what they're doing. Only an outright
    missing tool is treated as blocking.
    """
    return any(s.status == "missing" for s in statuses)


def has_any_issues(statuses: list[ToolStatus]) -> bool:
    """True if any status is not "ok" -- used by startup nag heuristic."""
    return any(s.status != "ok" for s in statuses)
