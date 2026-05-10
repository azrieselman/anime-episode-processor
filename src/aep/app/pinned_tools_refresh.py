"""Map verification statuses to pinned tool archives needing (re-)download.

Several adapters resolve from a single downloaded archive (e.g. ffmpeg + ffprobe
share :data:`scripts._tool_manifest.FFMPEG_PIN`). :func:`pins_to_refresh` collapses those
relations so ``fetch_one(..., force=True)`` reinstalls the right archive.
"""

from __future__ import annotations

from pathlib import Path

from aep.adapters.verification import ToolStatus
from aep.app.tools_fetcher import ALL_PINS, ToolPin, missing_pins

_ADAPTER_TOOL_TO_PIN_TOOL: dict[str, str] = {
    "ffprobe": "ffmpeg",
    "mkvpropedit": "mkvmerge",
    "mkvinfo": "mkvmerge",
}


def pin_for_adapter_tool_id(tool_id: str) -> ToolPin | None:
    """Return the manifest pin that supplies binaries for this adapter ``tool_id``."""

    lookup_id = _ADAPTER_TOOL_TO_PIN_TOOL.get(tool_id, tool_id)
    for pin in ALL_PINS:
        if pin.tool_id == lookup_id:
            return pin
    return None


def pins_to_refresh(
    statuses: list[ToolStatus],
    *,
    install_root: Path | None = None,
) -> list[ToolPin]:
    """Pins to (re-)fetch for the current probe table.

    * If every verified adapter row is ``ok``, ``version_unknown``, or ``mismatch``,
      there is **no** ``missing`` adapter row, and **all** ``mismatch`` rows map
      to exactly **one** archive pin (e.g. ffmpeg + ffprobe), reinstall **only**
      that pin — omit unrelated manifest pins reported by :func:`missing_pins`.
    * If any verified adapter row is ``missing``, or mismatches imply **multiple**
      distinct archives, manifest-missing pins are included alongside every pin
      behind ``missing`` or ``mismatch`` statuses.
    """

    by_tool_id = {p.tool_id: p for p in ALL_PINS}

    has_adapter_missing = any(s.status == "missing" for s in statuses)

    mismatch_pins: list[ToolPin] = []
    seen_mismatch: set[tuple[str, str]] = set()
    for st in statuses:
        if st.status != "mismatch":
            continue
        mapped = _ADAPTER_TOOL_TO_PIN_TOOL.get(st.tool_id, st.tool_id)
        pin = by_tool_id.get(mapped)
        if pin is None:
            continue
        key = (pin.tool_id, pin.subdir)
        if key in seen_mismatch:
            continue
        seen_mismatch.add(key)
        mismatch_pins.append(pin)

    if mismatch_pins and not has_adapter_missing and len(mismatch_pins) == 1:
        # Pure version drift for a single archive — do not pull every other unpinned CLI.
        return list(mismatch_pins)

    out: list[ToolPin] = []
    seen: set[tuple[str, str]] = set()

    def add(pin: ToolPin) -> None:
        key = (pin.tool_id, pin.subdir)
        if key in seen:
            return
        seen.add(key)
        out.append(pin)

    for pin in missing_pins(install_root):
        add(pin)

    for st in statuses:
        if st.status not in ("missing", "mismatch"):
            continue
        mapped = _ADAPTER_TOOL_TO_PIN_TOOL.get(st.tool_id, st.tool_id)
        pin = by_tool_id.get(mapped)
        if pin is not None:
            add(pin)

    return out


__all__ = ["pin_for_adapter_tool_id", "pins_to_refresh"]
