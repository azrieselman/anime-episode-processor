"""Parse and normalize ffmpeg argv tokens from presets and the GUI.

Preset ``encoder.extra_args`` must become a flat list of argv tokens for
``subprocess`` (no shell). Users often paste shell-quoted fragments (e.g.
``'-preset'`` from logs); on Windows, ``shlex.split(..., posix=False)`` would
keep those quotes and break ffmpeg. We always parse with POSIX rules and strip
one layer of matching outer quotes per token.
"""

from __future__ import annotations

import shlex
from typing import Any


def _strip_outer_shell_quotes(token: str) -> str:
    s = token.strip()
    while len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1].strip()
    return s


def normalize_ffmpeg_extra_args(value: Any) -> list[str]:
    """Turn preset/GUI input into ffmpeg argv tokens (no shell quoting).

    Accepts:
      * ``None`` → ``[]``
      * ``str`` → split on newlines, then per-line ``shlex.split``
      * ``list[str]`` → each element is one logical line (may contain spaces)
    """
    if value is None:
        return []
    if isinstance(value, str):
        lines = value.splitlines()
    elif isinstance(value, list):
        lines = [str(item) for item in value]
    else:
        raise TypeError(
            f"extra_args must be a list of strings or a string, got {type(value).__name__}",
        )

    tokens: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line, posix=True)
        except ValueError:
            parts = [line]
        for part in parts:
            cleaned = _strip_outer_shell_quotes(part)
            if cleaned:
                tokens.append(cleaned)
    return tokens
