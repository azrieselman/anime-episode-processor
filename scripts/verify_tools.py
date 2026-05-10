"""Verify pinned tool installation: presence + version probes for default adapters.

Usage:
    python scripts/verify_tools.py [--strict] [--fetch-mismatched]

Run after `fetch_tools.py` to confirm:
  * Every pinned tool resolves via the same logic the running app uses
    (`ToolAdapter.path`).
  * Every adapter's detected version starts with the pin's version stem.

With ``--fetch-mismatched``, download and reinstall archives for pins that are
outright missing or whose adapter row is ``missing`` / ``VERSION MISMATCH``,
using the same runtime fetcher as the GUI (SHA256-verified archives).

Exit codes:
    0   — everything OK
    1   — something is missing / version mismatch (in --strict mode), or fetch failed

Thin wrapper over `aep.adapters.verification.check_all()` so CI, the GUI's
Verify Tools dialog, and the first-launch hook all share one implementation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from aep.adapters.verification import (  # noqa: E402
    check_all,
    has_any_issues,
    has_blocking_issues,
)
from aep.app.pinned_tools_refresh import pins_to_refresh  # noqa: E402
from aep.app.tools_fetcher import FetchError, fetch_one  # noqa: E402

log = logging.getLogger("verify_tools")
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero on any mismatch (CI mode)",
    )
    parser.add_argument(
        "--fetch-mismatched",
        action="store_true",
        help="re-download pinned archives for missing or version-mismatched tools",
    )
    args = parser.parse_args()

    if args.fetch_mismatched:
        prior = check_all()
        pins = pins_to_refresh(prior)
        if not pins:
            log.info("[fetch-mismatched] nothing to install (no missing/mismatch rows)")
        else:
            for pin in pins:
                log.info("[fetch-mismatched] installing %s (%s)", pin.tool_id, pin.version)
                try:
                    fetch_one(pin, force=True)
                except FetchError as exc:
                    log.error("[fetch-mismatched] failed on %s: %s", pin.tool_id, exc)
                    return 1

    statuses = check_all()
    for s in statuses:
        if s.status == "ok":
            log.info("[OK] %-12s %s (%s)", s.tool_id, s.version, s.path)
        elif s.status == "mismatch":
            log.warning(
                "[MISMATCH] %-12s detected=%s pinned=%s path=%s",
                s.tool_id, s.version, s.expected, s.path,
            )
        elif s.status == "version_unknown":
            log.warning(
                "[VERSION-PROBE-FAILED] %s at %s: %s",
                s.tool_id, s.path, s.note,
            )
        else:  # missing
            log.error("[MISSING] %s: %s", s.tool_id, s.note)

    issue_count = sum(1 for s in statuses if s.status != "ok")
    if args.strict and has_any_issues(statuses):
        log.error("verification failed: %d issue(s)", issue_count)
        return 1
    if issue_count:
        if has_blocking_issues(statuses):
            log.warning(
                "verification finished with %d issue(s) including blocking ones "
                "(non-strict mode); pass --strict in CI to fail on these",
                issue_count,
            )
        else:
            log.warning(
                "verification finished with %d non-blocking issue(s)",
                issue_count,
            )
    else:
        log.info("all tools verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
