"""Worker process entry point.

Today this is a thin runner that executes
a single job synchronously, which is useful for debugging stage-by-stage behavior.
"""

from __future__ import annotations

import argparse
import sys

from aep.app.bootstrap import bootstrap


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("aep-worker")
    p.add_argument("job_id", help="Job ID to run synchronously")
    args = p.parse_args(argv)

    services = bootstrap()
    try:
        # The in-process broker handles execution; we just wait for it to pick up the job.
        # A future subprocess-worker contract can replace this entry point without
        # changing the surrounding CLI surface.
        from time import sleep

        from aep.jobs.queue import get_job
        for _ in range(600):  # up to 5 minutes for placeholder stages to march through
            job = get_job(args.job_id)
            if not job:
                print(f"job not found: {args.job_id}")
                return 2
            if job.is_terminal():
                print(f"job {job.id} -> {job.state.value} ({job.error or ''})")
                return 0 if job.state.value == "completed" else 1
            sleep(0.5)
        print("timed out")
        return 3
    finally:
        services.stop()


if __name__ == "__main__":
    sys.exit(main())
