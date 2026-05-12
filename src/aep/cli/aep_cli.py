"""Tiny CLI for headless use and CI smoke tests.

Subcommands:
  aep-cli probe <path>             — run the analyzer and print JSON
  aep-cli presets                  — list presets
  aep-cli enqueue <path> --preset  — add a job (does not block)
  aep-cli list                     — list jobs

This is also useful for testing the pipeline without launching Qt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aep.app.bootstrap import bootstrap


def _cmd_probe(args: argparse.Namespace) -> int:
    services = bootstrap()
    try:
        info = services.media.analyze(Path(args.path))
        print(info.model_dump_json(indent=2))
        return 0
    finally:
        services.stop()


def _cmd_presets(_: argparse.Namespace) -> int:
    services = bootstrap()
    try:
        for p in services.presets.list():
            tag = " (built-in)" if p.meta.builtin else ""
            print(f"{p.meta.id:24s} {p.meta.name}{tag}")
            if p.meta.description:
                print(f"    {p.meta.description.strip()}")
        return 0
    finally:
        services.stop()


def _cmd_enqueue(args: argparse.Namespace) -> int:
    services = bootstrap()
    try:
        overrides: dict | None = None
        if getattr(args, "frame_dedupe", False):
            overrides = {
                "frame_dedupe": {
                    "enabled": True,
                    "threshold": float(args.frame_dedupe_threshold),
                    "protect_scene_cuts": not bool(args.frame_dedupe_no_scene_protect),
                },
            }
        job = services.jobs.enqueue(
            Path(args.path),
            args.preset,
            preset_overrides=overrides,
        )
        print(json.dumps({"job_id": job.id, "state": job.state.value}, indent=2))
        return 0
    finally:
        services.stop()


def _cmd_list(_: argparse.Namespace) -> int:
    services = bootstrap()
    try:
        for j in services.jobs.list_jobs():
            print(f"{j.id}  {j.state.value:10s}  {j.progress*100:5.1f}%  {j.source_path}")
        return 0
    finally:
        services.stop()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("aep-cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("probe")
    pp.add_argument("path")
    pp.set_defaults(func=_cmd_probe)

    sub.add_parser("presets").set_defaults(func=_cmd_presets)

    pe = sub.add_parser("enqueue")
    pe.add_argument("path")
    pe.add_argument("--preset", default="anime_balanced")
    pe.add_argument(
        "--frame-dedupe",
        action="store_true",
        help="Enable perceptual frame dedupe (ffmpeg scene scores) for this job.",
    )
    pe.add_argument(
        "--frame-dedupe-threshold",
        type=float,
        default=0.02,
        help="Scene score below this marks a duplicate (default 0.02).",
    )
    pe.add_argument(
        "--frame-dedupe-no-scene-protect",
        action="store_true",
        help="Allow skipping frames adjacent to scene cuts (default: protect them).",
    )
    pe.set_defaults(func=_cmd_enqueue)

    sub.add_parser("list").set_defaults(func=_cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
