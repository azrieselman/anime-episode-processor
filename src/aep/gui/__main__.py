"""Allow `python -m aep.gui` to launch the GUI.

Mirrors the `aep-gui` console_script entry point declared in pyproject.toml,
so users who haven't installed the package (e.g. running straight from a
checkout with `PYTHONPATH=src`) can still launch the app the same way.
"""

from __future__ import annotations

import sys

from aep.gui.main import main

if __name__ == "__main__":
    sys.exit(main())
