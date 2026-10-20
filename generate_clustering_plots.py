#!/usr/bin/env python3
"""Thin shim — implementation lives in ``histodae.cli.plots``; prefer the
``histodae-plots`` console script or the unified ``histodae`` CLI.

The ``sys.path`` insert below is deliberate: it puts this checkout's
``src/`` ahead of any installed copy of ``histodae``, so a clone always runs
its own code (the remote JOIV sweep relies on this).
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from histodae.cli.plots import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main() or 0)
