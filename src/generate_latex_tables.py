#!/usr/bin/env python3
"""Legacy entry point — implementation now lives in ``histodae.cli.tables``.

Kept so ``python src/generate_latex_tables.py`` keeps working (the remote JOIV sweep invokes
the pipeline from a clone root) and so that clone-local code takes precedence
over any installed copy of ``histodae``.

Prefer the ``histodae-tables`` console script or ``histodae`` CLI group.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from histodae.cli.tables import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main() or 0)
