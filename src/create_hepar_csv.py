#!/usr/bin/env python3
"""Legacy entry point — implementation now lives in ``histodae.cli.make_hepar_csv``.

Kept so ``python src/create_hepar_csv.py`` keeps working (the remote JOIV sweep invokes
the pipeline from a clone root) and so that clone-local code takes precedence
over any installed copy of ``histodae``.

Prefer the ``histodae-make-hepar-csv`` console script or ``histodae`` CLI group.
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from histodae.cli.make_hepar_csv import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main() or 0)
