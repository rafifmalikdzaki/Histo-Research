"""Locate the ``histodae`` package when running a standalone script.

The package is normally importable (installed, or ``PYTHONPATH`` set). This
helper is the fallback for scripts that are copied next to a checkout --
e.g. the copies that live beside the sweep clones on the GPU box -- so they
work regardless of where they are dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_package_on_path() -> str | None:
    """Make ``histodae`` importable; return the directory that was added.

    Searches, in order:

    1. an already-importable ``histodae``,
    2. ``<dir>/src/histodae`` for the script's directory and each of its
       ancestors,
    3. ``<dir>/code*/src/histodae`` for the same directories -- this covers
       the loose copies that sit beside the sweep clones.
    """
    try:
        import histodae  # noqa: F401

        return None
    except ImportError:
        pass

    here = Path(__file__).resolve()
    roots = [here.parent, *here.parents]

    for root in roots:
        candidate = root / "src"
        if (candidate / "histodae").is_dir():
            sys.path.insert(0, str(candidate))
            return str(candidate)

    for root in roots:
        for clone in sorted(root.glob("code*")):
            candidate = clone / "src"
            if (candidate / "histodae").is_dir():
                sys.path.insert(0, str(candidate))
                return str(candidate)

    return None


__all__ = ["ensure_package_on_path"]
