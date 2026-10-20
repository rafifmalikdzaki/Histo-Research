"""Deprecated alias for :mod:`histodae.data.metadata`.

The canonical import path is::

    from histodae.data.metadata import ...

This module keeps the old top-level name importable and, crucially, makes it
the *same module object* as the canonical one (rather than a re-export), so
``isinstance`` checks and pickled references stay consistent across both
import styles.
"""

import sys

from histodae.data import metadata as _canonical

sys.modules[__name__] = _canonical
