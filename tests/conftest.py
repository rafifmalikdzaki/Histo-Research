"""Pytest setup.

The public namespace is ``histodae.*``. ``pythonpath = ["src"]`` in
``pyproject.toml`` puts the source root on ``sys.path`` so the tests run
against the working tree without requiring ``pip install -e .``.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
