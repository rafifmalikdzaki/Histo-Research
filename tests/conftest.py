"""Pytest path setup: run from the repo root so `from src.<mod> import ...` works."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
