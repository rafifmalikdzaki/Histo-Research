#!/usr/bin/env python3
"""Thin shim — implementation lives in src/; prefer the `histodae-*` console script."""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
runpy.run_module("compute_statistical_significance", run_name="__main__")
