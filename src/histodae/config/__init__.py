"""Packaged configuration files (experiment + W&B defaults).

These ship inside the package so an installed ``histodae`` never depends on
the current working directory to find its configuration. Resolution order for
a given config name:

1. an explicit path supplied by the caller,
2. a user override in ``<repo-root>/config/<name>`` (only when running from a
   source checkout),
3. the packaged default in ``histodae/config/<name>``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

#: Config file names shipped with the package.
EXPERIMENT_CONFIG = "experiment.yaml"
WANDB_CONFIG = "wandb.yaml"


def config_path(name: str) -> Path:
    """Return the filesystem path of a packaged config file.

    Falls back to the packaged copy when the package is installed as a
    zipimport (no real path) — callers that need the *contents* should use
    :func:`load_config`.
    """
    return Path(str(resources.files("histodae.config").joinpath(name)))


def user_config_path(name: str) -> Path | None:
    """Return a user override for ``name`` inside a source checkout, if any."""
    # histodae/config/<name> -> histodae/ -> src/ -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "config" / name
    return candidate if candidate.is_file() else None


def load_config(name: str) -> dict[str, Any]:
    """Load a packaged config (or its user override) as a dict."""
    override = user_config_path(name)
    if override is not None:
        return yaml.safe_load(override.read_text()) or {}
    text = resources.files("histodae.config").joinpath(name).read_text()
    return yaml.safe_load(text) or {}


__all__ = [
    "EXPERIMENT_CONFIG",
    "WANDB_CONFIG",
    "config_path",
    "load_config",
    "user_config_path",
]
