"""Run provenance: what exactly produced this run?

Every experiment directory gets a ``manifest.json`` describing the code, the
data and the settings that produced it. Without it, a results table cannot
be traced back to a specific commit or a specific split — which is exactly
the kind of ambiguity that makes a multi-seed sweep untrustworthy.

The manifest is best-effort: a missing git checkout or an unreadable CSV
degrades a field to ``None`` rather than failing the run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    "data_fingerprint",
    "git_commit",
    "write_run_manifest",
]


def git_commit(repo_root: Path | None = None) -> str | None:
    """Return the current git commit SHA, or ``None`` outside a checkout."""
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _hash_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def data_fingerprint(csv_paths: list[Path] | None) -> dict[str, Any]:
    """Fingerprint the split CSVs that define a run's data.

    Hashing the CSVs (rather than the image files) captures what actually
    determines the split — the specimen assignment — without reading tens of
    thousands of tiles.
    """
    fingerprint: dict[str, Any] = {"csv_sha256": {}}
    for path in csv_paths or []:
        p = Path(path)
        if not p.is_file():
            continue
        sha = _hash_file(p)
        if sha:
            fingerprint["csv_sha256"][str(p)] = sha
    return fingerprint


def write_run_manifest(
    exp_dir: Path,
    *,
    seed: int | None = None,
    model_name: str | None = None,
    dataset: str | None = None,
    hyperparameters: dict[str, Any] | None = None,
    split_csvs: list[Path] | None = None,
    repo_root: Path | None = None,
) -> Path:
    """Write ``<exp_dir>/manifest.json`` and return its path."""
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "git_commit": git_commit(repo_root),
        "seed": seed,
        "model_name": model_name,
        "dataset": dataset,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "hyperparameters": hyperparameters or {},
        "data": data_fingerprint(split_csvs),
    }

    try:
        import torch

        manifest["torch_version"] = torch.__version__
    except Exception:  # noqa: BLE001 - provenance is best-effort
        manifest["torch_version"] = None

    path = Path(exp_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
