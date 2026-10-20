"""Run-directory discovery for the evaluate-stats stage.

Finds training run directories, filters them by model/dataset tags, and loads
per-run metric files. Pure scanning logic (no argparse, no printing of final
reports): the ``evaluate_stats`` console script wires it to CLI flags.
"""

import re
from pathlib import Path
from typing import Any

import pandas as pd

from histodae.stats.tables import METRICS
from histodae.utils.reproducibility import read_config_file

__all__ = [
    "find_experiment_runs",
    "load_experiment_metrics",
]


def _normalize_dataset_name(name: str) -> str:
    """Lowercase alphanumeric fold so 'hepar' matches 'HeparUnifiedPNG'."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def _read_run_dataset(exp_dir: Path) -> list[str]:
    """Best-effort dataset tags for a run dir (directory name, then config)."""
    candidates = [_normalize_dataset_name(exp_dir.name)]
    for config_path in exp_dir.rglob('experiment.yaml'):
        config = read_config_file(config_path)
        for key in ('dataset', 'dataset_name', 'data'):
            if config.get(key):
                candidates.append(_normalize_dataset_name(str(config[key])))
                break
        break  # first config file only, mirroring load_experiment_metrics
    return [c for c in candidates if c]


def _dataset_matches(wanted: str, candidates: list[str]) -> bool:
    """Folded match so 'hepar' and 'HeparUnifiedPNG' match either way.

    Substring covers dataset names embedded in directory names; the prefix
    fallback covers short config tags ('hepar') against long canonical names.
    """
    return any(
        wanted in c or c in wanted or wanted.startswith(c) or c.startswith(wanted)
        for c in candidates
    )


def find_experiment_runs(
    base_dir: str = "outputs",
    model_name: str | None = None,
    dataset: str | None = None
) -> list[Path]:
    """
    Find all experiment run directories matching criteria.

    Args:
        base_dir: Base outputs directory
        model_name: Model name to filter by
        dataset: Dataset name to filter by (matched against the directory
            name and the run's experiment.yaml, so 'HeparUnifiedPNG'
            matches runs tagged 'hepar')

    Returns:
        List of paths to experiment directories
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"⚠️  Base directory {base_dir} does not exist")
        return []

    # Find all experiment directories
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('exp_')]

    # Filter by model name
    if model_name:
        exp_dirs = [d for d in exp_dirs if model_name in d.name]

    # Filter by dataset (dir name first — run dirs rarely carry the dataset;
    # fall back to the experiment.yaml tag recorded at training time)
    if dataset:
        wanted = _normalize_dataset_name(dataset)
        kept = []
        for d in exp_dirs:
            if _dataset_matches(wanted, _read_run_dataset(d)):
                kept.append(d)
        skipped = len(exp_dirs) - len(kept)
        if skipped:
            print(f"ℹ️  Skipped {skipped} run(s) not matching dataset '{dataset}'")
        exp_dirs = kept

    return sorted(exp_dirs)


def load_experiment_metrics(exp_dir: Path) -> dict[str, Any] | None:
    """
    Load metrics from an experiment directory.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        Dictionary containing metrics or None if not found
    """
    # Look for metrics file
    metrics_files = list(exp_dir.rglob('clustering_metrics.csv'))

    if not metrics_files:
        # Try alternative locations
        metrics_files = list(exp_dir.rglob('metrics.csv'))

    if not metrics_files:
        print(f"⚠️  No metrics file found in {exp_dir}")
        return None

    # Load the first metrics file found
    metrics_path = metrics_files[0]

    try:
        df = pd.read_csv(metrics_path)

        # Extract key metrics
        metrics = {}

        # Look for clustering metrics
        for metric in METRICS.keys():
            if metric in df.columns:
                metrics[metric] = df[metric].iloc[-1] if len(df) > 0 else None

        # Load config if available (best-effort: {} when unreadable; the key
        # is informational metadata, never consumed by aggregation)
        config_files = list(exp_dir.rglob('experiment.yaml'))
        if config_files:
            metrics['config'] = read_config_file(config_files[0])

        # Extract seed from directory name
        if 'seed' in exp_dir.name:
            parts = exp_dir.name.split('_')
            for part in parts:
                if part.startswith('seed'):
                    try:
                        metrics['seed'] = int(part.replace('seed', ''))
                    except ValueError:
                        pass

        return metrics

    except Exception as e:
        print(f"⚠️  Error loading metrics from {exp_dir}: {e}")
        return None


