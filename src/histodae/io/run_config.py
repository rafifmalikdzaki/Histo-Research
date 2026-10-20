"""Reading and writing run configuration files.

Two flavours live here:

* the strict helpers (:func:`load_config`, :func:`save_config`) raise on
  malformed input, and
* the forgiving scanners (:func:`read_config_file`, :func:`read_run_config`)
  never raise, because they walk run directories that may be partially
  written.
"""

import json
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "load_config",
    "load_json",
    "read_config_file",
    "read_run_config",
    "save_config",
    "save_json",
]


def save_config(config: dict[str, Any], save_path: Path) -> None:
    """
    Save experiment configuration to a YAML file.

    Args:
        config (Dict[str, Any]): Configuration dictionary to save.
        save_path (Path): Path to save the configuration file.

    Example:
        >>> config = {'latent_dim': 128, 'n_clusters': 5}
        >>> save_config(config, Path('outputs/exp_001/configs/experiment.yaml'))
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"✓ Saved configuration to {save_path}")


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.

    Example:
        >>> config = load_config(Path('outputs/exp_001/configs/experiment.yaml'))
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: dict[str, Any], save_path: Path) -> None:
    """
    Save data to a JSON file.

    Args:
        data (Dict[str, Any]): Data dictionary to save.
        save_path (Path): Path to save the JSON file.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"✓ Saved JSON to {save_path}")


def load_json(load_path: Path) -> dict[str, Any]:
    """
    Load data from a JSON file.

    Args:
        load_path (Path): Path to the JSON file.

    Returns:
        Dict[str, Any]: Loaded data dictionary.
    """
    with open(load_path) as f:
        data = json.load(f)
    return data


def read_config_file(config_path: Path) -> dict[str, Any]:
    """Best-effort parse of one YAML/JSON config file; ``{}`` on any failure.

    Unlike :func:`load_config` (strict, YAML-only, raises), this never raises:
    missing files, parse errors, unknown suffixes, and non-mapping documents
    all yield ``{}``. Used by run-dir scanners that must survive partial runs.
    """
    try:
        text = Path(config_path).read_text()
    except OSError:
        return {}
    try:
        suffix = Path(config_path).suffix.lower()
        if suffix in ('.yaml', '.yml'):
            meta = yaml.safe_load(text)
        elif suffix == '.json':
            meta = json.loads(text)
        else:
            return {}
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


def read_run_config(exp_dir: Path) -> dict[str, Any]:
    """Best-effort merged training metadata recorded inside a run dir.

    Reads ``configs/experiment.yaml`` and ``configs/experiment.json`` (written
    by current training via :func:`create_experiment_config` and stamped by the
    pipeline with the seed), merged with JSON winning on conflict; in practice
    both are written together with identical content. Returns ``{}`` when
    absent or unreadable — callers treat a missing key as "unknown", never as
    an error.
    """
    merged: dict[str, Any] = {}
    base = Path(exp_dir) / 'configs'
    for name in ('experiment.yaml', 'experiment.json'):
        merged.update(read_config_file(base / name))
    return merged
