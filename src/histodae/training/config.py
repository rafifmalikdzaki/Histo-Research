"""Translation between the experiment YAML and training CLI flags.

The YAML (``histodae/config/experiment.yaml``) is the single source of truth
for *experiment* hyperparameters; the training CLI owns the *runtime* knobs.
:func:`config_defaults` projects the YAML onto the argparse flag names so a
config file can seed the parser defaults, with explicit CLI flags still
taking precedence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from histodae.config import EXPERIMENT_CONFIG, load_config


def _get(cfg: dict[str, Any], *path: str, default: Any = None) -> Any:
    """Fetch a nested value, returning ``default`` when any level is missing."""
    node: Any = cfg
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _load_config_file(config_path: str | Path) -> dict[str, Any]:
    """Load a config from a path, or from a packaged config by name.

    ``--config-path`` accepts either a filesystem path or the bare name of the
    packaged experiment config, so the shipped defaults can
    be used without depending on the working directory.
    """
    import yaml

    path = Path(config_path)
    if path.is_file():
        config = yaml.safe_load(path.read_text())
        if not isinstance(config, dict):
            raise ValueError(f"Experiment config must be a YAML mapping: {path}")
        return config

    if str(config_path) in ("experiment", EXPERIMENT_CONFIG):
        return load_config(EXPERIMENT_CONFIG)

    raise FileNotFoundError(
        f"Config not found: {config_path!r}; supply a YAML file or 'experiment'"
    )


def config_defaults(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return argparse defaults derived from an experiment config file.

    Only keys that map onto an existing CLI flag are returned, so this can be
    passed straight to ``parser.set_defaults()``. Unknown / unmapped YAML
    keys are ignored rather than raising, which keeps the config file
    forward-compatible.
    """
    if config_path is None:
        cfg = load_config(EXPERIMENT_CONFIG)
    else:
        cfg = _load_config_file(config_path)

    defaults: dict[str, Any] = {}

    def put(flag: str, value: Any) -> None:
        if value is not None:
            defaults[flag] = value

    put("seed", _get(cfg, "experiment", "seed"))
    put("model_name", _get(cfg, "model", "name"))
    put("batch_size", _get(cfg, "data", "batch_size"))
    put("num_workers", _get(cfg, "data", "num_workers"))
    put("max_epochs", _get(cfg, "training", "epochs"))
    put("min_epochs", _get(cfg, "training", "min_epochs"))
    put("learning_rate", _get(cfg, "training", "learning_rate"))
    put("latent_dim", _get(cfg, "model", "latent_dim"))
    put("n_clusters", _get(cfg, "clustering", "n_clusters"))
    put("cluster_weight", _get(cfg, "clustering", "objective", "cluster_weight"))
    put("recon_weight", _get(cfg, "clustering", "objective", "recon_weight"))
    put("cluster_warmup_epochs", _get(cfg, "clustering", "objective", "cluster_warmup_epochs"))
    put("output_dir", _get(cfg, "output", "base_dir"))

    return defaults


__all__ = ["config_defaults"]
