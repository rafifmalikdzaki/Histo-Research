"""histodae — attention-enhanced dual autoencoder clustering for liver histopathology.

This is the single public namespace of the project. Everything lives under
``histodae.<layer>``:

======================  ==========================================================
Layer                   Contents
======================  ==========================================================
``histodae.data``       dataset resolution, splits, tile metadata parsing
``histodae.clustering`` robust clustering metrics, normalization, DEC/IDEC loss
``histodae.models``     dual autoencoder family, KAN + attention building blocks
``histodae.training``   Lightning module, optimizer groups, embedding IO, reports
``histodae.analysis``   in-training attention analysis and visualisation
``histodae.stats``      effect sizes, seed pairing, result tables
``histodae.io``         run-artifact IO: raw metric loading, run discovery
``histodae.utils``      reproducibility (seeding) and W&B integration
``histodae.cli``        command-line entry points
``histodae.config``     packaged YAML defaults
======================  ==========================================================

Attributes are resolved lazily (PEP 562) so that ``import histodae`` does not
pull in torch, lightning or wandb until a heavy symbol is actually touched.
"""

from importlib import import_module
from typing import Any

# Public symbol -> defining submodule.
_EXPORTS: dict[str, str] = {
    # data
    "ImageDataset": "histodae.data.datasets",
    "create_dataset": "histodae.data.datasets",
    "load_metadata": "histodae.data.datasets",
    "parse_hepar_filename": "histodae.data.metadata",
    "CANONICAL_GROUPS": "histodae.data.metadata",
    # clustering
    "NORMALIZERS": "histodae.clustering.metrics",
    "external_metrics": "histodae.clustering.metrics",
    "internal_metrics": "histodae.clustering.metrics",
    "normalize_embeddings": "histodae.clustering.metrics",
    "xie_beni": "histodae.clustering.metrics",
    "DECClusteringLoss": "histodae.clustering.dec",
    "LatentProjectionHead": "histodae.clustering.dec",
    # models
    "MODELS": "histodae.models.factory",
    "get_model": "histodae.models.factory",
    "DualAutoencoder": "histodae.models.dae",
    "VARIANT_CONFIGS": "histodae.models.dae",
    # io
    "load_data": "histodae.io.reporting",
    "load_raw_metrics": "histodae.io.reporting",
    "find_experiment_runs": "histodae.io.run_discovery",
    "load_experiment_metrics": "histodae.io.run_discovery",
    # stats
    "METRICS": "histodae.stats.tables",
    "compute_cohens_d": "histodae.stats.metrics",
    "format_mean_std": "histodae.stats.metrics",
    "summarize_values": "histodae.stats.metrics",
    # utils
    "set_seed": "histodae.utils.reproducibility",
    "read_run_config": "histodae.utils.reproducibility",
    "read_config_file": "histodae.utils.reproducibility",
}

__all__ = sorted(_EXPORTS) + ["__version__"]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    """Resolve a public symbol on first use (PEP 562)."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
