"""histodae — installed package surface.

Thin re-exports over the ``src``-layout modules (importable once the project
is installed, e.g. ``uv pip install -e .``). Frozen JOIV scripts keep using
the ``src.*`` / ``models.*`` paths; both resolve to the same modules.
"""

from clustering_metrics import (
    NORMALIZERS,
    external_metrics,
    internal_metrics,
    normalize_embeddings,
    xie_beni,
)
from histodata import ImageDataset, create_dataset, load_metadata
from models.factory import MODELS, get_model
from utils.reporting import load_data, load_raw_metrics
from utils.stats import compute_cohens_d, format_mean_std, summarize_values
from utils.stats_tables import METRICS

__all__ = [
    "MODELS",
    "NORMALIZERS",
    "ImageDataset",
    "compute_cohens_d",
    "create_dataset",
    "external_metrics",
    "format_mean_std",
    "get_model",
    "internal_metrics",
    "load_data",
    "load_metadata",
    "load_raw_metrics",
    "METRICS",
    "normalize_embeddings",
    "summarize_values",
    "xie_beni",
]
