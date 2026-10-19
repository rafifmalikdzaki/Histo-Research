"""Model registry: string key -> variant class (registry/factory pattern).

New ablation variants register by subclassing :class:`DualAutoencoder` and
adding one entry here; the training/eval runners never branch on model names.
"""

from __future__ import annotations

from .dae import (
    BAMOnly_DAE_KAN_Attention,
    BaselineDAE_KAN_Attention,
    DAE_KAN_Attention,
    DualAutoencoder,
    KANOnly_DAE_KAN_Attention,
    NoBAM_DAE_KAN_Attention,
    NoEKA_DAE_KAN_Attention,
    NoKAN_DAE_KAN_Attention,
)

MODELS: dict[str, type[DualAutoencoder]] = {
    "dae_kan_attention": DAE_KAN_Attention,
    "baseline": BaselineDAE_KAN_Attention,
    "bam_only": BAMOnly_DAE_KAN_Attention,
    "kan_only": KANOnly_DAE_KAN_Attention,
    "no_bam": NoBAM_DAE_KAN_Attention,
    "no_kan": NoKAN_DAE_KAN_Attention,
    "no_eka": NoEKA_DAE_KAN_Attention,
}


def get_model(model_name: str) -> type[DualAutoencoder]:
    """Return the variant class registered under ``model_name`` (case-insensitive).

    Args:
        model_name: Registry key from :data:`MODELS`.

    Returns:
        The :class:`DualAutoencoder` subclass (uninstantiated).

    Raises:
        ValueError: If the name is not registered.
    """
    key = model_name.lower()
    if key not in MODELS:
        raise ValueError(
            f"Unknown model name: {model_name}. Available models are: {list(MODELS.keys())}"
        )
    return MODELS[key]


__all__ = ["MODELS", "get_model"]
