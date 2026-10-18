from .dae import (
    BAMOnly_DAE_KAN_Attention,
    BaselineDAE_KAN_Attention,
    DAE_KAN_Attention,
    KANOnly_DAE_KAN_Attention,
    NoBAM_DAE_KAN_Attention,
    NoEKA_DAE_KAN_Attention,
    NoKAN_DAE_KAN_Attention,
)

MODELS = {
    "dae_kan_attention": DAE_KAN_Attention,
    "baseline": BaselineDAE_KAN_Attention,
    "bam_only": BAMOnly_DAE_KAN_Attention,
    "kan_only": KANOnly_DAE_KAN_Attention,
    "no_bam": NoBAM_DAE_KAN_Attention,
    "no_kan": NoKAN_DAE_KAN_Attention,
    "no_eka": NoEKA_DAE_KAN_Attention,
}

def get_model(model_name: str):
    """
    Model factory to get a model by name.

    Args:
        model_name (str): The name of the model to get.

    Returns:
        torch.nn.Module: The model class.
    """
    model_name = model_name.lower()
    if model_name not in MODELS:
        raise ValueError(f"Unknown model name: {model_name}. Available models are: {list(MODELS.keys())}")
    return MODELS[model_name]
