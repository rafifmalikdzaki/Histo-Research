"""Global seeding and run timestamps.

These are the primitives every reproducibility guarantee rests on: one
process-wide seed for python/numpy/torch, and a stable timestamp used to
name run directories.
"""

import os
import random
from datetime import datetime

import numpy as np
import torch

__all__ = ["get_experiment_timestamp", "set_seed"]


def set_seed(seed: int = 42) -> None:
    """
    Set global seeds for reproducibility across Python, NumPy, and PyTorch.

    This function ensures that all random operations are deterministic
    by setting seeds for:
    - Python's random module
    - NumPy's random number generator
    - PyTorch's CPU and CUDA random number generators
    - PyTorch's cuDNN backend (deterministic mode)

    Args:
        seed (int): The seed value to use. Default is 42.

    Example:
        >>> set_seed(42)
        ✓ Set seed 42 for reproducible experiments
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Enable deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable auto-tuning for reproducibility

    # Set environment variable for additional reproducibility
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"✓ Set seed {seed} for reproducible experiments")


def get_experiment_timestamp() -> str:
    """
    Generate a timestamp string for experiment naming.

    Returns:
        str: Timestamp in format 'YYYYMMDD_HHMMSS'

    Example:
        >>> get_experiment_timestamp()
        '20260331_143052'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')
