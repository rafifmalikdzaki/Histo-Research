"""
Reproducibility Utilities for DAE-KAN Experiments

This module provides utilities for ensuring reproducible experiments
by setting global seeds and managing experiment configurations.
"""

import random
import os
import numpy as np
import torch
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


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


def create_experiment_directory(
    base_dir: str = "outputs",
    experiment_name: Optional[str] = None,
    timestamp: Optional[str] = None
) -> Path:
    """
    Create a timestamped experiment directory with organized subdirectories.
    
    Creates the following structure:
    ```
    outputs/
    └── exp_<timestamp>_<experiment_name>/
        ├── checkpoints/
        ├── metrics/
        ├── visualizations/
        ├── configs/
        ├── embeddings/
        └── logs/
    ```
    
    Args:
        base_dir (str): Base directory for experiments. Default is "outputs".
        experiment_name (Optional[str]): Optional experiment name to append.
        timestamp (Optional[str]): Optional pre-generated timestamp.
    
    Returns:
        Path: Path to the created experiment directory.
    
    Example:
        >>> exp_dir = create_experiment_directory(experiment_name="dae_kan_liver")
        >>> print(exp_dir)
        PosixPath('outputs/exp_20260331_143052_dae_kan_liver')
    """
    if timestamp is None:
        timestamp = get_experiment_timestamp()
    
    # Create experiment directory name
    if experiment_name:
        exp_dir_name = f"exp_{timestamp}_{experiment_name}"
    else:
        exp_dir_name = f"exp_{timestamp}"
    
    exp_path = Path(base_dir) / exp_dir_name
    
    # Create subdirectories
    subdirs = ['checkpoints', 'metrics', 'visualizations', 'configs', 'embeddings', 'logs']
    for subdir in subdirs:
        (exp_path / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created experiment directory: {exp_path}")
    return exp_path


def save_config(config: Dict[str, Any], save_path: Path) -> None:
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


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load experiment configuration from a YAML file.
    
    Args:
        config_path (Path): Path to the configuration file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    
    Example:
        >>> config = load_config(Path('outputs/exp_001/configs/experiment.yaml'))
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Dict[str, Any], save_path: Path) -> None:
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


def load_json(load_path: Path) -> Dict[str, Any]:
    """
    Load data from a JSON file.
    
    Args:
        load_path (Path): Path to the JSON file.
    
    Returns:
        Dict[str, Any]: Loaded data dictionary.
    """
    with open(load_path, 'r') as f:
        data = json.load(f)
    return data


def get_default_hyperparameters() -> Dict[str, Any]:
    """
    Get default hyperparameters for DAE-KAN experiments.
    
    Returns:
        Dict[str, Any]: Dictionary containing all default hyperparameters.
    
    Example:
        >>> params = get_default_hyperparameters()
        >>> print(params['latent_dim'])
        128
    """
    return {
        # Model architecture
        'latent_dim': 128,
        'n_clusters': 5,
        
        # KAN parameters
        'kan_grid_size': 3,
        'kan_spline_order': 2,
        
        # Attention parameters
        'eca_kernel_size': 3,
        'bam_reduction': 16,
        
        # Training parameters
        'learning_rate': 0.002,
        'batch_size': 8,
        'epochs': 30,
        'weight_decay': 1e-5,
        
        # Optimization
        'optimizer': 'adam',
        'scheduler': 'reduce_on_plateau',
        'scheduler_factor': 0.7,
        'scheduler_patience': 2,
        
        # Reproducibility
        'seed': 42,
        
        # Data
        'image_size': 128,
        'num_workers': 4,
        
        # Logging
        'log_frequency': 10,
        'checkpoint_frequency': 5,
    }


def create_experiment_config(
    experiment_name: str,
    custom_params: Optional[Dict[str, Any]] = None,
    base_dir: str = "outputs"
) -> tuple[Path, Dict[str, Any]]:
    """
    Create a complete experiment configuration with all hyperparameters.
    
    This function:
    1. Creates an experiment directory
    2. Loads default hyperparameters
    3. Overrides with custom parameters
    4. Saves the configuration
    5. Returns the path and config dictionary
    
    Args:
        experiment_name (str): Name of the experiment.
        custom_params (Optional[Dict[str, Any]]): Custom parameters to override defaults.
        base_dir (str): Base directory for experiments.
    
    Returns:
        tuple[Path, Dict[str, Any]]: Path to experiment directory and config dictionary.
    
    Example:
        >>> exp_dir, config = create_experiment_config(
        ...     "dae_kan_test",
        ...     custom_params={'latent_dim': 256, 'seed': 123}
        ... )
    """
    # Create experiment directory
    exp_dir = create_experiment_directory(base_dir, experiment_name)
    
    # Get default hyperparameters
    config = get_default_hyperparameters()
    
    # Override with custom parameters
    if custom_params:
        config.update(custom_params)
    
    # Add experiment metadata
    config['experiment_name'] = experiment_name
    config['timestamp'] = get_experiment_timestamp()
    config['experiment_dir'] = str(exp_dir)
    
    # Save configuration
    config_path = exp_dir / 'configs' / 'experiment.yaml'
    save_config(config, config_path)
    
    # Also save as JSON for easier programmatic access
    config_json_path = exp_dir / 'configs' / 'experiment.json'
    save_json(config, config_json_path)
    
    return exp_dir, config


def get_experiment_config_from_dir(exp_dir: Path) -> Dict[str, Any]:
    """
    Load experiment configuration from an existing experiment directory.
    
    Args:
        exp_dir (Path): Path to the experiment directory.
    
    Returns:
        Dict[str, Any]: Configuration dictionary.
    
    Raises:
        FileNotFoundError: If no configuration file is found.
    """
    # Try YAML first
    yaml_path = exp_dir / 'configs' / 'experiment.yaml'
    if yaml_path.exists():
        return load_config(yaml_path)
    
    # Try JSON
    json_path = exp_dir / 'configs' / 'experiment.json'
    if json_path.exists():
        return load_json(json_path)
    
    raise FileNotFoundError(
        f"No configuration file found in {exp_dir}/configs/"
    )


if __name__ == "__main__":
    # Test the reproducibility utilities
    print("Testing reproducibility utilities...\n")
    
    # Test seed setting
    set_seed(42)
    
    # Test experiment directory creation
    exp_dir, config = create_experiment_config(
        "test_experiment",
        custom_params={'latent_dim': 256}
    )
    
    print(f"\n✓ Experiment directory: {exp_dir}")
    print(f"✓ Configuration saved with {len(config)} parameters")
    
    # Test loading config
    loaded_config = get_experiment_config_from_dir(exp_dir)
    print(f"✓ Loaded configuration: {loaded_config['experiment_name']}")
