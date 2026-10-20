"""Experiment directory layout and the default hyperparameter set."""

from pathlib import Path
from typing import Any

from histodae.io.run_config import load_config, load_json, save_config, save_json
from histodae.utils.seeding import get_experiment_timestamp

__all__ = [
    "create_experiment_config",
    "create_experiment_directory",
    "get_default_hyperparameters",
    "get_experiment_config_from_dir",
]


def create_experiment_directory(
    base_dir: str = "outputs",
    experiment_name: str | None = None,
    timestamp: str | None = None
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


def get_default_hyperparameters() -> dict[str, Any]:
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
    custom_params: dict[str, Any] | None = None,
    base_dir: str = "outputs"
) -> tuple[Path, dict[str, Any]]:
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


def get_experiment_config_from_dir(exp_dir: Path) -> dict[str, Any]:
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
