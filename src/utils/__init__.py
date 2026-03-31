"""
Utilities for DAE-KAN Experiments

This package provides utility functions for:
- Reproducibility (seed setting, experiment tracking)
- Configuration management
- Metrics computation
- Data processing
- Weights & Biases integration
"""

from .reproducibility import (
    set_seed,
    get_experiment_timestamp,
    create_experiment_directory,
    save_config,
    load_config,
    save_json,
    load_json,
    get_default_hyperparameters,
    create_experiment_config,
    get_experiment_config_from_dir,
)

from .wandb_utils import (
    get_wandb_config,
    get_group_for_model,
    get_tags_for_model,
    format_run_name,
    setup_wandb_logger,
    log_metric_with_namespace,
    get_standard_metric_name,
    check_wandb_login,
    print_wandb_dashboard_url,
)

__all__ = [
    # Reproducibility
    'set_seed',
    'get_experiment_timestamp',
    'create_experiment_directory',
    'save_config',
    'load_config',
    'save_json',
    'load_json',
    'get_default_hyperparameters',
    'create_experiment_config',
    'get_experiment_config_from_dir',
    # W&B utilities
    'get_wandb_config',
    'get_group_for_model',
    'get_tags_for_model',
    'format_run_name',
    'setup_wandb_logger',
    'log_metric_with_namespace',
    'get_standard_metric_name',
    'check_wandb_login',
    'print_wandb_dashboard_url',
]
