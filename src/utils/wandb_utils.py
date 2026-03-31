"""
Weights & Biases Utilities for DAE-KAN Experiments

This module provides standardized W&B configuration and logging utilities
to ensure consistent experiment tracking across all scripts.

Usage:
    from utils.wandb_utils import get_wandb_config, setup_wandb_logger
    
    # Load configuration
    config = get_wandb_config()
    
    # Setup logger with standard settings
    logger = setup_wandb_logger(
        model_name='dae_kan_attention',
        dataset='HeparUnifiedPNG',
        seed=42,
        custom_config={...}
    )
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml


def get_wandb_config(config_path: str = "config/wandb_config.yaml") -> Dict[str, Any]:
    """
    Load W&B configuration from YAML file.
    
    Args:
        config_path: Path to W&B configuration file.
    
    Returns:
        Dictionary containing W&B configuration.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # Try alternative paths
        for alt_path in [
            Path(__file__).parent.parent / "config" / "wandb_config.yaml",
            Path("config") / "wandb_config.yaml",
        ]:
            if alt_path.exists():
                config_file = alt_path
                break
        else:
            raise FileNotFoundError(
                f"W&B config not found at {config_path}. "
                "Please create it or set WANDB_PROJECT environment variable."
            )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if set
    if os.getenv('WANDB_ENTITY'):
        config['wandb']['entity'] = os.getenv('WANDB_ENTITY')
    if os.getenv('WANDB_PROJECT'):
        config['wandb']['project'] = os.getenv('WANDB_PROJECT')
    
    return config


def get_group_for_model(model_name: str, config: Optional[Dict] = None) -> str:
    """
    Get the W&B group name for a given model.
    
    Args:
        model_name: Name of the model.
        config: Optional W&B config (loads default if None).
    
    Returns:
        Group name string.
    """
    if config is None:
        config = get_wandb_config()
    
    model_to_group = config.get('model_to_group', {})
    group_key = model_to_group.get(model_name, 'main_experiments')
    
    groups = config.get('groups', {})
    return groups.get(group_key, f"00_{group_key}")


def get_tags_for_model(model_name: str, 
                       custom_tags: Optional[List[str]] = None,
                       config: Optional[Dict] = None) -> List[str]:
    """
    Get standard tags for a given model.
    
    Args:
        model_name: Name of the model.
        custom_tags: Optional custom tags to add.
        config: Optional W&B config (loads default if None).
    
    Returns:
        List of tags.
    """
    if config is None:
        config = get_wandb_config()
    
    # Start with default tags
    tags = list(config.get('default_tags', []))
    
    # Add model-specific tags
    model_tags = config.get('model_tags', {})
    if model_name in model_tags:
        tags.extend(model_tags[model_name])
    
    # Add custom tags
    if custom_tags:
        tags.extend(custom_tags)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    
    return unique_tags


def format_run_name(model_name: str, dataset: str, seed: int,
                   config: Optional[Dict] = None,
                   additional_info: Optional[str] = None) -> str:
    """
    Format run name according to W&B convention.
    
    Args:
        model_name: Name of the model.
        dataset: Dataset name.
        seed: Random seed.
        config: Optional W&B config (loads default if None).
        additional_info: Optional additional info to append.
    
    Returns:
        Formatted run name.
    """
    if config is None:
        config = get_wandb_config()
    
    # Get format string from config
    format_str = config.get('run_name_format', "{model}_{dataset}_seed{seed}")
    
    # Format the name
    run_name = format_str.format(
        model=model_name,
        dataset=dataset,
        seed=seed
    )
    
    # Add additional info if provided
    if additional_info:
        run_name = f"{run_name}_{additional_info}"
    
    return run_name


def setup_wandb_logger(model_name: str,
                      dataset: str,
                      seed: int,
                      custom_config: Optional[Dict] = None,
                      custom_tags: Optional[List[str]] = None,
                      group_override: Optional[str] = None,
                      **kwargs) -> 'WandbLogger':
    """
    Setup W&B logger with standardized configuration.
    
    This function:
    1. Loads W&B configuration
    2. Determines correct project, group, and tags
    3. Formats run name according to convention
    4. Creates and returns WandbLogger
    
    Args:
        model_name: Name of the model being trained.
        dataset: Dataset name.
        seed: Random seed.
        custom_config: Custom configuration to log.
        custom_tags: Custom tags to add.
        group_override: Override automatic group selection.
        **kwargs: Additional arguments to pass to WandbLogger.
    
    Returns:
        Configured WandbLogger instance.
    
    Example:
        >>> logger = setup_wandb_logger(
        ...     model_name='dae_kan_attention',
        ...     dataset='HeparUnifiedPNG',
        ...     seed=42,
        ...     custom_config={'epochs': 30, 'batch_size': 8}
        ... )
    """
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:
        raise ImportError(
            "pytorch_lightning not installed. "
            "Install with: pip install pytorch-lightning"
        )
    
    # Load configuration
    config = get_wandb_config()
    wandb_config = config.get('wandb', {})
    
    # Get entity and project
    entity = wandb_config.get('entity')
    project = wandb_config.get('project')
    
    # Check if entity is still placeholder
    if entity == "PLACEHOLDER_REPLACE_WITH_YOUR_USERNAME":
        import warnings
        warnings.warn(
            "W&B entity not configured. Please edit config/wandb_config.yaml "
            "and replace 'PLACEHOLDER_REPLACE_WITH_YOUR_USERNAME' with your "
            "W&B username or team name. Logging to personal workspace."
        )
        entity = None  # Will use personal workspace
    
    # Get group
    group = group_override or get_group_for_model(model_name, config)
    
    # Get tags
    tags = get_tags_for_model(model_name, custom_tags, config)
    
    # Format run name
    run_name = format_run_name(model_name, dataset, seed, config)
    
    # Prepare config to log
    logged_config = {
        'model_name': model_name,
        'dataset': dataset,
        'seed': seed,
        'run_name': run_name,
        'group': group,
    }
    
    if custom_config:
        logged_config.update(custom_config)
    
    # Create logger
    logger = WandbLogger(
        project=project,
        entity=entity,
        name=run_name,
        group=group,
        tags=tags,
        config=logged_config,
        **kwargs
    )
    
    print(f"✓ W&B Logger initialized:")
    print(f"  Project: {project}")
    print(f"  Entity: {entity or '(personal workspace)'}")
    print(f"  Group: {group}")
    print(f"  Run name: {run_name}")
    print(f"  Tags: {', '.join(tags[:5])}{'...' if len(tags) > 5 else ''}")
    
    return logger


def log_metric_with_namespace(logger: 'WandbLogger',
                             namespace: str,
                             metric_name: str,
                             value: float,
                             step: Optional[int] = None,
                             **kwargs):
    """
    Log a metric with proper namespace.
    
    Args:
        logger: W&B logger instance.
        namespace: Metric namespace (e.g., 'train', 'val', 'clustering').
        metric_name: Name of the metric.
        value: Metric value.
        step: Optional step number.
        **kwargs: Additional arguments for logger.log_metrics.
    
    Example:
        >>> log_metric_with_namespace(
        ...     logger, 'train', 'loss', 0.0234, step=epoch
        ... )
    """
    full_name = f"{namespace}/{metric_name}"
    logger.log_metrics({full_name: value}, step=step, **kwargs)


def get_standard_metric_name(metric_type: str, metric_name: str,
                            config: Optional[Dict] = None) -> str:
    """
    Get standard metric name with namespace.
    
    Args:
        metric_type: Type of metric (e.g., 'training', 'clustering').
        metric_name: Specific metric name.
        config: Optional W&B config.
    
    Returns:
        Standard metric name with namespace.
    
    Example:
        >>> get_standard_metric_name('clustering', 'silhouette')
        'clustering/silhouette_score'
    """
    if config is None:
        config = get_wandb_config()
    
    namespaces = config.get('metric_namespaces', {})
    namespace = namespaces.get(metric_type, metric_type)
    
    standard_metrics = config.get('standard_metrics', {})
    
    # Try exact match first
    key = f"{metric_type}_{metric_name}"
    if key in standard_metrics:
        return standard_metrics[key]
    
    # Fall back to namespace/metric_name format
    return f"{namespace}/{metric_name}"


def check_wandb_login() -> bool:
    """
    Check if W&B is properly configured and logged in.
    
    Returns:
        True if configured, False otherwise.
    """
    try:
        import wandb
        
        # Check if API key is set
        if not wandb.api.api_key:
            print("⚠️  W&B not logged in. Run: wandb login")
            return False
        
        return True
    
    except ImportError:
        print("⚠️  wandb not installed. Run: pip install wandb")
        return False


def print_wandb_dashboard_url(project: str, entity: Optional[str] = None) -> None:
    """
    Print the URL to access the W&B dashboard.
    
    Args:
        project: Project name.
        entity: Entity name (optional).
    """
    import wandb
    
    if entity:
        url = f"https://wandb.ai/{entity}/{project}"
    else:
        # Get current user's entity
        try:
            api = wandb.Api()
            entity = api.viewer['entity']
            url = f"https://wandb.ai/{entity}/{project}"
        except:
            url = f"https://wandb.ai/{project}"
    
    print(f"\n📊 View results on W&B: {url}")
    print(f"💡 Tip: Use W&B Groups to organize experiments by category\n")


if __name__ == "__main__":
    # Test W&B utilities
    print("Testing W&B utilities...\n")
    
    # Load config
    config = get_wandb_config()
    print(f"✓ Loaded W&B config")
    print(f"  Project: {config['wandb']['project']}")
    print(f"  Entity: {config['wandb']['entity']}")
    print(f"  Default tags: {len(config['wandb']['default_tags'])}")
    
    # Test run name formatting
    run_name = format_run_name('dae_kan_attention', 'HeparUnifiedPNG', 42, config)
    print(f"\n✓ Example run name: {run_name}")
    
    # Test group assignment
    group = get_group_for_model('dae_kan_attention', config)
    print(f"✓ Group for dae_kan_attention: {group}")
    
    # Test tags
    tags = get_tags_for_model('dae_kan_attention', config=config)
    print(f"✓ Tags for dae_kan_attention: {tags}")
    
    # Test metric naming
    metric_name = get_standard_metric_name('clustering', 'silhouette', config)
    print(f"✓ Standard metric name: {metric_name}")
    
    print("\n✅ All W&B utilities tested successfully!")
