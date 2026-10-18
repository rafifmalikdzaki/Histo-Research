"""Hyperparameter helpers (batch-size probe, ablation configs)."""

import torch
from datetime import datetime


def get_optimal_batch_size(model, device, input_size=(3, 128, 128)):
    """
    Dynamically determine optimal batch size based on available GPU memory
    """
    model.eval()
    torch.cuda.empty_cache()

    # Empirically probe batch sizes until memory is full
    if torch.cuda.is_available():
        # Start with batch size and increase until memory is full
        batch_size = 4
        max_batch_size = 32  # Upper limit

        while batch_size <= max_batch_size:
            try:
                # Test with current batch size
                test_input = torch.randn(batch_size, *input_size).to(device)
                with torch.no_grad():
                    _ = model(test_input)

                # If successful, try larger batch
                del test_input
                batch_size *= 2

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e

        # Return the largest successful batch size divided by 2 for safety
        return max(4, batch_size // 2)
    else:
        return 4  # Default for CPU


def generate_ablation_config(args, batch_size, device):
    """Generate comprehensive ablation study configuration"""

    # Base configuration
    config = {
        'ablation_mode': args.ablation_mode,
        'model_name': args.model_name,
        'batch_size': batch_size,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }

    # Mode-specific configurations
    mode_configs = {
        'baseline': {
            'description': 'Baseline DAE-KAN with attention mechanisms',
            'tags': ['baseline', 'attention', 'complete-model'],
            'group_name': 'baseline-experiments'
        },
        'no-attention': {
            'description': 'DAE-KAN without attention mechanisms',
            'tags': ['no-attention', 'ablation', 'attention-removed'],
            'group_name': 'attention-ablation'
        },
        'different-batch-size': {
            'description': f'DAE-KAN with batch size {batch_size}',
            'tags': ['batch-size', 'ablation', f'bs-{batch_size}'],
            'group_name': 'batch-size-ablation'
        },
        'different-lr': {
            'description': 'DAE-KAN with modified learning rate',
            'tags': ['learning-rate', 'ablation', 'lr-modified'],
            'group_name': 'learning-rate-ablation'
        },
        'different-architecture': {
            'description': f'DAE-KAN with {args.model_name} architecture',
            'tags': ['architecture', 'ablation', args.model_name],
            'group_name': 'architecture-ablation'
        },
        'custom': {
            'description': args.run_description or 'Custom ablation experiment',
            'tags': ['custom', 'ablation'] + args.ablation_tags,
            'group_name': args.ablation_group or 'custom-ablation'
        }
    }

    mode_config = mode_configs.get(args.ablation_mode, mode_configs['baseline'])

    # Generate experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{args.ablation_mode}_{args.model_name}_{timestamp}"

    # Generate W&B group
    if args.ablation_group:
        wandb_group = args.ablation_group
    else:
        wandb_group = mode_config['group_name']

    # Generate tags
    base_tags = ['dae-kan', 'optimized', 'analysis', 'histopathology']
    mode_tags = mode_config['tags']
    user_tags = args.ablation_tags

    all_tags = list(set(base_tags + mode_tags + user_tags))

    # Combine everything
    ablation_config = {
        **config,
        **mode_config,
        'experiment_name': experiment_name,
        'wandb_group': wandb_group,
        'tags': all_tags,
        'run_name': experiment_name,
        'notes': args.run_description or mode_config['description']
    }

    return ablation_config
