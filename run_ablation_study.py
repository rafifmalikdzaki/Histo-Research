#!/usr/bin/env python3
"""
Ablation Study Runner for DAE-KAN Models

This script provides an easy way to run comprehensive ablation studies
with proper organization in both local directories and W&B.

Usage Examples:
    # Run baseline experiment
    python run_ablation_study.py --mode baseline

    # Run batch size ablation
    python run_ablation_study.py --mode different-batch-size --batch-sizes 4 8 16

    # Run architecture comparison
    python run_ablation_study.py --mode different-architecture --models dae_kan_attention kan_conv

    # Run custom ablation
    python run_ablation_study.py --mode custom --group "my-experiment" --tags "test" "debug"
"""

import argparse
import subprocess
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Configuration for different ablation modes
ABLATION_CONFIGS = {
    'baseline': {
        'description': 'Baseline DAE-KAN with standard configuration',
        'base_args': [
            '--model-name', 'dae_kan_attention',
            '--batch-size', '8',
            '--max-epochs', '1',  # Default to 1 epoch for quick testing
            '--analysis-freq', '20',   # More frequent for 1 epoch
            '--wandb-metrics-freq', '5',
            '--wandb-viz-freq', '20',
            '--wandb-paper-freq', '50'
        ]
    },
    'no-attention': {
        'description': 'DAE-KAN without attention mechanisms',
        'base_args': [
            '--model-name', 'dae_kan_no_attention',  # Would need to implement this model
            '--batch-size', '8',
            '--analysis-freq', '100'
        ]
    },
    'different-batch-size': {
        'description': 'Ablation study on different batch sizes',
        'base_args': [
            '--model-name', 'dae_kan_attention',
            '--max-epochs', '1',  # Quick testing
            '--analysis-freq', '20',
            '--wandb-metrics-freq', '5',
            '--wandb-viz-freq', '20',
            '--wandb-paper-freq', '50'
        ],
        'parameter_ranges': {
            'batch-size': [4, 8, 16, 32]
        }
    },
    'different-learning-rate': {
        'description': 'Ablation study on different learning rates',
        'base_args': [
            '--model-name', 'dae_kan_attention',
            '--batch-size', '8',
            '--max-epochs', '1',  # Quick testing
            '--analysis-freq', '20',
            '--wandb-metrics-freq', '5',
            '--wandb-viz-freq', '20'
        ],
        'parameter_ranges': {
            'learning-rate': [0.001, 0.002, 0.005, 0.01]
        }
    },
    'different-architecture': {
        'description': 'Ablation study on different model architectures',
        'base_args': [
            '--batch-size', '8',
            '--max-epochs', '1',  # Quick testing
            '--analysis-freq', '20',
            '--wandb-metrics-freq', '5',
            '--wandb-viz-freq', '20'
        ],
        'parameter_ranges': {
            'model-name': ['dae_kan_attention', 'kan_conv']  # Add more models as available
        }
    }
}


def generate_experiment_configs(mode: str, **kwargs) -> List[Dict[str, Any]]:
    """Generate experiment configurations for the specified mode"""

    if mode not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation mode: {mode}")

    config = ABLATION_CONFIGS[mode]
    base_args = config['base_args'].copy()

    # Add custom arguments
    if 'extra_args' in kwargs:
        base_args.extend(kwargs['extra_args'])

    # Generate parameter combinations
    experiments = []

    if 'parameter_ranges' in config:
        # Generate experiments for parameter ranges
        param_ranges = config['parameter_ranges']

        # Override with provided parameters
        for param, values in kwargs.items():
            if param in param_ranges:
                param_ranges[param] = values

        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())

        if len(param_names) == 1:
            # Simple case: one parameter
            for value in param_values[0]:
                args = base_args.copy()
                args.extend([f'--{param_names[0]}', str(value)])

                experiment_name = f"{mode}_{param_names[0]}_{value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                experiments.append({
                    'name': experiment_name,
                    'args': args,
                    'parameters': {param_names[0]: value}
                })
        else:
            # Multiple parameters - generate combinations
            import itertools
            for combination in itertools.product(*param_values):
                args = base_args.copy()

                param_dict = {}
                for i, param_name in enumerate(param_names):
                    value = combination[i]
                    args.extend([f'--{param_name}', str(value)])
                    param_dict[param_name] = value

                # Generate experiment name
                param_str = '_'.join([f"{k}-{v}" for k, v in param_dict.items()])
                experiment_name = f"{mode}_{param_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                experiments.append({
                    'name': experiment_name,
                    'args': args,
                    'parameters': param_dict
                })
    else:
        # Single experiment
        experiment_name = f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiments.append({
            'name': experiment_name,
            'args': base_args,
            'parameters': {}
        })

    return experiments


def run_single_experiment(experiment: Dict[str, Any], base_command: List[str],
                         project_name: str, dry_run: bool = False, epoch_args: Dict = None) -> bool:
    """Run a single experiment"""

    cmd = base_command + experiment['args']

    # Add epoch control arguments
    if epoch_args:
        cmd.extend(['--max-epochs', str(epoch_args['max_epochs'])])
        cmd.extend(['--min-epochs', str(epoch_args['min_epochs'])])

        if epoch_args.get('early_stopping'):
            cmd.extend(['--early-stopping'])
            cmd.extend(['--patience', str(epoch_args['patience'])])

        if epoch_args.get('fast_mode'):
            cmd.extend(['--fast-mode'])

    cmd.extend([
        '--project-name', project_name,
        '--ablation-mode', 'custom',  # Use custom for generated experiments
        '--experiment-name', experiment['name'],
        '--run-description', f"Ablation study: {json.dumps(experiment['parameters'])}"
    ])

    print(f"\n{'='*80}")
    print(f"🚀 Running Experiment: {experiment['name']}")
    print(f"📋 Parameters: {experiment['parameters']}")
    print(f"🔧 Command: {' '.join(cmd)}")

    if dry_run:
        print("🔍 DRY RUN - Not executing")
        return True

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ Experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed with return code {e.returncode}")
        print(f"Error output: {e}")
        return False


def save_ablation_plan(experiments: List[Dict[str, Any]], mode: str,
                       project_name: str, output_file: str = None):
    """Save the ablation study plan to a file"""

    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"ablation_plan_{mode}_{timestamp}.json"

    plan = {
        'mode': mode,
        'project_name': project_name,
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(experiments),
        'experiments': experiments
    }

    with open(output_file, 'w') as f:
        json.dump(plan, f, indent=2)

    print(f"📝 Ablation plan saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description='DAE-KAN Ablation Study Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run baseline experiment
  python run_ablation_study.py --mode baseline

  # Run batch size ablation with custom sizes
  python run_ablation_study.py --mode different-batch-size --batch-sizes 4 8 16 32

  # Run architecture comparison with custom models
  python run_ablation_study.py --mode different-architecture --models dae_kan_attention kan_conv

  # Run custom experiment with specific settings
  python run_ablation_study.py --mode custom --group "my-study" --tags "test" "debug" \\
    --extra-args "--batch-size 16 --analysis-freq 50"

  # Dry run to see what would be executed
  python run_ablation_study.py --mode different-batch-size --dry-run
        """
    )

    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                       choices=list(ABLATION_CONFIGS.keys()),
                       help='Ablation study mode')

    # Project configuration
    parser.add_argument('--project-name', type=str,
                       default='dae-kan-ablation-study',
                       help='W&B project name for the ablation study')

    # Custom experiment settings
    parser.add_argument('--experiment-name', type=str,
                       help='Custom name for the experiment group')
    parser.add_argument('--group', type=str,
                       help='W&B group for organizing related experiments')
    parser.add_argument('--tags', nargs='+', default=[],
                       help='Additional tags for W&B organization')
    parser.add_argument('--description', type=str,
                       help='Description for the ablation study')
    parser.add_argument('--extra-args', type=str, default='',
                       help='Additional arguments to pass to training script (space-separated)')

    # Parameter ranges for different modes
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       help='Batch sizes to test (for different-batch-size mode)')
    parser.add_argument('--learning-rates', type=float, nargs='+',
                       help='Learning rates to test (for different-learning-rate mode)')
    parser.add_argument('--models', type=str, nargs='+',
                       help='Models to test (for different-architecture mode)')

    # Training duration control
    parser.add_argument('--max-epochs', type=int, default=1,
                       help='Maximum number of epochs for each experiment')
    parser.add_argument('--min-epochs', type=int, default=1,
                       help='Minimum number of epochs for each experiment')
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping for faster convergence')
    parser.add_argument('--patience', type=int, default=3,
                       help='Patience for early stopping (epochs)')
    parser.add_argument('--fast-mode', action='store_true',
                       help='Ultra-fast mode: minimal analysis, quick training')

    # Execution control
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be executed without running')
    parser.add_argument('--save-plan', action='store_true',
                       help='Save the ablation plan to a file')
    parser.add_argument('--base-command', type=str,
                       default='python src/pl_training_with_analysis_and_optimization.py',
                       help='Base command for running experiments')

    args = parser.parse_args()

    print("🔬 DAE-KAN Ablation Study Runner")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Project: {args.project_name}")

    # Prepare parameters for configuration generation
    generation_kwargs = {}

    if args.extra_args:
        generation_kwargs['extra_args'] = args.extra_args.split()

    # Add mode-specific parameters
    if args.mode == 'different-batch-size' and args.batch_sizes:
        generation_kwargs['batch-size'] = args.batch_sizes
    elif args.mode == 'different-learning-rate' and args.learning_rates:
        generation_kwargs['learning-rate'] = args.learning_rates
    elif args.mode == 'different-architecture' and args.models:
        generation_kwargs['model-name'] = args.models

    # Generate experiment configurations
    experiments = generate_experiment_configs(args.mode, **generation_kwargs)

    print(f"📋 Generated {len(experiments)} experiment configurations")

    # Save ablation plan
    plan_file = None
    if args.save_plan:
        plan_file = save_ablation_plan(experiments, args.mode, args.project_name)

    # Show experiment overview
    print(f"\n📊 Experiment Overview:")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i:2d}. {exp['name']}")
        print(f"      Parameters: {exp['parameters']}")

    if not args.dry_run:
        print(f"\n🚀 Starting ablation study...")
        print(f"Training configuration:")
        print(f"  - Max epochs: {args.max_epochs}")
        print(f"  - Min epochs: {args.min_epochs}")
        print(f"  - Early stopping: {args.early_stopping}")
        if args.early_stopping:
            print(f"  - Patience: {args.patience}")
        print(f"  - Fast mode: {args.fast_mode}")
        print(f"Note: This will run {len(experiments)} experiments sequentially.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            return

    # Prepare epoch arguments
    epoch_args = {
        'max_epochs': args.max_epochs,
        'min_epochs': args.min_epochs,
        'early_stopping': args.early_stopping,
        'patience': args.patience,
        'fast_mode': args.fast_mode
    }

    # Run experiments
    base_command = args.base_command.split()
    successful_runs = 0
    failed_runs = 0

    for i, experiment in enumerate(experiments, 1):
        print(f"\n📍 Progress: {i}/{len(experiments)}")

        if run_single_experiment(experiment, base_command, args.project_name, args.dry_run, epoch_args):
            successful_runs += 1
        else:
            failed_runs += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 Ablation Study Summary")
    print(f"{'='*80}")
    print(f"Mode: {args.mode}")
    print(f"Total Experiments: {len(experiments)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")

    if plan_file:
        print(f"Plan saved to: {plan_file}")

    if not args.dry_run:
        print(f"\n💡 To compare results, run:")
        print(f"python ablation_study_comparison.py --base_dir auto_analysis --all_runs")


if __name__ == "__main__":
    main()