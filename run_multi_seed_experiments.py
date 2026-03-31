#!/usr/bin/env python3
"""
Multi-Seed Experiment Runner for Statistical Significance Analysis

This script runs experiments across multiple random seeds to ensure
statistical significance of results and report mean ± std deviations.

Usage:
    # Run with default seeds (42, 123, 456, 789, 1011)
    python run_multi_seed_experiments.py --model-name dae_kan_attention
    
    # Run with custom seeds
    python run_multi_seed_experiments.py --seeds 42 123 456 789 1011
    
    # Run for specific dataset
    python run_multi_seed_experiments.py --dataset PanNuke --model-name dae_kan_attention
    
    # Dry run to see what would be executed
    python run_multi_seed_experiments.py --dry-run
"""

import argparse
import subprocess
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import json


# Default seeds for statistical significance (Reviewer 1 #2, Reviewer 2)
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]


def run_single_seed_experiment(
    seed: int,
    model_name: str,
    dataset: str,
    config_path: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    dry_run: bool = False
) -> bool:
    """
    Run a single experiment with a specific seed.
    
    Args:
        seed: Random seed for this run
        model_name: Model architecture to use
        dataset: Dataset name
        config_path: Path to configuration file
        output_dir: Base output directory
        epochs: Number of training epochs
        batch_size: Batch size
        dry_run: If True, only print command without executing
    
    Returns:
        True if successful, False otherwise
    """
    # Create experiment name with seed
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"{model_name}_seed{seed}_{dataset}"
    
    # Build command
    cmd = [
        "python", "src/pl_training_with_analysis_and_optimization.py",
        "--model-name", model_name,
        "--seed", str(seed),
        "--dataset-name", dataset,
        "--config-path", config_path,
        "--experiment-name", experiment_name,
        "--output-dir", output_dir,
        "--max-epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    
    print(f"\n{'='*80}")
    print(f"🔬 Running Experiment - Seed {seed}")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset}")
    print(f"Seed: {seed}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("🔍 DRY RUN - Not executing")
        return True
    
    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ Seed {seed} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Seed {seed} failed with return code {e.returncode}")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Seed {seed} failed with error: {e}")
        return False


def save_experiment_plan(
    seeds: List[int],
    model_name: str,
    dataset: str,
    output_path: str
) -> None:
    """
    Save the multi-seed experiment plan to a JSON file.
    
    Args:
        seeds: List of seeds to use
        model_name: Model architecture
        dataset: Dataset name
        output_path: Path to save the plan
    """
    plan = {
        "model_name": model_name,
        "dataset": dataset,
        "seeds": seeds,
        "n_experiments": len(seeds),
        "created_at": datetime.now().isoformat(),
        "status": "planned"
    }
    
    with open(output_path, 'w') as f:
        json.dump(plan, f, indent=2)
    
    print(f"📝 Experiment plan saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-seed experiments for statistical significance analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default seeds
  python run_multi_seed_experiments.py --model-name dae_kan_attention
  
  # Run with custom seeds
  python run_multi_seed_experiments.py --seeds 42 123 456 --model-name baseline
  
  # Run for PanNuke dataset
  python run_multi_seed_experiments.py --dataset PanNuke --model-name dae_kan_attention
  
  # Dry run to see commands
  python run_multi_seed_experiments.py --dry-run
        """
    )
    
    # Model and data arguments
    parser.add_argument(
        '--model-name',
        type=str,
        required=True,
        choices=['dae_kan_attention', 'baseline', 'bam_only', 'kan_only', 
                 'no_bam', 'no_kan', 'no_eka'],
        help='Model architecture to evaluate'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='HeparUnifiedPNG',
        choices=['PanNuke', 'HeparUnifiedPNG'],
        help='Dataset to use for experiments'
    )
    
    # Seed arguments
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=DEFAULT_SEEDS,
        help=f'Random seeds to use (default: {DEFAULT_SEEDS})'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=None,
        help='Number of runs (alternative to specifying seeds directly)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs per seed'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--config-path',
        type=str,
        default='config/experiment_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Base output directory for experiments'
    )
    
    # Execution arguments
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be executed without running'
    )
    parser.add_argument(
        '--save-plan',
        action='store_true',
        help='Save the experiment plan to a file'
    )
    parser.add_argument(
        '--continue-on-failure',
        action='store_true',
        help='Continue with next seed if one fails'
    )
    
    args = parser.parse_args()
    
    # Generate seeds if n_runs specified
    if args.n_runs is not None:
        import random
        random.seed(42)
        args.seeds = random.sample(range(1000, 9999), args.n_runs)
    
    print("🔬 Multi-Seed Experiment Runner")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Number of seeds: {len(args.seeds)}")
    print(f"Seeds: {args.seeds}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    
    # Save experiment plan
    if args.save_plan:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plan_path = f"multi_seed_plan_{args.model_name}_{timestamp}.json"
        save_experiment_plan(args.seeds, args.model_name, args.dataset, plan_path)
    
    if not args.dry_run:
        print(f"\n⚠️  This will run {len(args.seeds)} experiments sequentially.")
        print(f"⏱️  Estimated time: ~{len(args.seeds) * args.epochs * 5} minutes")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted by user")
            return
    
    # Run experiments
    successful_runs = 0
    failed_runs = 0
    failed_seeds = []
    
    for i, seed in enumerate(args.seeds, 1):
        print(f"\n📍 Progress: {i}/{len(args.seeds)}")
        
        success = run_single_seed_experiment(
            seed=seed,
            model_name=args.model_name,
            dataset=args.dataset,
            config_path=args.config_path,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
        
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
            failed_seeds.append(seed)
            
            if not args.continue_on_failure:
                print(f"\n❌ Stopping due to failure at seed {seed}")
                break
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 Multi-Seed Experiment Summary")
    print(f"{'='*80}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Total runs: {len(args.seeds)}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    
    if failed_seeds:
        print(f"Failed seeds: {failed_seeds}")
    
    if successful_runs > 0:
        print(f"\n💡 Next step: Run evaluate_stats.py to aggregate results")
        print(f"   python evaluate_stats.py --model {args.model_name} --dataset {args.dataset}")


if __name__ == "__main__":
    main()
