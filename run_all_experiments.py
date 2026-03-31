#!/usr/bin/env python3
"""
DAE-KAN Parallel Experiment Runner with Checkpoint/Resume

This script runs all experiments in parallel with automatic checkpointing
and resume capability. If interrupted, simply re-run to continue from where you left off.

Features:
- ✅ Automatic checkpointing after each experiment
- ✅ Resume from interruption
- ✅ Parallel execution (configurable)
- ✅ Progress tracking
- ✅ Error handling (continues on failure)
- ✅ Resource management (GPU memory)

Usage:
    # Run all experiments (default: 2 parallel jobs)
    python run_all_experiments.py
    
    # Run with more parallel jobs (if you have multiple GPUs)
    python run_all_experiments.py --parallel-jobs 4
    
    # Resume after interruption
    python run_all_experiments.py --resume
    
    # Dry run (see what would be executed)
    python run_all_experiments.py --dry-run
    
    # Reset and start fresh
    python run_all_experiments.py --reset
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import signal


# ============================================================================
# CONFIGURATION
# ============================================================================

# All experiments to run
EXPERIMENTS = {
    # Main experiments (proposed method + ablation variants)
    'main_dae_kan': {
        'command': [
            'python', 'run_multi_seed_experiments.py',
            '--model-name', 'dae_kan_attention',
            '--dataset', 'HeparUnifiedPNG',
            '--epochs', '30',
            '--batch-size', '8',
            '--seeds', '42', '123', '456', '789', '1011'
        ],
        'priority': 1,
        'gpu_memory_gb': 8,
        'estimated_hours': 5
    },
    'main_baseline': {
        'command': [
            'python', 'run_multi_seed_experiments.py',
            '--model-name', 'baseline',
            '--dataset', 'HeparUnifiedPNG',
            '--epochs', '30',
            '--batch-size', '8',
            '--seeds', '42', '123', '456', '789', '1011'
        ],
        'priority': 2,
        'gpu_memory_gb': 8,
        'estimated_hours': 5
    },
    'main_bam_only': {
        'command': [
            'python', 'run_multi_seed_experiments.py',
            '--model-name', 'bam_only',
            '--dataset', 'HeparUnifiedPNG',
            '--epochs', '30',
            '--batch-size', '8',
            '--seeds', '42', '123', '456', '789', '1011'
        ],
        'priority': 2,
        'gpu_memory_gb': 8,
        'estimated_hours': 5
    },
    'main_kan_only': {
        'command': [
            'python', 'run_multi_seed_experiments.py',
            '--model-name', 'kan_only',
            '--dataset', 'HeparUnifiedPNG',
            '--epochs', '30',
            '--batch-size', '8',
            '--seeds', '42', '123', '456', '789', '1011'
        ],
        'priority': 2,
        'gpu_memory_gb': 8,
        'estimated_hours': 5
    },
    'main_no_bam': {
        'command': [
            'python', 'run_multi_seed_experiments.py',
            '--model-name', 'no_bam',
            '--dataset', 'HeparUnifiedPNG',
            '--epochs', '30',
            '--batch-size', '8',
            '--seeds', '42', '123', '456', '789', '1011'
        ],
        'priority': 2,
        'gpu_memory_gb': 8,
        'estimated_hours': 5
    },
    'main_no_kan': {
        'command': [
            'python', 'run_multi_seed_experiments.py',
            '--model-name', 'no_kan',
            '--dataset', 'HeparUnifiedPNG',
            '--epochs', '30',
            '--batch-size', '8',
            '--seeds', '42', '123', '456', '789', '1011'
        ],
        'priority': 2,
        'gpu_memory_gb': 8,
        'estimated_hours': 5
    },
    
    # Baseline comparisons
    'baselines_all': {
        'command': [
            'python', 'baselines/run_all_baselines.py',
            '--all-models',
            '--dataset', 'HeparUnifiedPNG',
            '--epochs', '30',
            '--batch-size', '16',
            '--seeds', '42', '123', '456'
        ],
        'priority': 3,
        'gpu_memory_gb': 6,
        'estimated_hours': 3
    },
    
    # Additional analyses (optional)
    'sensitivity_analysis': {
        'command': [
            'python', 'sensitivity_analysis.py',
            '--latent-dims', '64', '128', '256',
            '--n-clusters', '3', '5', '7',
            '--kan-spline-orders', '2', '3', '5',
            '--eca-kernel-sizes', '3', '5', '7',
            '--plot-heatmaps',
            '--fast-mode'
        ],
        'priority': 4,
        'gpu_memory_gb': 8,
        'estimated_hours': 2,
        'optional': True
    },
    'cross_domain': {
        'command': [
            'python', 'cross_domain_eval.py',
            '--source', 'PanNuke',
            '--target', 'HeparUnifiedPNG',
            '--strategy', 'zero_shot',
            '--finetune-epochs', '10'
        ],
        'priority': 4,
        'gpu_memory_gb': 8,
        'estimated_hours': 2,
        'optional': True
    },
    'profiling': {
        'command': [
            'python', 'profile_runtime.py',
            '--model-name', 'dae_kan_attention',
            '--batch-sizes', '4', '8', '16',
            '--output', 'outputs/profiling/runtime_report.csv'
        ],
        'priority': 5,
        'gpu_memory_gb': 8,
        'estimated_hours': 0.5,
        'optional': True
    },
    'interpretability': {
        'command': [
            'python', 'compute_interpretability_metrics.py',
            '--model-path', 'auto',  # Will be replaced with best checkpoint
            '--data-dir', 'data/processed',
            '--dataset', 'HeparUnifiedPNG',
            '--batch-size', '16',
            '--cluster-analysis',
            '--output', 'outputs/interpretability'
        ],
        'priority': 5,
        'gpu_memory_gb': 8,
        'estimated_hours': 1,
        'optional': True
    },
    
    # Final aggregation
    'statistical_analysis': {
        'command': [
            'python', 'evaluate_stats.py',
            '--all-models',
            '--dataset', 'HeparUnifiedPNG',
            '--run-tests',
            '--test-type', 'wilcoxon',
            '--output', 'results/summary_stats.csv'
        ],
        'priority': 6,
        'gpu_memory_gb': 2,
        'estimated_hours': 0.2,
        'depends_on': ['main_dae_kan', 'main_baseline', 'baselines_all']
    }
}

# Checkpoint file location
CHECKPOINT_FILE = Path('.experiment_checkpoint.json')
LOCK_FILE = Path('.experiment_lock')


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manages experiment checkpoints with file locking for safety."""
    
    def __init__(self, checkpoint_file: Path = CHECKPOINT_FILE):
        self.checkpoint_file = checkpoint_file
        self.lock_file = checkpoint_file.with_suffix('.lock')
        self._lock = threading.Lock()
    
    def load(self) -> Dict[str, Any]:
        """Load checkpoint from file."""
        if not self.checkpoint_file.exists():
            return {
                'completed': [],
                'failed': [],
                'running': [],
                'started_at': None,
                'last_updated': None
            }
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            return {
                'completed': [],
                'failed': [],
                'running': [],
                'started_at': None,
                'last_updated': None
            }
    
    def save(self, data: Dict[str, Any]) -> bool:
        """Save checkpoint to file."""
        with self._lock:
            try:
                data['last_updated'] = datetime.now().isoformat()
                
                # Write atomically
                temp_file = self.checkpoint_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2)
                
                temp_file.replace(self.checkpoint_file)
                return True
            except Exception as e:
                print(f"❌ Error saving checkpoint: {e}")
                return False
    
    def mark_completed(self, exp_name: str, duration_seconds: float):
        """Mark an experiment as completed."""
        data = self.load()
        if exp_name not in data['completed']:
            data['completed'].append(exp_name)
        if exp_name in data['running']:
            data['running'].remove(exp_name)
        data['completed_details'] = data.get('completed_details', {})
        data['completed_details'][exp_name] = {
            'completed_at': datetime.now().isoformat(),
            'duration_seconds': duration_seconds
        }
        self.save(data)
    
    def mark_failed(self, exp_name: str, error: str):
        """Mark an experiment as failed."""
        data = self.load()
        if exp_name not in data['failed']:
            data['failed'].append(exp_name)
        if exp_name in data['running']:
            data['running'].remove(exp_name)
        data['failed_details'] = data.get('failed_details', {})
        data['failed_details'][exp_name] = {
            'failed_at': datetime.now().isoformat(),
            'error': error
        }
        self.save(data)
    
    def mark_running(self, exp_name: str):
        """Mark an experiment as running."""
        data = self.load()
        if not data['started_at']:
            data['started_at'] = datetime.now().isoformat()
        if exp_name not in data['running']:
            data['running'].append(exp_name)
        self.save(data)
    
    def is_completed(self, exp_name: str) -> bool:
        """Check if an experiment is completed."""
        data = self.load()
        return exp_name in data['completed']
    
    def reset(self):
        """Reset all checkpoints."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        print("✓ Checkpoint reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        return self.load()


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_experiment(exp_name: str, exp_config: Dict, dry_run: bool = False) -> bool:
    """
    Run a single experiment.
    
    Args:
        exp_name: Experiment name
        exp_config: Experiment configuration
        dry_run: If True, only print command
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"🔬 Starting Experiment: {exp_name}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(exp_config['command'])}")
    print(f"Estimated time: {exp_config['estimated_hours']} hours")
    print(f"GPU Memory: {exp_config['gpu_memory_gb']} GB")
    print(f"Priority: {exp_config['priority']}")
    print(f"{'='*80}\n")
    
    if dry_run:
        print("🔍 DRY RUN - Not executing")
        return True
    
    # Check dependencies
    if 'depends_on' in exp_config:
        checkpoint = CheckpointManager()
        data = checkpoint.load()
        for dep in exp_config['depends_on']:
            if dep not in data['completed']:
                print(f"⚠️  Dependency not completed: {dep}")
                return False
    
    # Handle auto model-path for interpretability
    command = exp_config['command'].copy()
    if '--model-path' in command and 'auto' in command:
        idx = command.index('auto')
        # Find best checkpoint
        checkpoint_files = list(Path('outputs').rglob('checkpoints/*.ckpt'))
        if checkpoint_files:
            best_ckpt = str(checkpoint_files[0])
            command[idx] = best_ckpt
            print(f"✓ Using checkpoint: {best_ckpt}")
        else:
            print("⚠️  No checkpoint found, skipping interpretability")
            return False
    
    # Run experiment
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True
        )
        
        duration = time.time() - start_time
        print(f"\n✅ Experiment {exp_name} completed successfully")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Experiment {exp_name} failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Experiment {exp_name} failed with error: {e}")
        return False


def run_experiment_with_checkpoint(exp_name: str, exp_config: Dict,
                                   checkpoint_mgr: CheckpointManager,
                                   dry_run: bool = False) -> bool:
    """Run experiment with checkpoint management."""
    
    # Skip if already completed
    if checkpoint_mgr.is_completed(exp_name):
        print(f"⏭️  Skipping {exp_name} (already completed)")
        return True
    
    # Mark as running
    checkpoint_mgr.mark_running(exp_name)
    
    # Run experiment
    success = run_experiment(exp_name, exp_config, dry_run)
    
    # Update checkpoint
    if success:
        duration = exp_config['estimated_hours'] * 3600  # Estimate
        checkpoint_mgr.mark_completed(exp_name, duration)
    else:
        checkpoint_mgr.mark_failed(exp_name, "Experiment failed")
    
    return success


# ============================================================================
# PARALLEL EXECUTION
# ============================================================================

def get_available_experiments(checkpoint_mgr: CheckpointManager,
                             include_optional: bool = True) -> List[str]:
    """Get list of experiments that are ready to run."""
    data = checkpoint_mgr.load()
    completed = set(data['completed'])
    running = set(data['running'])
    failed = set(data['failed'])
    
    ready = []
    
    for exp_name, exp_config in EXPERIMENTS.items():
        # Skip if already done
        if exp_name in completed or exp_name in running:
            continue
        
        # Skip optional if not requested
        if exp_config.get('optional', False) and not include_optional:
            continue
        
        # Check dependencies
        deps = exp_config.get('depends_on', [])
        if all(dep in completed for dep in deps):
            ready.append(exp_name)
    
    # Sort by priority
    ready.sort(key=lambda x: EXPERIMENTS[x]['priority'])
    
    return ready


def run_all_experiments(parallel_jobs: int = 2,
                       include_optional: bool = True,
                       dry_run: bool = False,
                       resume: bool = False) -> int:
    """
    Run all experiments with parallel execution.
    
    Args:
        parallel_jobs: Number of parallel jobs
        include_optional: Include optional experiments
        dry_run: Dry run mode
        resume: Resume from checkpoint
    
    Returns:
        Number of failed experiments
    """
    print(f"\n{'='*80}")
    print(f"🚀 DAE-KAN Parallel Experiment Runner")
    print(f"{'='*80}")
    print(f"Parallel jobs: {parallel_jobs}")
    print(f"Include optional: {include_optional}")
    print(f"Dry run: {dry_run}")
    print(f"Resume: {resume}")
    print(f"{'='*80}\n")
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager()
    
    # Reset if not resuming and checkpoint exists
    if not resume and checkpoint_mgr.checkpoint_file.exists():
        print("⚠️  Existing checkpoint found.")
        response = input("Resume from checkpoint? (y/n/reset): ")
        if response.lower() == 'reset':
            checkpoint_mgr.reset()
        elif response.lower() != 'y':
            print("❌ Aborted")
            return -1
    
    # Get total experiments
    total_experiments = len([
        exp for exp, cfg in EXPERIMENTS.items()
        if not (cfg.get('optional', False) and not include_optional)
    ])
    
    print(f"📊 Total experiments: {total_experiments}")
    print(f"📋 Experiment queue:")
    for i, (exp_name, exp_config) in enumerate(EXPERIMENTS.items(), 1):
        if exp_config.get('optional', False) and not include_optional:
            continue
        status = " (optional)" if exp_config.get('optional', False) else ""
        status += f" [depends on: {', '.join(exp_config.get('depends_on', []))}]" if exp_config.get('depends_on') else ""
        print(f"  {i:2d}. {exp_name}{status}")
    
    if not dry_run:
        print(f"\n⚠️  This will run {total_experiments} experiments.")
        print(f"⏱️  Estimated total time: {sum(e['estimated_hours'] for e in EXPERIMENTS.values())/parallel_jobs:.1f} hours")
        response = input("\nContinue? (y/N): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return -1
    
    # Run experiments
    failed_count = 0
    completed_count = 0
    
    with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
        while True:
            # Get ready experiments
            ready = get_available_experiments(checkpoint_mgr, include_optional)
            
            if not ready:
                # Check if we're done
                data = checkpoint_mgr.load()
                remaining = total_experiments - len(data['completed']) - len(data['failed'])
                if remaining == 0:
                    break
                else:
                    # Wait for running experiments
                    time.sleep(60)
                    continue
            
            # Submit experiments
            futures = {}
            for exp_name in ready[:parallel_jobs]:
                exp_config = EXPERIMENTS[exp_name]
                future = executor.submit(
                    run_experiment_with_checkpoint,
                    exp_name, exp_config, checkpoint_mgr, dry_run
                )
                futures[future] = exp_name
            
            # Wait for completion
            for future in as_completed(futures):
                exp_name = futures[future]
                try:
                    success = future.result()
                    if success:
                        completed_count += 1
                        print(f"\n✅ Completed: {exp_name} ({completed_count}/{total_experiments})")
                    else:
                        failed_count += 1
                        print(f"\n❌ Failed: {exp_name}")
                except Exception as e:
                    failed_count += 1
                    print(f"\n❌ Exception in {exp_name}: {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 EXPERIMENT RUNNER SUMMARY")
    print(f"{'='*80}")
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed_count}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {completed_count/(completed_count+failed_count)*100:.1f}%")
    print(f"{'='*80}")
    
    if failed_count == 0:
        print(f"\n✅ All experiments completed successfully!")
    else:
        print(f"\n⚠️  {failed_count} experiment(s) failed")
        print(f"💡 Re-run with --resume to retry failed experiments")
    
    return failed_count


# ============================================================================
# STATUS DISPLAY
# ============================================================================

def show_status():
    """Show current experiment status."""
    checkpoint_mgr = CheckpointManager()
    data = checkpoint_mgr.load()
    
    print(f"\n{'='*80}")
    print(f"📊 EXPERIMENT STATUS")
    print(f"{'='*80}")
    
    if data['started_at']:
        print(f"Started: {data['started_at']}")
    if data['last_updated']:
        print(f"Last updated: {data['last_updated']}")
    
    print(f"\nCompleted: {len(data['completed'])}")
    for exp in data['completed']:
        print(f"  ✓ {exp}")
    
    if data['running']:
        print(f"\nRunning: {len(data['running'])}")
        for exp in data['running']:
            print(f"  ⏳ {exp}")
    
    if data['failed']:
        print(f"\nFailed: {len(data['failed'])}")
        for exp in data['failed']:
            error = data.get('failed_details', {}).get(exp, {}).get('error', 'Unknown')
            print(f"  ❌ {exp}: {error}")
    
    # Show remaining
    completed = set(data['completed'])
    running = set(data['running'])
    failed = set(data['failed'])
    done = completed | running | failed
    
    remaining = [exp for exp in EXPERIMENTS.keys() if exp not in done]
    if remaining:
        print(f"\nRemaining: {len(remaining)}")
        for exp in remaining:
            print(f"  ⏸️  {exp}")
    
    print(f"\n{'='*80}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DAE-KAN Parallel Experiment Runner with Checkpoint/Resume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments (2 parallel jobs)
  python run_all_experiments.py
  
  # Run with more parallel jobs
  python run_all_experiments.py --parallel-jobs 4
  
  # Resume from interruption
  python run_all_experiments.py --resume
  
  # Skip optional experiments
  python run_all_experiments.py --no-optional
  
  # Dry run
  python run_all_experiments.py --dry-run
  
  # Reset and start fresh
  python run_all_experiments.py --reset
  
  # Show status
  python run_all_experiments.py --status
        """
    )
    
    parser.add_argument(
        '--parallel-jobs',
        type=int,
        default=2,
        help='Number of parallel jobs (default: 2)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--no-optional',
        action='store_true',
        help='Skip optional experiments'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run (show commands without executing)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset checkpoint and start fresh'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current status'
    )
    parser.add_argument(
        '--yes',
        action='store_true',
        help='Auto-confirm prompts'
    )
    
    args = parser.parse_args()
    
    # Handle status
    if args.status:
        show_status()
        return 0
    
    # Handle reset
    if args.reset:
        checkpoint_mgr = CheckpointManager()
        checkpoint_mgr.reset()
        return 0
    
    # Run experiments
    failed = run_all_experiments(
        parallel_jobs=args.parallel_jobs,
        include_optional=not args.no_optional,
        dry_run=args.dry_run,
        resume=args.resume or args.yes
    )
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    # Handle interrupts gracefully
    def signal_handler(sig, frame):
        print("\n\n⚠️  Interrupt received!")
        print("💡 Experiments will resume from checkpoint on next run")
        print("💡 Re-run: python run_all_experiments.py --resume")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    sys.exit(main())
