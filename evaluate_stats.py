#!/usr/bin/env python3
"""
Statistical Significance Evaluation Script

This script aggregates results from multi-seed experiments and performs
statistical significance testing to validate clustering improvements.

Addresses Reviewer 1 #2 and Reviewer 2 concerns about:
- Statistical significance testing (repeated runs with mean ± std)
- Hypothesis testing to validate improvements are not due to random initialization
- Comparison against baseline methods

Usage:
    # Evaluate a specific model
    python evaluate_stats.py --model dae_kan_attention --dataset HeparUnifiedPNG
    
    # Compare multiple models
    python evaluate_stats.py --models dae_kan_attention baseline bam_only
    
    # Run statistical tests
    python evaluate_stats.py --model dae_kan_attention --run-tests
    
    # Generate summary table
    python evaluate_stats.py --all-models --output results/summary_stats.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel


# Default seeds matching multi-seed runner
DEFAULT_SEEDS = [42, 123, 456, 789, 1011]

# Clustering metrics
METRICS = {
    'davies_bouldin': {'direction': 'lower', 'description': 'Davies-Bouldin Index (lower is better)'},
    'calinski_harabasz': {'direction': 'higher', 'description': 'Calinski-Harabasz Index (higher is better)'},
    'silhouette': {'direction': 'higher', 'description': 'Silhouette Score (higher is better)'},
    'xie_beni': {'direction': 'lower', 'description': 'Xie-Beni Index (lower is better)'},
    'adjusted_mutual_info': {'direction': 'higher', 'description': 'Adjusted Mutual Information'},
    'adjusted_rand_index': {'direction': 'higher', 'description': 'Adjusted Rand Index'},
}


def find_experiment_runs(
    base_dir: str = "outputs",
    model_name: Optional[str] = None,
    dataset: Optional[str] = None
) -> List[Path]:
    """
    Find all experiment run directories matching criteria.
    
    Args:
        base_dir: Base outputs directory
        model_name: Model name to filter by
        dataset: Dataset name to filter by
    
    Returns:
        List of paths to experiment directories
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"⚠️  Base directory {base_dir} does not exist")
        return []
    
    # Find all experiment directories
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('exp_')]
    
    # Filter by model name
    if model_name:
        exp_dirs = [d for d in exp_dirs if model_name in d.name]
    
    # Filter by dataset
    if dataset:
        exp_dirs = [d for d in exp_dirs if dataset in d.name]
    
    return sorted(exp_dirs)


def load_experiment_metrics(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load metrics from an experiment directory.
    
    Args:
        exp_dir: Path to experiment directory
    
    Returns:
        Dictionary containing metrics or None if not found
    """
    # Look for metrics file
    metrics_files = list(exp_dir.rglob('clustering_metrics.csv'))
    
    if not metrics_files:
        # Try alternative locations
        metrics_files = list(exp_dir.rglob('metrics.csv'))
    
    if not metrics_files:
        print(f"⚠️  No metrics file found in {exp_dir}")
        return None
    
    # Load the first metrics file found
    metrics_path = metrics_files[0]
    
    try:
        df = pd.read_csv(metrics_path)
        
        # Extract key metrics
        metrics = {}
        
        # Look for clustering metrics
        for metric in METRICS.keys():
            if metric in df.columns:
                metrics[metric] = df[metric].iloc[-1] if len(df) > 0 else None
        
        # Load config if available
        config_files = list(exp_dir.rglob('experiment.yaml'))
        if config_files:
            import yaml
            with open(config_files[0], 'r') as f:
                config = yaml.safe_load(f)
                metrics['config'] = config
        
        # Extract seed from directory name
        if 'seed' in exp_dir.name:
            parts = exp_dir.name.split('_')
            for part in parts:
                if part.startswith('seed'):
                    try:
                        metrics['seed'] = int(part.replace('seed', ''))
                    except ValueError:
                        pass
        
        return metrics
    
    except Exception as e:
        print(f"⚠️  Error loading metrics from {exp_dir}: {e}")
        return None


def aggregate_multi_seed_results(
    exp_dirs: List[Path]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple seeds.
    
    Args:
        exp_dirs: List of experiment directories
    
    Returns:
        Dictionary with aggregated metrics (mean ± std)
    """
    # Collect metrics per seed
    seed_metrics = {}
    
    for exp_dir in exp_dirs:
        metrics = load_experiment_metrics(exp_dir)
        
        if metrics and 'seed' in metrics:
            seed = metrics['seed']
            seed_metrics[seed] = metrics
    
    # Compute aggregated statistics
    aggregated = {}
    
    for metric in METRICS.keys():
        values = []
        
        for seed_data in seed_metrics.values():
            if metric in seed_data and seed_data[metric] is not None:
                values.append(seed_data[metric])
        
        if len(values) > 0:
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n_runs': len(values),
                'values': values  # Keep individual values for statistical tests
            }
        else:
            aggregated[metric] = {
                'mean': None,
                'std': None,
                'min': None,
                'max': None,
                'n_runs': 0,
                'values': []
            }
    
    return aggregated


def compute_statistical_tests(
    model_a_metrics: Dict[str, Dict[str, float]],
    model_b_metrics: Dict[str, Dict[str, float]],
    test_type: str = 'wilcoxon'
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical tests comparing two models.
    
    Args:
        model_a_metrics: Metrics for model A (proposed method)
        model_b_metrics: Metrics for model B (baseline)
        test_type: Type of test ('wilcoxon' or 'ttest')
    
    Returns:
        Dictionary with test statistics and p-values
    """
    results = {}
    
    for metric in METRICS.keys():
        values_a = model_a_metrics.get(metric, {}).get('values', [])
        values_b = model_b_metrics.get(metric, {}).get('values', [])
        
        # Need paired samples
        if len(values_a) < 2 or len(values_b) < 2:
            results[metric] = {
                'test': test_type,
                'statistic': None,
                'p_value': None,
                'significant': None,
                'note': 'Insufficient samples'
            }
            continue
        
        # Ensure same length (paired test)
        min_len = min(len(values_a), len(values_b))
        values_a = values_a[:min_len]
        values_b = values_b[:min_len]
        
        try:
            if test_type == 'wilcoxon':
                statistic, p_value = wilcoxon(values_a, values_b)
            elif test_type == 'ttest':
                statistic, p_value = ttest_rel(values_a, values_b)
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Determine significance (α = 0.05)
            significant = p_value < 0.05
            
            results[metric] = {
                'test': test_type,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
                'mean_diff': float(np.mean(values_a) - np.mean(values_b)),
                'effect_size': compute_cohens_d(values_a, values_b)
            }
        
        except Exception as e:
            results[metric] = {
                'test': test_type,
                'statistic': None,
                'p_value': None,
                'significant': None,
                'error': str(e)
            }
    
    return results


def compute_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group1: First group values
        group2: Second group values
    
    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def create_summary_table(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str
) -> pd.DataFrame:
    """
    Create a summary table of all results.
    
    Args:
        all_results: Dictionary mapping model names to aggregated metrics
        output_path: Path to save the summary CSV
    
    Returns:
        DataFrame with summary statistics
    """
    rows = []
    
    for model_name, metrics in all_results.items():
        row = {'model': model_name}
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                if metric_data.get('mean') is not None:
                    row[f'{metric_name}_mean'] = metric_data['mean']
                    row[f'{metric_name}_std'] = metric_data['std']
                    row[f'{metric_name}_n'] = metric_data['n_runs']
            else:
                row[metric_name] = metric_data
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Summary table saved to: {output_path}")
    
    return df


def format_mean_std(mean: float, std: float) -> str:
    """Format mean ± std as a string."""
    if mean is None or std is None:
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}"


def print_summary_report(
    all_results: Dict[str, Dict[str, Dict[str, float]]],
    statistical_tests: Optional[Dict[str, Dict[str, float]]] = None
) -> None:
    """
    Print a formatted summary report.
    
    Args:
        all_results: Dictionary mapping model names to aggregated metrics
        statistical_tests: Optional statistical test results
    """
    print("\n" + "="*80)
    print("📊 STATISTICAL SIGNIFICANCE ANALYSIS REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Model comparison table
    print("\n📈 CLUSTERING METRICS COMPARISON (Mean ± Std Dev)")
    print("-"*80)
    
    models = list(all_results.keys())
    metrics_list = list(METRICS.keys())
    
    # Header
    header = f"{'Metric':<25}"
    for model in models:
        header += f"{model[:15]:>18}"
    print(header)
    print("-" * len(header))
    
    # Rows
    for metric in metrics_list:
        row = f"{METRICS[metric]['description'][:25]:<25}"
        
        for model in models:
            metric_data = all_results[model].get(metric, {})
            mean = metric_data.get('mean')
            std = metric_data.get('std')
            n = metric_data.get('n_runs', 0)
            
            if mean is not None and n > 0:
                row += f"{format_mean_std(mean, std):>18}"
            else:
                row += f"{'N/A':>18}"
        
        print(row)
    
    # Statistical tests
    if statistical_tests:
        print("\n" + "="*80)
        print("🧪 STATISTICAL SIGNIFICANCE TESTS")
        print("-"*80)
        print("Null hypothesis (H0): No difference between proposed model and baseline")
        print("Alternative hypothesis (H1): Significant difference exists")
        print("Significance level (α): 0.05")
        print("-"*80)
        
        print(f"\n{'Metric':<25} {'Test':<12} {'Statistic':>12} {'p-value':>12} {'Significant?':>12} {'Effect Size':>12}")
        print("-"*85)
        
        for metric, test_results in statistical_tests.items():
            metric_name = METRICS[metric]['description'][:25]
            test_type = test_results.get('test', 'N/A')
            statistic = test_results.get('statistic')
            p_value = test_results.get('p_value')
            significant = test_results.get('significant')
            effect_size = test_results.get('effect_size')
            
            if statistic is not None:
                stat_str = f"{statistic:>12.4f}"
                p_str = f"{p_value:>12.4f}"
                sig_str = "✓ YES" if significant else "✗ NO"
                eff_str = f"{effect_size:>12.4f}" if effect_size is not None else f"{'N/A':>12}"
            else:
                stat_str = f"{'N/A':>12}"
                p_str = f"{'N/A':>12}"
                sig_str = f"{'N/A':>12}"
                eff_str = f"{'N/A':>12}"
            
            print(f"{metric_name:<25} {test_type:<12} {stat_str} {p_str} {sig_str} {eff_str}")
        
        print("\n" + "="*80)
        print("INTERPRETATION:")
        print("  - p < 0.05: Reject H0, significant difference exists")
        print("  - p ≥ 0.05: Fail to reject H0, no significant difference")
        print("  - Effect size (Cohen's d):")
        print("      • 0.2 = small effect")
        print("      • 0.5 = medium effect")
        print("      • 0.8 = large effect")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Statistical significance evaluation for multi-seed experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single model
  python evaluate_stats.py --model dae_kan_attention --dataset HeparUnifiedPNG
  
  # Compare multiple models
  python evaluate_stats.py --models dae_kan_attention baseline bam_only
  
  # Run with statistical tests (requires baseline)
  python evaluate_stats.py --model dae_kan_attention --baseline baseline --run-tests
  
  # Generate summary for all models
  python evaluate_stats.py --all-models --output results/summary_stats.csv
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        help='Model to evaluate (single model mode)'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Models to compare (multi-model mode)'
    )
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Evaluate all models found in outputs directory'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='baseline',
        help='Baseline model name for statistical comparison'
    )
    
    # Dataset
    parser.add_argument(
        '--dataset',
        type=str,
        default='HeparUnifiedPNG',
        help='Dataset to evaluate'
    )
    
    # Statistical tests
    parser.add_argument(
        '--run-tests',
        action='store_true',
        help='Run statistical significance tests'
    )
    parser.add_argument(
        '--test-type',
        type=str,
        choices=['wilcoxon', 'ttest'],
        default='wilcoxon',
        help='Type of statistical test to run'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='results/summary_stats.csv',
        help='Output path for summary CSV'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='outputs',
        help='Base directory containing experiment runs'
    )
    
    args = parser.parse_args()
    
    # Determine which models to evaluate
    if args.all_models:
        # Find all models in outputs
        exp_dirs = find_experiment_runs(args.base_dir)
        models_found = set()
        
        for exp_dir in exp_dirs:
            for model in ['dae_kan_attention', 'baseline', 'bam_only', 'kan_only', 
                         'no_bam', 'no_kan', 'no_eka']:
                if model in exp_dir.name:
                    models_found.add(model)
        
        args.models = list(models_found)
        print(f"🔍 Found models: {args.models}")
    
    elif args.models:
        pass  # Use provided models
    
    elif args.model:
        args.models = [args.model]
    
    else:
        print("❌ Error: Must specify --model, --models, or --all-models")
        return
    
    # Aggregate results for each model
    all_results = {}
    
    for model_name in args.models:
        print(f"\n🔍 Evaluating model: {model_name}")
        
        # Find experiment runs
        exp_dirs = find_experiment_runs(args.base_dir, model_name, args.dataset)
        
        if not exp_dirs:
            print(f"⚠️  No experiments found for {model_name}")
            continue
        
        print(f"✓ Found {len(exp_dirs)} experiment runs")
        
        # Aggregate multi-seed results
        aggregated = aggregate_multi_seed_results(exp_dirs)
        all_results[model_name] = aggregated
    
    if not all_results:
        print("\n❌ No results to analyze. Run experiments first.")
        return
    
    # Create summary table
    summary_df = create_summary_table(all_results, args.output)
    
    # Run statistical tests if requested
    statistical_tests = None
    
    if args.run_tests and len(all_results) >= 2:
        print(f"\n🧪 Running statistical tests (proposed: {args.models[0]} vs baseline: {args.baseline})")
        
        if args.models[0] in all_results and args.baseline in all_results:
            statistical_tests = compute_statistical_tests(
                all_results[args.models[0]],
                all_results[args.baseline],
                args.test_type
            )
        else:
            print(f"⚠️  Cannot run tests: missing {args.models[0]} or {args.baseline}")
    
    # Print report
    print_summary_report(all_results, statistical_tests)
    
    print(f"\n✅ Analysis complete!")
    print(f"📄 Summary saved to: {args.output}")


if __name__ == "__main__":
    main()
