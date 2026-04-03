#!/usr/bin/env python3
"""
Clustering Metrics Visualization Script

Generates publication-quality box plots and comparison charts for clustering metrics
with statistical significance annotations.

Usage:
    python generate_clustering_plots.py \
        --input results/clustering_summary.csv \
        --tests results/statistical_tests.csv \
        --output-dir figures/
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_data(summary_path: str, tests_path: Optional[str] = None):
    """Load summary and statistical tests data."""
    summary_df = pd.read_csv(summary_path)
    print(f"✓ Loaded summary: {len(summary_df)} rows")
    
    tests_df = None
    if tests_path and Path(tests_path).exists():
        tests_df = pd.read_csv(tests_path)
        print(f"✓ Loaded statistical tests: {len(tests_df)} rows")
    
    return summary_df, tests_df


def get_significance_stars(p_value: float) -> str:
    """Convert p-value to significance stars."""
    if pd.isna(p_value):
        return ''
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    return ''


def plot_metric_comparison(
    summary_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    metric: str,
    metric_display: str,
    method: str,
    output_path: str,
    higher_is_better: bool = True
):
    """Create bar plot comparison for a specific metric."""
    method_df = summary_df[summary_df['clustering_method'] == method]
    
    if len(method_df) == 0:
        print(f"⚠️  No data for method: {method}")
        return
    
    models = method_df['model'].tolist()
    means = method_df[f'{metric}_mean'].tolist()
    stds = method_df[f'{metric}_std'].tolist()
    
    stars = []
    if tests_df is not None:
        method_tests = tests_df[tests_df['clustering_method'] == method]
        for model in models:
            model_tests = method_tests[
                (method_tests['model'] == model) & 
                (method_tests['metric'] == metric)
            ]
            if len(model_tests) > 0:
                pval = model_tests['ttest_pvalue'].iloc[0]
                stars.append(get_significance_stars(pval))
            else:
                stars.append('')
    else:
        stars = [''] * len(models)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(models))
    
    colors = ['#2196F3' if m == 'dae_kan_attention' else '#9E9E9E' for m in models]
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, 
                  alpha=0.8, edgecolor='black', linewidth=1.2)
    
    for i, (bar, star) in enumerate(zip(bars, stars)):
        if star:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + stds[i] + 0.01,
                   star, ha='center', va='bottom', fontsize=14, fontweight='bold',
                   color='red')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric_display, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric_display} Comparison ({method.replace("_", " ").title()})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center')
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_all_metrics_combined(
    summary_df: pd.DataFrame,
    tests_df: pd.DataFrame,
    method: str,
    output_path: str
):
    """Create combined plot with all metrics."""
    method_df = summary_df[summary_df['clustering_method'] == method]
    
    if len(method_df) == 0:
        return
    
    models = method_df['model'].tolist()
    
    metrics_to_plot = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    normalized_data = {}
    
    for metric in metrics_to_plot:
        values = method_df[f'{metric}_mean'].values
        min_val = values.min()
        max_val = values.max()
        if max_val - min_val > 0:
            normalized_data[metric] = (values - min_val) / (max_val - min_val)
        else:
            normalized_data[metric] = np.zeros_like(values)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x_pos = np.arange(len(models))
    width = 0.25
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x_pos + i*width, normalized_data[metric], width, 
              label=metric.replace('_', ' ').title(), color=colors[i], 
              alpha=0.8, edgecolor='black', linewidth=1.0)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Score (0-1)', fontsize=12, fontweight='bold')
    ax.set_title(f'Clustering Metrics Comparison ({method.replace("_", " ").title()})', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=0, ha='center')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1.1)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate clustering metrics visualization plots')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input summary CSV file')
    parser.add_argument('--tests', type=str, default=None,
                        help='Statistical tests CSV file (optional)')
    parser.add_argument('--output-dir', type=str, default='figures/',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📊 GENERATING CLUSTERING METRICS PLOTS")
    print("="*80)
    
    summary_df, tests_df = load_data(args.input, args.tests)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_config = {
        'silhouette': ('Silhouette Score', True),
        'davies_bouldin': ('Davies-Bouldin Index', False),
        'calinski_harabasz': ('Calinski-Harabasz Index', True),
        'xie_beni': ('Xie-Beni Index', False),
    }
    
    for method in summary_df['clustering_method'].unique():
        print(f"\n📈 Generating plots for: {method}")
        
        for metric, (display_name, higher_better) in metrics_config.items():
            if f'{metric}_mean' not in summary_df.columns:
                continue
            
            output_path = output_dir / f'{metric}_{method}_comparison.png'
            plot_metric_comparison(
                summary_df, tests_df, metric, display_name,
                method, str(output_path), higher_better
            )
        
        combined_path = output_dir / f'all_metrics_{method}_comparison.png'
        plot_all_metrics_combined(summary_df, tests_df, method, str(combined_path))
    
    print("\n" + "="*80)
    print("✅ PLOT GENERATION COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
