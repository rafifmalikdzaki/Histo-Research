#!/usr/bin/env python3
"""
Ablation Study Comparison Script

This script creates comprehensive comparisons across multiple training runs
for ablation studies. It compares metrics, attention patterns, and performance
across different model configurations and hyperparameters.

Usage:
    python ablation_study_comparison.py --base_dir auto_analysis --runs run1 run2 run3
    python ablation_study_comparison.py --base_dir auto_analysis --all_runs
    python ablation_study_comparison.py --config ablation_config.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.analysis.auto_analysis import AutomaticAnalyzer


class AblationStudyComparator:
    """
    Comprehensive ablation study comparison system
    """

    def __init__(self, base_dir: str = "auto_analysis"):
        self.base_dir = base_dir
        self.runs_data = {}
        self.comparison_results = {}

    def discover_runs(self) -> List[str]:
        """Discover all available runs in the base directory"""
        if not os.path.exists(self.base_dir):
            print(f"❌ Base directory {self.base_dir} does not exist")
            return []

        runs = []
        for item in os.listdir(self.base_dir):
            run_path = os.path.join(self.base_dir, item)
            if os.path.isdir(run_path) and not item.startswith('.'):
                # Check if it's a valid run directory (has experiment_config.json)
                config_path = os.path.join(run_path, 'experiment_config.json')
                if os.path.exists(config_path):
                    runs.append(item)

        return sorted(runs)

    def load_run_data(self, run_id: str) -> Dict:
        """Load all data for a specific run"""
        run_path = os.path.join(self.base_dir, run_id)

        if not os.path.exists(run_path):
            print(f"❌ Run directory {run_path} does not exist")
            return {}

        run_data = {'run_id': run_id, 'path': run_path}

        # Load experiment configuration
        config_path = os.path.join(run_path, 'experiment_config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    run_data['config'] = json.load(f)
            except Exception as e:
                print(f"⚠ Could not load config for {run_id}: {e}")
                run_data['config'] = {}
        else:
            run_data['config'] = {}

        # Load latest metrics
        metrics_dir = os.path.join(run_path, 'metrics')
        if os.path.exists(metrics_dir):
            metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
            if metrics_files:
                latest_metrics_file = sorted(metrics_files)[-1]
                metrics_path = os.path.join(metrics_dir, latest_metrics_file)
                try:
                    with open(metrics_path, 'r') as f:
                        run_data['metrics'] = json.load(f)
                except Exception as e:
                    print(f"⚠ Could not load metrics for {run_id}: {e}")
                    run_data['metrics'] = {}

        # Load checkpoints info
        checkpoints_dir = os.path.join(run_path, 'checkpoints')
        if os.path.exists(checkpoints_dir):
            checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
            run_data['checkpoints'] = {
                'count': len(checkpoint_files),
                'files': checkpoint_files
            }
        else:
            run_data['checkpoints'] = {'count': 0, 'files': []}

        return run_data

    def load_multiple_runs(self, run_ids: List[str]) -> Dict:
        """Load data for multiple runs"""
        print(f"Loading data for {len(run_ids)} runs...")

        for run_id in run_ids:
            print(f"  Loading {run_id}...")
            self.runs_data[run_id] = self.load_run_data(run_id)

        successful_runs = len([r for r in self.runs_data.values() if r])
        print(f"✓ Successfully loaded {successful_runs}/{len(run_ids)} runs")

        return self.runs_data

    def compare_configurations(self) -> Dict:
        """Compare experiment configurations across runs"""
        config_comparison = {}

        # Get all unique config keys
        all_keys = set()
        for run_data in self.runs_data.values():
            if 'config' in run_data:
                all_keys.update(run_data['config'].keys())

        # Compare each key
        for key in sorted(all_keys):
            values = {}
            for run_id, run_data in self.runs_data.items():
                if 'config' in run_data and key in run_data['config']:
                    values[run_id] = run_data['config'][key]

            # Check if values differ (interesting for ablation)
            if len(set(str(v) for v in values.values())) > 1:
                config_comparison[key] = values

        return config_comparison

    def compare_performance_metrics(self) -> Dict:
        """Compare performance metrics across runs"""
        performance_comparison = {}

        for run_id, run_data in self.runs_data.items():
            if 'metrics' not in run_data or 'summary' not in run_data['metrics']:
                continue

            summary = run_data['metrics']['summary']
            performance_comparison[run_id] = {
                'final_loss': run_data['metrics']['batch_metrics'][-1]['loss'] if run_data['metrics']['batch_metrics'] else None,
                'avg_loss': summary.get('loss_mean', None),
                'final_ssim': run_data['metrics']['batch_metrics'][-1]['ssim'] if run_data['metrics']['batch_metrics'] else None,
                'avg_ssim': summary.get('ssim_mean', None),
                'total_batches': summary.get('total_batches', 0),
                'num_checkpoints': run_data['checkpoints']['count']
            }

        return performance_comparison

    def create_comparison_visualization(self, output_path: str = None) -> str:
        """Create comprehensive comparison visualization"""
        if output_path is None:
            output_path = os.path.join(self.base_dir, 'ablation_study_comparison.png')

        # Prepare data
        config_comparison = self.compare_configurations()
        performance_comparison = self.compare_performance_metrics()

        if not performance_comparison:
            print("⚠ No performance data available for comparison")
            return None

        # Create figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. Performance overview (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        runs = list(performance_comparison.keys())
        losses = [performance_comparison[r]['final_loss'] for r in runs]
        ssims = [performance_comparison[r]['final_ssim'] for r in runs]

        x = np.arange(len(runs))
        width = 0.35

        ax1_twin = ax1.twinx()
        bars1 = ax1.bar(x - width/2, losses, width, label='Final Loss', alpha=0.8, color='coral')
        bars2 = ax1_twin.bar(x + width/2, ssims, width, label='Final SSIM', alpha=0.8, color='steelblue')

        ax1.set_xlabel('Runs')
        ax1.set_ylabel('Loss', color='coral')
        ax1_twin.set_ylabel('SSIM', color='steelblue')
        ax1.set_title('Performance Comparison Across Runs', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([r[:15] + '...' if len(r) > 15 else r for r in runs], rotation=45)
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars1, losses):
            if value is not None:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(losses)*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=8)

        for bar, value in zip(bars2, ssims):
            if value is not None:
                ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ssims)*0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # 2. Training curves (top right)
        ax2 = fig.add_subplot(gs[0, 2:])
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.runs_data)))

        for i, (run_id, run_data) in enumerate(self.runs_data.items()):
            if 'metrics' in run_data and 'batch_metrics' in run_data['metrics']:
                batch_metrics = run_data['metrics']['batch_metrics']
                losses = [m['loss'] for m in batch_metrics if 'loss' in m]
                if losses:
                    ax2.plot(losses, label=run_id[:10] + '...', color=colors[i], linewidth=2, alpha=0.8)

        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Configuration differences table
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.axis('off')

        if config_comparison:
            # Create table data
            table_data = [['Configuration', 'Values']]
            for key, values in config_comparison.items():
                value_str = ', '.join([f"{run[:8]}: {str(v)[:10]}" for run, v in values.items()])
                table_data.append([key[:20], value_str[:40]])

            if len(table_data) > 1:
                table = ax3.table(cellText=table_data[1:], colLabels=table_data[0],
                                loc='center', cellLoc='left')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
        else:
            ax3.text(0.5, 0.5, 'No configuration differences found',
                    ha='center', va='center', transform=ax3.transAxes)

        ax3.set_title('Configuration Differences', fontsize=14, fontweight='bold')

        # 4. Attention metrics comparison
        ax4 = fig.add_subplot(gs[1, 2:])
        attention_metrics = ['attention_bam_384_attention_entropy_mean',
                           'attention_bam_384_attention_concentration_10_mean',
                           'attention_bam_384_attention_sparsity_mean']

        metric_labels = ['Entropy', 'Concentration', 'Sparsity']
        x = np.arange(len(metric_labels))
        width = 0.8 / len(self.runs_data)

        for i, (run_id, run_data) in enumerate(self.runs_data.items()):
            if 'metrics' in run_data and 'summary' in run_data['metrics']:
                summary = run_data['metrics']['summary']
                values = [summary.get(metric, 0) for metric in attention_metrics]
                ax4.bar(x + i * width, values, width, label=run_id[:10] + '...',
                       color=colors[i], alpha=0.8)

        ax4.set_xlabel('Attention Metrics')
        ax4.set_ylabel('Value')
        ax4.set_title('Attention Metrics Comparison', fontsize=14, fontweight='bold')
        ax4.set_xticks(x + width * (len(self.runs_data) - 1) / 2)
        ax4.set_xticklabels(metric_labels)
        if len(self.runs_data) <= 5:
            ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Summary statistics table
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')

        # Create summary table
        summary_data = [['Run ID', 'Model', 'Final Loss', 'Final SSIM', 'Total Batches', 'Checkpoints']]

        for run_id, perf in performance_comparison.items():
            model_name = self.runs_data[run_id]['config'].get('model_name', 'unknown')
            summary_data.append([
                run_id[:12] + '...' if len(run_id) > 12 else run_id,
                model_name[:10] + '...' if len(model_name) > 10 else model_name,
                f"{perf['final_loss']:.4f}" if perf['final_loss'] else 'N/A',
                f"{perf['final_ssim']:.3f}" if perf['final_ssim'] else 'N/A',
                str(perf['total_batches']),
                str(perf['num_checkpoints'])
            ])

        table = ax5.table(cellText=summary_data[1:], colLabels=summary_data[0],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)

        # Style the table
        for i in range(len(summary_data)):
            for j in range(len(summary_data[0])):
                cell = table[(i, j) if i > 0 else (0, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2')

        ax5.set_title('Ablation Study Summary', fontsize=14, fontweight='bold')

        plt.suptitle('DAE-KAN Ablation Study Comparison', fontsize=18, fontweight='bold', y=0.98)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Comparison visualization saved: {output_path}")
        return output_path

    def generate_comparison_report(self, output_path: str = None) -> str:
        """Generate detailed comparison report"""
        if output_path is None:
            output_path = os.path.join(self.base_dir, 'ablation_study_report.md')

        config_comparison = self.compare_configurations()
        performance_comparison = self.compare_performance_metrics()

        report = []
        report.append("# DAE-KAN Ablation Study Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive summary
        report.append("## Executive Summary\n")
        report.append(f"- **Total Runs Compared**: {len(self.runs_data)}")
        report.append(f"- **Base Directory**: {self.base_dir}")

        if performance_comparison:
            best_loss_run = min(performance_comparison.items(), key=lambda x: x[1]['final_loss'] if x[1]['final_loss'] else float('inf'))
            best_ssim_run = max(performance_comparison.items(), key=lambda x: x[1]['final_ssim'] if x[1]['final_ssim'] else 0)

            report.append(f"- **Best Loss Performance**: {best_loss_run[0]} (Loss: {best_loss_run[1]['final_loss']:.6f})")
            report.append(f"- **Best SSIM Performance**: {best_ssim_run[0]} (SSIM: {best_ssim_run[1]['final_ssim']:.4f})\n")

        # Configuration analysis
        report.append("## Configuration Analysis\n")
        if config_comparison:
            report.append("### Key Configuration Differences\n")
            for key, values in config_comparison.items():
                report.append(f"**{key}**:")
                for run_id, value in values.items():
                    report.append(f"  - {run_id}: {value}")
                report.append("")
        else:
            report.append("No significant configuration differences found between runs.\n")

        # Performance comparison
        report.append("## Performance Comparison\n")
        if performance_comparison:
            report.append("| Run ID | Model | Final Loss | Final SSIM | Total Batches |\n")
            report.append("|--------|-------|------------|------------|---------------|\n")

            for run_id, perf in sorted(performance_comparison.items(), key=lambda x: x[1]['final_loss'] or float('inf')):
                model_name = self.runs_data[run_id]['config'].get('model_name', 'unknown')
                report.append(f"| {run_id[:15]} | {model_name} | {perf['final_loss']:.6f} | {perf['final_ssim']:.4f} | {perf['total_batches']} |\n")

        # Recommendations
        report.append("## Recommendations\n")
        if performance_comparison:
            # Identify best performing configuration
            best_run = min(performance_comparison.items(), key=lambda x: x[1]['final_loss'] if x[1]['final_loss'] else float('inf'))
            best_config = self.runs_data[best_run[0]]['config']

            report.append(f"### Best Performing Configuration\n")
            report.append(f"The best performing run is **{best_run[0]}** with:\n")

            for key, value in best_config.items():
                if key not in ['created_at', 'device', 'pytorch_version']:
                    report.append(f"- **{key}**: {value}\n")

        report.append("### Next Steps\n")
        report.append("1. **Analyze attention patterns**: Compare attention mechanisms across runs")
        report.append("2. **Hyperparameter optimization**: Focus on promising configurations")
        report.append("3. **Extended training**: Train best configurations for more epochs")
        report.append("4. **Cross-validation**: Validate results on multiple data splits")

        report_text = "\n".join(report)

        with open(output_path, 'w') as f:
            f.write(report_text)

        print(f"✓ Comparison report saved: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description='DAE-KAN Ablation Study Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare specific runs
  python ablation_study_comparison.py --base_dir auto_analysis --runs run1 run2 run3

  # Compare all available runs
  python ablation_study_comparison.py --base_dir auto_analysis --all_runs

  # Use configuration file
  python ablation_study_comparison.py --config ablation_config.json
        """
    )

    parser.add_argument('--base_dir', type=str, default='auto_analysis',
                       help='Base directory containing runs (default: auto_analysis)')
    parser.add_argument('--runs', nargs='+',
                       help='List of run IDs to compare')
    parser.add_argument('--all_runs', action='store_true',
                       help='Compare all available runs')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Output directory for results (default: current directory)')
    parser.add_argument('--config', type=str,
                       help='Configuration file for comparison settings')

    args = parser.parse_args()

    # Validate arguments
    if not args.runs and not args.all_runs:
        parser.error("Either --runs or --all_runs must be specified")

    if args.runs and args.all_runs:
        parser.error("Cannot use both --runs and --all_runs")

    # Initialize comparator
    comparator = AblationStudyComparator(args.base_dir)

    # Get runs to compare
    if args.all_runs:
        runs = comparator.discover_runs()
        if not runs:
            print("❌ No runs found in base directory")
            return
        print(f"Found {len(runs)} runs: {', '.join(runs)}")
    else:
        runs = args.runs

    # Load run data
    comparator.load_multiple_runs(runs)

    if len(comparator.runs_data) < 2:
        print("❌ Need at least 2 runs for comparison")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate comparison
    print("\n📊 Generating ablation study comparison...")

    # Create visualization
    viz_path = os.path.join(args.output_dir, 'ablation_study_comparison.png')
    comparator.create_comparison_visualization(viz_path)

    # Generate report
    report_path = os.path.join(args.output_dir, 'ablation_study_report.md')
    comparator.generate_comparison_report(report_path)

    print(f"\n✅ Ablation study comparison completed!")
    print(f"📈 Visualization: {viz_path}")
    print(f"📄 Report: {report_path}")


if __name__ == "__main__":
    main()