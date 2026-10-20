"""Cross-run comparison figures for ablation studies."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["create_ablation_study_comparison"]


def create_ablation_study_comparison(base_dir: str, run_ids: list[str],
                                     save_path: str = None) -> str:
    """
    Create ablation study comparison across multiple runs

    Args:
        base_dir: Base directory containing all runs
        run_ids: List of run IDs to compare
        save_path: Path to save comparison figure

    Returns:
        Path to saved comparison figure
    """

    if save_path is None:
        save_path = os.path.join(base_dir, 'ablation_study_comparison.png')

    # Collect data from all runs
    all_data = {}

    for run_id in run_ids:
        run_dir = os.path.join(base_dir, run_id)
        metrics_dir = os.path.join(run_dir, 'metrics')

        if not os.path.exists(metrics_dir):
            continue

        # Load latest metrics file
        metrics_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
        if not metrics_files:
            continue

        latest_metrics_file = sorted(metrics_files)[-1]
        metrics_path = os.path.join(metrics_dir, latest_metrics_file)

        try:
            with open(metrics_path) as f:
                data = json.load(f)
                all_data[run_id] = data
        except Exception as e:
            print(f"⚠ Could not load metrics for {run_id}: {e}")
            continue

    if len(all_data) < 2:
        print("⚠ Need at least 2 runs for comparison")
        return None

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ablation Study Comparison Across Runs', fontsize=16, fontweight='bold')

    # Colors for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))

    # 1. Loss comparison
    ax1 = axes[0, 0]
    for i, (run_id, data) in enumerate(all_data.items()):
        if 'batch_metrics' in data:
            losses = [m['loss'] for m in data['batch_metrics'] if 'loss' in m]
            ax1.plot(losses, label=run_id, color=colors[i], linewidth=2, alpha=0.8)
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. SSIM comparison
    ax2 = axes[0, 1]
    for i, (run_id, data) in enumerate(all_data.items()):
        if 'batch_metrics' in data:
            ssims = [m['ssim'] for m in data['batch_metrics'] if 'ssim' in m]
            ax2.plot(ssims, label=run_id, color=colors[i], linewidth=2, alpha=0.8)
    ax2.set_title('SSIM Comparison')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('SSIM')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Attention concentration comparison
    ax3 = axes[0, 2]
    for i, (run_id, data) in enumerate(all_data.items()):
        if 'batch_metrics' in data:
            concentrations = [m.get('attention_bam_384_attention_concentration_10', 0)
                            for m in data['batch_metrics']]
            ax3.plot(concentrations, label=run_id, color=colors[i], linewidth=2, alpha=0.8)
    ax3.set_title('Attention Concentration Comparison')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Concentration')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Final metrics comparison (bar chart)
    ax4 = axes[1, 0]
    metrics_to_compare = ['loss', 'ssim']

    x = np.arange(len(metrics_to_compare))
    width = 0.8 / len(all_data)

    for i, (run_id, data) in enumerate(all_data.items()):
        if 'summary' in data:
            values = []
            for metric in metrics_to_compare:
                if metric == 'loss':
                    # Get final loss value
                    final_loss = data['batch_metrics'][-1]['loss'] if data['batch_metrics'] else 0
                    values.append(final_loss)
                elif f'{metric}_mean' in data['summary']:
                    values.append(data['summary'][f'{metric}_mean'])
                else:
                    values.append(0)

            ax4.bar(x + i * width, values, width, label=run_id,
                   color=colors[i], alpha=0.8)

    ax4.set_title('Final Metrics Comparison')
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Values')
    ax4.set_xticks(x + width * (len(all_data) - 1) / 2)
    ax4.set_xticklabels(['Loss', 'SSIM'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Attention concentration comparison
    ax5 = axes[1, 1]
    for i, (run_id, data) in enumerate(all_data.items()):
        if 'batch_metrics' in data:
            concentrations = [m.get('attention_bam_384_attention_concentration_10', 0)
                            for m in data['batch_metrics']]
            ax5.plot(concentrations, label=run_id, color=colors[i], linewidth=2, alpha=0.8)
    ax5.set_title('Attention Concentration Comparison')
    ax5.set_xlabel('Batch')
    ax5.set_ylabel('Concentration')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Summary statistics table
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Create summary table
    table_data = [['Run ID', 'Final Loss', 'Final SSIM', 'Total Batches']]

    for run_id, data in all_data.items():
        if 'batch_metrics' in data and len(data['batch_metrics']) > 0:
            final_metrics = data['batch_metrics'][-1]

            table_data.append([
                run_id[:15] + '...' if len(run_id) > 15 else run_id,
                f"{final_metrics.get('loss', 0):.4f}",
                f"{final_metrics.get('ssim', 0):.4f}",
                str(len(data['batch_metrics']))
            ])

    if len(table_data) > 1:
        table = ax6.table(cellText=table_data[1:], colLabels=table_data[0],
                        loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style the table
        for i in range(len(table_data)):
            for j in range(len(table_data[0])):
                cell = table[(i, j) if i > 0 else (0, j)]
                if i == 0:  # Header row
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Ablation study comparison saved: {save_path}")
    return save_path
