"""Paper-dashboard builder (mixin for AutomaticAnalyzer).

Split out of ``viz.py``; runs on shared analyzer state (batch metrics,
attention maps, subdir paths, wandb helpers) owned by AutomaticAnalyzer.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec


class VizDashboardMixin:
    """See AutomaticAnalyzer."""

    def create_comprehensive_paper_dashboard(self, batch_idx: int, input_tensor: torch.Tensor,
                                           output_tensor: torch.Tensor, phase: str = "train", global_step: int = None) -> dict[str, str]:
        """Create comprehensive paper-ready dashboard like the reference image"""

        # Acquire lock to prevent concurrent access
        if not self._acquire_lock():
            print(f"⚠ Could not acquire lock for batch {batch_idx}, skipping analysis")
            return {}

        try:
            # Create main dashboard figure (matches reference layout)
            fig = plt.figure(figsize=(24, 16))

            # Create more complex layout to match reference
            gs = GridSpec(4, 6, figure=fig, hspace=0.25, wspace=0.25,
                         height_ratios=[1.2, 1, 1, 0.8], width_ratios=[1, 1, 1, 1, 1, 1])

            # Get data
            input_np = input_tensor[0].detach().cpu().numpy()
            output_np = output_tensor[0].detach().cpu().numpy()
            img = input_np.transpose(1, 2, 0)
            img_norm = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
            recon = output_np.transpose(1, 2, 0)
            recon_norm = ((recon - recon.min()) / (recon.max() - recon.min())).astype(np.float32)

            # === TOP ROW: Images and Reconstruction ===

            # Original image
            ax_orig = fig.add_subplot(gs[0, 0])
            ax_orig.imshow(img_norm)
            ax_orig.set_title('Original Image', fontsize=12, fontweight='bold')
            ax_orig.axis('off')

            # Reconstructed image
            ax_recon = fig.add_subplot(gs[0, 1])
            ax_recon.imshow(recon_norm)
            ax_recon.set_title('Reconstructed Image', fontsize=12, fontweight='bold')
            ax_recon.axis('off')

            # Error map
            ax_error = fig.add_subplot(gs[0, 2])
            error_map = np.mean((img_norm.astype(np.float32) - recon_norm.astype(np.float32)) ** 2, axis=2).astype(np.float32)
            im_error = ax_error.imshow(error_map, cmap='hot')
            ax_error.set_title('Reconstruction Error', fontsize=12, fontweight='bold')
            ax_error.axis('off')
            plt.colorbar(im_error, ax=ax_error, fraction=0.046)

            # Attention maps (multiple)
            bam_keys = [k for k in self.attention_data.keys() if 'bam' in k]
            attention_figs = {}

            for i, bam_key in enumerate(bam_keys[:3]):  # Up to 3 attention maps
                if i < 3:  # Only first 3 columns
                    ax_attn = fig.add_subplot(gs[0, 3+i])
                    attention = self.attention_data[bam_key]['data']
                    attention_map = attention[0]
                    if attention_map.ndim == 4:
                        attention_map = np.mean(attention_map, axis=0)
                    elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                        attention_map = np.mean(attention_map, axis=0)

                    # Ensure attention map is float32 for matplotlib compatibility
                    attention_map_float = attention_map.astype(np.float32)
                    im_attn = ax_attn.imshow(attention_map_float, cmap='jet')
                    ax_attn.set_title(f'Attention: {bam_key}', fontsize=12, fontweight='bold')
                    ax_attn.axis('off')
                    plt.colorbar(im_attn, ax=ax_attn, fraction=0.046)
                    attention_figs[bam_key] = attention_map_float

            # === SECOND ROW: Training Progress ===

            # Loss and SSIM curves
            ax_progress = fig.add_subplot(gs[1, 0:3])
            if len(self.batch_metrics) > 1:
                losses = [m['loss'] for m in self.batch_metrics]
                ssims = [m['ssim'] for m in self.batch_metrics]

                ax_progress_twin = ax_progress.twinx()
                line1 = ax_progress.plot(range(len(losses)), losses, 'b-', label='Loss', linewidth=2, alpha=0.8)
                line2 = ax_progress_twin.plot(range(len(ssims)), ssims, 'r-', label='SSIM', linewidth=2, alpha=0.8)

                ax_progress.set_xlabel('Training Batches', fontsize=11)
                ax_progress.set_ylabel('Loss', color='b', fontsize=11)
                ax_progress_twin.set_ylabel('SSIM', color='r', fontsize=11)
                ax_progress.set_title('Training Progress', fontsize=12, fontweight='bold')
                ax_progress.grid(True, alpha=0.3)

                # Combine legends
                lines = line1 + line2
                labels = [ln.get_label() for ln in lines]
                ax_progress.legend(lines, labels, loc='upper left')

            # Attention quality metrics
            ax_metrics = fig.add_subplot(gs[1, 3:])
            if len(self.batch_metrics) > 0:
                latest_metrics = self.batch_metrics[-1]
                attention_metrics = {k: v for k, v in latest_metrics.items()
                                   if 'attention_' in k and any(word in k for word in ['concentration', 'sparsity', 'mean'])}

                if attention_metrics:
                    names = [k.replace('attention_bam_384_', '').replace('attention_bam_16_', '').replace('_', ' ').title()
                            for k in attention_metrics.keys()]
                    values = list(attention_metrics.values())

                    # Create bar chart with better colors
                    colors = ['#A23B72', '#F18F01', '#C73E1D', '#2E86AB']
                    bars = ax_metrics.bar(range(len(values)), values, color=colors[:len(values)], alpha=0.8)
                    ax_metrics.set_xticks(range(len(names)))
                    ax_metrics.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
                    ax_metrics.set_ylabel('Value', fontsize=11)
                    ax_metrics.set_title('Attention Quality Metrics', fontsize=12, fontweight='bold')
                    ax_metrics.grid(True, alpha=0.3, axis='y')

                    # Add value labels
                    for bar, value in zip(bars, values):
                        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)

            # === THIRD ROW: Attention Analysis ===

            # Attention evolution over time
            ax_evolution = fig.add_subplot(gs[2, 0:3])
            if len(self.batch_metrics) > 20:
                concentration_values = []
                sparsity_values = []

                for metrics in self.batch_metrics[-100:]:  # Last 100 batches
                    if 'attention_bam_384_attention_concentration_10' in metrics:
                        concentration_values.append(metrics['attention_bam_384_attention_concentration_10'])
                    if 'attention_bam_384_attention_sparsity' in metrics:
                        sparsity_values.append(metrics['attention_bam_384_attention_sparsity'])

                if concentration_values:
                    ax_evolution.plot(range(len(concentration_values)), concentration_values, 'r-', label='Concentration', linewidth=2, alpha=0.8)
                if sparsity_values:
                    ax_evolution.plot(range(len(sparsity_values)), sparsity_values, 'b-', label='Sparsity', linewidth=2, alpha=0.8)

                ax_evolution.set_xlabel('Recent Batches', fontsize=11)
                ax_evolution.set_ylabel('Value', fontsize=11)
                ax_evolution.set_title('Attention Evolution (Last 100 Batches)', fontsize=12, fontweight='bold')
                ax_evolution.legend()
                ax_evolution.grid(True, alpha=0.3)

            # Spatial attention analysis
            ax_spatial = fig.add_subplot(gs[2, 3:])
            if bam_keys and bam_keys[0] in attention_figs:
                attention_map = attention_figs[bam_keys[0]]

                # Create spatial distribution plot
                h, w = attention_map.shape
                y_coords, x_coords = np.mgrid[0:h, 0:w]

                # Flatten coordinates weighted by attention
                attention_flat = attention_map.flatten()
                y_flat = y_coords.flatten()
                x_flat = x_coords.flatten()

                # Create scatter plot
                scatter = ax_spatial.scatter(x_flat, y_flat, c=attention_flat, cmap='viridis',
                                           s=1, alpha=0.5)
                ax_spatial.set_xlabel('X Coordinate', fontsize=11)
                ax_spatial.set_ylabel('Y Coordinate', fontsize=11)
                ax_spatial.set_title(f'Spatial Attention Distribution ({bam_keys[0]})', fontsize=12, fontweight='bold')
                ax_spatial.invert_yaxis()  # Match image coordinates
                plt.colorbar(scatter, ax=ax_spatial, fraction=0.046)

            # === BOTTOM ROW: Summary Statistics ===

            ax_summary = fig.add_subplot(gs[3, :])
            ax_summary.axis('off')

            if len(self.batch_metrics) > 0:
                latest_metrics = self.batch_metrics[-1]

                # Create formatted summary text
                summary_text = f"""
DAE-KAN Training Analysis Summary | {phase.upper()} Batch {batch_idx:06d} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:    Loss: {latest_metrics['loss']:.6f}    |    MSE: {latest_metrics['mse']:.6f}    |    SSIM: {latest_metrics['ssim']:.4f}
Attention Quality:     Concentration: {latest_metrics.get('attention_bam_384_attention_concentration_10', 0):.3f}    |    Sparsity: {latest_metrics.get('attention_bam_384_attention_sparsity', 0):.3f}
Spatial Analysis:      Center Distance: {latest_metrics.get('attention_bam_384_attention_center_distance', 0):.3f}    |    Spatial Variance: {latest_metrics.get('attention_bam_384_attention_spatial_variance', 0):.3f}    |    Peak Count: {latest_metrics.get('attention_bam_384_num_attention_peaks', 0):.0f}
Training Progress:     Total Batches: {len(self.batch_metrics):04d}    |    Current Epoch: {self._estimate_epoch(batch_idx)}    |    Attention Layers: {len(latest_metrics.get('attention_layers', []))}
                """

                ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                              fontsize=11, verticalalignment='center', horizontalalignment='center',
                              fontfamily='monospace',
                              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

            # Main title
            plt.suptitle('DAE-KAN Attention Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)

            # Don't save the combined dashboard - only save individual figures
            paper_figs_dir = self.get_subdir_path('paper_figures')

            individual_figs = {}

            # 1. Reconstruction quality figure
            fig1 = plt.figure(figsize=(15, 5))
            gs1 = GridSpec(1, 4, figure=fig1, wspace=0.1)

            ax1 = fig1.add_subplot(gs1[0])
            ax1.imshow(img_norm)
            ax1.set_title('Original', fontsize=14, fontweight='bold')
            ax1.axis('off')

            ax2 = fig1.add_subplot(gs1[1])
            ax2.imshow(recon_norm)
            ax2.set_title('Reconstructed', fontsize=14, fontweight='bold')
            ax2.axis('off')

            ax3 = fig1.add_subplot(gs1[2])
            im3 = ax3.imshow(error_map, cmap='hot')
            ax3.set_title('Error Map', fontsize=14, fontweight='bold')
            ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046)

            ax4 = fig1.add_subplot(gs1[3])
            if bam_keys:
                attention_map = attention_figs.get(bam_keys[0])
                if attention_map is not None:
                    # Ensure attention map is float32 for matplotlib compatibility
                    attention_map_float = attention_map.astype(np.float32)
                    im4 = ax4.imshow(attention_map_float, cmap='jet')
                    ax4.set_title('Attention Map', fontsize=14, fontweight='bold')
                    ax4.axis('off')
                    plt.colorbar(im4, ax=ax4, fraction=0.046)

            plt.suptitle('DAE-KAN Reconstruction Quality Analysis', fontsize=16, fontweight='bold')
            recon_fig_path = os.path.join(paper_figs_dir, 'figure1_reconstruction_quality.png')
            plt.savefig(recon_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            individual_figs['reconstruction_quality'] = recon_fig_path

            # 2. Training progress figure
            fig2 = plt.figure(figsize=(12, 8))
            gs2 = GridSpec(2, 2, figure=fig2, hspace=0.25, wspace=0.25)

            # Loss curve
            ax_loss = fig2.add_subplot(gs2[0, 0])
            if len(self.batch_metrics) > 1:
                losses = [m['loss'] for m in self.batch_metrics]
                ax_loss.plot(range(len(losses)), losses, 'b-', linewidth=2, alpha=0.8)
                ax_loss.set_xlabel('Batch', fontsize=12)
                ax_loss.set_ylabel('Loss', fontsize=12)
                ax_loss.set_title('Training Loss', fontsize=14, fontweight='bold')
                ax_loss.grid(True, alpha=0.3)

            # SSIM curve
            ax_ssim = fig2.add_subplot(gs2[0, 1])
            if len(self.batch_metrics) > 1:
                ssims = [m['ssim'] for m in self.batch_metrics]
                ax_ssim.plot(range(len(ssims)), ssims, 'r-', linewidth=2, alpha=0.8)
                ax_ssim.set_xlabel('Batch', fontsize=12)
                ax_ssim.set_ylabel('SSIM', fontsize=12)
                ax_ssim.set_title('Structural Similarity (SSIM)', fontsize=14, fontweight='bold')
                ax_ssim.grid(True, alpha=0.3)

            # Attention evolution
            ax_attn_evo = fig2.add_subplot(gs2[1, :])
            if len(self.batch_metrics) > 20:
                concentration_values = []

                for metrics in self.batch_metrics[-200:]:
                    if 'attention_bam_384_attention_concentration_10' in metrics:
                        concentration_values.append(metrics['attention_bam_384_attention_concentration_10'])

                if concentration_values:
                    ax_attn_evo.plot(range(len(concentration_values)), concentration_values, 'r-', label='Concentration', linewidth=2, alpha=0.8)

                ax_attn_evo.set_xlabel('Batches', fontsize=12)
                ax_attn_evo.set_ylabel('Value', fontsize=12)
                ax_attn_evo.set_title('Attention Quality Evolution', fontsize=14, fontweight='bold')
                ax_attn_evo.legend()
                ax_attn_evo.grid(True, alpha=0.3)

            plt.suptitle('DAE-KAN Training Progress Analysis', fontsize=16, fontweight='bold')
            progress_fig_path = os.path.join(paper_figs_dir, 'figure2_training_progress.png')
            plt.savefig(progress_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            individual_figs['training_progress'] = progress_fig_path

            # 3. Attention analysis figure
            fig3 = plt.figure(figsize=(15, 10))
            gs3 = GridSpec(3, 4, figure=fig3, hspace=0.3, wspace=0.3)

            # Multiple attention maps
            for i, bam_key in enumerate(bam_keys[:2]):
                attention_map = attention_figs.get(bam_key)
                if attention_map is not None:
                    # Attention heatmap
                    ax_attn_map = fig3.add_subplot(gs3[i, 0])
                    # Ensure attention map is float32 for matplotlib compatibility
                    attention_map_float = attention_map.astype(np.float32)
                    im = ax_attn_map.imshow(attention_map_float, cmap='jet')
                    ax_attn_map.set_title(f'{bam_key} Heatmap', fontsize=12, fontweight='bold')
                    ax_attn_map.axis('off')
                    plt.colorbar(im, ax=ax_attn_map, fraction=0.046)

                    # Spatial distribution
                    ax_spatial = fig3.add_subplot(gs3[i, 1])
                    h, w = attention_map.shape
                    y_coords, x_coords = np.mgrid[0:h, 0:w]
                    attention_flat = attention_map.flatten()
                    y_flat = y_coords.flatten()
                    x_flat = x_coords.flatten()

                    scatter = ax_spatial.scatter(x_flat, y_flat, c=attention_flat, cmap='viridis',
                                               s=1, alpha=0.5)
                    ax_spatial.set_title(f'{bam_key} Spatial Distribution', fontsize=12, fontweight='bold')
                    ax_spatial.invert_yaxis()
                    plt.colorbar(scatter, ax=ax_spatial, fraction=0.046)

                    # Attention intensity histogram
                    ax_hist = fig3.add_subplot(gs3[i, 2])
                    ax_hist.hist(attention_flat.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                    ax_hist.set_xlabel('Attention Intensity', fontsize=10)
                    ax_hist.set_ylabel('Frequency', fontsize=10)
                    ax_hist.set_title(f'{bam_key} Intensity Distribution', fontsize=12, fontweight='bold')
                    ax_hist.grid(True, alpha=0.3)

            # Metrics comparison
            ax_metrics_comp = fig3.add_subplot(gs3[:2, 3])
            if len(self.batch_metrics) > 10:
                recent_metrics = self.batch_metrics[-10:]

                concentration_vals = [m.get('attention_bam_384_attention_concentration_10', 0) for m in recent_metrics]
                sparsity_vals = [m.get('attention_bam_384_attention_sparsity', 0) for m in recent_metrics]

                x = range(len(recent_metrics))
                width = 0.4

                ax_metrics_comp.bar(x, concentration_vals, width, label='Concentration', alpha=0.8)
                ax_metrics_comp.bar([i + width for i in x], sparsity_vals, width, label='Sparsity', alpha=0.8)

                ax_metrics_comp.set_xlabel('Recent Batches', fontsize=10)
                ax_metrics_comp.set_ylabel('Value', fontsize=10)
                ax_metrics_comp.set_title('Recent Attention Metrics', fontsize=12, fontweight='bold')
                ax_metrics_comp.legend()
                ax_metrics_comp.grid(True, alpha=0.3)

            # Summary statistics table
            ax_table = fig3.add_subplot(gs3[2, :])
            ax_table.axis('off')

            if len(self.batch_metrics) > 0:
                latest_metrics = self.batch_metrics[-1]

                # Create table data
                table_data = [
                    ['Metric', 'BAM-384', 'BAM-16'],
                    ['Concentration', f"{latest_metrics.get('attention_bam_384_attention_concentration_10', 0):.3f}",
                     f"{latest_metrics.get('attention_bam_16_attention_concentration_10', 0):.3f}"],
                    ['Sparsity', f"{latest_metrics.get('attention_bam_384_attention_sparsity', 0):.3f}",
                     f"{latest_metrics.get('attention_bam_16_attention_sparsity', 0):.3f}"],
                    ['Peak Count', f"{latest_metrics.get('attention_bam_384_num_attention_peaks', 0):.0f}",
                     f"{latest_metrics.get('attention_bam_16_num_attention_peaks', 0):.0f}"],
                    ['Center Distance', f"{latest_metrics.get('attention_bam_384_attention_center_distance', 0):.3f}",
                     f"{latest_metrics.get('attention_bam_16_attention_center_distance', 0):.3f}"]
                ]

                table = ax_table.table(cellText=table_data, loc='center', cellLoc='center',
                                     colWidths=[0.25, 0.375, 0.375])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)

                # Style the table
                for i in range(len(table_data)):
                    for j in range(len(table_data[0])):
                        cell = table[i, j]
                        if i == 0:  # Header row
                            cell.set_facecolor('#40466e')
                            cell.set_text_props(weight='bold', color='white')
                        else:
                            cell.set_facecolor('#f1f1f2')

            plt.suptitle('DAE-KAN Attention Mechanism Analysis', fontsize=16, fontweight='bold')
            attention_fig_path = os.path.join(paper_figs_dir, 'figure3_attention_analysis.png')
            plt.savefig(attention_fig_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            individual_figs['attention_analysis'] = attention_fig_path

            plt.close(fig)  # Close main dashboard figure without saving

            # Log paper figures to W&B if enabled and at the right frequency
            if self.should_log_to_wandb('paper_figures', batch_idx):
                log_step = global_step if global_step is not None else batch_idx
                for fig_name, fig_path in individual_figs.items():
                    self.log_image_to_wandb_safe(fig_path, f"{phase}/paper_{fig_name}", step=log_step)

            return {
                'individual_figures': individual_figs
            }
        finally:
            # Release lock
            self._release_lock()
