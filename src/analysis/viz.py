"""Batch visualisation and paper-dashboard builders (mixin)."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from matplotlib.gridspec import GridSpec


class VizMixin:
    """Visualisation builders; see AutomaticAnalyzer."""

    def create_batch_visualization(self, batch_idx: int, input_tensor: torch.Tensor,
                                 output_tensor: torch.Tensor, phase: str = "train", global_step: int = None) -> str:
        """Create visualization for current batch"""

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Original and reconstructed images
        input_np = input_tensor[0].detach().cpu().numpy()
        output_np = output_tensor[0].detach().cpu().numpy()

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        img = input_np.transpose(1, 2, 0)
        img_norm = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
        ax1.imshow(img_norm)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Reconstructed image
        ax2 = fig.add_subplot(gs[0, 1])
        recon = output_np.transpose(1, 2, 0)
        recon_norm = ((recon - recon.min()) / (recon.max() - recon.min())).astype(np.float32)
        ax2.imshow(recon_norm)
        ax2.set_title('Reconstructed Image')
        ax2.axis('off')

        # Error map
        ax3 = fig.add_subplot(gs[0, 2])
        error_map = np.mean((img_norm.astype(np.float32) - recon_norm.astype(np.float32)) ** 2, axis=2).astype(np.float32)
        im = ax3.imshow(error_map, cmap='hot')
        ax3.set_title('Reconstruction Error')
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)

        # Attention heatmap (if available)
        ax4 = fig.add_subplot(gs[0, 3])
        bam_keys = [k for k in self.attention_data.keys() if 'bam' in k]
        if bam_keys:
            attention = self.attention_data[bam_keys[0]]['data']
            attention_map = attention[0]
            if attention_map.ndim == 4:
                attention_map = np.mean(attention_map, axis=0)
            elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                attention_map = np.mean(attention_map, axis=0)

            # Ensure attention map is float32 for matplotlib compatibility
            attention_map_float = attention_map.astype(np.float32)
            im = ax4.imshow(attention_map_float, cmap='jet')
            ax4.set_title(f'Attention: {bam_keys[0]}')
            ax4.axis('off')
            plt.colorbar(im, ax=ax4, fraction=0.046)

        # Training metrics over time
        ax5 = fig.add_subplot(gs[1, 0:2])
        if len(self.batch_metrics) > 1:
            losses = [m['loss'] for m in self.batch_metrics]
            ssims = [m['ssim'] for m in self.batch_metrics]

            ax5_twin = ax5.twinx()
            line1 = ax5.plot(range(len(losses)), losses, 'b-', label='Loss', alpha=0.7)
            line2 = ax5_twin.plot(range(len(ssims)), ssims, 'r-', label='SSIM', alpha=0.7)

            ax5.set_xlabel('Batch')
            ax5.set_ylabel('Loss', color='b')
            ax5_twin.set_ylabel('SSIM', color='r')
            ax5.set_title('Training Progress')
            ax5.grid(True, alpha=0.3)

            # Combine legends
            lines = line1 + line2
            labels = [ln.get_label() for ln in lines]
            ax5.legend(lines, labels, loc='upper left')

        # Attention metrics
        ax6 = fig.add_subplot(gs[1, 2:4])
        if len(self.batch_metrics) > 0:
            latest_metrics = self.batch_metrics[-1]
            attention_metrics = {k: v for k, v in latest_metrics.items() if 'attention_' in k and '_mean' in k}

            if attention_metrics:
                names = [k.replace('attention_', '').replace('_mean', '') for k in attention_metrics.keys()]
                values = list(attention_metrics.values())

                bars = ax6.bar(range(len(values)), values, alpha=0.7)
                ax6.set_xticks(range(len(names)))
                ax6.set_xticklabels(names, rotation=45, ha='right')
                ax6.set_ylabel('Value')
                ax6.set_title('Latest Attention Metrics')
                ax6.grid(True, alpha=0.3)

                # Add value labels
                for bar, value in zip(bars, values):
                    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)

        # Attention evolution
        ax7 = fig.add_subplot(gs[2, 0:2])
        if len(self.batch_metrics) > 10:
            concentration_values = []

            for metrics in self.batch_metrics[-50:]:  # Last 50 batches
                if 'attention_bam_384_attention_concentration_10' in metrics:
                    concentration_values.append(metrics['attention_bam_384_attention_concentration_10'])

            if concentration_values:
                ax7.plot(range(len(concentration_values)), concentration_values, 'r-', label='Concentration', alpha=0.7)

            ax7.set_xlabel('Recent Batches')
            ax7.set_ylabel('Value')
            ax7.set_title('Attention Evolution (Recent 50 Batches)')
            ax7.legend()
            ax7.grid(True, alpha=0.3)

        # Summary statistics
        ax8 = fig.add_subplot(gs[2, 2:4])
        ax8.axis('off')

        if len(self.batch_metrics) > 0:
            latest_metrics = self.batch_metrics[-1]
            stats_text = f"""
            Batch {batch_idx} - {phase.upper()} Summary:

            Performance:
            - Loss: {latest_metrics['loss']:.6f}
            - MSE: {latest_metrics['mse']:.6f}
            - SSIM: {latest_metrics['ssim']:.4f}

            Attention Quality:
            - Concentration: {latest_metrics.get('attention_bam_384_attention_concentration_10', 'N/A')}
            - Sparsity: {latest_metrics.get('attention_bam_384_attention_sparsity', 'N/A')}
            - Num Peaks: {latest_metrics.get('attention_bam_384_num_attention_peaks', 'N/A')}

            Progress:
            - Total Batches: {len(self.batch_metrics)}
            - Current Time: {datetime.now().strftime('%H:%M:%S')}
            """

            ax8.text(0.05, 0.95, stats_text, transform=ax8.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace')

        plt.suptitle(f'DAE-KAN Analysis - {phase.upper()} Batch {batch_idx}', fontsize=16, fontweight='bold')

        # Save visualization to visualizations subdirectory
        save_path = os.path.join(self.get_subdir_path('visualizations'), f'live_batch_visualization_{phase}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Log to W&B if enabled and at the right frequency
        if self.should_log_to_wandb('visualizations', batch_idx):
            log_step = global_step if global_step is not None else batch_idx
            self.log_image_to_wandb_safe(save_path, f"{phase}/batch_visualization", step=log_step)

        return save_path
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
    def save_individual_components(self, batch_idx: int, input_tensor: torch.Tensor,
                                 output_tensor: torch.Tensor, phase: str = "train", global_step: int = None) -> dict[str, str]:
        """Save each analysis component as individual figure files"""

        # Acquire lock to prevent concurrent access
        if not self._acquire_lock():
            print(f"⚠ Could not acquire lock for batch {batch_idx}, skipping analysis")
            return {}

        try:
            # Create directory for individual components
            components_dir = self.get_subdir_path('individual_components')

            # Create batch-specific directory with timestamp for better organization
            batch_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_dir = os.path.join(components_dir, f'batch_{batch_idx:06d}_{phase}_{batch_timestamp}')
            os.makedirs(batch_dir, exist_ok=True)

            # Create attention-specific subdirectory
            attention_dir = os.path.join(batch_dir, 'attention_maps')
            os.makedirs(attention_dir, exist_ok=True)

            # Create reconstruction subdirectory
            reconstruction_dir = os.path.join(batch_dir, 'reconstruction_analysis')
            os.makedirs(reconstruction_dir, exist_ok=True)

            # Get data
            input_np = input_tensor[0].detach().cpu().numpy()
            output_np = output_tensor[0].detach().cpu().numpy()
            img = input_np.transpose(1, 2, 0)
            img_norm = ((img - img.min()) / (img.max() - img.min())).astype(np.float32)
            recon = output_np.transpose(1, 2, 0)
            recon_norm = ((recon - recon.min()) / (recon.max() - recon.min())).astype(np.float32)

            saved_files = {}

            # 1. Original Image
            fig_orig = plt.figure(figsize=(8, 8))
            ax_orig = fig_orig.add_subplot(111)
            ax_orig.imshow(img_norm)
            ax_orig.set_title('Original Image', fontsize=14, fontweight='bold')
            ax_orig.axis('off')
            orig_path = os.path.join(reconstruction_dir, 'original.png')
            plt.savefig(orig_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            saved_files['original'] = orig_path

            # 2. Reconstructed Image
            fig_recon = plt.figure(figsize=(8, 8))
            ax_recon = fig_recon.add_subplot(111)
            ax_recon.imshow(recon_norm)
            ax_recon.set_title('Reconstructed Image', fontsize=14, fontweight='bold')
            ax_recon.axis('off')
            recon_path = os.path.join(reconstruction_dir, 'reconstructed.png')
            plt.savefig(recon_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            saved_files['reconstructed'] = recon_path

            # 3. Error Map
            fig_error = plt.figure(figsize=(8, 8))
            ax_error = fig_error.add_subplot(111)
            error_map = np.mean((img_norm.astype(np.float32) - recon_norm.astype(np.float32)) ** 2, axis=2).astype(np.float32)
            im_error = ax_error.imshow(error_map, cmap='hot')
            ax_error.set_title('Reconstruction Error Map', fontsize=14, fontweight='bold')
            ax_error.axis('off')
            plt.colorbar(im_error, ax=ax_error, fraction=0.046)
            error_path = os.path.join(reconstruction_dir, 'error_map.png')
            plt.savefig(error_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            saved_files['error_map'] = error_path

            # 4. Attention Maps (one for each BAM layer)
            bam_keys = [k for k in self.attention_data.keys() if 'bam' in k]

            for bam_key in bam_keys:
                attention = self.attention_data[bam_key]['data']
                attention_map = attention[0]
                if attention_map.ndim == 4:
                    attention_map = np.mean(attention_map, axis=0)
                elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                    attention_map = np.mean(attention_map, axis=0)

                # Attention heatmap
                fig_attn = plt.figure(figsize=(8, 8))
                ax_attn = fig_attn.add_subplot(111)
                # Ensure attention map is float32 for matplotlib compatibility
                attention_map_float = attention_map.astype(np.float32)
                im_attn = ax_attn.imshow(attention_map_float, cmap='jet')
                ax_attn.set_title(f'Attention Map: {bam_key}', fontsize=14, fontweight='bold')
                ax_attn.axis('off')
                plt.colorbar(im_attn, ax=ax_attn, fraction=0.046)
                attn_path = os.path.join(attention_dir, f'attention_map_{bam_key}.png')
                plt.savefig(attn_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files[f'attention_map_{bam_key}'] = attn_path

                # Attention intensity histogram
                fig_hist = plt.figure(figsize=(8, 6))
                ax_hist = fig_hist.add_subplot(111)
                ax_hist.hist(attention_map.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax_hist.set_xlabel('Attention Intensity', fontsize=12)
                ax_hist.set_ylabel('Frequency', fontsize=12)
                ax_hist.set_title(f'Attention Intensity Distribution: {bam_key}', fontsize=14, fontweight='bold')
                ax_hist.grid(True, alpha=0.3)
                hist_path = os.path.join(attention_dir, f'attention_histogram_{bam_key}.png')
                plt.savefig(hist_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files[f'attention_histogram_{bam_key}'] = hist_path

                # Spatial attention distribution
                fig_spatial = plt.figure(figsize=(8, 8))
                ax_spatial = fig_spatial.add_subplot(111)
                h, w = attention_map.shape
                y_coords, x_coords = np.mgrid[0:h, 0:w]
                attention_flat = attention_map.flatten()
                y_flat = y_coords.flatten()
                x_flat = x_coords.flatten()

                scatter = ax_spatial.scatter(x_flat, y_flat, c=attention_flat, cmap='viridis',
                                           s=1, alpha=0.5)
                ax_spatial.set_xlabel('X Coordinate', fontsize=12)
                ax_spatial.set_ylabel('Y Coordinate', fontsize=12)
                ax_spatial.set_title(f'Spatial Attention Distribution: {bam_key}', fontsize=14, fontweight='bold')
                ax_spatial.invert_yaxis()  # Match image coordinates
                plt.colorbar(scatter, ax=ax_spatial, fraction=0.046)
                spatial_path = os.path.join(attention_dir, f'attention_spatial_{bam_key}.png')
                plt.savefig(spatial_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files[f'attention_spatial_{bam_key}'] = spatial_path

                # 4b. Attention Overlay Visualization
                fig_overlay = plt.figure(figsize=(12, 6))

                # Original image
                ax_orig = fig_overlay.add_subplot(1, 3, 1)
                ax_orig.imshow(img_norm)
                ax_orig.set_title('Original Image', fontsize=12, fontweight='bold')
                ax_orig.axis('off')

                # Attention overlay
                ax_overlay = fig_overlay.add_subplot(1, 3, 2)

                # Resize attention map to match image dimensions if needed
                if attention_map.shape[:2] != img_norm.shape[:2]:
                    from skimage.transform import resize
                    attention_resized = resize(attention_map, (img_norm.shape[0], img_norm.shape[1]),
                                           preserve_range=True, anti_aliasing=True)
                else:
                    attention_resized = attention_map

                # Create overlay
                ax_overlay.imshow(img_norm)
                im_overlay = ax_overlay.imshow(attention_resized, cmap='jet', alpha=0.5)
                ax_overlay.set_title(f'Attention Overlay: {bam_key}', fontsize=12, fontweight='bold')
                ax_overlay.axis('off')
                plt.colorbar(im_overlay, ax=ax_overlay, fraction=0.046)

                # Attention only
                ax_attn_only = fig_overlay.add_subplot(1, 3, 3)
                im_attn_only = ax_attn_only.imshow(attention_resized, cmap='jet')
                ax_attn_only.set_title(f'Attention Map: {bam_key}', fontsize=12, fontweight='bold')
                ax_attn_only.axis('off')
                plt.colorbar(im_attn_only, ax=ax_attn_only, fraction=0.046)

                plt.suptitle(f'Attention Analysis: {bam_key}', fontsize=14, fontweight='bold')
                plt.tight_layout()

                overlay_path = os.path.join(attention_dir, f'attention_overlay_{bam_key}.png')
                plt.savefig(overlay_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files[f'attention_overlay_{bam_key}'] = overlay_path

                # 4c. Attention Heatmap with Contours
                fig_contour = plt.figure(figsize=(10, 8))
                ax_contour = fig_contour.add_subplot(111)

                # Background: original image
                ax_contour.imshow(img_norm, alpha=0.7)

                # Attention heatmap with contours
                attention_resized_float = attention_resized.astype(np.float32)
                im_contour = ax_contour.imshow(attention_resized_float, cmap='jet', alpha=0.6)

                # Add contour lines
                contour_levels = np.percentile(attention_resized_float[attention_resized_float > 0],
                                             [20, 40, 60, 80])
                if len(contour_levels) > 1:
                    contours = ax_contour.contour(attention_resized_float, levels=contour_levels,
                                                 colors='white', linewidths=1.5, alpha=0.8)
                    ax_contour.clabel(contours, inline=True, fontsize=10)

                ax_contour.set_title(f'Attention Heatmap with Contours: {bam_key}', fontsize=14, fontweight='bold')
                ax_contour.axis('off')
                plt.colorbar(im_contour, ax=ax_contour, fraction=0.046)

                contour_path = os.path.join(attention_dir, f'attention_contours_{bam_key}.png')
                plt.savefig(contour_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files[f'attention_contours_{bam_key}'] = contour_path

            # 5. Training Progress (Loss and SSIM curves)
            if len(self.batch_metrics) > 1:
                fig_progress = plt.figure(figsize=(12, 6))
                ax_progress = fig_progress.add_subplot(111)
                losses = [m['loss'] for m in self.batch_metrics]
                ssims = [m['ssim'] for m in self.batch_metrics]

                ax_progress_twin = ax_progress.twinx()
                line1 = ax_progress.plot(range(len(losses)), losses, 'b-', label='Loss', linewidth=2, alpha=0.8)
                line2 = ax_progress_twin.plot(range(len(ssims)), ssims, 'r-', label='SSIM', linewidth=2, alpha=0.8)

                ax_progress.set_xlabel('Training Batches', fontsize=12)
                ax_progress.set_ylabel('Loss', color='b', fontsize=12)
                ax_progress_twin.set_ylabel('SSIM', color='r', fontsize=12)
                ax_progress.set_title('Training Progress', fontsize=14, fontweight='bold')
                ax_progress.grid(True, alpha=0.3)

                # Combine legends
                lines = line1 + line2
                labels = [ln.get_label() for ln in lines]
                ax_progress.legend(lines, labels, loc='upper left')

                progress_path = os.path.join(batch_dir, 'training_progress.png')
                plt.savefig(progress_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files['training_progress'] = progress_path

            # 6. Attention Evolution
            if len(self.batch_metrics) > 20:
                fig_evolution = plt.figure(figsize=(12, 6))
                ax_evolution = fig_evolution.add_subplot(111)
                entropy_values = []
                concentration_values = []
                sparsity_values = []

                for metrics in self.batch_metrics[-100:]:  # Last 100 batches
                    if 'attention_bam_384_attention_entropy' in metrics:
                        entropy_values.append(metrics['attention_bam_384_attention_entropy'])
                    if 'attention_bam_384_attention_concentration_10' in metrics:
                        concentration_values.append(metrics['attention_bam_384_attention_concentration_10'])
                    if 'attention_bam_384_attention_sparsity' in metrics:
                        sparsity_values.append(metrics['attention_bam_384_attention_sparsity'])

                if entropy_values:
                    ax_evolution.plot(range(len(entropy_values)), entropy_values, 'g-', label='Entropy', linewidth=2, alpha=0.8)
                if concentration_values:
                    ax_evolution.plot(range(len(concentration_values)), concentration_values, 'r-', label='Concentration', linewidth=2, alpha=0.8)
                if sparsity_values:
                    ax_evolution.plot(range(len(sparsity_values)), sparsity_values, 'b-', label='Sparsity', linewidth=2, alpha=0.8)

                ax_evolution.set_xlabel('Recent Batches', fontsize=12)
                ax_evolution.set_ylabel('Value', fontsize=12)
                ax_evolution.set_title('Attention Evolution (Last 100 Batches)', fontsize=14, fontweight='bold')
                ax_evolution.legend()
                ax_evolution.grid(True, alpha=0.3)

                evolution_path = os.path.join(batch_dir, 'attention_evolution.png')
                plt.savefig(evolution_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files['attention_evolution'] = evolution_path

            # 7. Metrics Summary
            if len(self.batch_metrics) > 0:
                fig_metrics = plt.figure(figsize=(10, 8))
                ax_metrics = fig_metrics.add_subplot(111)
                latest_metrics = self.batch_metrics[-1]
                attention_metrics = {k: v for k, v in latest_metrics.items()
                                   if 'attention_' in k and any(word in k for word in ['entropy', 'concentration', 'sparsity', 'mean'])}

                if attention_metrics:
                    names = [k.replace('attention_bam_384_', '').replace('attention_bam_16_', '').replace('_', ' ').title()
                            for k in attention_metrics.keys()]
                    values = list(attention_metrics.values())

                    # Create bar chart with better colors
                    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
                    bars = ax_metrics.bar(range(len(values)), values, color=colors[:len(values)], alpha=0.8)
                    ax_metrics.set_xticks(range(len(names)))
                    ax_metrics.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
                    ax_metrics.set_ylabel('Value', fontsize=12)
                    ax_metrics.set_title('Attention Quality Metrics', fontsize=14, fontweight='bold')
                    ax_metrics.grid(True, alpha=0.3, axis='y')

                    # Add value labels
                    for bar, value in zip(bars, values):
                        ax_metrics.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                                      f'{value:.3f}', ha='center', va='bottom', fontsize=9)

                metrics_path = os.path.join(batch_dir, 'attention_metrics.png')
                plt.savefig(metrics_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files['attention_metrics'] = metrics_path

            # 8. Summary Statistics
            if len(self.batch_metrics) > 0:
                fig_summary = plt.figure(figsize=(12, 8))
                ax_summary = fig_summary.add_subplot(111)
                ax_summary.axis('off')

                latest_metrics = self.batch_metrics[-1]

                # Create formatted summary text
                summary_text = f"""
DAE-KAN Training Analysis Summary | {phase.upper()} Batch {batch_idx:06d} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Performance Metrics:
- Loss: {latest_metrics['loss']:.6f}
- MSE: {latest_metrics['mse']:.6f}
- SSIM: {latest_metrics['ssim']:.4f}

Attention Quality (BAM-384):
- Entropy: {latest_metrics.get('attention_bam_384_attention_entropy', 0):.3f}
- Concentration: {latest_metrics.get('attention_bam_384_attention_concentration_10', 0):.3f}
- Sparsity: {latest_metrics.get('attention_bam_384_attention_sparsity', 0):.3f}
- Peak Count: {latest_metrics.get('attention_bam_384_num_attention_peaks', 0):.0f}

Spatial Analysis (BAM-384):
- Center Distance: {latest_metrics.get('attention_bam_384_attention_center_distance', 0):.3f}
- Spatial Variance: {latest_metrics.get('attention_bam_384_attention_spatial_variance', 0):.3f}

Training Progress:
- Total Batches: {len(self.batch_metrics):04d}
- Current Epoch: {self._estimate_epoch(batch_idx)}
                """

                ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes,
                              fontsize=11, verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

                summary_path = os.path.join(batch_dir, 'summary_statistics.png')
                plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files['summary_statistics'] = summary_path

            # 9. Create HTML Summary Page
            html_content = self._create_html_summary(batch_idx, phase, batch_dir, saved_files, img_norm, attention_maps=bam_keys)
            html_path = os.path.join(batch_dir, 'attention_analysis_summary.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            saved_files['html_summary'] = html_path

            # Log individual components to W&B if enabled and at the right frequency
            if self.should_log_to_wandb('individual_components', batch_idx):
                log_step = global_step if global_step is not None else batch_idx
                for component_name, component_path in saved_files.items():
                    if component_name != 'html_summary':  # Don't log HTML to W&B
                        self.log_image_to_wandb_safe(component_path, f"{phase}/components_{component_name}", step=log_step)

            return saved_files
        finally:
            # Release lock
            self._release_lock()
    def _create_html_summary(self, batch_idx: int, phase: str, batch_dir: str,
                           saved_files: dict, img_norm: np.ndarray, attention_maps: list[str]) -> str:
        """Create an HTML summary page for easy browsing of attention analysis results"""

        # Calculate relative paths
        def get_relative_path(file_path):
            # Get relative path from batch_dir to file
            if os.path.isabs(file_path):
                return os.path.relpath(file_path, batch_dir)
            return file_path

        # Group files by category
        reconstruction_files = {}
        attention_files = {}
        training_files = {}

        for file_name, file_path in saved_files.items():
            if file_name in ['original', 'reconstructed', 'error_map']:
                reconstruction_files[file_name] = get_relative_path(file_path)
            elif 'attention' in file_name:
                attention_files[file_name] = get_relative_path(file_path)
            elif file_name in ['training_progress', 'attention_evolution', 'attention_metrics', 'summary_statistics']:
                training_files[file_name] = get_relative_path(file_path)

        # Generate HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DAE-KAN Attention Analysis - Batch {batch_idx} ({phase})</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .info {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .image-item {{
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .image-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .image-item h3 {{
            margin-top: 10px;
            margin-bottom: 5px;
            color: #2c3e50;
            font-size: 16px;
        }}
        .attention-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .attention-item {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
        }}
        .attention-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            margin-bottom: 10px;
        }}
        .attention-item h4 {{
            margin: 5px 0;
            color: #495057;
            font-size: 14px;
            font-weight: bold;
        }}
        .file-list {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
        }}
        .file-list ul {{
            list-style-type: none;
            padding: 0;
        }}
        .file-list li {{
            padding: 5px 0;
            border-bottom: 1px solid #e9ecef;
        }}
        .file-list li:last-child {{
            border-bottom: none;
        }}
        .file-list a {{
            color: #007bff;
            text-decoration: none;
        }}
        .file-list a:hover {{
            text-decoration: underline;
        }}
        .timestamp {{
            color: #6c757d;
            font-style: italic;
            text-align: center;
            margin-top: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DAE-KAN Attention Analysis</h1>

        <div class="info">
            <h3>Experiment Information</h3>
            <p><strong>Batch ID:</strong> {batch_idx:06d}</p>
            <p><strong>Phase:</strong> {phase.upper()}</p>
            <p><strong>Run ID:</strong> {self.run_id}</p>
            <p><strong>Timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Total Analysis Files:</strong> {len(saved_files)}</p>
        </div>

        <h2>🖼️ Reconstruction Analysis</h2>
        <div class="image-grid">
            {''.join([f'''
            <div class="image-item">
                <img src="{path}" alt="{title}" loading="lazy">
                <h3>{title.replace('_', ' ').title()}</h3>
            </div>''' for title, path in reconstruction_files.items()])}
        </div>

        <h2>🎯 Attention Map Analysis</h2>
        <div class="attention-grid">
            {''.join([
                f'''<div class="attention-item">
                    <h4>{map_type.replace('attention_', '').replace('_', ' ').title()} - {layer}</h4>
                    <img src="{path}" alt="{map_type}" loading="lazy">
                </div>'''
                for layer in attention_maps
                for map_type, path in [
                    (f'attention_map_{layer}', saved_files.get(f'attention_map_{layer}')),
                    (f'attention_overlay_{layer}', saved_files.get(f'attention_overlay_{layer}')),
                    (f'attention_contours_{layer}', saved_files.get(f'attention_contours_{layer}')),
                    (f'attention_histogram_{layer}', saved_files.get(f'attention_histogram_{layer}')),
                    (f'attention_spatial_{layer}', saved_files.get(f'attention_spatial_{layer}'))
                ]
                if path is not None
            ])}
        </div>

        <h2>📈 Training Progress</h2>
        <div class="image-grid">
            {''.join([f'''
            <div class="image-item">
                <img src="{path}" alt="{title}" loading="lazy">
                <h3>{title.replace('_', ' ').title()}</h3>
            </div>''' for title, path in training_files.items()])}
        </div>

        <h2>📁 All Generated Files</h2>
        <div class="file-list">
            <ul>
                {''.join([f'<li><a href="{get_relative_path(path)}" target="_blank">{file_name}</a></li>'
                         for file_name, path in saved_files.items()])}
            </ul>
        </div>

        <div class="timestamp">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><em>Open this HTML file in a web browser for interactive browsing of all attention analysis results</em></p>
        </div>
    </div>
</body>
</html>
        """

        return html_content
