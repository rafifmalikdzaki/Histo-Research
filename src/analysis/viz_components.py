"""Per-component figure writer; calls VizMixin._create_html_summary (mixin for AutomaticAnalyzer).

Split out of ``viz.py``; runs on shared analyzer state (batch metrics,
attention maps, subdir paths, wandb helpers) owned by AutomaticAnalyzer.
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


class VizComponentsMixin:
    """See AutomaticAnalyzer."""

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
