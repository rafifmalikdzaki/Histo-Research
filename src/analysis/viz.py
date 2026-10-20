"""Batch visualisation and paper-dashboard builders (mixin)."""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
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
