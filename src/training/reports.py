"""Epoch/embedding analysis reports and visualisations (mixin)."""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.manifold import TSNE


class ReportsMixin:
    """Analysis report builders; see the Lightning module."""

    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        # Simple SSIM calculation
        mu1 = torch.mean(img1, dim=[1, 2, 3])
        mu2 = torch.mean(img2, dim=[1, 2, 3])

        sigma1_sq = torch.var(img1, dim=[1, 2, 3], unbiased=False)
        sigma2_sq = torch.var(img2, dim=[1, 2, 3], unbiased=False)
        sigma12 = torch.mean((img1 - mu1.unsqueeze(1).unsqueeze(2).unsqueeze(3)) *
                              (img2 - mu2.unsqueeze(1).unsqueeze(2).unsqueeze(3)),
                              dim=[1, 2, 3])

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_num = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2))
        ssim_den = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

        ssim = ssim_num / ssim_den
        return torch.mean(ssim)
    def _create_epoch_evolution_analysis(self):
        """Create analysis of how embeddings evolve across epochs"""

        if not hasattr(self, 'auto_analyzer') or not self.auto_analyzer:
            return

        try:
            import glob
            # Create embeddings directory
            embeddings_dir = os.path.join(self.auto_analyzer.save_dir, 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)

            # Find all epoch embedding files
            train_epoch_files = sorted(glob.glob(os.path.join(embeddings_dir, 'train_embeddings_epoch_*.npy')))
            val_epoch_files = sorted(glob.glob(os.path.join(embeddings_dir, 'val_embeddings_epoch_*.npy')))

            if not train_epoch_files:
                print("⚠ No epoch embedding files found for evolution analysis")
                return

            # Load all epoch embeddings
            train_embeddings_by_epoch = []
            val_embeddings_by_epoch = []

            for file_path in train_epoch_files:
                epoch_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
                embeddings = np.load(file_path)
                metadata = pd.read_csv(file_path.replace('_embeddings_epoch_', '_metadata_epoch_').replace('.npy', '.csv'))
                train_embeddings_by_epoch.append({
                    'epoch': epoch_num,
                    'embeddings': embeddings,
                    'metadata': metadata,
                    'file_path': file_path
                })

            for file_path in val_epoch_files:
                epoch_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
                embeddings = np.load(file_path)
                metadata = pd.read_csv(file_path.replace('_embeddings_epoch_', '_metadata_epoch_').replace('.npy', '.csv'))
                val_embeddings_by_epoch.append({
                    'epoch': epoch_num,
                    'embeddings': embeddings,
                    'metadata': metadata,
                    'file_path': file_path
                })

            print(f"Found {len(train_embeddings_by_epoch)} training epochs and {len(val_embeddings_by_epoch)} validation epochs")

            # Create epoch evolution visualizations
            self._create_epoch_evolution_visualizations(train_embeddings_by_epoch, val_embeddings_by_epoch, embeddings_dir)

            # Create epoch evolution HTML report
            self._create_epoch_evolution_html_report(train_embeddings_by_epoch, val_embeddings_by_epoch, embeddings_dir)

        except Exception as e:
            print(f"⚠ Failed to create epoch evolution analysis: {e}")
    def _create_epoch_evolution_visualizations(self, train_epochs: list, val_epochs: list, embeddings_dir: str):
        """Create visualizations showing how embeddings evolve across epochs"""

        try:
            import matplotlib.pyplot as plt

            # Create evolution plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Embedding Evolution Across Training Epochs', fontsize=16, fontweight='bold')

            # 1. Embedding statistics evolution (training)
            ax1 = axes[0, 0]
            epochs = [epoch['epoch'] for epoch in train_epochs]
            means = [np.mean(epoch['embeddings']) for epoch in train_epochs]
            stds = [np.std(epoch['embeddings']) for epoch in train_epochs]
            norms = [np.mean(np.linalg.norm(epoch['embeddings'], axis=1)) for epoch in train_epochs]

            ax1_twin = ax1.twinx()
            ax1.plot(epochs, means, 'b-o', label='Mean', linewidth=2, markersize=6)
            ax1_twin.plot(epochs, stds, 'r-s', label='Std Dev', linewidth=2, markersize=6)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Mean Value', color='b')
            ax1_twin.set_ylabel('Standard Deviation', color='r')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            ax1.set_title('Training Embedding Statistics Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')

            # 2. Embedding norms evolution (training)
            ax2 = axes[0, 1]
            ax2.plot(epochs, norms, 'g-^', linewidth=2, markersize=6)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean L2 Norm')
            ax2.set_title('Training Embedding Norm Evolution')
            ax2.grid(True, alpha=0.3)

            # 3. Embedding sparsity evolution (training)
            ax3 = axes[0, 2]
            sparsities = [np.mean(epoch['embeddings'] == 0) for epoch in train_epochs]
            ax3.plot(epochs, sparsities, 'm-d', linewidth=2, markersize=6)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Sparsity (fraction of zeros)')
            ax3.set_title('Training Embedding Sparsity Evolution')
            ax3.grid(True, alpha=0.3)

            # 4-6. Validation evolution plots (if available)
            if val_epochs:
                val_epochs_list = [epoch['epoch'] for epoch in val_epochs]
                val_means = [np.mean(epoch['embeddings']) for epoch in val_epochs]
                val_stds = [np.std(epoch['embeddings']) for epoch in val_epochs]
                val_norms = [np.mean(np.linalg.norm(epoch['embeddings'], axis=1)) for epoch in val_epochs]
                val_sparsities = [np.mean(epoch['embeddings'] == 0) for epoch in val_epochs]

                # Validation statistics
                ax4 = axes[1, 0]
                ax4_twin = ax4.twinx()
                ax4.plot(val_epochs_list, val_means, 'b-o', label='Mean', linewidth=2, markersize=6)
                ax4_twin.plot(val_epochs_list, val_stds, 'r-s', label='Std Dev', linewidth=2, markersize=6)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Mean Value', color='b')
                ax4_twin.set_ylabel('Standard Deviation', color='r')
                ax4.tick_params(axis='y', labelcolor='b')
                ax4_twin.tick_params(axis='y', labelcolor='r')
                ax4.set_title('Validation Embedding Statistics Evolution')
                ax4.grid(True, alpha=0.3)
                ax4.legend(loc='upper left')

                # Validation norms
                ax5 = axes[1, 1]
                ax5.plot(val_epochs_list, val_norms, 'g-^', linewidth=2, markersize=6)
                ax5.set_xlabel('Epoch')
                ax5.set_ylabel('Mean L2 Norm')
                ax5.set_title('Validation Embedding Norm Evolution')
                ax5.grid(True, alpha=0.3)

                # Validation sparsity
                ax6 = axes[1, 2]
                ax6.plot(val_epochs_list, val_sparsities, 'm-d', linewidth=2, markersize=6)
                ax6.set_xlabel('Epoch')
                ax6.set_ylabel('Sparsity (fraction of zeros)')
                ax6.set_title('Validation Embedding Sparsity Evolution')
                ax6.grid(True, alpha=0.3)
            else:
                # Hide validation plots if not available
                for ax in axes[1, :]:
                    ax.text(0.5, 0.5, 'No validation\nembeddings\navailable',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Validation Data (Not Available)')

            plt.tight_layout()
            evolution_path = os.path.join(embeddings_dir, 'embedding_evolution_analysis.png')
            plt.savefig(evolution_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ Epoch evolution visualizations saved to {embeddings_dir}")

        except ImportError:
            print("⚠ Matplotlib not available for evolution visualizations")
        except Exception as e:
            print(f"⚠ Evolution visualization failed: {e}")
    def _create_epoch_evolution_html_report(self, train_epochs: list, val_epochs: list, embeddings_dir: str):
        """Create HTML report for epoch evolution analysis"""

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Evolution Analysis</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
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
        .stats {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #3498db;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .section {{
            margin: 30px 0;
        }}
        .section h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .epoch-list {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
        }}
        .epoch-item {{
            padding: 10px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .epoch-item:last-child {{
            border-bottom: none;
        }}
        .file-link {{
            color: #007bff;
            text-decoration: none;
            padding: 5px 10px;
            background-color: #e9ecef;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .file-link:hover {{
            background-color: #dee2e6;
            text-decoration: none;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 Embedding Evolution Analysis</h1>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{len(train_epochs)}</div>
                <div class="stat-label">Training Epochs Analyzed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(val_epochs) if val_epochs else 0}</div>
                <div class="stat-label">Validation Epochs Analyzed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{train_epochs[0]['embeddings'].shape[0] if train_epochs else 0}</div>
                <div class="stat-label">Embeddings per Epoch</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{train_epochs[0]['embeddings'].shape[1] if train_epochs else 0}</div>
                <div class="stat-label">Embedding Dimensions</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Evolution Visualizations</h2>
            <div class="image-container">
                <img src="embedding_evolution_analysis.png" alt="Embedding Evolution Analysis" loading="lazy">
            </div>
        </div>

        <div class="section">
            <h2>📁 Training Epoch Files</h2>
            <div class="epoch-list">
                {''.join([f'''
                <div class="epoch-item">
                    <span>Epoch {epoch['epoch']}</span>
                    <a href="{os.path.basename(epoch['file_path'])}" class="file-link" download>Embeddings</a>
                    <a href="{os.path.basename(epoch['file_path']).replace('_embeddings_epoch_', '_metadata_epoch_').replace('.npy', '.csv')}" class="file-link" download>Metadata</a>
                </div>''' for epoch in train_epochs])}
            </div>
        </div>

        <div class="section">
            <h2>📁 Validation Epoch Files</h2>
            <div class="epoch-list">
                {''.join([f'''
                <div class="epoch-item">
                    <span>Epoch {epoch['epoch']}</span>
                    <a href="{os.path.basename(epoch['file_path'])}" class="file-link" download>Embeddings</a>
                    <a href="{os.path.basename(epoch['file_path']).replace('_embeddings_epoch_', '_metadata_epoch_').replace('.npy', '.csv')}" class="file-link" download>Metadata</a>
                </div>''' for epoch in val_epochs]) if val_epochs else '<p>No validation epoch files available</p>'}
            </div>
        </div>

        <div style="text-align: center; margin-top: 30px; color: #6c757d; font-style: italic;">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><em>This report shows how DAE-KAN embeddings evolve across training epochs</em></p>
        </div>
    </div>
</body>
</html>
        """

        html_path = os.path.join(embeddings_dir, 'embedding_evolution_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Epoch evolution HTML report saved to {html_path}")
    def _create_embedding_visualizations(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                       phase: str, embeddings_dir: str):
        """Create various embedding visualizations and analysis"""

        if len(embeddings) == 0:
            return

        print(f"Creating embedding visualizations for {len(embeddings)} {phase} embeddings...")

        # 1. Basic embedding statistics
        self._create_embedding_statistics(embeddings, metadata, phase, embeddings_dir)

        # 2. Dimensionality reduction (UMAP/t-SNE)
        self._create_dimensionality_reduction_plots(embeddings, metadata, phase, embeddings_dir)

        # 3. Embedding clustering analysis
        if self.auto_analyzer:
            self.auto_analyzer.analyze_embedding_clustering(embeddings, metadata, phase, embeddings_dir)

        # 4. Embedding quality metrics
        self._create_embedding_quality_analysis(embeddings, metadata, phase, embeddings_dir)

        # 5. Create HTML summary
        self._create_embedding_html_summary(embeddings, metadata, phase, embeddings_dir)
    def _create_embedding_statistics(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                   phase: str, embeddings_dir: str):
        """Create basic embedding statistics visualization"""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Embedding Statistics - {phase.upper()} Set', fontsize=16, fontweight='bold')

        # 1. Embedding dimension distribution
        ax1 = axes[0, 0]
        embedding_means = np.mean(embeddings, axis=1)
        ax1.hist(embedding_means, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Distribution of Embedding Means')
        ax1.set_xlabel('Mean Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # 2. Embedding variance distribution
        ax2 = axes[0, 1]
        embedding_vars = np.var(embeddings, axis=1)
        ax2.hist(embedding_vars, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_title('Distribution of Embedding Variances')
        ax2.set_xlabel('Variance')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # 3. Embedding sparsity
        ax3 = axes[1, 0]
        sparsity = np.mean(embeddings == 0, axis=1)
        ax3.hist(sparsity, bins=50, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax3.set_title('Embedding Sparsity Distribution')
        ax3.set_xlabel('Sparsity (fraction of zeros)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. Batch progression
        ax4 = axes[1, 1]
        batch_means = []
        for batch_idx in sorted(metadata['batch_idx'].unique()):
            batch_mask = metadata['batch_idx'] == batch_idx
            batch_means.append(np.mean(embeddings[batch_mask]))

        ax4.plot(batch_means, 'o-', color='purple', linewidth=2, markersize=6)
        ax4.set_title('Embedding Means vs Batch Progress')
        ax4.set_xlabel('Batch Index')
        ax4.set_ylabel('Mean Embedding Value')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        stats_path = os.path.join(embeddings_dir, f'{phase}_embedding_statistics.png')
        plt.savefig(stats_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    def create_final_summary_report(self, final_metrics):
        """Create final summary report"""
        if not final_metrics:
            return

        save_dir = self.auto_analyzer.save_dir if self.auto_analyzer else "final_analysis"
        os.makedirs(save_dir, exist_ok=True)

        # Create text report
        report = []
        report.append("# DAE-KAN Training Final Summary\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Training Summary\n")
        report.append(f"- **Total Batches**: {final_metrics.get('total_batches', 'N/A')}")
        report.append(f"- **Final Batch**: {final_metrics.get('latest_batch', 'N/A')}")

        if 'latest_metrics' in final_metrics:
            latest = final_metrics['latest_metrics']
            report.append(f"- **Final Loss**: {latest.get('loss', 'N/A'):.6f}")
            report.append(f"- **Final MSE**: {latest.get('mse', 'N/A'):.6f}")
            report.append(f"- **Final SSIM**: {latest.get('ssim', 'N/A'):.4f}")

        report.append("\n## Attention Analysis Summary\n")
        attention_metrics = [k for k in final_metrics.keys() if 'attention_' in k and 'mean' in k]
        if attention_metrics:
            report.append("Key attention metrics tracked:")
            for metric in sorted(attention_metrics)[:10]:  # Top 10 metrics
                clean_name = metric.replace('attention_', '').replace('_mean', '')
                value = final_metrics[metric]
                report.append(f"- **{clean_name}**: {value:.4f}")

        report.append("\n## Files Generated\n")
        report.append(f"- **Analysis Directory**: {save_dir}")
        report.append("- **Metrics JSON**: final_training_metrics.json")
        report.append("- **Visualizations**: Batch analysis images")

        if self.training_times and self.inference_times:
            report.append("\n## Performance Summary\n")
            report.append(f"- **Avg Training Time**: {np.mean(self.training_times):.2f} ms")
            report.append(f"- **Avg Inference Time**: {np.mean(self.inference_times):.2f} ms")
            report.append(f"- **Throughput**: {1000/np.mean(self.inference_times):.1f} images/sec")

            if self.memory_usage:
                report.append(f"- **Avg Memory Usage**: {np.mean(self.memory_usage):.1f} MB")

        report_text = "\n".join(report)

        # Save report
        report_path = os.path.join(save_dir, "final_summary_report.md")
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"✓ Final summary report saved: {report_path}")
    def _create_dimensionality_reduction_plots(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                             phase: str, embeddings_dir: str):
        """Create UMAP/t-SNE visualizations"""
        try:
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler

            # Sample if too many embeddings
            max_samples = 2000
            if len(embeddings) > max_samples:
                indices = np.random.choice(len(embeddings), max_samples, replace=False)
                embeddings_sample = embeddings[indices]
            else:
                embeddings_sample = embeddings

            # Standardize
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_sample)

            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'{phase.capitalize()} Embedding Dimensionality Reduction', fontsize=24, fontweight='bold')

            # Try UMAP first
            try:
                import umap
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                embedding_2d = reducer.fit_transform(embeddings_scaled)

                axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.7, s=40)
                axes[0].set_title('UMAP Projection', fontsize=22, fontweight='bold')
                axes[0].set_xlabel('UMAP 1', fontsize=20)
                axes[0].set_ylabel('UMAP 2', fontsize=20)
                axes[0].grid(True, alpha=0.3)

            except ImportError:
                axes[0].text(0.5, 0.5, 'UMAP not available\n(pip install umap-learn)',
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('UMAP Projection')

            # t-SNE
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
            embedding_2d = tsne.fit_transform(embeddings_scaled)

            axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6, s=30)
            axes[1].set_title('t-SNE Projection', fontsize=18, fontweight='bold')
            axes[1].set_xlabel('t-SNE 1', fontsize=16)
            axes[1].set_ylabel('t-SNE 2', fontsize=16)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            viz_path = os.path.join(embeddings_dir, f'{phase}_dimensionality_reduction.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        except Exception as e:
            print(f"⚠ Failed to create dimensionality reduction plots: {e}")
    def _create_embedding_quality_analysis(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                         phase: str, embeddings_dir: str):
        """Create embedding quality metrics analysis"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{phase.capitalize()} Embedding Quality Analysis', fontsize=16, fontweight='bold')

            # Embedding norms
            norms = np.linalg.norm(embeddings, axis=1)
            axes[0, 0].hist(norms, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Distribution of Embedding Norms')
            axes[0, 0].set_xlabel('L2 Norm')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)

            # Embedding means
            means = np.mean(embeddings, axis=1)
            axes[0, 1].hist(means, bins=50, alpha=0.7, edgecolor='black', color='coral')
            axes[0, 1].set_title('Distribution of Embedding Means')
            axes[0, 1].set_xlabel('Mean Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)

            # Embedding variances
            variances = np.var(embeddings, axis=1)
            axes[1, 0].hist(variances, bins=50, alpha=0.7, edgecolor='black', color='green')
            axes[1, 0].set_title('Distribution of Embedding Variances')
            axes[1, 0].set_xlabel('Variance')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)

            # Sparsity
            sparsity = np.mean(embeddings == 0, axis=1)
            axes[1, 1].hist(sparsity, bins=50, alpha=0.7, edgecolor='black', color='purple')
            axes[1, 1].set_title('Embedding Sparsity Distribution')
            axes[1, 1].set_xlabel('Fraction of Zeros')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            viz_path = os.path.join(embeddings_dir, f'{phase}_quality_analysis.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        except Exception as e:
            print(f"⚠ Failed to create quality analysis: {e}")
    def _create_embedding_html_summary(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                     phase: str, embeddings_dir: str):
        """Create HTML summary for embedding analysis"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{phase.capitalize()} Embedding Analysis</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #333; }}
                    .stats {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                    .stats th {{ text-align: left; padding: 5px; }}
                    .stats td {{ padding: 5px; }}
                    .viz-container {{ text-align: center; margin: 20px 0; }}
                    .viz-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{phase.capitalize()} Embedding Analysis</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="stats">
                    <h2>Embedding Statistics</h2>
                    <table>
                        <tr><th>Total Embeddings</th><td>{len(embeddings):,}</td></tr>
                        <tr><th>Embedding Dimension</th><td>{embeddings.shape[1]}</td></tr>
                        <tr><th>Average Norm</th><td>{np.mean(np.linalg.norm(embeddings, axis=1)):.4f}</td></tr>
                        <tr><th>Average Mean</th><td>{np.mean(np.mean(embeddings, axis=1)):.4f}</td></tr>
                        <tr><th>Average Variance</th><td>{np.mean(np.var(embeddings, axis=1)):.4f}</td></tr>
                        <tr><th>Sparsity</th><td>{np.mean(np.mean(embeddings == 0, axis=1)):.4f}</td></tr>
                    </table>
                </div>

                <div class="viz-container">
                    <h2>Visualizations</h2>
                    <img src="{phase}_dimensionality_reduction.png" alt="Dimensionality Reduction">
                    <img src="{phase}_clustering_analysis.png" alt="Clustering Analysis">
                    <img src="{phase}_quality_analysis.png" alt="Quality Analysis">
                </div>
            </body>
            </html>
            """

            html_path = os.path.join(embeddings_dir, f'{phase}_embedding_analysis.html')
            with open(html_path, 'w') as f:
                f.write(html_content)

        except Exception as e:
            print(f"⚠ Failed to create HTML summary: {e}")
