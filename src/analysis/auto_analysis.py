"""
Automatic Analysis Module for DAE-KAN Training

This module provides streamlined, automatic analysis functionality that integrates
all analysis components into a single cohesive system for real-time monitoring
during training.
"""

import fcntl  # For file locking
import json
import os
import time
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from skimage import measure

warnings.filterwarnings('ignore')

# Try to import wandb for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not available - W&B logging will be disabled")

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    from models.dae import DAE_KAN_Attention  # noqa: F401  (availability probe)
except ImportError:
    pass


from analysis.embedding_cluster import EmbeddingClusterMixin
from analysis.viz import VizMixin


class AutomaticAnalyzer(VizMixin, EmbeddingClusterMixin):
    """
    Automatic analysis system that integrates all analysis components with run-based organization
    """

    def __init__(self, model: nn.Module, device: str, save_dir: str = "auto_analysis",
                 run_id: str = None, experiment_config: dict = None, logger=None):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_config = experiment_config or {}
        self.logger = logger  # W&B logger reference

        # Create organized directory structure
        self.save_dir = self._create_organized_directory_structure(save_dir)

        # Initialize analysis data storage
        self.batch_metrics = []
        self.attention_data = []
        self.performance_data = []
        self.timing_data = []

        # Attention hooks
        self.attention_hooks = []
        self.setup_attention_hooks()

        # Analysis frequency
        self.analysis_frequency = 100  # Run full analysis every N batches
        self.quick_analysis_frequency = 20  # Run quick analysis every N batches
        self.dashboard_frequency = 200  # Run comprehensive dashboard every N batches

        # W&B logging settings
        self.log_to_wandb = WANDB_AVAILABLE and logger is not None
        self.wandb_log_frequencies = {
            'metrics': 10,  # Log metrics every N batches
            'visualizations': 100,  # Log basic visualizations every N batches
            'paper_figures': 200,  # Log paper figures every N batches
            'individual_components': 500  # Log individual components every N batches
        }

        # File locking to prevent concurrent access
        self.lock_file = os.path.join(self.save_dir, ".analysis.lock")

        # Save experiment configuration
        self._save_experiment_config()

        print(f"✓ W&B logging enabled: {self.log_to_wandb}")

    def _create_organized_directory_structure(self, base_dir: str) -> str:
        """
        Create organized directory structure for this run:
        base_dir/
        ├── run_id/
        │   ├── metrics/
        │   ├── visualizations/
        │   ├── paper_figures/
        │   ├── individual_components/
        │   └── logs/
        """
        # Create run-specific directory
        run_dir = os.path.join(base_dir, self.run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Create subdirectories
        subdirs = ['metrics', 'visualizations', 'paper_figures', 'individual_components', 'logs']
        for subdir in subdirs:
            os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)

        return run_dir

    def _save_experiment_config(self):
        """Save experiment configuration to run directory"""
        config_path = os.path.join(self.save_dir, 'experiment_config.json')

        config_data = {
            'run_id': self.run_id,
            'created_at': datetime.now().isoformat(),
            'device': str(self.device),
            'analysis_frequency': self.analysis_frequency,
            'quick_analysis_frequency': self.quick_analysis_frequency,
            **self.experiment_config
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"✓ Experiment config saved: {config_path}")

    def get_subdir_path(self, subdir: str) -> str:
        """Get path to a specific subdirectory"""
        return os.path.join(self.save_dir, subdir)

    def log_to_wandb_safe(self, data: dict, prefix: str = "", step: int = None):
        """Safely log data to W&B with error handling"""
        if not self.log_to_wandb:
            return

        try:
            # Add prefix to all keys
            if prefix:
                prefixed_data = {f"{prefix}/{k}": v for k, v in data.items()}
            else:
                prefixed_data = data

            # Log to W&B (skip if step is provided and might be outdated to prevent warnings)
            if step is not None and step < 0:
                return
            wandb.log(prefixed_data, step=step)
        except Exception as e:
            print(f"⚠ Failed to log to W&B: {e}")

    def log_image_to_wandb_safe(self, image_path: str, caption: str, step: int = None):
        """Safely log image to W&B with error handling"""
        if not self.log_to_wandb or not os.path.exists(image_path):
            return

        try:
            # Skip if step is provided and might be outdated to prevent warnings
            if step is not None and step < 0:
                return
            wandb.log({caption: wandb.Image(image_path)}, step=step)
        except Exception as e:
            print(f"⚠ Failed to log image {caption} to W&B: {e}")

    def log_table_to_wandb_safe(self, df: pd.DataFrame, table_name: str, step: int = None):
        """Safely log table to W&B with error handling"""
        if not self.log_to_wandb:
            return

        try:
            # Skip if step is provided and might be outdated to prevent warnings
            if step is not None and step < 0:
                return
            table = wandb.Table(dataframe=df)
            wandb.log({table_name: table}, step=step)
        except Exception as e:
            print(f"⚠ Failed to log table {table_name} to W&B: {e}")

    def should_log_to_wandb(self, log_type: str, batch_idx: int) -> bool:
        """Check if should log to W&B based on frequency"""
        if not self.log_to_wandb:
            return False

        frequency = self.wandb_log_frequencies.get(log_type, self.analysis_frequency)
        return batch_idx % frequency == 0

    def _acquire_lock(self):
        """Acquire file lock to prevent concurrent access"""
        try:
            self.lock_fd = open(self.lock_file, 'w')
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX)
            return True
        except Exception:
            return False

    def _release_lock(self):
        """Release file lock"""
        try:
            if hasattr(self, 'lock_fd'):
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
        except Exception:
            pass

    def setup_attention_hooks(self):
        """Setup hooks for automatic attention extraction"""
        self.attention_data = {}

        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                # Store attention data
                if hasattr(output, 'detach'):
                    data = output.detach().float().cpu().numpy()
                elif hasattr(output, 'cpu'):
                    data = output.float().cpu().numpy()
                else:
                    data = output

                self.attention_data[layer_name] = {
                    'data': data,
                    'shape': data.shape,
                    'timestamp': time.time()
                }

            return hook_fn

        # Get the actual DAE_KAN_Attention model (skip the Lightning wrapper)
        actual_model = self.model.model if hasattr(self.model, 'model') else self.model

        # Register hooks only for modules that exist. Ablation variants drop
        # whole components (kan_only has no BAM attn, no_kan has no KAN, etc.), so
        # probe attribute paths defensively rather than assuming the full model.
        def _maybe(name, getter):
            try:
                layer = getter()
            except AttributeError:
                return
            if layer is None:
                return
            hook = layer.register_forward_hook(create_hook_fn(name))
            self.attention_hooks.append(hook)

        _maybe('bam_384', lambda: actual_model.bottleneck.attn1)
        _maybe('bam_16', lambda: actual_model.bottleneck.attn2)
        _maybe('encoder_kan', lambda: actual_model.ae_encoder.kan)
        _maybe('decoder_kan', lambda: actual_model.ae_decoder.kan)

        if not self.attention_hooks:
            print("ℹ auto_analysis: no BAM/KAN submodules found to hook "
                  "(expected for some ablation variants).")

    def compute_attention_metrics(self, attention_map: np.ndarray) -> dict[str, float]:
        """Compute comprehensive attention metrics"""
        # Normalize attention map
        if attention_map.ndim == 4:
            attention_map = np.mean(attention_map, axis=0)
        elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
            attention_map = np.mean(attention_map, axis=0)

        attention_flat = attention_map.flatten()
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Basic statistics
        metrics = {
            'mean_attention': float(np.mean(attention_flat)),
            'std_attention': float(np.std(attention_flat)),
            'max_attention': float(np.max(attention_flat)),
            'min_attention': float(np.min(attention_flat)),
        }

        # Distribution metrics
        metrics['attention_kurtosis'] = float(stats.kurtosis(attention_flat))
        metrics['attention_skewness'] = float(stats.skew(attention_flat))

        # Sparsity and concentration
        metrics['attention_sparsity'] = float(np.mean(attention_flat < 0.1))
        metrics['attention_concentration_10'] = self._compute_concentration(attention_flat, 0.10)
        metrics['attention_concentration_20'] = self._compute_concentration(attention_flat, 0.20)

        # Spatial metrics
        spatial_metrics = self._compute_spatial_metrics(attention_norm)
        metrics.update(spatial_metrics)

        # Peak metrics
        peak_metrics = self._compute_peak_metrics(attention_norm)
        metrics.update(peak_metrics)

        return metrics



    def _compute_concentration(self, attention_flat: np.ndarray, top_percent: float) -> float:
        """Compute attention concentration in top-k regions"""
        k = int(len(attention_flat) * top_percent)
        if k == 0:
            return 0.0
        top_k_values = np.sort(attention_flat)[-k:]
        return float(np.sum(top_k_values) / (np.sum(attention_flat) + 1e-8))

    def _compute_spatial_metrics(self, attention_map: np.ndarray) -> dict[str, float]:
        """Compute spatial distribution metrics"""
        # Ensure attention_map is 2D
        if attention_map.ndim > 2:
            attention_map = np.mean(attention_map, axis=0)
        if attention_map.ndim > 2:
            attention_map = np.mean(attention_map, axis=0)

        h, w = attention_map.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]

        center_y = np.sum(y_coords * attention_map) / (np.sum(attention_map) + 1e-8)
        center_x = np.sum(x_coords * attention_map) / (np.sum(attention_map) + 1e-8)

        image_center_y, image_center_x = h / 2, w / 2
        distance_from_center = np.sqrt((center_y - image_center_y)**2 + (center_x - image_center_x)**2)
        normalized_distance = distance_from_center / np.sqrt(image_center_y**2 + image_center_x**2)

        spatial_variance = np.sum(((y_coords - center_y)**2 + (x_coords - center_x)**2) * attention_map) / (np.sum(attention_map) + 1e-8)

        return {
            'attention_center_y': float(center_y / h),
            'attention_center_x': float(center_x / w),
            'attention_center_distance': float(normalized_distance),
            'attention_spatial_variance': float(spatial_variance),
            'attention_spatial_std': float(np.sqrt(spatial_variance))
        }

    def _compute_peak_metrics(self, attention_map: np.ndarray) -> dict[str, float]:
        """Compute peak-related metrics"""
        # Ensure attention_map is 2D
        if attention_map.ndim > 2:
            attention_map = np.mean(attention_map, axis=0)
        if attention_map.ndim > 2:
            attention_map = np.mean(attention_map, axis=0)

        threshold = np.percentile(attention_map, 90)
        significant_attention = attention_map > threshold

        if not np.any(significant_attention):
            return {
                'num_attention_peaks': 0,
                'peak_separation_mean': 0.0,
                'peak_intensity_mean': 0.0,
                'peak_coverage': 0.0
            }

        labeled_regions = measure.label(significant_attention)
        regions = measure.regionprops(labeled_regions)

        if len(regions) == 0:
            return {
                'num_attention_peaks': 0,
                'peak_separation_mean': 0.0,
                'peak_intensity_mean': 0.0,
                'peak_coverage': 0.0
            }

        peak_intensities = [region.mean_intensity if hasattr(region, 'mean_intensity') else np.mean(attention_map[region.coords]) for region in regions]
        peak_coverage = np.sum(significant_attention) / attention_map.size

        # Compute peak separation
        peak_separation = []
        if len(regions) > 1:
            centroids = [region.centroid for region in regions]
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    dist = np.sqrt((centroids[i][0] - centroids[j][0])**2 +
                                (centroids[i][1] - centroids[j][1])**2)
                    peak_separation.append(dist)

        return {
            'num_attention_peaks': len(regions),
            'peak_separation_mean': float(np.mean(peak_separation) if peak_separation else 0.0),
            'peak_intensity_mean': float(np.mean(peak_intensities)),
            'peak_coverage': float(peak_coverage)
        }

    def analyze_batch(self, batch_idx: int, input_tensor: torch.Tensor,
                     output_tensor: torch.Tensor, loss: float,
                     phase: str = "train", global_step: int = None) -> dict:
        """Analyze a single batch and return metrics"""

        # Clear attention data for this batch
        self.attention_data.clear()

        # Run forward pass to collect attention data
        with torch.no_grad():
            # Always call the actual DAE_KAN_Attention model
            actual_model = self.model.model if hasattr(self.model, 'model') else self.model
            _ = actual_model(input_tensor)

        # Compute basic metrics
        metrics = {
            'batch_idx': batch_idx,
            'phase': phase,
            'loss': loss,
            'timestamp': time.time(),
            'attention_layers': list(self.attention_data.keys())
        }

        # Add reconstruction metrics
        mse_loss = nn.functional.mse_loss(input_tensor, output_tensor)
        ssim_score = self._calculate_ssim(input_tensor, output_tensor)
        metrics['mse'] = float(mse_loss.item())
        metrics['ssim'] = float(ssim_score)

        # Analyze attention patterns
        for layer_name, attention_info in self.attention_data.items():
            if 'bam' in layer_name:  # Focus on BAM attention
                attention_map = attention_info['data']
                attention_metrics = self.compute_attention_metrics(attention_map)

                # Add layer-specific metrics
                for metric_name, value in attention_metrics.items():
                    metrics[f'attention_{layer_name}_{metric_name}'] = value

        # Store batch metrics
        self.batch_metrics.append(metrics)

        # Log to W&B if enabled and at the right frequency
        if self.should_log_to_wandb('metrics', batch_idx):
            wandb_metrics = {
                'loss': loss,
                'mse': metrics['mse'],
                'ssim': metrics['ssim']
            }

            # Add attention metrics
            for key, value in metrics.items():
                if 'attention_' in key and isinstance(value, int | float):
                    wandb_metrics[key] = value

            log_step = global_step if global_step is not None else batch_idx
            self.log_to_wandb_safe(wandb_metrics, f"{phase}/analysis", step=log_step)

        return metrics

    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images"""
        # Ensure tensors are detached before computing
        img1 = img1.detach()
        img2 = img2.detach()

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

        ssim = ssim_num / (ssim_den + 1e-8)
        return float(torch.mean(ssim).item())

    def _estimate_epoch(self, batch_idx):
        """Estimate current epoch from batch index"""
        # Rough estimate assuming 4233 batches per epoch (from training data)
        batches_per_epoch = 4233
        epoch = batch_idx // batches_per_epoch + 1
        return epoch

    def get_batch_summary(self) -> dict:
        """Get summary of all batch metrics"""
        if not self.batch_metrics:
            return {}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.batch_metrics)

        summary = {
            'total_batches': len(self.batch_metrics),
            'latest_batch': self.batch_metrics[-1]['batch_idx'],
            'latest_metrics': self.batch_metrics[-1],
        }

        # Compute statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['batch_idx', 'timestamp']:
                summary[f'{col}_mean'] = float(df[col].mean())
                summary[f'{col}_std'] = float(df[col].std())
                summary[f'{col}_min'] = float(df[col].min())
                summary[f'{col}_max'] = float(df[col].max())

        return summary

    def save_metrics(self, filename: str = None):
        """Save all metrics to file"""
        if filename is None:
            filename = f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        save_path = os.path.join(self.get_subdir_path('metrics'), filename)

        # Prepare data for JSON serialization
        data = {
            'batch_metrics': self.batch_metrics,
            'summary': self.get_batch_summary(),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_batches': len(self.batch_metrics),
                'device': str(self.device)
            }
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return save_path

    def cleanup(self):
        """Cleanup hooks and data"""
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks.clear()
        self.attention_data.clear()

    @staticmethod
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
