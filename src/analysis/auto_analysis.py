"""
Automatic Analysis Module for DAE-KAN Training

This module provides streamlined, automatic analysis functionality that integrates
all analysis components into a single cohesive system for real-time monitoring
during training.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from skimage import measure, filters, morphology
import time
import json
import os
from datetime import datetime
import warnings
import fcntl  # For file locking
import tempfile
warnings.filterwarnings('ignore')

try:
    from models.model import DAE_KAN_Attention
except ImportError:
    from src.models.model import DAE_KAN_Attention


class AutomaticAnalyzer:
    """
    Automatic analysis system that integrates all analysis components
    """

    def __init__(self, model: nn.Module, device: str, save_dir: str = "auto_analysis"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

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
        
        # File locking to prevent concurrent access
        self.lock_file = os.path.join(save_dir, ".analysis.lock")

    def _acquire_lock(self):
        """Acquire file lock to prevent concurrent access"""
        try:
            self.lock_fd = open(self.lock_file, 'w')
            fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_EX)
            return True
        except:
            return False

    def _release_lock(self):
        """Release file lock"""
        try:
            if hasattr(self, 'lock_fd'):
                fcntl.flock(self.lock_fd.fileno(), fcntl.LOCK_UN)
                self.lock_fd.close()
        except:
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

        # Register hooks for BAM layers
        bam_layers = [
            ('bam_384', actual_model.bottleneck.attn1),
            ('bam_16', actual_model.bottleneck.attn2),
        ]

        for name, layer in bam_layers:
            hook = layer.register_forward_hook(create_hook_fn(name))
            self.attention_hooks.append(hook)

        # Register hooks for KAN layers
        kan_layers = [
            ('encoder_kan', actual_model.ae_encoder.kan),
            ('decoder_kan', actual_model.ae_decoder.kan),
        ]

        for name, layer in kan_layers:
            hook = layer.register_forward_hook(create_hook_fn(name))
            self.attention_hooks.append(hook)

    def compute_attention_metrics(self, attention_map: np.ndarray) -> Dict[str, float]:
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
        metrics['attention_entropy'] = self._compute_entropy(attention_flat)
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

    def _compute_entropy(self, attention_flat: np.ndarray) -> float:
        """Compute entropy of attention distribution"""
        attention_prob = attention_flat / (np.sum(attention_flat) + 1e-8)
        return float(-np.sum(attention_prob * np.log2(attention_prob + 1e-8)))

    def _compute_concentration(self, attention_flat: np.ndarray, top_percent: float) -> float:
        """Compute attention concentration in top-k regions"""
        k = int(len(attention_flat) * top_percent)
        if k == 0:
            return 0.0
        top_k_values = np.sort(attention_flat)[-k:]
        return float(np.sum(top_k_values) / (np.sum(attention_flat) + 1e-8))

    def _compute_spatial_metrics(self, attention_map: np.ndarray) -> Dict[str, float]:
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

    def _compute_peak_metrics(self, attention_map: np.ndarray) -> Dict[str, float]:
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
                     phase: str = "train") -> Dict:
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

    def create_batch_visualization(self, batch_idx: int, input_tensor: torch.Tensor,
                                 output_tensor: torch.Tensor, phase: str = "train") -> str:
        """Create visualization for current batch"""

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Original and reconstructed images
        input_np = input_tensor[0].detach().cpu().numpy()
        output_np = output_tensor[0].detach().cpu().numpy()

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        img = input_np.transpose(1, 2, 0)
        img_norm = (img - img.min()) / (img.max() - img.min())
        ax1.imshow(img_norm)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Reconstructed image
        ax2 = fig.add_subplot(gs[0, 1])
        recon = output_np.transpose(1, 2, 0)
        recon_norm = (recon - recon.min()) / (recon.max() - recon.min())
        ax2.imshow(recon_norm)
        ax2.set_title('Reconstructed Image')
        ax2.axis('off')

        # Error map
        ax3 = fig.add_subplot(gs[0, 2])
        error_map = np.mean((img_norm - recon_norm) ** 2, axis=2)
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

            im = ax4.imshow(attention_map, cmap='jet')
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
            labels = [l.get_label() for l in lines]
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
            entropy_values = []
            concentration_values = []

            for metrics in self.batch_metrics[-50:]:  # Last 50 batches
                if 'attention_bam_384_attention_entropy' in metrics:
                    entropy_values.append(metrics['attention_bam_384_attention_entropy'])
                if 'attention_bam_384_attention_concentration_10' in metrics:
                    concentration_values.append(metrics['attention_bam_384_attention_concentration_10'])

            if entropy_values:
                ax7.plot(range(len(entropy_values)), entropy_values, 'g-', label='Entropy', alpha=0.7)
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
            - Entropy: {latest_metrics.get('attention_bam_384_attention_entropy', 'N/A')}
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

        # Save visualization
        save_path = os.path.join(self.save_dir, f'{phase}_batch_{batch_idx:06d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    def create_comprehensive_paper_dashboard(self, batch_idx: int, input_tensor: torch.Tensor,
                                           output_tensor: torch.Tensor, phase: str = "train") -> Dict[str, str]:
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
            img_norm = (img - img.min()) / (img.max() - img.min())
            recon = output_np.transpose(1, 2, 0)
            recon_norm = (recon - recon.min()) / (recon.max() - recon.min())

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
            error_map = np.mean((img_norm - recon_norm) ** 2, axis=2)
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

                    im_attn = ax_attn.imshow(attention_map, cmap='jet')
                    ax_attn.set_title(f'Attention: {bam_key}', fontsize=12, fontweight='bold')
                    ax_attn.axis('off')
                    plt.colorbar(im_attn, ax=ax_attn, fraction=0.046)
                    attention_figs[bam_key] = attention_map

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
                labels = [l.get_label() for l in lines]
                ax_progress.legend(lines, labels, loc='upper left')

            # Attention quality metrics
            ax_metrics = fig.add_subplot(gs[1, 3:])
            if len(self.batch_metrics) > 0:
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
Attention Quality:     Entropy: {latest_metrics.get('attention_bam_384_attention_entropy', 0):.3f}    |    Concentration: {latest_metrics.get('attention_bam_384_attention_concentration_10', 0):.3f}    |    Sparsity: {latest_metrics.get('attention_bam_384_attention_sparsity', 0):.3f}
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
            paper_figs_dir = os.path.join(self.save_dir, 'paper_figures')
            os.makedirs(paper_figs_dir, exist_ok=True)

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
                    im4 = ax4.imshow(attention_map, cmap='jet')
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
                entropy_values = []
                concentration_values = []

                for metrics in self.batch_metrics[-200:]:
                    if 'attention_bam_384_attention_entropy' in metrics:
                        entropy_values.append(metrics['attention_bam_384_attention_entropy'])
                    if 'attention_bam_384_attention_concentration_10' in metrics:
                        concentration_values.append(metrics['attention_bam_384_attention_concentration_10'])

                if entropy_values:
                    ax_attn_evo.plot(range(len(entropy_values)), entropy_values, 'g-', label='Entropy', linewidth=2, alpha=0.8)
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
                    im = ax_attn_map.imshow(attention_map, cmap='jet')
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

                entropy_vals = [m.get('attention_bam_384_attention_entropy', 0) for m in recent_metrics]
                concentration_vals = [m.get('attention_bam_384_attention_concentration_10', 0) for m in recent_metrics]
                sparsity_vals = [m.get('attention_bam_384_attention_sparsity', 0) for m in recent_metrics]

                x = range(len(recent_metrics))
                width = 0.25

                ax_metrics_comp.bar([i - width for i in x], entropy_vals, width, label='Entropy', alpha=0.8)
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
                    ['Entropy', f"{latest_metrics.get('attention_bam_384_attention_entropy', 0):.3f}",
                     f"{latest_metrics.get('attention_bam_16_attention_entropy', 0):.3f}"],
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

            return {
                'individual_figures': individual_figs
            }
        finally:
            # Release lock
            self._release_lock()

    def save_individual_components(self, batch_idx: int, input_tensor: torch.Tensor,
                                 output_tensor: torch.Tensor, phase: str = "train") -> Dict[str, str]:
        """Save each analysis component as individual figure files"""
        
        # Acquire lock to prevent concurrent access
        if not self._acquire_lock():
            print(f"⚠ Could not acquire lock for batch {batch_idx}, skipping analysis")
            return {}
        
        try:
            # Create directory for individual components
            components_dir = os.path.join(self.save_dir, 'individual_components')
            os.makedirs(components_dir, exist_ok=True)
            
            # Create subdirectory for this batch
            batch_dir = os.path.join(components_dir, f'{phase}_batch_{batch_idx:06d}')
            os.makedirs(batch_dir, exist_ok=True)
            
            # Get data
            input_np = input_tensor[0].detach().cpu().numpy()
            output_np = output_tensor[0].detach().cpu().numpy()
            img = input_np.transpose(1, 2, 0)
            img_norm = (img - img.min()) / (img.max() - img.min())
            recon = output_np.transpose(1, 2, 0)
            recon_norm = (recon - recon.min()) / (recon.max() - recon.min())
            
            saved_files = {}
            
            # 1. Original Image
            fig_orig = plt.figure(figsize=(8, 8))
            ax_orig = fig_orig.add_subplot(111)
            ax_orig.imshow(img_norm)
            ax_orig.set_title('Original Image', fontsize=14, fontweight='bold')
            ax_orig.axis('off')
            orig_path = os.path.join(batch_dir, 'original.png')
            plt.savefig(orig_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            saved_files['original'] = orig_path
            
            # 2. Reconstructed Image
            fig_recon = plt.figure(figsize=(8, 8))
            ax_recon = fig_recon.add_subplot(111)
            ax_recon.imshow(recon_norm)
            ax_recon.set_title('Reconstructed Image', fontsize=14, fontweight='bold')
            ax_recon.axis('off')
            recon_path = os.path.join(batch_dir, 'reconstructed.png')
            plt.savefig(recon_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            saved_files['reconstructed'] = recon_path
            
            # 3. Error Map
            fig_error = plt.figure(figsize=(8, 8))
            ax_error = fig_error.add_subplot(111)
            error_map = np.mean((img_norm - recon_norm) ** 2, axis=2)
            im_error = ax_error.imshow(error_map, cmap='hot')
            ax_error.set_title('Reconstruction Error Map', fontsize=14, fontweight='bold')
            ax_error.axis('off')
            plt.colorbar(im_error, ax=ax_error, fraction=0.046)
            error_path = os.path.join(batch_dir, 'error_map.png')
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
                im_attn = ax_attn.imshow(attention_map, cmap='jet')
                ax_attn.set_title(f'Attention Map: {bam_key}', fontsize=14, fontweight='bold')
                ax_attn.axis('off')
                plt.colorbar(im_attn, ax=ax_attn, fraction=0.046)
                attn_path = os.path.join(batch_dir, f'attention_map_{bam_key}.png')
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
                hist_path = os.path.join(batch_dir, f'attention_histogram_{bam_key}.png')
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
                spatial_path = os.path.join(batch_dir, f'attention_spatial_{bam_key}.png')
                plt.savefig(spatial_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                saved_files[f'attention_spatial_{bam_key}'] = spatial_path
            
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
                labels = [l.get_label() for l in lines]
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
            
            return saved_files
        finally:
            # Release lock
            self._release_lock()

    def _estimate_epoch(self, batch_idx):
        """Estimate current epoch from batch index"""
        # Rough estimate assuming 4233 batches per epoch (from training data)
        batches_per_epoch = 4233
        epoch = batch_idx // batches_per_epoch + 1
        return epoch

    def get_batch_summary(self) -> Dict:
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

        save_path = os.path.join(self.save_dir, filename)

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
