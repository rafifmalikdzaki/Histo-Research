"""
Enhanced Attention Analysis with Quantitative Metrics

This module provides enhanced attention visualization and analysis capabilities
with quantitative metrics for pathological correlation validation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.patches import Rectangle, Circle
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import scipy.stats as stats
from skimage import measure, filters, morphology, segmentation
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

from models.model import DAE_KAN_Attention
from analysis.attention_visualizer import AttentionExtractor


class EnhancedAttentionAnalyzer:
    """
    Enhanced attention analyzer with quantitative metrics and pathologist validation features
    """

    def __init__(self, extractor: AttentionExtractor):
        self.extractor = extractor
        self.attention_metrics = {}
        self.pathological_regions = {}

    def compute_attention_saliency_metrics(self, attention_map: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive saliency metrics for attention maps
        """
        # Normalize attention map
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        attention_flat = attention_norm.flatten()

        # Basic statistics
        metrics = {
            'mean_attention': np.mean(attention_flat),
            'std_attention': np.std(attention_flat),
            'max_attention': np.max(attention_flat),
            'min_attention': np.min(attention_flat),
            'median_attention': np.median(attention_flat)
        }

        # Distribution metrics
        metrics.update({
            'attention_entropy': self._compute_entropy(attention_flat),
            'attention_kurtosis': stats.kurtosis(attention_flat),
            'attention_skewness': stats.skew(attention_flat),
        })

        # Sparsity and concentration metrics
        metrics.update({
            'attention_sparsity': np.mean(attention_flat < 0.1),  # Fraction of low-attention pixels
            'attention_concentration_5': self._compute_concentration(attention_flat, 0.05),
            'attention_concentration_10': self._compute_concentration(attention_flat, 0.10),
            'attention_concentration_20': self._compute_concentration(attention_flat, 0.20),
        })

        # Spatial distribution metrics
        metrics.update(self._compute_spatial_metrics(attention_norm))

        # Peak detection metrics
        metrics.update(self._compute_peak_metrics(attention_norm))

        return metrics

    def _compute_entropy(self, attention_flat: np.ndarray) -> float:
        """Compute entropy of attention distribution"""
        attention_prob = attention_flat / (np.sum(attention_flat) + 1e-8)
        return -np.sum(attention_prob * np.log2(attention_prob + 1e-8))

    def _compute_concentration(self, attention_flat: np.ndarray, top_percent: float) -> float:
        """Compute attention concentration in top-k percent"""
        k = int(len(attention_flat) * top_percent)
        top_k_values = np.sort(attention_flat)[-k:]
        return np.sum(top_k_values) / (np.sum(attention_flat) + 1e-8)

    def _compute_spatial_metrics(self, attention_map: np.ndarray) -> Dict[str, float]:
        """Compute spatial distribution metrics"""
        # Center of mass
        h, w = attention_map.shape
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        center_y = np.sum(y_coords * attention_map) / (np.sum(attention_map) + 1e-8)
        center_x = np.sum(x_coords * attention_map) / (np.sum(attention_map) + 1e-8)

        # Distance from image center
        image_center_y, image_center_x = h / 2, w / 2
        distance_from_center = np.sqrt((center_y - image_center_y)**2 + (center_x - image_center_x)**2)
        normalized_distance = distance_from_center / np.sqrt(image_center_y**2 + image_center_x**2)

        # Spatial variance (spread)
        spatial_variance = np.sum(((y_coords - center_y)**2 + (x_coords - center_x)**2) * attention_map) / (np.sum(attention_map) + 1e-8)

        return {
            'attention_center_y': center_y / h,  # Normalized
            'attention_center_x': center_x / w,  # Normalized
            'attention_center_distance': normalized_distance,
            'attention_spatial_variance': spatial_variance,
            'attention_spatial_std': np.sqrt(spatial_variance)
        }

    def _compute_peak_metrics(self, attention_map: np.ndarray) -> Dict[str, float]:
        """Compute peak-related metrics"""
        # Find local maxima
        local_maxima = measure.maximum_position(attention_map, labels=measure.label(attention_map > 0.1))

        if len(local_maxima) == 0:
            return {
                'num_attention_peaks': 0,
                'peak_separation_mean': 0,
                'peak_intensity_mean': 0,
                'peak_coverage': 0
            }

        # Extract peak intensities
        peak_intensities = [attention_map[y, x] for y, x in local_maxima]

        # Compute peak coverage (area above threshold)
        threshold = np.percentile(attention_map, 90)
        peak_coverage = np.sum(attention_map > threshold) / attention_map.size

        # Compute peak separation if multiple peaks
        peak_separation = []
        if len(local_maxima) > 1:
            for i in range(len(local_maxima)):
                for j in range(i + 1, len(local_maxima)):
                    dist = np.sqrt((local_maxima[i][0] - local_maxima[j][0])**2 +
                                (local_maxima[i][1] - local_maxima[j][1])**2)
                    peak_separation.append(dist)

        return {
            'num_attention_peaks': len(local_maxima),
            'peak_separation_mean': np.mean(peak_separation) if peak_separation else 0,
            'peak_intensity_mean': np.mean(peak_intensities),
            'peak_intensity_std': np.std(peak_intensities),
            'peak_coverage': peak_coverage
        }

    def identify_attention_regions(self, attention_map: np.ndarray,
                                 num_regions: int = 5) -> Dict[str, np.ndarray]:
        """
        Identify distinct attention regions using clustering
        """
        # Normalize attention map
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Threshold to get significant attention areas
        threshold = np.percentile(attention_norm, 75)
        significant_attention = attention_norm > threshold

        if not np.any(significant_attention):
            return {'regions': np.zeros_like(attention_norm, dtype=int)}

        # Label connected components
        labeled_regions = measure.label(significant_attention)
        regions = measure.regionprops(labeled_regions)

        # Sort regions by mean attention intensity
        regions.sort(key=lambda x: x.mean_intensity, reverse=True)

        # Create region map
        region_map = np.zeros_like(attention_norm, dtype=int)
        for i, region in enumerate(regions[:num_regions]):
            region_map[labeled_regions == region.label] = i + 1

        return {
            'regions': region_map,
            'region_props': regions[:num_regions],
            'num_regions': min(num_regions, len(regions))
        }

    def compute_attention_region_correlation(self, attention_map: np.ndarray,
                                           feature_map: np.ndarray) -> Dict[str, float]:
        """
        Compute correlation between attention regions and pathological features
        """
        # Identify attention regions
        attention_regions = self.identify_attention_regions(attention_map)
        region_map = attention_regions['regions']

        if attention_regions['num_regions'] == 0:
            return {'region_correlation_mean': 0, 'region_correlation_std': 0, 'num_significant_regions': 0}

        correlations = []
        for region_id in range(1, attention_regions['num_regions'] + 1):
            region_mask = region_map == region_id
            if np.sum(region_mask) > 0:
                region_attention = attention_map[region_mask]
                region_feature = feature_map[region_mask]

                if len(region_attention) > 1 and len(region_feature) > 1:
                    corr, _ = stats.pearsonr(region_attention, region_feature)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))  # Use absolute correlation

        return {
            'region_correlation_mean': np.mean(correlations) if correlations else 0,
            'region_correlation_std': np.std(correlations) if correlations else 0,
            'num_significant_regions': len([c for c in correlations if c > 0.3]),
            'region_correlations': correlations
        }

    def create_enhanced_attention_visualization(self, attention_data: Dict[str, np.ndarray],
                                              original_image: np.ndarray,
                                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Create enhanced attention visualization with quantitative metrics
        """
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        img = original_image.transpose(1, 2, 0)
        img_norm = (img - img.min()) / (img.max() - img.min())
        ax1.imshow(img_norm)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Process attention maps
        bam_attention_keys = [k for k in attention_data.keys() if 'bam' in k and 'weighted' in k]

        if not bam_attention_keys:
            print("No BAM attention found in attention_data")
            return fig

        # Use first available BAM attention
        attention_key = bam_attention_keys[0]
        attention_map = attention_data[attention_key][0]

        if attention_map.ndim == 4:
            attention_map = np.mean(attention_map, axis=0)
        elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
            attention_map = np.mean(attention_map, axis=0)

        # Attention heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(attention_map, cmap='jet')
        ax2.set_title(f'Attention Heatmap\n({attention_key})', fontsize=12, fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # Attention overlay on original image
        ax3 = fig.add_subplot(gs[0, 2])
        attention_resized = cv2.resize(attention_map, (img_norm.shape[1], img_norm.shape[0]))
        ax3.imshow(img_norm)
        im3 = ax3.imshow(attention_resized, cmap='jet', alpha=0.5)
        ax3.set_title('Attention Overlay', fontsize=12, fontweight='bold')
        ax3.axis('off')

        # Attention regions
        ax4 = fig.add_subplot(gs[0, 3])
        attention_regions = self.identify_attention_regions(attention_map)
        region_map = attention_regions['regions']

        # Create custom colormap for regions
        colors = plt.cm.Set3(np.linspace(0, 1, attention_regions['num_regions'] + 1))
        cmap = LinearSegmentedColormap.from_list('regions', colors, N=attention_regions['num_regions'] + 1)

        im4 = ax4.imshow(region_map, cmap=cmap, vmin=0, vmax=attention_regions['num_regions'])
        ax4.set_title(f'Attention Regions\n({attention_regions["num_regions"]} regions)', fontsize=12, fontweight='bold')
        ax4.axis('off')

        # Attention histogram
        ax5 = fig.add_subplot(gs[0, 4])
        attention_flat = attention_map.flatten()
        ax5.hist(attention_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax5.axvline(np.mean(attention_flat), color='red', linestyle='--', label=f'Mean: {np.mean(attention_flat):.3f}')
        ax5.axvline(np.median(attention_flat), color='green', linestyle='--', label=f'Median: {np.median(attention_flat):.3f}')
        ax5.set_xlabel('Attention Value')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Attention Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Attention statistics
        ax6 = fig.add_subplot(gs[0, 5])
        ax6.axis('off')

        metrics = self.compute_attention_saliency_metrics(attention_map)
        stats_text = f"""
        Attention Statistics:

        Basic Metrics:
        - Mean: {metrics['mean_attention']:.3f}
        - Std: {metrics['std_attention']:.3f}
        - Max: {metrics['max_attention']:.3f}
        - Min: {metrics['min_attention']:.3f}

        Distribution:
        - Entropy: {metrics['attention_entropy']:.3f}
        - Skewness: {metrics['attention_skewness']:.3f}
        - Kurtosis: {metrics['attention_kurtosis']:.3f}

        Concentration:
        - Top 5%: {metrics['attention_concentration_5']:.3f}
        - Top 10%: {metrics['attention_concentration_10']:.3f}
        - Top 20%: {metrics['attention_concentration_20']:.3f}

        Spatial:
        - Center Distance: {metrics['attention_center_distance']:.3f}
        - Spatial Std: {metrics['attention_spatial_std']:.3f}

        Peaks:
        - Num Peaks: {metrics['num_attention_peaks']}
        - Peak Coverage: {metrics['peak_coverage']:.3f}
        """

        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', family='monospace')

        # Attention intensity heatmap (different colormaps)
        ax7 = fig.add_subplot(gs[1, 0])
        im7 = ax7.imshow(attention_map, cmap='hot')
        ax7.set_title('Attention (Hot)', fontsize=12, fontweight='bold')
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)

        ax8 = fig.add_subplot(gs[1, 1])
        im8 = ax8.imshow(attention_map, cmap='viridis')
        ax8.set_title('Attention (Viridis)', fontsize=12, fontweight='bold')
        ax8.axis('off')
        plt.colorbar(im8, ax=ax8, fraction=0.046)

        # Attention gradient
        ax9 = fig.add_subplot(gs[1, 2])
        grad_y, grad_x = np.gradient(attention_map)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        im9 = ax9.imshow(grad_magnitude, cmap='plasma')
        ax9.set_title('Attention Gradient', fontsize=12, fontweight='bold')
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046)

        # Binary attention threshold
        ax10 = fig.add_subplot(gs[1, 3])
        threshold = np.percentile(attention_map, 75)
        binary_attention = (attention_map > threshold).astype(float)
        im10 = ax10.imshow(binary_attention, cmap='gray')
        ax10.set_title(f'Binary Attention\n(> {threshold:.3f})', fontsize=12, fontweight='bold')
        ax10.axis('off')

        # Attention contour
        ax11 = fig.add_subplot(gs[1, 4])
        ax11.imshow(img_norm, alpha=0.7)
        contours = ax11.contour(attention_resized, levels=10, colors='red', alpha=0.8, linewidths=2)
        ax11.clabel(contours, inline=True, fontsize=8)
        ax11.set_title('Attention Contours', fontsize=12, fontweight='bold')
        ax11.axis('off')

        # Region analysis
        ax12 = fig.add_subplot(gs[1, 5])
        if attention_regions['num_regions'] > 0:
            region_props = attention_regions['region_props']
            region_sizes = [prop.area for prop in region_props]
            region_intensities = [prop.mean_intensity for prop in region_props]

            ax12.scatter(region_sizes, region_intensities, alpha=0.7, s=100)
            ax12.set_xlabel('Region Size (pixels)')
            ax12.set_ylabel('Mean Intensity')
            ax12.set_title('Region Analysis', fontsize=12, fontweight='bold')
            ax12.grid(True, alpha=0.3)
        else:
            ax12.text(0.5, 0.5, 'No significant\nattention regions', ha='center', va='center', transform=ax12.transAxes)
            ax12.set_title('Region Analysis', fontsize=12, fontweight='bold')

        # Attention evolution through layers (if multiple layers available)
        layer_idx = 2
        for layer_name in sorted(bam_attention_keys[1:4], key=lambda x: int(''.join(filter(str.isdigit, x)) if ''.join(filter(str.isdigit, x)) else '0')):
            if layer_idx >= 4:
                break

            row = (layer_idx - 2) // 3
            col = layer_idx % 3

            ax = fig.add_subplot(gs[row + 2, col])
            layer_attention = attention_data[layer_name][0]

            if layer_attention.ndim == 4:
                layer_attention = np.mean(layer_attention, axis=0)
            elif layer_attention.ndim == 3 and layer_attention.shape[0] > 1:
                layer_attention = np.mean(layer_attention, axis=0)

            im = ax.imshow(layer_attention, cmap='jet')
            ax.set_title(f'{layer_name}', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)

            layer_idx += 1

        # Pathologist validation annotations placeholder
        ax_validation = fig.add_subplot(gs[3, 3:])
        ax_validation.axis('off')

        validation_text = """
        Pathologist Validation Framework:

        ✓ Quantitative attention metrics computed
        ✓ Attention regions identified and analyzed
        ✓ Spatial distribution characteristics measured
        ✓ Peak detection and concentration analysis
        ✓ Multi-layer attention evolution tracked

        Next Steps for Validation:
        1. Compare attention regions with pathologist annotations
        2. Validate attention focus on diagnostically relevant areas
        3. Assess attention consistency across similar cases
        4. Evaluate attention interpretability for clinical use

        Metrics for Pathologist Review:
        - Attention center location and spread
        - Number and size of attention regions
        - Correlation with known pathological features
        - Attention intensity gradients and patterns
        """

        ax_validation.text(0.05, 0.95, validation_text, transform=ax_validation.transAxes,
                         fontsize=10, verticalalignment='top', family='monospace')

        plt.suptitle('Enhanced Attention Analysis with Quantitative Metrics', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_attention_comparison_plot(self, attention_data_list: List[Dict[str, np.ndarray]],
                                      labels: List[str],
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comparison plot for multiple attention maps
        """
        num_samples = len(attention_data_list)
        if num_samples == 0:
            return None

        fig, axes = plt.subplots(3, num_samples, figsize=(5*num_samples, 12))
        if num_samples == 1:
            axes = axes.reshape(-1, 1)

        # Process each sample
        for i, (attention_data, label) in enumerate(zip(attention_data_list, labels)):
            bam_attention_keys = [k for k in attention_data.keys() if 'bam' in k and 'weighted' in k]

            if not bam_attention_keys:
                continue

            attention_key = bam_attention_keys[0]
            attention_map = attention_data[attention_key][0]

            if attention_map.ndim == 4:
                attention_map = np.mean(attention_map, axis=0)
            elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                attention_map = np.mean(attention_map, axis=0)

            # Attention heatmap
            im1 = axes[0, i].imshow(attention_map, cmap='jet')
            axes[0, i].set_title(f'{label}\nAttention Heatmap', fontsize=12, fontweight='bold')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

            # Attention regions
            attention_regions = self.identify_attention_regions(attention_map)
            region_map = attention_regions['regions']
            colors = plt.cm.Set3(np.linspace(0, 1, attention_regions['num_regions'] + 1))
            cmap = LinearSegmentedColormap.from_list('regions', colors, N=attention_regions['num_regions'] + 1)

            im2 = axes[1, i].imshow(region_map, cmap=cmap, vmin=0, vmax=attention_regions['num_regions'])
            axes[1, i].set_title(f'Attention Regions\n({attention_regions["num_regions"]} regions)', fontsize=12, fontweight='bold')
            axes[1, i].axis('off')

            # Metrics bar chart
            metrics = self.compute_attention_saliency_metrics(attention_map)
            metric_names = ['Entropy', 'Concentration', 'Sparsity', 'Center Distance']
            metric_values = [
                metrics['attention_entropy'],
                metrics['attention_concentration_10'],
                metrics['attention_sparsity'],
                metrics['attention_center_distance']
            ]

            bars = axes[2, i].bar(metric_names, metric_values, alpha=0.7)
            axes[2, i].set_title('Quantitative Metrics', fontsize=12, fontweight='bold')
            axes[2, i].set_ylabel('Value')
            axes[2, i].tick_params(axis='x', rotation=45)
            axes[2, i].grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, value in zip(bars, metric_values):
                axes[2, i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_attention_report(self, attention_data: Dict[str, np.ndarray],
                                save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive attention analysis report
        """
        report = []
        report.append("# Enhanced Attention Analysis Report\n")

        bam_attention_keys = [k for k in attention_data.keys() if 'bam' in k and 'weighted' in k]

        if not bam_attention_keys:
            report.append("No BAM attention data found for analysis.\n")
            return "\n".join(report)

        for attention_key in bam_attention_keys:
            report.append(f"## {attention_key} Analysis\n")

            attention_map = attention_data[attention_key][0]
            if attention_map.ndim == 4:
                attention_map = np.mean(attention_map, axis=0)
            elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                attention_map = np.mean(attention_map, axis=0)

            metrics = self.compute_attention_saliency_metrics(attention_map)
            attention_regions = self.identify_attention_regions(attention_map)

            report.append("### Quantitative Metrics\n")
            report.append(f"- **Mean Attention**: {metrics['mean_attention']:.4f}")
            report.append(f"- **Attention Entropy**: {metrics['attention_entropy']:.4f}")
            report.append(f"- **Attention Concentration (Top 10%)**: {metrics['attention_concentration_10']:.4f}")
            report.append(f"- **Attention Sparsity**: {metrics['attention_sparsity']:.4f}")
            report.append(f"- **Spatial Standard Deviation**: {metrics['attention_spatial_std']:.4f}")
            report.append(f"- **Number of Attention Peaks**: {metrics['num_attention_peaks']}")
            report.append(f"- **Peak Coverage**: {metrics['peak_coverage']:.4f}")

            report.append("\n### Spatial Analysis\n")
            report.append(f"- **Attention Center Distance**: {metrics['attention_center_distance']:.4f}")
            report.append(f"- **Attention Center X**: {metrics['attention_center_x']:.4f}")
            report.append(f"- **Attention Center Y**: {metrics['attention_center_y']:.4f}")

            report.append("\n### Region Analysis\n")
            report.append(f"- **Number of Regions**: {attention_regions['num_regions']}")

            if attention_regions['num_regions'] > 0:
                region_props = attention_regions['region_props']
                region_sizes = [prop.area for prop in region_props]
                region_intensities = [prop.mean_intensity for prop in region_props]

                report.append(f"- **Average Region Size**: {np.mean(region_sizes):.1f} pixels")
                report.append(f"- **Region Size Range**: {np.min(region_sizes):.1f} - {np.max(region_sizes):.1f} pixels")
                report.append(f"- **Average Region Intensity**: {np.mean(region_intensities):.4f}")

            report.append("\n### Interpretability Assessment\n")

            # Assessment based on metrics
            if metrics['attention_concentration_10'] > 0.5:
                report.append("✅ **High Concentration**: Attention is well-focused on specific regions")
            elif metrics['attention_concentration_10'] > 0.3:
                report.append("⚡ **Moderate Concentration**: Attention shows some focus but could be more specific")
            else:
                report.append("⚠️ **Low Concentration**: Attention is diffuse, may indicate unclear focus")

            if metrics['attention_entropy'] > 3:
                report.append("✅ **High Entropy**: Attention is distributed across diverse regions")
            elif metrics['attention_entropy'] > 2:
                report.append("⚡ **Moderate Entropy**: Balanced attention distribution")
            else:
                report.append("⚠️ **Low Entropy**: Attention is highly concentrated in few areas")

            if metrics['num_attention_peaks'] > 1 and metrics['num_attention_peaks'] <= 5:
                report.append("✅ **Multiple Focused Regions**: Attention identifies multiple relevant areas")
            elif metrics['num_attention_peaks'] == 1:
                report.append("⚡ **Single Focus**: Attention concentrates on one primary region")
            else:
                report.append("⚠️ **Too Many/Few Peaks**: Attention pattern may be noisy or overly diffuse")

            report.append("\n### Pathologist Validation Recommendations\n")
            report.append("1. **Verify Attention Focus**: Confirm that attention regions correspond to diagnostically relevant structures")
            report.append("2. **Assess Regional Appropriateness**: Evaluate if the number and size of attention regions are clinically meaningful")
            report.append("3. **Check Spatial Consistency**: Ensure attention patterns align with expected pathological locations")
            report.append("4. **Compare with Expert Annotations**: Validate against pathologist markings of relevant regions")

            report.append("\n" + "="*50 + "\n")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text


def main():
    """
    Main function to run enhanced attention analysis
    """
    print("Initializing Enhanced Attention Analysis...")

    from analysis.attention_visualizer import AttentionExtractor
    from models.model import DAE_KAN_Attention
    from histodata import create_dataset, ImageDataset
    from torch.utils.data import DataLoader

    # Create model and extractor
    model = DAE_KAN_Attention()
    extractor = AttentionExtractor(model, device="cuda")
    enhanced_analyzer = EnhancedAttentionAnalyzer(extractor)

    # Create dataset
    train_ds = ImageDataset(*create_dataset('train'))
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # Analyze a few samples
    print("Running enhanced attention analysis...")
    attention_data_list = []
    labels = []

    for i, batch in enumerate(dataloader):
        if i >= 3:  # Analyze 3 samples
            break

        x, y = batch

        # Extract attention patterns
        bam_attention = extractor.extract_bam_attention(x)
        attention_data_list.append(bam_attention)
        labels.append(f"Sample {i+1}")

        # Create enhanced visualization
        print(f"Creating enhanced visualization for sample {i+1}...")
        fig = enhanced_analyzer.create_enhanced_attention_visualization(
            bam_attention,
            x.numpy(),
            save_path=f"../analysis/visualizations/enhanced_attention_sample_{i+1}.png"
        )
        plt.close(fig)

        # Generate report
        report = enhanced_analyzer.generate_attention_report(
            bam_attention,
            save_path=f"../analysis/reports/enhanced_attention_report_sample_{i+1}.md"
        )

    # Create comparison visualization
    print("Creating attention comparison visualization...")
    comparison_fig = enhanced_analyzer.create_attention_comparison_plot(
        attention_data_list,
        labels,
        save_path="../analysis/visualizations/attention_comparison.png"
    )
    if comparison_fig:
        plt.close(comparison_fig)

    print("\nEnhanced Attention Analysis Complete!")
    print("Results saved to:")
    print("- ../analysis/visualizations/enhanced_attention_sample_*.png")
    print("- ../analysis/visualizations/attention_comparison.png")
    print("- ../analysis/reports/enhanced_attention_report_sample_*.md")

    return enhanced_analyzer


if __name__ == "__main__":
    analyzer = main()