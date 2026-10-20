"""Attention extraction hooks and the metrics computed from them.

The hooks are registered defensively: ablation variants drop whole
components (kan_only has no BAM attention, no_kan has no KAN, ...), so the
attribute paths are probed rather than assumed.
"""

import time

import numpy as np
import scipy.stats as stats
from skimage import measure

__all__ = ["AttentionMetricsMixin"]


class AttentionMetricsMixin:
    """Hook the model's attention submodules and summarise what they produce."""

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
