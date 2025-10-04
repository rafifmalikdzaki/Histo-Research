"""
Quantitative Pathology Correlation Analysis

This module provides tools to quantitatively analyze the correlation between
model attention patterns and histopathologically relevant features.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
from skimage import filters, measure, morphology, segmentation
try:
    from skimage.feature import greycomatrix, greycoprops
except ImportError:
    # For newer versions of scikit-image
    try:
        from skimage.feature.texture import graycomatrix, graycoprops
        # Create aliases for backward compatibility
        greycomatrix = graycomatrix
        greycoprops = graycoprops
    except ImportError:
        print("Warning: greycomatrix/greycoprops not available. Texture analysis will be disabled.")
        greycomatrix = None
        greycoprops = None
from skimage.measure import regionprops
import cv2
from scipy import stats
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available. Some correlation analysis will be disabled.")
    KMeans = None
    adjusted_rand_score = None
    normalized_mutual_info_score = None
    SKLEARN_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

from models.model import DAE_KAN_Attention
from histodata import create_dataset, ImageDataset
from torch.utils.data import DataLoader
try:
    from analysis.attention_visualizer import AttentionExtractor
except ImportError:
    try:
        from attention_visualizer import AttentionExtractor
    except ImportError:
        print("Warning: attention_visualizer not available. Some features will be disabled.")
        AttentionExtractor = None


class HistopathologyFeatureExtractor:
    """
    Extract histopathologically relevant features from tissue images
    """

    def __init__(self):
        self.feature_names = []

    def extract_nuclear_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract nuclear features (nuclei detection, density, morphology)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Enhance contrast
        gray_enhanced = cv2.equalizeHist(gray.astype(np.uint8))

        # Nuclei detection using adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Find nuclei
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create nuclei mask
        nuclei_mask = np.zeros_like(gray)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 1000:  # Filter by size
                cv2.drawContours(nuclei_mask, [contour], -1, 255, -1)

        # Calculate nuclear density
        kernel_size = 50
        nuclei_density = cv2.GaussianBlur(nuclei_mask.astype(float), (kernel_size, kernel_size), 0)

        # Extract morphological features
        labeled_nuclei = measure.label(nuclei_mask > 0)
        regions = regionprops(labeled_nuclei)

        nuclear_features = {
            'nuclei_mask': nuclei_mask,
            'nuclei_density': nuclei_density,
            'nuclei_count': len(regions),
            'nuclear_areas': [r.area for r in regions],
            'nuclear_eccentricity': [r.eccentricity for r in regions],
            'nuclear_solidity': [r.solidity for r in regions]
        }

        return nuclear_features

    def extract_tissue_architecture_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract tissue architecture features (glands, stroma, texture)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Texture features using GLCM
        if greycomatrix is None or greycoprops is None:
            print("Warning: greycomatrix/greycoprops not available. Skipping texture features.")
            texture_features = {}
        else:
            glcm = greycomatrix(gray, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

            texture_features = {
                'contrast': greycoprops(glcm, 'contrast')[0, 0],
                'dissimilarity': greycoprops(glcm, 'dissimilarity')[0, 0],
                'homogeneity': greycoprops(glcm, 'homogeneity')[0, 0],
                'energy': greycoprops(glcm, 'energy')[0, 0],
                'correlation': greycoprops(glcm, 'correlation')[0, 0]
            }

        # Edge detection for structural features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(float), (21, 21), 0) / 255

        # Gland-like structure detection (using morphological operations)
        morphed = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
        gland_like = cv2.threshold(morphed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        architecture_features = {
            **texture_features,
            'edge_density': edge_density,
            'gland_like_mask': gland_like,
            'structural_complexity': np.std(edges)
        }

        return architecture_features

    def extract_color_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract color-based features (H&E staining patterns)
        """
        if len(image.shape) != 3:
            return {}

        # Convert to different color spaces
        rgb = image.transpose(1, 2, 0)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        # H&E deconvolution (simplified)
        # Hematoxylin channel (typically bluish)
        h_channel = hsv[:, :, 0] / 179.0  # Normalize
        # Eosin channel (typically reddish)
        e_channel = (255 - hsv[:, :, 0]) / 179.0

        color_features = {
            'hematoxylin_intensity': h_channel,
            'eosin_intensity': e_channel,
            'color_variance': np.var(rgb, axis=2),
            'saturation': hsv[:, :, 1] / 255.0,
            'lightness': lab[:, :, 0] / 255.0
        }

        return color_features


class AttentionPathologyCorrelator:
    """
    Correlate attention patterns with histopathological features
    """

    def __init__(self, extractor: AttentionExtractor):
        self.extractor = extractor
        self.feature_extractor = HistopathologyFeatureExtractor()

    def compute_attention_feature_correlation(self, attention_map: np.ndarray,
                                            feature_map: np.ndarray,
                                            method: str = 'pearson') -> Dict[str, float]:
        """
        Compute correlation between attention and pathological features
        """
        # Ensure both maps have the same shape
        if attention_map.shape != feature_map.shape:
            attention_map = cv2.resize(attention_map, (feature_map.shape[1], feature_map.shape[0]))

        # Flatten both maps
        attention_flat = attention_map.flatten()
        feature_flat = feature_map.flatten()

        # Remove NaN and infinite values
        mask = np.isfinite(attention_flat) & np.isfinite(feature_flat)
        attention_clean = attention_flat[mask]
        feature_clean = feature_flat[mask]

        if len(attention_clean) == 0:
            return {'correlation': 0.0, 'p_value': 1.0, 'method': method}

        # Compute correlation
        if method == 'pearson':
            corr, p_value = stats.pearsonr(attention_clean, feature_clean)
        elif method == 'spearman':
            corr, p_value = stats.spearmanr(attention_clean, feature_clean)
        elif method == 'kendall':
            corr, p_value = stats.kendalltau(attention_clean, feature_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        return {
            'correlation': corr,
            'p_value': p_value,
            'method': method,
            'sample_size': len(attention_clean)
        }

    def compute_attention_overlap_with_regions(self, attention_map: np.ndarray,
                                             region_mask: np.ndarray) -> Dict[str, float]:
        """
        Compute how much attention overlaps with specific regions (e.g., nuclei, glands)
        """
        # Ensure both maps have the same shape
        if attention_map.shape != region_mask.shape:
            attention_map = cv2.resize(attention_map, (region_mask.shape[1], region_mask.shape[0]))

        # Normalize attention map
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Binary region mask
        region_binary = (region_mask > 0).astype(float)

        # Compute overlap metrics
        attention_in_region = attention_norm * region_binary
        attention_out_region = attention_norm * (1 - region_binary)

        total_attention = np.sum(attention_norm)
        attention_in_region_ratio = np.sum(attention_in_region) / (total_attention + 1e-8)
        attention_out_region_ratio = np.sum(attention_out_region) / (total_attention + 1e-8)

        # Compute weighted attention concentration
        region_area = np.sum(region_binary)
        total_area = region_binary.size
        region_area_ratio = region_area / total_area

        # Attention concentration ratio
        attention_concentration = attention_in_region_ratio / (region_area_ratio + 1e-8)

        return {
            'attention_in_region_ratio': attention_in_region_ratio,
            'attention_out_region_ratio': attention_out_region_ratio,
            'region_area_ratio': region_area_ratio,
            'attention_concentration': attention_concentration,
            'total_attention_in_region': np.sum(attention_in_region),
            'total_attention_out_region': np.sum(attention_out_region)
        }

    def analyze_sample_correlations(self, input_tensor: torch.Tensor) -> Dict[str, Dict]:
        """
        Comprehensive correlation analysis for a single sample
        """
        # Extract model attention
        bam_attention = self.extractor.extract_bam_attention(input_tensor)

        # Convert input to numpy
        image_np = input_tensor[0].cpu().numpy()

        # Extract pathological features
        nuclear_features = self.feature_extractor.extract_nuclear_features(image_np)
        architecture_features = self.feature_extractor.extract_tissue_architecture_features(image_np)
        color_features = self.feature_extractor.extract_color_features(image_np)

        # Analyze correlations for each attention layer
        correlations = {}

        for layer_name, attention in bam_attention.items():
            if 'bam' not in layer_name:
                continue

            attention_map = attention[0]
            if attention_map.ndim == 4:
                attention_map = np.mean(attention_map, axis=0)
            elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                attention_map = np.mean(attention_map, axis=0)

            layer_correlations = {
                'nuclear_correlations': {},
                'architecture_correlations': {},
                'color_correlations': {},
                'region_overlaps': {}
            }

            # Nuclear feature correlations
            for feature_name, feature_data in nuclear_features.items():
                if isinstance(feature_data, np.ndarray) and feature_data.dtype != object:
                    if feature_data.ndim == 2:
                        corr = self.compute_attention_feature_correlation(attention_map, feature_data)
                        layer_correlations['nuclear_correlations'][feature_name] = corr

            # Architecture feature correlations
            for feature_name, feature_data in architecture_features.items():
                if isinstance(feature_data, (int, float)):
                    # For scalar features, create a uniform map
                    uniform_map = np.full_like(attention_map, feature_data)
                    corr = self.compute_attention_feature_correlation(attention_map, uniform_map)
                    layer_correlations['architecture_correlations'][feature_name] = corr
                elif isinstance(feature_data, np.ndarray) and feature_data.dtype != object:
                    if feature_data.ndim == 2:
                        corr = self.compute_attention_feature_correlation(attention_map, feature_data)
                        layer_correlations['architecture_correlations'][feature_name] = corr

            # Color feature correlations
            for feature_name, feature_data in color_features.items():
                if isinstance(feature_data, np.ndarray) and feature_data.dtype != object:
                    if feature_data.ndim == 2:
                        corr = self.compute_attention_feature_correlation(attention_map, feature_data)
                        layer_correlations['color_correlations'][feature_name] = corr

            # Region overlaps
            if 'nuclei_mask' in nuclear_features:
                overlap = self.compute_attention_overlap_with_regions(
                    attention_map, nuclear_features['nuclei_mask']
                )
                layer_correlations['region_overlaps']['nuclei'] = overlap

            if 'gland_like_mask' in architecture_features:
                overlap = self.compute_attention_overlap_with_regions(
                    attention_map, architecture_features['gland_like_mask']
                )
                layer_correlations['region_overlaps']['glands'] = overlap

            correlations[layer_name] = layer_correlations

        return correlations

    def batch_correlation_analysis(self, dataloader: DataLoader,
                                 num_samples: int = 50) -> pd.DataFrame:
        """
        Perform correlation analysis on multiple samples
        """
        all_results = []

        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break

            x, y = batch
            if x.shape[0] > 1:
                x = x[:1]  # Take single image

            print(f"Analyzing correlations for sample {i+1}/{num_samples}...")

            try:
                correlations = self.analyze_sample_correlations(x)

                # Flatten results for DataFrame
                for layer_name, layer_corrs in correlations.items():
                    for category, features in layer_corrs.items():
                        if category == 'region_overlaps':
                            for region_name, metrics in features.items():
                                for metric_name, value in metrics.items():
                                    all_results.append({
                                        'sample_id': i,
                                        'layer': layer_name,
                                        'category': category,
                                        'feature': f"{region_name}_{metric_name}",
                                        'value': value
                                    })
                        else:
                            for feature_name, corr_data in features.items():
                                all_results.append({
                                    'sample_id': i,
                                    'layer': layer_name,
                                    'category': category,
                                    'feature': feature_name,
                                    'correlation': corr_data.get('correlation', 0),
                                    'p_value': corr_data.get('p_value', 1),
                                    'method': corr_data.get('method', 'pearson')
                                })

            except Exception as e:
                print(f"Error analyzing sample {i}: {e}")
                continue

        return pd.DataFrame(all_results)

    def visualize_correlations(self, correlation_df: pd.DataFrame,
                             save_path: Optional[str] = None):
        """
        Create comprehensive visualization of correlation results
        """
        fig = plt.figure(figsize=(20, 12))

        # Filter out region overlaps for correlation plots
        corr_df = correlation_df[correlation_df['category'] != 'region_overlaps'].copy()

        # 1. Correlation distribution by category
        ax1 = plt.subplot(2, 3, 1)
        sns.boxplot(data=corr_df, x='category', y='correlation', ax=ax1)
        ax1.set_title('Correlation Distribution by Feature Category')
        ax1.set_xlabel('Feature Category')
        ax1.set_ylabel('Correlation Coefficient')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Top correlations heatmap
        ax2 = plt.subplot(2, 3, 2)
        # Get top correlations by absolute value
        top_corrs = corr_df.reindex(corr_df['correlation'].abs().sort_values(ascending=False).index)
        top_15 = top_corrs.head(15)

        pivot_table = top_15.pivot_table(
            index='feature', columns='layer', values='correlation', fill_value=0
        )
        sns.heatmap(pivot_table, annot=True, cmap='RdBu_r', center=0, ax=ax2)
        ax2.set_title('Top 15 Feature-Layer Correlations')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Feature')

        # 3. Significant correlations (p < 0.05)
        ax3 = plt.subplot(2, 3, 3)
        sig_df = corr_df[corr_df['p_value'] < 0.05].copy()
        if len(sig_df) > 0:
            sig_df['abs_correlation'] = sig_df['correlation'].abs()
            top_sig = sig_df.nlargest(10, 'abs_correlation')
            sns.barplot(data=top_sig, x='correlation', y='feature', hue='layer', ax=ax3)
            ax3.set_title('Top 10 Significant Correlations (p < 0.05)')
            ax3.set_xlabel('Correlation Coefficient')
            ax3.set_ylabel('Feature')
        else:
            ax3.text(0.5, 0.5, 'No significant correlations\n(p < 0.05)', ha='center', va='center')
            ax3.set_title('Significant Correlations (p < 0.05)')

        # 4. Region overlap analysis
        ax4 = plt.subplot(2, 3, 4)
        region_df = correlation_df[correlation_df['category'] == 'region_overlaps'].copy()
        if len(region_df) > 0:
            # Focus on attention concentration
            conc_df = region_df[region_df['feature'].str.contains('concentration')]
            if len(conc_df) > 0:
                sns.boxplot(data=conc_df, x='layer', y='value', ax=ax4)
                ax4.set_title('Attention Concentration in Pathological Regions')
                ax4.set_xlabel('Layer')
                ax4.set_ylabel('Concentration Ratio')
                ax4.tick_params(axis='x', rotation=45)
            else:
                ax4.text(0.5, 0.5, 'No concentration data', ha='center', va='center')
        else:
            ax4.text(0.5, 0.5, 'No region overlap data', ha='center', va='center')

        # 5. Layer-wise correlation summary
        ax5 = plt.subplot(2, 3, 5)
        layer_summary = corr_df.groupby('layer')['correlation'].agg(['mean', 'std']).reset_index()
        x_pos = np.arange(len(layer_summary))
        ax5.bar(x_pos, layer_summary['mean'], yerr=layer_summary['std'], capsize=5)
        ax5.set_xlabel('Layer')
        ax5.set_ylabel('Mean Correlation')
        ax5.set_title('Mean Correlation by Layer')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([layer.replace('bam_', '') for layer in layer_summary['layer']], rotation=45)
        ax5.grid(True, alpha=0.3)

        # 6. P-value distribution
        ax6 = plt.subplot(2, 3, 6)
        sns.histplot(data=corr_df, x='p_value', bins=20, ax=ax6)
        ax6.axvline(x=0.05, color='red', linestyle='--', label='p = 0.05')
        ax6.set_title('P-value Distribution')
        ax6.set_xlabel('P-value')
        ax6.set_ylabel('Count')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_correlation_report(self, correlation_df: pd.DataFrame,
                                  save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive correlation analysis report
        """
        report = []
        report.append("# DAE-KAN Attention-Pathology Correlation Analysis Report\n")

        # Summary statistics
        report.append("## Summary Statistics\n")
        report.append(f"- **Total Samples Analyzed**: {correlation_df['sample_id'].nunique()}")
        report.append(f"- **Total Correlations Computed**: {len(correlation_df)}")
        report.append(f"- **Attention Layers Analyzed**: {correlation_df['layer'].nunique()}")
        report.append(f"- **Feature Categories**: {correlation_df['category'].nunique()}\n")

        # Filter correlation data
        corr_df = correlation_df[correlation_df['category'] != 'region_overlaps'].copy()

        # Overall correlation statistics
        report.append("## Overall Correlation Statistics\n")
        report.append(f"- **Mean Correlation**: {corr_df['correlation'].mean():.3f}")
        report.append(f"- **Std Correlation**: {corr_df['correlation'].std():.3f}")
        report.append(f"- **Max Correlation**: {corr_df['correlation'].max():.3f}")
        report.append(f"- **Min Correlation**: {corr_df['correlation'].min():.3f}\n")

        # Significant correlations
        sig_df = corr_df[corr_df['p_value'] < 0.05]
        report.append("## Significant Correlations (p < 0.05)\n")
        report.append(f"- **Number of Significant Correlations**: {len(sig_df)}")
        report.append(f"- **Percentage of Significant**: {len(sig_df) / len(corr_df) * 100:.1f}%")
        if len(sig_df) > 0:
            report.append(f"- **Mean Significant Correlation**: {sig_df['correlation'].mean():.3f}")
        report.append("")

        # Top correlations by category
        report.append("## Top Correlations by Category\n")
        for category in corr_df['category'].unique():
            category_df = corr_df[corr_df['category'] == category]
            if len(category_df) > 0:
                top_corr = category_df.reindex(category_df['correlation'].abs().sort_values(ascending=False).index).iloc[0]
                report.append(f"### {category.title()}")
                report.append(f"- **Top Feature**: {top_corr['feature']}")
                report.append(f"- **Layer**: {top_corr['layer']}")
                report.append(f"- **Correlation**: {top_corr['correlation']:.3f}")
                report.append(f"- **P-value**: {top_corr['p_value']:.4f}\n")

        # Region overlap analysis
        region_df = correlation_df[correlation_df['category'] == 'region_overlaps'].copy()
        if len(region_df) > 0:
            report.append("## Region Overlap Analysis\n")
            conc_df = region_df[region_df['feature'].str.contains('concentration')]
            if len(conc_df) > 0:
                report.append("### Attention Concentration in Pathological Regions")
                for layer in conc_df['layer'].unique():
                    layer_conc = conc_df[conc_df['layer'] == layer]
                    mean_conc = layer_conc['value'].mean()
                    report.append(f"- **{layer}**: {mean_conc:.3f} concentration ratio")
                report.append("")

        # Interpretability assessment
        report.append("## Interpretability Assessment\n")
        if len(sig_df) > 0:
            high_corr = sig_df[abs(sig_df['correlation']) > 0.3]
            if len(high_corr) > 0:
                report.append("### Strong Correlations Found (> 0.3)")
                for _, row in high_corr.iterrows():
                    report.append(f"- **{row['layer']}** - {row['category']}: {row['feature']} (r = {row['correlation']:.3f})")
                report.append("")
                report.append("✅ **The model shows meaningful correlations with pathological features**")
            else:
                report.append("⚠️ **Correlations are weak but statistically significant**")
        else:
            report.append("❌ **No significant correlations found**")
        report.append("")

        # Recommendations
        report.append("## Recommendations\n")
        if len(sig_df) > 0:
            report.append("### For Pathologist Validation")
            report.append("1. Focus on validating top correlated features")
            report.append("2. Examine attention maps for high-concentration regions")
            report.append("3. Compare model attention with expert annotations")
            report.append("")
            report.append("### For Model Improvement")
            report.append("1. Consider attention mechanisms that focus more on pathological features")
            report.append("2. Incorporate pathological feature constraints during training")
            report.append("3. Use attention patterns for weakly supervised learning")
        else:
            report.append("### Next Steps")
            report.append("1. Verify attention extraction is working correctly")
            report.append("2. Consider different feature extraction methods")
            report.append("3. Increase sample size for analysis")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text


def main():
    """
    Main function to run pathology correlation analysis
    """
    print("Initializing Pathology Correlation Analysis...")

    # Create model and extractor
    model = DAE_KAN_Attention()
    extractor = AttentionExtractor(model, device="cuda")
    correlator = AttentionPathologyCorrelator(extractor)

    # Create dataset
    train_ds = ImageDataset(*create_dataset('train'))
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # Run correlation analysis
    print("Running correlation analysis...")
    correlation_df = correlator.batch_correlation_analysis(dataloader, num_samples=20)

    # Save results
    correlation_df.to_csv('../analysis/reports/pathology_correlations.csv', index=False)

    # Create visualizations
    print("Creating correlation visualizations...")
    correlator.visualize_correlations(
        correlation_df,
        save_path='../analysis/visualizations/pathology_correlations.png'
    )

    # Generate report
    print("Generating correlation report...")
    report = correlator.generate_correlation_report(
        correlation_df,
        save_path='../analysis/reports/pathology_correlation_report.md'
    )

    print("\nPathology Correlation Analysis Complete!")
    print("Results saved to:")
    print("- ../analysis/reports/pathology_correlations.csv")
    print("- ../analysis/visualizations/pathology_correlations.png")
    print("- ../analysis/reports/pathology_correlation_report.md")

    return correlator, correlation_df


if __name__ == "__main__":
    correlator, results = main()