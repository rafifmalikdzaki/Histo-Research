"""
Pathological Correlation Validation Framework

This module provides a comprehensive framework for validating attention maps
against pathological features, enabling expert pathologist collaboration
and quantitative assessment of model interpretability.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from typing import Dict, List, Tuple, Optional, Union
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime
import scipy.stats as stats
from skimage import measure, filters, morphology, segmentation, draw
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

from models.model import DAE_KAN_Attention
from analysis.attention_visualizer import AttentionExtractor
from analysis.enhanced_attention_analysis import EnhancedAttentionAnalyzer


class PathologicalFeatureAnnotator:
    """
    Tool for annotating and managing pathological features
    """

    def __init__(self):
        self.annotations = {}
        self.feature_categories = {
            'nuclei': ['normal_nuclei', 'atypical_nuclei', 'mitotic_figures'],
            'glands': ['normal_glands', 'distorted_glands', 'necrotic_glands'],
            'stroma': ['normal_stroma', 'desmoplastic_stroma', 'inflammatory_stroma'],
            'architecture': ['tumor_boundaries', 'necrosis', 'vascular_invasion']
        }

    def create_annotation_template(self, image_shape: Tuple[int, int]) -> Dict:
        """Create empty annotation template for an image"""
        return {
            'image_shape': image_shape,
            'annotations': {},
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'annotator': 'pathologist',
                'version': '1.0'
            }
        }

    def add_annotation(self, image_id: str, feature_type: str,
                      coordinates: List[Tuple[int, int]],
                      annotation_type: str = 'polygon',
                      confidence: float = 1.0,
                      notes: str = ""):
        """Add annotation for a specific pathological feature"""
        if image_id not in self.annotations:
            self.annotations[image_id] = self.create_annotation_template((128, 128))  # Default size

        self.annotations[image_id]['annotations'][feature_type] = {
            'coordinates': coordinates,
            'type': annotation_type,  # 'polygon', 'circle', 'rectangle', 'point'
            'confidence': confidence,
            'notes': notes,
            'created_at': datetime.now().isoformat()
        }

    def load_annotations_from_file(self, file_path: str):
        """Load annotations from JSON file"""
        with open(file_path, 'r') as f:
            self.annotations = json.load(f)

    def save_annotations_to_file(self, file_path: str):
        """Save annotations to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)

    def create_annotation_mask(self, image_id: str, feature_type: str) -> np.ndarray:
        """Create binary mask for a specific annotation"""
        if image_id not in self.annotations or feature_type not in self.annotations[image_id]['annotations']:
            return np.zeros((128, 128), dtype=np.uint8)

        annotation = self.annotations[image_id]['annotations'][feature_type]
        mask = np.zeros((128, 128), dtype=np.uint8)

        if annotation['type'] == 'polygon':
            polygon = np.array(annotation['coordinates'], dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 1)
        elif annotation['type'] == 'circle':
            center = annotation['coordinates'][0]
            radius = annotation['coordinates'][1]
            cv2.circle(mask, center, radius, 1, -1)
        elif annotation['type'] == 'rectangle':
            pt1, pt2 = annotation['coordinates']
            cv2.rectangle(mask, pt1, pt2, 1, -1)
        elif annotation['type'] == 'point':
            for point in annotation['coordinates']:
                cv2.circle(mask, point, 3, 1, -1)

        return mask


class AttentionPathologyValidator:
    """
    Comprehensive validation of attention patterns against pathological annotations
    """

    def __init__(self, extractor: AttentionExtractor, annotator: PathologicalFeatureAnnotator):
        self.extractor = extractor
        self.annotator = annotator
        self.enhanced_analyzer = EnhancedAttentionAnalyzer(extractor)
        self.validation_results = {}

    def compute_attention_annotation_overlap(self, attention_map: np.ndarray,
                                          annotation_mask: np.ndarray) -> Dict[str, float]:
        """
        Compute overlap metrics between attention and pathological annotations
        """
        # Normalize attention map
        attention_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        # Resize annotation mask to match attention map if needed
        if attention_norm.shape != annotation_mask.shape:
            annotation_mask = cv2.resize(annotation_mask.astype(float),
                                        (attention_norm.shape[1], attention_norm.shape[0]))
            annotation_mask = (annotation_mask > 0.5).astype(int)

        # Compute overlap metrics
        attention_binary = (attention_norm > 0.5).astype(int)  # High attention threshold
        annotation_binary = annotation_mask.astype(int)

        # Basic overlap
        intersection = np.sum(attention_binary & annotation_binary)
        union = np.sum(attention_binary | annotation_binary)
        iou = intersection / (union + 1e-8)

        # Dice coefficient
        dice = 2 * intersection / (np.sum(attention_binary) + np.sum(annotation_binary) + 1e-8)

        # Precision and recall
        precision = intersection / (np.sum(attention_binary) + 1e-8)
        recall = intersection / (np.sum(annotation_binary) + 1e-8)

        # Weighted overlap (attention intensity weighted)
        weighted_overlap = np.sum(attention_norm * annotation_binary) / (np.sum(annotation_binary) + 1e-8)

        # Attention coverage of annotation
        coverage = np.sum(attention_norm * annotation_binary) / (np.sum(attention_norm) + 1e-8)

        return {
            'iou': iou,
            'dice': dice,
            'precision': precision,
            'recall': recall,
            'f1_score': 2 * (precision * recall) / (precision + recall + 1e-8),
            'weighted_overlap': weighted_overlap,
            'coverage': coverage,
            'annotation_size': np.sum(annotation_binary),
            'attention_size': np.sum(attention_binary)
        }

    def validate_single_sample(self, image_id: str, input_tensor: torch.Tensor) -> Dict:
        """
        Comprehensive validation for a single sample
        """
        # Extract attention patterns
        bam_attention = self.extractor.extract_bam_attention(input_tensor)

        if not bam_attention:
            return {'error': 'No attention data extracted'}

        # Get attention map
        attention_key = list(bam_attention.keys())[0]  # Use first available
        attention_map = bam_attention[attention_key][0]

        if attention_map.ndim == 4:
            attention_map = np.mean(attention_map, axis=0)
        elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
            attention_map = np.mean(attention_map, axis=0)

        # Enhanced attention analysis
        attention_metrics = self.enhanced_analyzer.compute_attention_saliency_metrics(attention_map)
        attention_regions = self.enhanced_analyzer.identify_attention_regions(attention_map)

        # Validate against all available annotations
        validation_results = {
            'image_id': image_id,
            'attention_key': attention_key,
            'attention_metrics': attention_metrics,
            'attention_regions': attention_regions,
            'annotation_validations': {}
        }

        # Check each feature type in annotations
        if image_id in self.annotator.annotations:
            for feature_type in self.annotator.annotations[image_id]['annotations'].keys():
                annotation_mask = self.annotator.create_annotation_mask(image_id, feature_type)

                if np.sum(annotation_mask) > 0:  # Only validate if annotation exists
                    overlap_metrics = self.compute_attention_annotation_overlap(attention_map, annotation_mask)
                    validation_results['annotation_validations'][feature_type] = overlap_metrics

        return validation_results

    def batch_validation(self, samples_data: List[Tuple[str, torch.Tensor]]) -> pd.DataFrame:
        """
        Perform batch validation across multiple samples
        """
        all_results = []

        for image_id, input_tensor in samples_data:
            print(f"Validating sample: {image_id}")

            try:
                validation_result = self.validate_single_sample(image_id, input_tensor)

                # Flatten results for DataFrame
                if 'annotation_validations' in validation_result:
                    for feature_type, metrics in validation_result['annotation_validations'].items():
                        row = {
                            'image_id': image_id,
                            'attention_layer': validation_result['attention_key'],
                            'feature_type': feature_type,
                            **metrics
                        }

                        # Add attention metrics
                        if 'attention_metrics' in validation_result:
                            for metric_name, value in validation_result['attention_metrics'].items():
                                row[f'attention_{metric_name}'] = value

                        all_results.append(row)

            except Exception as e:
                print(f"Error validating sample {image_id}: {e}")
                continue

        return pd.DataFrame(all_results)

    def compute_validation_statistics(self, validation_df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive validation statistics
        """
        if len(validation_df) == 0:
            return {'error': 'No validation data available'}

        stats = {}

        # Overall statistics by feature type
        for feature_type in validation_df['feature_type'].unique():
            feature_df = validation_df[validation_df['feature_type'] == feature_type]

            stats[feature_type] = {
                'sample_count': len(feature_df),
                'mean_iou': feature_df['iou'].mean(),
                'std_iou': feature_df['iou'].std(),
                'mean_dice': feature_df['dice'].mean(),
                'std_dice': feature_df['dice'].std(),
                'mean_precision': feature_df['precision'].mean(),
                'std_precision': feature_df['precision'].std(),
                'mean_recall': feature_df['recall'].mean(),
                'std_recall': feature_df['recall'].std(),
                'mean_f1_score': feature_df['f1_score'].mean(),
                'std_f1_score': feature_df['f1_score'].std(),
                'mean_weighted_overlap': feature_df['weighted_overlap'].mean(),
                'std_weighted_overlap': feature_df['weighted_overlap'].std()
            }

        # Overall aggregated statistics
        stats['overall'] = {
            'total_samples': len(validation_df),
            'unique_feature_types': validation_df['feature_type'].nunique(),
            'mean_iou': validation_df['iou'].mean(),
            'std_iou': validation_df['iou'].std(),
            'mean_dice': validation_df['dice'].mean(),
            'std_dice': validation_df['dice'].std(),
            'mean_f1_score': validation_df['f1_score'].mean(),
            'std_f1_score': validation_df['f1_score'].std()
        }

        return stats

    def create_validation_visualization(self, validation_df: pd.DataFrame,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive validation visualization
        """
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

        # 1. IoU distribution by feature type
        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=validation_df, x='feature_type', y='iou', ax=ax1)
        ax1.set_title('IoU Distribution by Feature Type')
        ax1.set_ylabel('Intersection over Union')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # 2. Dice coefficient distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=validation_df, x='feature_type', y='dice', ax=ax2)
        ax2.set_title('Dice Coefficient Distribution')
        ax2.set_ylabel('Dice Coefficient')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        # 3. Precision-Recall scatter
        ax3 = fig.add_subplot(gs[0, 2])
        scatter = ax3.scatter(validation_df['precision'], validation_df['recall'],
                            c=validation_df['iou'], cmap='viridis', alpha=0.7, s=100)
        ax3.set_xlabel('Precision')
        ax3.set_ylabel('Recall')
        ax3.set_title('Precision-Recall Scatter (colored by IoU)')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.colorbar(scatter, ax=ax3, label='IoU')
        ax3.grid(True, alpha=0.3)

        # 4. Weighted overlap distribution
        ax4 = fig.add_subplot(gs[0, 3])
        sns.boxplot(data=validation_df, x='feature_type', y='weighted_overlap', ax=ax4)
        ax4.set_title('Weighted Overlap Distribution')
        ax4.set_ylabel('Weighted Overlap')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)

        # 5. Feature type performance heatmap
        ax5 = fig.add_subplot(gs[1, 0:2])
        feature_stats = validation_df.groupby('feature_type')[['iou', 'dice', 'precision', 'recall', 'f1_score']].mean()
        sns.heatmap(feature_stats.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax5)
        ax5.set_title('Performance Metrics by Feature Type')
        ax5.set_xlabel('Feature Type')
        ax5.set_ylabel('Metric')

        # 6. Attention metrics vs overlap correlation
        ax6 = fig.add_subplot(gs[1, 2:4])
        attention_metrics = ['attention_attention_concentration_10', 'attention_attention_entropy',
                           'attention_num_attention_peaks', 'attention_peak_coverage']

        for metric in attention_metrics:
            if metric in validation_df.columns:
                corr = validation_df[metric].corr(validation_df['iou'])
                if not np.isnan(corr) and abs(corr) > 0.1:  # Only plot meaningful correlations
                    ax6.scatter(validation_df[metric], validation_df['iou'],
                              alpha=0.6, label=f'{metric} (r={corr:.3f})')

        ax6.set_xlabel('Attention Metric Value')
        ax6.set_ylabel('IoU')
        ax6.set_title('Attention Metrics vs IoU Correlation')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 7. Sample-wise performance
        ax7 = fig.add_subplot(gs[2, 0:2])
        sample_stats = validation_df.groupby('image_id')['iou'].mean().sort_values()
        ax7.bar(range(len(sample_stats)), sample_stats.values)
        ax7.set_xlabel('Sample (sorted by performance)')
        ax7.set_ylabel('Mean IoU')
        ax7.set_title('Sample-wise Performance Distribution')
        ax7.grid(True, alpha=0.3)

        # 8. Performance categories
        ax8 = fig.add_subplot(gs[2, 2:4])
        performance_categories = pd.cut(validation_df['iou'],
                                      bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                      labels=['Poor (0-0.2)', 'Fair (0.2-0.4)', 'Good (0.4-0.6)',
                                             'Very Good (0.6-0.8)', 'Excellent (0.8-1.0)'])
        category_counts = performance_categories.value_counts()
        ax8.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        ax8.set_title('IoU Performance Categories')

        # 9. Validation summary statistics
        ax9 = fig.add_subplot(gs[3, :])
        ax9.axis('off')

        validation_stats = self.compute_validation_statistics(validation_df)

        summary_text = f"""
        Pathological Validation Summary:

        Total Samples Validated: {validation_stats['overall']['total_samples']}
        Feature Types Evaluated: {validation_stats['overall']['unique_feature_types']}

        Overall Performance:
        - Mean IoU: {validation_stats['overall']['mean_iou']:.3f} ± {validation_stats['overall']['std_iou']:.3f}
        - Mean Dice: {validation_stats['overall']['mean_dice']:.3f} ± {validation_stats['overall']['std_dice']:.3f}
        - Mean F1-Score: {validation_stats['overall']['mean_f1_score']:.3f} ± {validation_stats['overall']['std_f1_score']:.3f}

        Feature Type Performance:
        """

        for feature_type, stats in validation_stats.items():
            if feature_type != 'overall':
                summary_text += f"""
        {feature_type}:
        - IoU: {stats['mean_iou']:.3f} ± {stats['std_iou']:.3f}
        - F1-Score: {stats['mean_f1_score']:.3f} ± {stats['std_f1_score']:.3f}
        - Samples: {stats['sample_count']}
        """

        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')

        plt.suptitle('Pathological Attention Validation Results', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def generate_validation_report(self, validation_df: pd.DataFrame,
                                save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report
        """
        report = []
        report.append("# Pathological Attention Validation Report\n")

        validation_stats = self.compute_validation_statistics(validation_df)

        # Executive summary
        report.append("## Executive Summary\n")
        report.append(f"This report presents the validation of DAE-KAN attention patterns against pathological annotations.")
        report.append(f"**Total Validated Samples**: {validation_stats['overall']['total_samples']}")
        report.append(f"**Feature Types**: {validation_stats['overall']['unique_feature_types']}")
        report.append(f"**Overall Mean IoU**: {validation_stats['overall']['mean_iou']:.3f}")
        report.append(f"**Overall Mean Dice**: {validation_stats['overall']['mean_dice']:.3f}")
        report.append(f"**Overall Mean F1-Score**: {validation_stats['overall']['mean_f1_score']:.3f}\n")

        # Performance assessment
        report.append("## Performance Assessment\n")
        if validation_stats['overall']['mean_iou'] >= 0.6:
            report.append("✅ **Excellent Performance**: The model attention shows strong correlation with pathological features")
        elif validation_stats['overall']['mean_iou'] >= 0.4:
            report.append("⚡ **Good Performance**: The model attention demonstrates reasonable correlation with pathological features")
        elif validation_stats['overall']['mean_iou'] >= 0.2:
            report.append("⚠️ **Fair Performance**: The model attention shows some correlation but needs improvement")
        else:
            report.append("❌ **Poor Performance**: The model attention does not align well with pathological features")

        report.append("")

        # Detailed feature analysis
        report.append("## Feature Type Analysis\n")
        for feature_type, stats in validation_stats.items():
            if feature_type == 'overall':
                continue

            report.append(f"### {feature_type.replace('_', ' ').title()}\n")
            report.append(f"- **Samples**: {stats['sample_count']}")
            report.append(f"- **Mean IoU**: {stats['mean_iou']:.3f} ± {stats['std_iou']:.3f}")
            report.append(f"- **Mean Dice**: {stats['mean_dice']:.3f} ± {stats['std_dice']:.3f}")
            report.append(f"- **Mean Precision**: {stats['mean_precision']:.3f} ± {stats['std_precision']:.3f}")
            report.append(f"- **Mean Recall**: {stats['mean_recall']:.3f} ± {stats['std_recall']:.3f}")
            report.append(f"- **Mean F1-Score**: {stats['mean_f1_score']:.3f} ± {stats['std_f1_score']:.3f}")

            # Performance category
            if stats['mean_iou'] >= 0.6:
                category = "Excellent"
                emoji = "✅"
            elif stats['mean_iou'] >= 0.4:
                category = "Good"
                emoji = "⚡"
            elif stats['mean_iou'] >= 0.2:
                category = "Fair"
                emoji = "⚠️"
            else:
                category = "Poor"
                emoji = "❌"

            report.append(f"- **Performance Category**: {emoji} {category}\n")

        # Pathologist recommendations
        report.append("## Pathologist Recommendations\n")

        # Identify best and worst performing features
        feature_performances = [(feature_type, stats['mean_iou'])
                               for feature_type, stats in validation_stats.items()
                               if feature_type != 'overall']

        if feature_performances:
            feature_performances.sort(key=lambda x: x[1], reverse=True)
            best_feature = feature_performances[0]
            worst_feature = feature_performances[-1]

            report.append(f"### Best Performing Feature")
            report.append(f"- **{best_feature[0].replace('_', ' ').title()}**: IoU = {best_feature[1]:.3f}")
            report.append("  - The model attention aligns well with this feature type")
            report.append("  - Consider this feature as a baseline for model interpretability\n")

            report.append(f"### Areas for Improvement")
            report.append(f"- **{worst_feature[0].replace('_', ' ').title()}**: IoU = {worst_feature[1]:.3f}")
            report.append("  - The model attention shows poor alignment with this feature")
            report.append("  - Investigate why the model fails to focus on this pathology")
            report.append("  - Consider targeted training or attention mechanism improvements\n")

        # Clinical validation recommendations
        report.append("### Clinical Validation Protocol\n")
        report.append("1. **Expert Review**: Have pathologists review attention patterns for diagnostically challenging cases")
        report.append("2. **Feature Importance**: Validate that attention focuses on clinically relevant features")
        report.append("3. **Consistency Check**: Ensure attention patterns are consistent across similar pathological cases")
        report.append("4. **False Positive Analysis**: Investigate areas of high attention that don't correspond to pathology")
        report.append("5. **False Negative Analysis**: Identify pathological features that the model attention misses")

        report.append("\n### Model Improvement Recommendations\n")
        report.append("1. **Attention Refinement**: Adjust attention mechanisms to better focus on pathological features")
        report.append("2. **Training Data**: Consider adding more examples of poorly performing feature types")
        report.append("3. **Loss Function**: Incorporate attention-based loss terms for pathological features")
        report.append("4. **Multi-scale Analysis**: Implement attention at multiple spatial scales")
        report.append("5. **Feature Integration**: Directly integrate pathological feature detection into the model")

        # Conclusion
        report.append("## Conclusion\n")
        overall_iou = validation_stats['overall']['mean_iou']
        if overall_iou >= 0.5:
            report.append("The DAE-KAN model demonstrates **good to excellent interpretability** with attention patterns that meaningfully correlate with pathological features. This suggests the model is learning clinically relevant features and could be valuable for diagnostic support.")
        else:
            report.append("The DAE-KAN model shows **limited interpretability** with attention patterns that don't consistently align with pathological features. Further work is needed to improve the clinical relevance of the attention mechanisms.")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text

    def create_sample_annotation_interface(self, image: np.ndarray,
                                         attention_map: np.ndarray,
                                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Create interface for pathologist annotation
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Original image
        axes[0, 0].imshow(image.transpose(1, 2, 0))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Attention overlay
        img_norm = (image - image.min()) / (image.max() - image.min())
        attention_resized = cv2.resize(attention_map, (img_norm.shape[1], img_norm.shape[0]))
        axes[0, 1].imshow(img_norm.transpose(1, 2, 0))
        im = axes[0, 1].imshow(attention_resized, cmap='jet', alpha=0.5)
        axes[0, 1].set_title('Attention Overlay')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)

        # High attention regions
        high_attention_threshold = np.percentile(attention_map, 90)
        high_attention = (attention_map > high_attention_threshold).astype(float)
        axes[0, 2].imshow(high_attention, cmap='hot')
        axes[0, 2].set_title(f'High Attention Regions (> {high_attention_threshold:.3f})')
        axes[0, 2].axis('off')

        # Annotation guide
        axes[1, 0].axis('off')
        guide_text = """
        Pathologist Annotation Guide:

        1. Review the attention patterns
        2. Identify pathological features
        3. Annotate regions of interest:
           - Normal vs atypical nuclei
           - Glandular structures
           - Stromal changes
           - Architectural abnormalities

        Use the annotation tool to mark:
        - Areas where attention correctly focuses
        - Areas where attention incorrectly focuses
        - Pathological features missed by attention
        - False positive attention regions
        """
        axes[1, 0].text(0.05, 0.95, guide_text, transform=axes[1, 0].transAxes,
                       fontsize=10, verticalalignment='top', family='monospace')

        # Annotation template
        axes[1, 1].imshow(img_norm.transpose(1, 2, 0))
        axes[1, 1].set_title('Annotation Template')
        axes[1, 1].axis('off')

        # Legend and categories
        axes[1, 2].axis('off')
        legend_text = """
        Annotation Categories:

        🟢 Correct Focus:
        - Attention aligns with pathology
        - Clinically relevant regions

        🔴 Incorrect Focus:
        - Attention on non-pathological areas
        - False positive regions

        🟡 Missed Pathology:
        - Pathological features not attended to
        - False negative regions

        🔵 Borderline:
        - Unclear clinical relevance
        - Requires expert discussion
        """
        axes[1, 2].text(0.05, 0.95, legend_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', family='monospace')

        plt.suptitle('Pathologist Annotation Interface', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig


def main():
    """
    Main function to run pathological validation framework
    """
    print("Initializing Pathological Validation Framework...")

    # Create components
    model = DAE_KAN_Attention()
    extractor = AttentionExtractor(model, device="cuda")
    annotator = PathologicalFeatureAnnotator()
    validator = AttentionPathologyValidator(extractor, annotator)

    # Create sample annotations (in practice, these would be created by pathologists)
    print("Creating sample annotations...")
    annotator.add_annotation("sample_1", "atypical_nuclei",
                           [(30, 30), (50, 30), (50, 50), (30, 50)],
                           annotation_type='polygon', confidence=0.9)
    annotator.add_annotation("sample_1", "normal_glands",
                           [(70, 70), (90, 70), (90, 90), (70, 90)],
                           annotation_type='polygon', confidence=0.8)

    annotator.add_annotation("sample_2", "desmoplastic_stroma",
                           [(20, 20), (40, 20), (40, 40), (20, 40)],
                           annotation_type='polygon', confidence=0.7)

    # Save sample annotations
    annotator.save_annotations_to_file("../analysis/annotations/sample_annotations.json")

    # Load dataset for validation
    from histodata import create_dataset, ImageDataset
    from torch.utils.data import DataLoader

    train_ds = ImageDataset(*create_dataset('train'))
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # Prepare samples for validation
    samples_data = []
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Validate 3 samples
            break
        x, y = batch
        samples_data.append((f"sample_{i+1}", x))

    # Run validation
    print("Running pathological validation...")
    validation_df = validator.batch_validation(samples_data)

    # Save validation results
    validation_df.to_csv("../analysis/reports/pathological_validation_results.csv", index=False)

    # Create visualization
    print("Creating validation visualization...")
    fig = validator.create_validation_visualization(
        validation_df,
        save_path="../analysis/visualizations/pathological_validation.png"
    )
    plt.close(fig)

    # Generate report
    print("Generating validation report...")
    report = validator.generate_validation_report(
        validation_df,
        save_path="../analysis/reports/pathological_validation_report.md"
    )

    # Create annotation interface for first sample
    print("Creating annotation interface...")
    x_sample = samples_data[0][1]
    bam_attention = extractor.extract_bam_attention(x_sample)
    if bam_attention:
        attention_key = list(bam_attention.keys())[0]
        attention_map = bam_attention[attention_key][0]
        if attention_map.ndim == 4:
            attention_map = np.mean(attention_map, axis=0)
        elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
            attention_map = np.mean(attention_map, axis=0)

        annotation_fig = validator.create_sample_annotation_interface(
            x_sample.numpy(),
            attention_map,
            save_path="../analysis/visualizations/annotation_interface_sample_1.png"
        )
        plt.close(annotation_fig)

    print("\nPathological Validation Framework Complete!")
    print("Results saved to:")
    print("- ../analysis/annotations/sample_annotations.json")
    print("- ../analysis/reports/pathological_validation_results.csv")
    print("- ../analysis/visualizations/pathological_validation.png")
    print("- ../analysis/reports/pathological_validation_report.md")
    print("- ../analysis/visualizations/annotation_interface_sample_1.png")

    return validator


if __name__ == "__main__":
    validator = main()