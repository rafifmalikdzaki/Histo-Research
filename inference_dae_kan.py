#!/usr/bin/env python3
"""
DAE-KAN Inference Script with Comprehensive Analysis

This script provides comprehensive inference capabilities for the trained DAE-KAN model,
including reconstruction, attention analysis, pathological validation, and clinical
interpretability assessment.

Usage:
    python inference_dae_kan.py --model_path path/to/model.pth --input_dir path/to/images/
    python inference_dae_kan.py --model_path path/to/model.pth --input_image path/to/image.png
    python inference_dae_kan.py --model_path path/to/model.pth --batch_mode --input_dir path/to/images/
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys
import json
import time
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.factory import get_model
from src.analysis.attention_visualizer import AttentionExtractor, AttentionVisualizer, AttentionAnalyzer
from src.analysis.enhanced_attention_analysis import EnhancedAttentionAnalyzer
from src.analysis.pathology_correlation import AttentionPathologyCorrelator
from src.analysis.pathological_validation_framework import PathologicalFeatureAnnotator, AttentionPathologyValidator
from src.analysis.model_complexity_analysis import ModelComplexityAnalyzer, PerformanceBenchmark
from src.histodata import ImageDataset
from torch.utils.data import DataLoader
import cv2
from PIL import Image
import pandas as pd


class DAEKANInference:
    """
    Comprehensive inference system for DAE-KAN models with analysis capabilities.
    Supports any model type for ablation studies.
    """

    def __init__(self, model_path: str, device: str = "auto", analysis_config: Optional[Dict] = None,
                 model_name: str = None, model_config: Dict = None):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model_name = model_name or self._detect_model_type()
        self.model_config = model_config or {}
        self.model = None
        self.analysis_config = analysis_config or self._default_analysis_config()

        # Initialize analysis components
        self.attention_extractor = None
        self.enhanced_analyzer = None
        self.pathology_correlator = None
        self.pathological_validator = None
        self.complexity_analyzer = None

        # Performance tracking
        self.inference_times = []
        self.memory_usage = []
        self.results = {}

    def _detect_model_type(self) -> str:
        """Detect model type from checkpoint path or metadata"""
        # Try to infer from path name
        path_lower = self.model_path.lower()
        if 'dae_kan' in path_lower or 'attention' in path_lower:
            return 'dae_kan_attention'
        elif 'kan' in path_lower:
            return 'kan_conv'
        elif 'ae' in path_lower or 'autoencoder' in path_lower:
            return 'autoencoder'
        else:
            # Default to DAE-KAN attention
            return 'dae_kan_attention'

    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        else:
            device = device

        return torch.device(device)

    def _default_analysis_config(self) -> Dict:
        """Default analysis configuration"""
        return {
            'enable_attention_analysis': True,
            'enhanced_visualization': True,
            'pathology_correlation': True,
            'complexity_analysis': True,
            'timing_analysis': True,
            'save_intermediate': True,
            'batch_analysis': True,
            'output_format': ['png', 'pdf'],  # Can also include 'svg'
            'dpi': 300,
            'create_report': True
        }

    def load_model(self) -> bool:
        """Load the trained model (supports any model type)"""
        try:
            print(f"Loading model '{self.model_name}' from {self.model_path}...")

            # Initialize model using factory
            self.model = get_model(self.model_name)(**self.model_config)

            # Load state dict
            if self.model_path.endswith('.pth'):
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                elif 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                elif 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                raise ValueError("Unsupported model format. Expected .pth file")

            # Move to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()

            # Initialize analysis components (if model supports attention analysis)
            self._initialize_analysis_components()

            print(f"✓ Model '{self.model_name}' loaded successfully!")
            print(f"✓ Model has {sum(p.numel() for p in self.model.parameters()):,} total parameters")
            return True

        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Available model types: {[name for name in dir(get_model._models) if not name.startswith('_')]}")
            return False

    def _initialize_analysis_components(self):
        """Initialize all analysis components (if supported by model)"""
        if self.analysis_config['enable_attention_analysis']:
            try:
                self.attention_extractor = AttentionExtractor(self.model, device=str(self.device))
                self.enhanced_analyzer = EnhancedAttentionAnalyzer(self.attention_extractor)
                print(f"✓ Attention analysis enabled for model '{self.model_name}'")
            except Exception as e:
                print(f"⚠ Could not initialize attention analysis for model '{self.model_name}': {e}")
                self.attention_extractor = None
                self.enhanced_analyzer = None

        if self.analysis_config['pathology_correlation'] and self.attention_extractor:
            try:
                self.pathology_correlator = AttentionPathologyCorrelator(self.attention_extractor)
                print(f"✓ Pathology correlation analysis enabled")
            except Exception as e:
                print(f"⚠ Could not initialize pathology correlation: {e}")
                self.pathology_correlator = None

        # Initialize pathological validator
        if self.attention_extractor:
            try:
                annotator = PathologicalFeatureAnnotator()
                self.pathological_validator = AttentionPathologyValidator(self.attention_extractor, annotator)
                print(f"✓ Pathological validation enabled")
            except Exception as e:
                print(f"⚠ Could not initialize pathological validation: {e}")
                self.pathological_validator = None

        if self.analysis_config['complexity_analysis']:
            try:
                self.complexity_analyzer = ModelComplexityAnalyzer(self.model, device=str(self.device))
                print(f"✓ Complexity analysis enabled")
            except Exception as e:
                print(f"⚠ Could not initialize complexity analysis: {e}")
                self.complexity_analyzer = None

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess input image for inference"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')

            # Resize to expected input size (128x128)
            image = image.resize((128, 128), Image.LANCZOS)

            # Convert to numpy and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0

            # Convert to tensor and add batch dimension
            image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).unsqueeze(0)

            return image_tensor

        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None

    def run_single_inference(self, image_path: str, output_dir: str,
                           sample_name: Optional[str] = None) -> Dict:
        """Run inference on a single image with comprehensive analysis"""
        if sample_name is None:
            sample_name = Path(image_path).stem

        print(f"Processing {sample_name}...")

        # Preprocess image
        input_tensor = self.preprocess_image(image_path)
        if input_tensor is None:
            return {'error': f'Failed to preprocess {image_path}'}

        input_tensor = input_tensor.to(self.device)

        # Track timing and memory
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(self.device)

        try:
            # Run inference
            with torch.no_grad():
                encoded, decoded, z = self.model(input_tensor)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            inference_time = (time.perf_counter() - start_time) * 1000  # ms

            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.device)
                memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB

            # Store timing data
            self.inference_times.append(inference_time)
            if torch.cuda.is_available():
                self.memory_usage.append(memory_used)

            # Convert to numpy for analysis
            input_np = input_tensor[0].cpu().numpy()
            decoded_np = decoded[0].cpu().numpy()
            encoded_np = encoded[0].cpu().numpy()

            # Calculate reconstruction metrics
            mse_loss = torch.nn.functional.mse_loss(input_tensor, decoded).item()
            ssim_score = self._calculate_ssim(input_np, decoded_np)

            # Initialize results dictionary
            results = {
                'sample_name': sample_name,
                'input_shape': input_np.shape,
                'encoded_shape': encoded_np.shape,
                'decoded_shape': decoded_np.shape,
                'inference_time_ms': inference_time,
                'memory_usage_mb': memory_used if torch.cuda.is_available() else 0,
                'mse_loss': mse_loss,
                'ssim_score': ssim_score,
                'timestamp': datetime.now().isoformat()
            }

            # Create output directory
            sample_output_dir = os.path.join(output_dir, sample_name)
            os.makedirs(sample_output_dir, exist_ok=True)

            # Save basic reconstruction results
            self._save_reconstruction_results(input_np, decoded_np, sample_output_dir)

            # Run comprehensive analysis if enabled
            if self.analysis_config['enable_attention_analysis']:
                attention_results = self._run_attention_analysis(
                    input_tensor, sample_output_dir, sample_name
                )
                results.update(attention_results)

            if self.analysis_config['pathology_correlation']:
                pathology_results = self._run_pathology_analysis(
                    input_tensor, sample_output_dir, sample_name
                )
                results.update(pathology_results)

            # Save results
            self._save_sample_results(results, sample_output_dir)

            print(f"✓ Completed {sample_name} (Time: {inference_time:.2f}ms, MSE: {mse_loss:.6f})")
            return results

        except Exception as e:
            print(f"Error during inference for {sample_name}: {e}")
            return {'error': str(e), 'sample_name': sample_name}

    def _save_reconstruction_results(self, input_np: np.ndarray, decoded_np: np.ndarray,
                                  output_dir: str):
        """Save basic reconstruction visualizations"""
        # Denormalize images
        input_denorm = np.clip(input_np * 255, 0, 255).astype(np.uint8)
        decoded_denorm = np.clip(decoded_np * 255, 0, 255).astype(np.uint8)

        # Save individual images
        self._save_image(input_denorm, os.path.join(output_dir, 'original.png'))
        self._save_image(decoded_denorm, os.path.join(output_dir, 'reconstructed.png'))

        # Create comparison visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(input_denorm.transpose(1, 2, 0))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Reconstructed
        axes[1].imshow(decoded_denorm.transpose(1, 2, 0))
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')

        # Error map
        error_map = np.mean((input_np - decoded_np) ** 2, axis=0)
        im = axes[2].imshow(error_map, cmap='hot')
        axes[2].set_title('Reconstruction Error')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'reconstruction_comparison.png'),
                   dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()

        # Save error map separately
        plt.imsave(os.path.join(output_dir, 'error_map.png'), error_map, cmap='hot')

    def _run_attention_analysis(self, input_tensor: torch.Tensor, output_dir: str,
                              sample_name: str) -> Dict:
        """Run comprehensive attention analysis"""
        attention_results = {}

        try:
            # Extract attention patterns
            print(f"  Running attention analysis for {sample_name}...")

            bam_attention = self.attention_extractor.extract_bam_attention(input_tensor)
            kan_activations = self.attention_extractor.extract_kan_activations(input_tensor)
            layer_features = self.attention_extractor.extract_layer_features(input_tensor)

            attention_results['attention_layers'] = list(bam_attention.keys())
            attention_results['num_attention_layers'] = len(bam_attention)

            # Enhanced attention visualization
            if self.analysis_config['enhanced_visualization']:
                fig = self.enhanced_analyzer.create_enhanced_attention_visualization(
                    bam_attention,
                    input_tensor[0].cpu().numpy(),
                    save_path=os.path.join(output_dir, 'enhanced_attention.png')
                )
                plt.close(fig)

                # Generate attention report
                report = self.enhanced_analyzer.generate_attention_report(
                    bam_attention,
                    save_path=os.path.join(output_dir, 'attention_report.md')
                )
                attention_results['attention_report_generated'] = True

            # Save attention data
            attention_data = {
                'bam_attention': {k: v.tolist() for k, v in bam_attention.items()},
                'kan_activations': {k: {sk: sv.tolist() for sk, sv in v.items()}
                                  for k, v in kan_activations.items()},
                'layer_features': {k: v.tolist() for k, v in layer_features.items()}
            }

            with open(os.path.join(output_dir, 'attention_data.json'), 'w') as f:
                json.dump(attention_data, f, indent=2)

            # Compute attention metrics
            for attention_key, attention in bam_attention.items():
                attention_map = attention[0]
                if attention_map.ndim == 4:
                    attention_map = np.mean(attention_map, axis=0)
                elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                    attention_map = np.mean(attention_map, axis=0)

                metrics = self.enhanced_analyzer.compute_attention_saliency_metrics(attention_map)
                attention_results[f'{attention_key}_metrics'] = metrics

                # Identify attention regions
                regions = self.enhanced_analyzer.identify_attention_regions(attention_map)
                attention_results[f'{attention_key}_regions'] = {
                    'num_regions': regions['num_regions'],
                    'region_properties': regions.get('region_props', [])
                }

            print(f"  ✓ Attention analysis completed for {sample_name}")

        except Exception as e:
            print(f"  ⚠ Attention analysis failed for {sample_name}: {e}")
            attention_results['attention_analysis_error'] = str(e)

        return attention_results

    def _run_pathology_analysis(self, input_tensor: torch.Tensor, output_dir: str,
                              sample_name: str) -> Dict:
        """Run pathology correlation analysis"""
        pathology_results = {}

        try:
            print(f"  Running pathology analysis for {sample_name}...")

            # Run pathology correlation
            correlations = self.pathology_correlator.analyze_sample_correlations(input_tensor)
            pathology_results['pathology_correlations'] = correlations

            # Create pathology visualization
            if correlations:
                # Get attention data for visualization
                bam_attention = self.attention_extractor.extract_bam_attention(input_tensor)

                # Create pathology correlation visualization
                self._create_pathology_visualization(
                    correlations, bam_attention, input_tensor[0].cpu().numpy(), output_dir
                )

                pathology_results['pathology_visualization_created'] = True

            # Pathological validation
            validation_result = self.pathological_validator.validate_single_sample(sample_name, input_tensor)
            pathology_results['pathological_validation'] = validation_result

            # Create annotation interface
            if bam_attention:
                attention_key = list(bam_attention.keys())[0]
                attention_map = bam_attention[attention_key][0]
                if attention_map.ndim == 4:
                    attention_map = np.mean(attention_map, axis=0)
                elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                    attention_map = np.mean(attention_map, axis=0)

                annotation_fig = self.pathological_validator.create_sample_annotation_interface(
                    input_tensor[0].cpu().numpy(),
                    attention_map,
                    save_path=os.path.join(output_dir, 'annotation_interface.png')
                )
                plt.close(annotation_fig)
                pathology_results['annotation_interface_created'] = True

            print(f"  ✓ Pathology analysis completed for {sample_name}")

        except Exception as e:
            print(f"  ⚠ Pathology analysis failed for {sample_name}: {e}")
            pathology_results['pathology_analysis_error'] = str(e)

        return pathology_results

    def _create_pathology_visualization(self, correlations: Dict, attention_data: Dict,
                                     image_np: np.ndarray, output_dir: str):
        """Create pathology correlation visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Pathology Correlation Analysis', fontsize=14, fontweight='bold')

        # Original image
        ax1 = axes[0, 0]
        img = image_np.transpose(1, 2, 0)
        img_norm = (img - img.min()) / (img.max() - img.min())
        ax1.imshow(img_norm)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # BAM attention overlay
        ax2 = axes[0, 1]
        if attention_data and 'bam_384_weighted' in attention_data:
            attention = attention_data['bam_384_weighted'][0]
            if attention.ndim == 4:
                attention_viz = np.mean(attention, axis=0)
            else:
                attention_viz = attention

            # Resize attention to match image
            if attention_viz.shape != img_norm.shape[:2]:
                attention_viz = np.resize(attention_viz, img_norm.shape[:2])

            ax2.imshow(img_norm)
            im = ax2.imshow(attention_viz, cmap='jet', alpha=0.5)
            ax2.set_title('BAM Attention Overlay')
            ax2.axis('off')

        # Correlation summary
        ax3 = axes[1, 0]
        ax3.axis('off')

        # Extract correlation summary
        correlation_summary = []
        for category, features in correlations.items():
            if isinstance(features, dict):
                for feature_name, corr_data in features.items():
                    if isinstance(corr_data, dict) and 'correlation' in corr_data:
                        correlation_summary.append({
                            'feature': f"{category}_{feature_name}",
                            'correlation': corr_data['correlation'],
                            'p_value': corr_data.get('p_value', 1.0)
                        })

        # Create correlation text
        if correlation_summary:
            # Sort by absolute correlation
            correlation_summary.sort(key=lambda x: abs(x['correlation']), reverse=True)

            text = "Top Correlations:\n\n"
            for i, corr in enumerate(correlation_summary[:10]):
                text += f"{i+1}. {corr['feature']}\n"
                text += f"   r = {corr['correlation']:.3f}, p = {corr['p_value']:.3f}\n\n"
        else:
            text = "No significant correlations found"

        ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')

        # Correlation histogram
        ax4 = axes[1, 1]
        if correlation_summary:
            correlations = [c['correlation'] for c in correlation_summary]
            ax4.hist(correlations, bins=20, alpha=0.7, edgecolor='black')
            ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax4.set_xlabel('Correlation Coefficient')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Correlation Distribution')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax4.transAxes)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pathology_correlation.png'),
                   dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images"""
        # Simple SSIM calculation
        mu1 = np.mean(img1, axis=(1, 2))
        mu2 = np.mean(img2, axis=(1, 2))

        sigma1_sq = np.var(img1, axis=(1, 2), ddof=0)
        sigma2_sq = np.var(img2, axis=(1, 2), ddof=0)
        sigma12 = np.mean((img1 - mu1.reshape(-1, 1, 1)) * (img2 - mu2.reshape(-1, 1, 1)), axis=(1, 2))

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_num = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2))
        ssim_den = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

        ssim = ssim_num / (ssim_den + 1e-8)
        return float(np.mean(ssim))

    def _save_image(self, image_array: np.ndarray, save_path: str):
        """Save image array to file"""
        if image_array.shape[0] == 3:  # CHW format
            image_array = image_array.transpose(1, 2, 0)  # Convert to HWC

        image = Image.fromarray(image_array.astype(np.uint8))
        image.save(save_path)

    def _save_sample_results(self, results: Dict, output_dir: str):
        """Save sample results to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_serializable = convert_numpy(results)

        with open(os.path.join(output_dir, 'inference_results.json'), 'w') as f:
            json.dump(results_serializable, f, indent=2)

    def run_batch_inference(self, input_dir: str, output_dir: str,
                          file_extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tiff']) -> Dict:
        """Run inference on all images in a directory"""
        print(f"Running batch inference on {input_dir}...")

        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(Path(input_dir).glob(f'*{ext}'))
            image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"No image files found in {input_dir}")
            return {'error': 'No image files found'}

        print(f"Found {len(image_files)} images")

        # Process each image
        all_results = []
        failed_samples = []

        for i, image_path in enumerate(image_files):
            print(f"\nProcessing {i+1}/{len(image_files)}: {image_path.name}")

            result = self.run_single_inference(
                str(image_path),
                output_dir,
                sample_name=image_path.stem
            )

            if 'error' in result:
                failed_samples.append(result)
            else:
                all_results.append(result)

        # Generate batch summary
        batch_results = {
            'total_samples': len(image_files),
            'successful_samples': len(all_results),
            'failed_samples': len(failed_samples),
            'success_rate': len(all_results) / len(image_files) * 100,
            'timestamp': datetime.now().isoformat()
        }

        if all_results:
            # Calculate aggregate metrics
            batch_results['aggregate_metrics'] = {
                'avg_inference_time_ms': np.mean([r['inference_time_ms'] for r in all_results]),
                'std_inference_time_ms': np.std([r['inference_time_ms'] for r in all_results]),
                'avg_mse_loss': np.mean([r['mse_loss'] for r in all_results]),
                'std_mse_loss': np.std([r['mse_loss'] for r in all_results]),
                'avg_ssim_score': np.mean([r['ssim_score'] for r in all_results]),
                'std_ssim_score': np.std([r['ssim_score'] for r in all_results])
            }

            if self.memory_usage:
                batch_results['aggregate_metrics']['avg_memory_mb'] = np.mean(self.memory_usage)
                batch_results['aggregate_metrics']['std_memory_mb'] = np.std(self.memory_usage)

        # Save batch results
        self._save_sample_results(batch_results, output_dir)

        # Create batch visualization
        self._create_batch_summary_visualization(all_results, output_dir)

        # Generate batch report
        if self.analysis_config['create_report']:
            self._generate_batch_report(batch_results, all_results, failed_samples, output_dir)

        print(f"\n✓ Batch inference completed!")
        print(f"  Successful: {len(all_results)}/{len(image_files)} samples")
        print(f"  Average inference time: {batch_results['aggregate_metrics']['avg_inference_time_ms']:.2f} ms")
        print(f"  Average MSE: {batch_results['aggregate_metrics']['avg_mse_loss']:.6f}")
        print(f"  Average SSIM: {batch_results['aggregate_metrics']['avg_ssim_score']:.4f}")

        return batch_results

    def _create_batch_summary_visualization(self, results: List[Dict], output_dir: str):
        """Create batch summary visualization"""
        if not results:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Batch Inference Summary', fontsize=16, fontweight='bold')

        # Inference time distribution
        ax1 = axes[0, 0]
        inference_times = [r['inference_time_ms'] for r in results]
        ax1.hist(inference_times, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Inference Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Inference Time Distribution')
        ax1.grid(True, alpha=0.3)

        # MSE loss distribution
        ax2 = axes[0, 1]
        mse_losses = [r['mse_loss'] for r in results]
        ax2.hist(mse_losses, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_xlabel('MSE Loss')
        ax2.set_ylabel('Frequency')
        ax2.set_title('MSE Loss Distribution')
        ax2.grid(True, alpha=0.3)

        # SSIM score distribution
        ax3 = axes[0, 2]
        ssim_scores = [r['ssim_score'] for r in results]
        ax3.hist(ssim_scores, bins=20, alpha=0.7, edgecolor='black', color='green')
        ax3.set_xlabel('SSIM Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('SSIM Score Distribution')
        ax3.grid(True, alpha=0.3)

        # Performance scatter plot
        ax4 = axes[1, 0]
        scatter = ax4.scatter(inference_times, mse_losses, c=ssim_scores,
                            cmap='viridis', alpha=0.7, s=50)
        ax4.set_xlabel('Inference Time (ms)')
        ax4.set_ylabel('MSE Loss')
        ax4.set_title('Performance Scatter (colored by SSIM)')
        plt.colorbar(scatter, ax=ax4, label='SSIM')
        ax4.grid(True, alpha=0.3)

        # Summary statistics
        ax5 = axes[1, 1]
        ax5.axis('off')

        stats_text = f"""
        Batch Statistics:

        Samples: {len(results)}

        Inference Time:
        - Mean: {np.mean(inference_times):.2f} ms
        - Std: {np.std(inference_times):.2f} ms
        - Min: {np.min(inference_times):.2f} ms
        - Max: {np.max(inference_times):.2f} ms

        MSE Loss:
        - Mean: {np.mean(mse_losses):.6f}
        - Std: {np.std(mse_losses):.6f}
        - Min: {np.min(mse_losses):.6f}
        - Max: {np.max(mse_losses):.6f}

        SSIM Score:
        - Mean: {np.mean(ssim_scores):.4f}
        - Std: {np.std(ssim_scores):.4f}
        - Min: {np.min(ssim_scores):.4f}
        - Max: {np.max(ssim_scores):.4f}
        """

        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')

        # Performance over time (if order matters)
        ax6 = axes[1, 2]
        sample_indices = range(len(results))
        ax6.plot(sample_indices, inference_times, 'b-', alpha=0.7, label='Inference Time')
        ax6_twin = ax6.twinx()
        ax6_twin.plot(sample_indices, mse_losses, 'r-', alpha=0.7, label='MSE Loss')
        ax6.set_xlabel('Sample Index')
        ax6.set_ylabel('Inference Time (ms)', color='b')
        ax6_twin.set_ylabel('MSE Loss', color='r')
        ax6.set_title('Performance Over Processing Order')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'batch_summary.png'),
                   dpi=self.analysis_config['dpi'], bbox_inches='tight')
        plt.close()

    def _generate_batch_report(self, batch_results: Dict, successful_results: List[Dict],
                             failed_samples: List[Dict], output_dir: str):
        """Generate comprehensive batch report"""
        report = []
        report.append("# DAE-KAN Batch Inference Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive summary
        report.append("## Executive Summary\n")
        report.append(f"- **Total Samples Processed**: {batch_results['total_samples']}")
        report.append(f"- **Successful Inferences**: {batch_results['successful_samples']}")
        report.append(f"- **Failed Inferences**: {batch_results['failed_samples']}")
        report.append(f"- **Success Rate**: {batch_results['success_rate']:.1f}%\n")

        # Performance metrics
        if 'aggregate_metrics' in batch_results:
            metrics = batch_results['aggregate_metrics']
            report.append("## Performance Metrics\n")
            report.append(f"### Inference Performance\n")
            report.append(f"- **Average Inference Time**: {metrics['avg_inference_time_ms']:.2f} ± {metrics['std_inference_time_ms']:.2f} ms")
            report.append(f"- **Throughput**: {1000 / metrics['avg_inference_time_ms']:.1f} images/second")

            if 'avg_memory_mb' in metrics:
                report.append(f"- **Average Memory Usage**: {metrics['avg_memory_mb']:.1f} ± {metrics['std_memory_mb']:.1f} MB")

            report.append(f"\n### Reconstruction Quality\n")
            report.append(f"- **Average MSE Loss**: {metrics['avg_mse_loss']:.6f} ± {metrics['std_mse_loss']:.6f}")
            report.append(f"- **Average SSIM Score**: {metrics['avg_ssim_score']:.4f} ± {metrics['std_ssim_score']:.4f}")

        # Sample analysis
        report.append(f"\n## Sample Analysis\n")

        if successful_results:
            # Best performing samples
            best_samples = sorted(successful_results, key=lambda x: x['ssim_score'], reverse=True)[:5]
            report.append("### Best Performing Samples (by SSIM)\n")
            for i, sample in enumerate(best_samples, 1):
                report.append(f"{i}. **{sample['sample_name']}**")
                report.append(f"   - SSIM: {sample['ssim_score']:.4f}")
                report.append(f"   - MSE: {sample['mse_loss']:.6f}")
                report.append(f"   - Time: {sample['inference_time_ms']:.2f} ms\n")

            # Worst performing samples
            worst_samples = sorted(successful_results, key=lambda x: x['ssim_score'])[:5]
            report.append("### Worst Performing Samples (by SSIM)\n")
            for i, sample in enumerate(worst_samples, 1):
                report.append(f"{i}. **{sample['sample_name']}**")
                report.append(f"   - SSIM: {sample['ssim_score']:.4f}")
                report.append(f"   - MSE: {sample['mse_loss']:.6f}")
                report.append(f"   - Time: {sample['inference_time_ms']:.2f} ms\n")

        # Failed samples
        if failed_samples:
            report.append("### Failed Samples\n")
            for sample in failed_samples:
                report.append(f"- **{sample['sample_name']}**: {sample.get('error', 'Unknown error')}")
            report.append("")

        # Attention analysis summary
        attention_samples = [r for r in successful_results if 'attention_layers' in r]
        if attention_samples:
            report.append("## Attention Analysis Summary\n")
            report.append(f"- **Samples with Attention Analysis**: {len(attention_samples)}")

            if attention_samples:
                avg_layers = np.mean([r['num_attention_layers'] for r in attention_samples])
                report.append(f"- **Average Attention Layers**: {avg_layers:.1f}")

                # Check for common attention patterns
                layer_names = []
                for r in attention_samples:
                    layer_names.extend(r['attention_layers'])

                if layer_names:
                    from collections import Counter
                    common_layers = Counter(layer_names).most_common(5)
                    report.append("\n### Most Common Attention Layers\n")
                    for layer_name, count in common_layers:
                        report.append(f"- **{layer_name}**: {count} samples")

        # Pathology analysis summary
        pathology_samples = [r for r in successful_results if 'pathology_correlations' in r]
        if pathology_samples:
            report.append(f"\n## Pathology Analysis Summary\n")
            report.append(f"- **Samples with Pathology Analysis**: {len(pathology_samples)}")
            report.append("- Pathology correlation analysis performed on processed samples")
            report.append("- Annotation interfaces generated for expert validation")

        # Recommendations
        report.append(f"\n## Recommendations\n")

        avg_inference_time = batch_results['aggregate_metrics']['avg_inference_time_ms']
        avg_ssim = batch_results['aggregate_metrics']['avg_ssim_score']

        if avg_inference_time < 100:
            report.append("✅ **Inference Speed**: Excellent for real-time applications")
        elif avg_inference_time < 500:
            report.append("⚡ **Inference Speed**: Good for batch processing applications")
        else:
            report.append("⚠️ **Inference Speed**: May be slow for time-sensitive applications")

        if avg_ssim > 0.9:
            report.append("✅ **Reconstruction Quality**: Excellent quality achieved")
        elif avg_ssim > 0.8:
            report.append("⚡ **Reconstruction Quality**: Good quality achieved")
        elif avg_ssim > 0.7:
            report.append("⚠️ **Reconstruction Quality**: Moderate quality, may need improvement")
        else:
            report.append("❌ **Reconstruction Quality**: Poor quality, needs significant improvement")

        report.append("\n### Next Steps\n")
        report.append("1. Review attention visualizations for clinical relevance")
        report.append("2. Validate attention patterns with pathologist expertise")
        report.append("3. Consider performance optimization if needed for deployment")
        report.append("4. Test on challenging cases to assess robustness")

        report_text = "\n".join(report)

        with open(os.path.join(output_dir, 'batch_report.md'), 'w') as f:
            f.write(report_text)

    def run_complexity_analysis(self, output_dir: str) -> Dict:
        """Run comprehensive model complexity analysis"""
        if not self.complexity_analyzer:
            print("Complexity analysis not enabled")
            return {}

        print("Running model complexity analysis...")

        try:
            complexity_results = self.complexity_analyzer.comprehensive_complexity_analysis()

            # Save complexity results
            with open(os.path.join(output_dir, 'model_complexity_analysis.json'), 'w') as f:
                json.dump(complexity_results, f, indent=2, default=str)

            # Run performance benchmark
            benchmark = PerformanceBenchmark(device=str(self.device))
            benchmark_df = benchmark.benchmark_models(num_runs=50)

            # Save benchmark results
            benchmark_df.to_csv(os.path.join(output_dir, 'performance_benchmark.csv'), index=False)

            # Create benchmark visualization
            benchmark.create_comparison_visualization(
                save_path=os.path.join(output_dir, 'benchmark_comparison.png')
            )

            # Generate efficiency report
            report = benchmark.generate_efficiency_report(
                benchmark_df,
                save_path=os.path.join(output_dir, 'efficiency_report.md')
            )

            print("✓ Complexity analysis completed")
            return complexity_results

        except Exception as e:
            print(f"Complexity analysis failed: {e}")
            return {'error': str(e)}

    def generate_final_report(self, output_dir: str, batch_results: Optional[Dict] = None):
        """Generate final comprehensive report"""
        report = []
        report.append("# DAE-KAN Inference Final Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Model: {self.model_path}")
        report.append(f"Device: {self.device}\n")

        # Model information
        report.append("## Model Information\n")
        if hasattr(self.model, 'parameters'):
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            report.append(f"- **Total Parameters**: {total_params:,}")
            report.append(f"- **Trainable Parameters**: {trainable_params:,}")
            report.append(f"- **Model Size**: {total_params * 4 / (1024**2):.1f} MB (FP32)")

        # Performance summary
        if self.inference_times:
            report.append(f"\n## Performance Summary\n")
            report.append(f"- **Average Inference Time**: {np.mean(self.inference_times):.2f} ± {np.std(self.inference_times):.2f} ms")
            report.append(f"- **Throughput**: {1000 / np.mean(self.inference_times):.1f} images/second")
            report.append(f"- **Total Samples Processed**: {len(self.inference_times)}")

            if self.memory_usage:
                report.append(f"- **Average Memory Usage**: {np.mean(self.memory_usage):.1f} ± {np.std(self.memory_usage):.1f} MB")

        # Batch results summary
        if batch_results:
            report.append(f"\n## Batch Results Summary\n")
            report.append(f"- **Success Rate**: {batch_results['success_rate']:.1f}%")
            report.append(f"- **Total Samples**: {batch_results['total_samples']}")
            report.append(f"- **Successful**: {batch_results['successful_samples']}")
            report.append(f"- **Failed**: {batch_results['failed_samples']}")

            if 'aggregate_metrics' in batch_results:
                metrics = batch_results['aggregate_metrics']
                report.append(f"\n### Quality Metrics\n")
                report.append(f"- **Average SSIM**: {metrics['avg_ssim_score']:.4f}")
                report.append(f"- **Average MSE**: {metrics['avg_mse_loss']:.6f}")

        # Analysis capabilities
        report.append(f"\n## Analysis Capabilities\n")
        report.append("✅ **Reconstruction Analysis**: Original vs reconstructed comparison")
        report.append("✅ **Attention Visualization**: Multi-layer attention pattern analysis")
        report.append("✅ **Enhanced Metrics**: Quantitative attention saliency metrics")
        report.append("✅ **Pathology Correlation**: Correlation with pathological features")
        report.append("✅ **Annotation Interface**: Tools for pathologist validation")
        report.append("✅ **Performance Benchmarking**: Model complexity and efficiency analysis")

        # Clinical deployment readiness
        report.append(f"\n## Clinical Deployment Readiness\n")

        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            if avg_time < 100:
                readiness = "✅ **Ready** for real-time clinical applications"
            elif avg_time < 1000:
                readiness = "⚡ **Suitable** for batch clinical processing"
            else:
                readiness = "⚠️ **Needs optimization** for clinical deployment"

            report.append(f"- **Inference Speed**: {readiness}")

        # Recommendations
        report.append(f"\n## Recommendations\n")
        report.append("### For Clinical Use\n")
        report.append("1. **Validate attention patterns** with expert pathologists")
        report.append("2. **Test on diverse cases** to assess robustness")
        report.append("3. **Consider integration** into existing clinical workflows")
        report.append("4. **Implement quality control** measures for automated analysis")

        report.append("\n### For Research\n")
        report.append("1. **Explore attention mechanisms** for improved interpretability")
        report.append("2. **Investigate multi-scale analysis** for different detail levels")
        report.append("3. **Develop domain-specific adaptations** for particular pathology types")
        report.append("4. **Create standardized evaluation protocols** for clinical validation")

        report_text = "\n".join(report)

        with open(os.path.join(output_dir, 'final_report.md'), 'w') as f:
            f.write(report_text)

        print(f"✓ Final report generated: {os.path.join(output_dir, 'final_report.md')}")


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description='DAE-KAN Inference with Comprehensive Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image inference
  python inference_dae_kan.py --model_path model.pth --input_image image.png

  # Batch inference
  python inference_dae_kan.py --model_path model.pth --input_dir images/ --batch_mode

  # With custom configuration
  python inference_dae_kan.py --model_path model.pth --input_dir images/ --batch_mode --config config.json
        """
    )

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--model_name', type=str,
                       help='Model type (auto-detected if not specified)')
    parser.add_argument('--input_image', type=str,
                       help='Path to single input image')
    parser.add_argument('--input_dir', type=str,
                       help='Path to directory containing images')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results (default: inference_results)')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Run batch inference on all images in input directory')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Computation device (default: auto)')
    parser.add_argument('--config', type=str,
                       help='Path to analysis configuration JSON file')
    parser.add_argument('--enable_analysis', action='store_true', default=True,
                       help='Enable comprehensive analysis (default: True)')
    parser.add_argument('--disable_analysis', action='store_true',
                       help='Disable analysis for faster inference')
    parser.add_argument('--list_models', action='store_true',
                       help='List available model types and exit')

    args = parser.parse_args()

    # Handle list models request
    if args.list_models:
        print("Available model types for ablation studies:")
        try:
            from models.factory import get_model
            available_models = []
            for name in dir(get_model._models):
                if not name.startswith('_'):
                    available_models.append(name)

            if available_models:
                print("📋 Available Models:")
                for i, model_name in enumerate(available_models, 1):
                    print(f"  {i:2d}. {model_name}")
                print(f"\n💡 Usage: python inference_dae_kan.py --model_path model.pth --model_name {available_models[0]} --input_image image.png")
            else:
                print("❌ No models found in factory")
        except Exception as e:
            print(f"❌ Error loading model factory: {e}")
        return

    # Validate arguments
    if not args.input_image and not args.input_dir:
        parser.error("Either --input_image or --input_dir must be specified")

    if args.input_image and args.batch_mode:
        parser.error("Cannot use --input_image with --batch_mode")

    if not os.path.exists(args.model_path):
        parser.error(f"Model file not found: {args.model_path}")

    if args.input_image and not os.path.exists(args.input_image):
        parser.error(f"Input image not found: {args.input_image}")

    if args.input_dir and not os.path.exists(args.input_dir):
        parser.error(f"Input directory not found: {args.input_dir}")

    # Load configuration
    analysis_config = None
    if args.config:
        try:
            with open(args.config, 'r') as f:
                analysis_config = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config file {args.config}: {e}")

    if args.disable_analysis:
        analysis_config = {
            'enable_attention_analysis': False,
            'enhanced_visualization': False,
            'pathology_correlation': False,
            'complexity_analysis': False,
            'timing_analysis': True,
            'create_report': True
        }

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize inference system
    print("🚀 Initializing DAE-KAN Inference System...")
    print(f"📋 Model: {args.model_name or 'auto-detect'}")
    print(f"📁 Model path: {args.model_path}")

    inference_system = DAEKANInference(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device,
        analysis_config=analysis_config
    )

    # Load model
    if not inference_system.load_model():
        print("❌ Failed to load model. Exiting.")
        return

    print(f"📁 Output directory: {args.output_dir}")

    # Run inference
    start_time = time.time()

    if args.batch_mode:
        # Batch inference
        results = inference_system.run_batch_inference(args.input_dir, args.output_dir)
    else:
        # Single image inference
        input_path = args.input_image or args.input_dir
        sample_name = Path(input_path).stem
        results = inference_system.run_single_inference(input_path, args.output_dir, sample_name)

    total_time = time.time() - start_time

    # Run complexity analysis if enabled
    if analysis_config is None or analysis_config.get('complexity_analysis', True):
        print("\n🔍 Running model complexity analysis...")
        complexity_results = inference_system.run_complexity_analysis(args.output_dir)

    # Generate final report
    print("\n📊 Generating final report...")
    inference_system.generate_final_report(args.output_dir, results if isinstance(results, dict) else None)

    # Print summary
    print(f"\n✅ Inference completed in {total_time:.2f} seconds")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📄 Final report: {os.path.join(args.output_dir, 'final_report.md')}")

    if isinstance(results, dict) and 'aggregate_metrics' in results:
        metrics = results['aggregate_metrics']
        print(f"\n📈 Performance Summary:")
        print(f"   Average inference time: {metrics['avg_inference_time_ms']:.2f} ms")
        print(f"   Throughput: {1000 / metrics['avg_inference_time_ms']:.1f} images/second")
        print(f"   Average SSIM: {metrics['avg_ssim_score']:.4f}")
        print(f"   Average MSE: {metrics['avg_mse_loss']:.6f}")


if __name__ == "__main__":
    main()
