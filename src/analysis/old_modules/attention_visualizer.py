"""
Attention Map Visualization for DAE-KAN Model

This module provides comprehensive tools for visualizing and analyzing attention mechanisms
in the DAE-KAN model, including BAM attention and KAN activation patterns.
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
# Optional imports that may not be available
try:
    from sklearn.manifold import TSNE
except ImportError:
    print("Warning: sklearn.manifold not available. Some visualizations will be disabled.")
    TSNE = None

try:
    from sklearn.decomposition import PCA
except ImportError:
    print("Warning: sklearn.decomposition not available. Some visualizations will be disabled.")
    PCA = None

try:
    import umap
except ImportError:
    print("Warning: umap-learn not available. Some visualizations will be disabled.")
    umap = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError:
    print("Warning: plotly not available. Interactive visualizations will be disabled.")
    go = None
    px = None
    make_subplots = None

from models.model import DAE_KAN_Attention
from models.attention_mechanisms.bam import BAM
from histodata import create_dataset, ImageDataset
from torch.utils.data import DataLoader


class AttentionExtractor:
    """
    Extract attention patterns from different layers of the DAE-KAN model
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.attention_data = {}

    def extract_bam_attention(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract spatial and channel attention from BAM layers
        """
        attentions = {}
        hooks = []

        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                # Store the attention-weighted features, convert to float32 for compatibility
                attentions[f"{layer_name}_weighted"] = output.detach().float().cpu().numpy()

                # Try to extract individual attention components if available
                if hasattr(module, 'channel_gate'):
                    channel_att = module.channel_gate(input[0]).detach().float().cpu().numpy()
                    attentions[f"{layer_name}_channel"] = channel_att

                if hasattr(module, 'spatial_gate'):
                    spatial_att = module.spatial_gate(input[0]).detach().float().cpu().numpy()
                    attentions[f"{layer_name}_spatial"] = spatial_att

            return hook_fn

        # Register hooks for BAM layers
        bam_layers = [
            ('bam_384', self.model.bottleneck.attn1),
            ('bam_16', self.model.bottleneck.attn2),
        ]

        for name, layer in bam_layers:
            hook = layer.register_forward_hook(create_hook_fn(name))
            hooks.append(hook)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return attentions

    def extract_kan_activations(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract KAN activation patterns
        """
        activations = {}
        hooks = []

        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                activations[layer_name] = {
                    'input': input[0].detach().float().cpu().numpy(),
                    'output': output.detach().float().cpu().numpy()
                }

                # Extract spline basis if available
                if hasattr(module, 'b_splines'):
                    try:
                        spline_basis = module.b_splines(input[0].flatten(0, 1)).float().cpu().numpy()
                        activations[layer_name]['spline_basis'] = spline_basis
                    except:
                        pass

            return hook_fn

        # Register hooks for KAN layers
        kan_layers = [
            ('encoder_kan', self.model.ae_encoder.kan),
            ('decoder_kan', self.model.ae_decoder.kan),
        ]

        for name, layer in kan_layers:
            hook = layer.register_forward_hook(create_hook_fn(name))
            hooks.append(hook)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations

    def extract_layer_features(self, input_tensor: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Extract features from different encoder/decoder layers
        """
        features = {}
        hooks = []

        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                features[layer_name] = output.detach().float().cpu().numpy()

            return hook_fn

        # Register hooks for key layers
        key_layers = [
            ('encoder1', self.model.ae_encoder.encoder1),
            ('encoder2', self.model.ae_encoder.encoder2),
            ('encoder3', self.model.ae_encoder.encoder3),
            ('bottleneck_enc1', self.model.bottleneck.encoder1),
            ('bottleneck_enc2', self.model.bottleneck.encoder2),
            ('decoder1', self.model.ae_decoder.decoder1),
            ('decoder2', self.model.ae_decoder.decoder2),
            ('decoder3', self.model.ae_decoder.decoder3),
        ]

        for name, layer in key_layers:
            hook = layer.register_forward_hook(create_hook_fn(name))
            hooks.append(hook)

        # Forward pass
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor.to(self.device))

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return features


class AttentionVisualizer:
    """
    Create comprehensive visualizations of attention patterns
    """

    def __init__(self, extractor: AttentionExtractor):
        self.extractor = extractor
        self.color_maps = {
            'attention': 'jet',
            'activation': 'viridis',
            'error': 'hot',
            'features': 'coolwarm'
        }

    def plot_bam_attention(self, attention_data: Dict[str, np.ndarray],
                          original_image: np.ndarray,
                          save_path: Optional[str] = None):
        """
        Visualize BAM attention patterns
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        img = original_image.transpose(1, 2, 0)
        img_norm = (img - img.min()) / (img.max() - img.min())
        ax1.imshow(img_norm)
        ax1.set_title('Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')

        # Plot BAM attention maps
        bam_layers = [key for key in attention_data.keys() if 'bam' in key]
        plot_idx = 1

        for layer_name in sorted(bam_layers):
            if plot_idx >= 12:  # Limit number of plots
                break

            ax = fig.add_subplot(gs[plot_idx // 4, plot_idx % 4])
            attention = attention_data[layer_name][0]  # First image in batch

            if attention.ndim == 4:  # [B, C, H, W]
                # Average across channels or select first channel
                attention_viz = np.mean(attention, axis=0)
            elif attention.ndim == 3:  # [B, H, W] or [C, H, W]
                attention_viz = attention[0] if attention.shape[0] == 1 else np.mean(attention, axis=0)
            else:  # [H, W]
                attention_viz = attention

            im = ax.imshow(attention_viz, cmap=self.color_maps['attention'])
            ax.set_title(f'{layer_name}', fontsize=10)
            ax.axis('off')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

            plot_idx += 1

        # Add attention statistics
        if plot_idx < 12:
            ax = fig.add_subplot(gs[2, 3])
            self._plot_attention_statistics(attention_data, bam_layers, ax)

        plt.suptitle('BAM Attention Patterns Analysis', fontsize=14, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_kan_activations(self, kan_data: Dict[str, np.ndarray],
                           save_path: Optional[str] = None):
        """
        Visualize KAN activation patterns
        """
        fig = plt.figure(figsize=(16, 10))

        num_layers = len(kan_data)
        cols = 3
        rows = (num_layers + cols - 1) // cols

        for i, (layer_name, layer_data) in enumerate(kan_data.items()):
            ax = plt.subplot(rows, cols, i + 1)

            if 'spline_basis' in layer_data:
                # Visualize spline basis functions
                spline_basis = layer_data['spline_basis']
                if spline_basis.ndim == 2:
                    # Average across samples
                    avg_basis = np.mean(spline_basis, axis=0)
                    ax.plot(avg_basis)
                    ax.set_title(f'{layer_name} - Spline Basis', fontsize=10)
                    ax.set_xlabel('Basis Index')
                    ax.set_ylabel('Activation')
                else:
                    # Visualize as heatmap
                    im = ax.imshow(spline_basis[0], cmap=self.color_maps['activation'], aspect='auto')
                    ax.set_title(f'{layer_name} - Spline Basis', fontsize=10)
                    plt.colorbar(im, ax=ax, fraction=0.046)

            elif 'output' in layer_data:
                # Visualize output activations
                output = layer_data['output'][0]  # First sample
                if output.ndim == 3:  # [C, H, W]
                    # Average spatial dimensions
                    activation_mean = np.mean(output, axis=(1, 2))
                    ax.bar(range(len(activation_mean)), activation_mean)
                    ax.set_title(f'{layer_name} - Channel Activations', fontsize=10)
                    ax.set_xlabel('Channel')
                    ax.set_ylabel('Mean Activation')
                elif output.ndim == 1:  # Flattened
                    ax.plot(output)
                    ax.set_title(f'{layer_name} - Activations', fontsize=10)
                    ax.set_xlabel('Neuron Index')
                    ax.set_ylabel('Activation')

            ax.grid(True, alpha=0.3)

        plt.suptitle('KAN Activation Patterns Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_feature_evolution(self, feature_data: Dict[str, np.ndarray],
                              save_path: Optional[str] = None):
        """
        Visualize feature evolution through encoder-decoder pipeline
        """
        fig = plt.figure(figsize=(20, 8))

        # Get feature dimensions
        encoder_layers = [k for k in feature_data.keys() if 'encoder' in k]
        decoder_layers = [k for k in feature_data.keys() if 'decoder' in k]

        # Plot encoder feature evolution
        plt.subplot(2, len(encoder_layers) + 2, 1)
        plt.text(0.5, 0.5, 'Input', ha='center', va='center', fontsize=12, fontweight='bold')
        plt.axis('off')

        for i, layer_name in enumerate(sorted(encoder_layers)):
            ax = plt.subplot(2, len(encoder_layers) + 2, i + 2)
            features = feature_data[layer_name][0]  # First sample

            if features.ndim == 3:  # [C, H, W]
                # Visualize first few channels
                n_channels = min(16, features.shape[0])
                channel_features = features[:n_channels].reshape(n_channels, -1)
                im = ax.imshow(channel_features, cmap=self.color_maps['features'], aspect='auto')
                ax.set_title(f'{layer_name}\n{features.shape}', fontsize=8)
                ax.set_xlabel('Spatial')
                ax.set_ylabel('Channel')
                plt.colorbar(im, ax=ax, fraction=0.046)

        # Plot decoder feature evolution
        plt.subplot(2, len(encoder_layers) + 2, len(encoder_layers) + 2)
        plt.text(0.5, 0.5, 'Latent', ha='center', va='center', fontsize=12, fontweight='bold')
        plt.axis('off')

        for i, layer_name in enumerate(sorted(decoder_layers)):
            ax = plt.subplot(2, len(encoder_layers) + 2, len(encoder_layers) + 3 + i)
            features = feature_data[layer_name][0]

            if features.ndim == 3:
                n_channels = min(16, features.shape[0])
                channel_features = features[:n_channels].reshape(n_channels, -1)
                im = ax.imshow(channel_features, cmap=self.color_maps['features'], aspect='auto')
                ax.set_title(f'{layer_name}\n{features.shape}', fontsize=8)
                ax.set_xlabel('Spatial')
                ax.set_ylabel('Channel')
                plt.colorbar(im, ax=ax, fraction=0.046)

        plt.subplot(2, len(encoder_layers) + 2, len(encoder_layers) + 2 + len(decoder_layers) + 1)
        plt.text(0.5, 0.5, 'Output', ha='center', va='center', fontsize=12, fontweight='bold')
        plt.axis('off')

        plt.suptitle('Feature Evolution Through Encoder-Decoder Pipeline', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_attention_statistics(self, attention_data: Dict[str, np.ndarray],
                                layer_names: List[str], ax):
        """
        Plot attention statistics
        """
        stats = {
            'mean': [],
            'std': [],
            'max': [],
            'min': [],
            'sparsity': []
        }

        for layer_name in layer_names:
            if layer_name in attention_data:
                attention = attention_data[layer_name][0]
                attention_flat = attention.flatten()

                stats['mean'].append(np.mean(attention_flat))
                stats['std'].append(np.std(attention_flat))
                stats['max'].append(np.max(attention_flat))
                stats['min'].append(np.min(attention_flat))
                # Sparsity: proportion of near-zero values
                sparsity = np.mean(np.abs(attention_flat) < 0.01)
                stats['sparsity'].append(sparsity)

        # Create bar plots
        x = np.arange(len(layer_names))
        width = 0.15

        ax.bar(x - 2*width, stats['mean'], width, label='Mean', alpha=0.8)
        ax.bar(x - width, stats['std'], width, label='Std', alpha=0.8)
        ax.bar(x, stats['max'], width, label='Max', alpha=0.8)
        ax.bar(x + width, stats['min'], width, label='Min', alpha=0.8)
        ax.bar(x + 2*width, stats['sparsity'], width, label='Sparsity', alpha=0.8)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Value')
        ax.set_title('Attention Statistics', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('bam_', '') for name in layer_names], rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def create_interactive_attention_plot(self, attention_data: Dict[str, np.ndarray],
                                         original_image: np.ndarray,
                                         save_path: Optional[str] = None):
        """
        Create interactive plot using Plotly
        """
        if go is None or px is None or make_subplots is None:
            print("Warning: plotly not available. Skipping interactive visualization.")
            return

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Original Image', 'BAM Attention 1', 'BAM Attention 2', 'Feature Comparison'),
            specs=[[{"type": "xy"}, {"type": "heatmap"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )

        # Original image
        img = original_image.transpose(1, 2, 0)
        img_norm = (img - img.min()) / (img.max() - img.min())
        fig.add_trace(
            go.Heatmap(z=img_norm[:, :, 0], colorscale='Viridis'),
            row=1, col=1
        )

        # BAM attention maps
        bam_layers = [k for k in attention_data.keys() if 'bam' in k]
        for i, layer_name in enumerate(bam_layers[:2]):
            if i >= 2:
                break
            attention = attention_data[layer_name][0]
            if attention.ndim == 4:
                attention_viz = np.mean(attention, axis=0)
            elif attention.ndim == 3:
                attention_viz = attention[0] if attention.shape[0] == 1 else np.mean(attention, axis=0)
            else:
                attention_viz = attention

            fig.add_trace(
                go.Heatmap(z=attention_viz, colorscale='Jet'),
                row=1, col=i+2
            )

        fig.update_layout(
            title='Interactive Attention Analysis',
            height=600,
            showlegend=False
        )

        if save_path:
            fig.write_html(save_path)

        fig.show()


class AttentionAnalyzer:
    """
    Quantitative analysis of attention patterns
    """

    def __init__(self, extractor: AttentionExtractor):
        self.extractor = extractor

    def compute_attention_entropy(self, attention_map: np.ndarray) -> float:
        """
        Compute entropy of attention map (higher = more distributed)
        """
        attention_flat = attention_map.flatten()
        # Normalize to probability distribution
        attention_prob = attention_flat / (np.sum(attention_flat) + 1e-8)
        # Compute entropy
        entropy = -np.sum(attention_prob * np.log2(attention_prob + 1e-8))
        return entropy

    def compute_attention_concentration(self, attention_map: np.ndarray,
                                       top_k_percent: float = 0.1) -> float:
        """
        Compute how concentrated attention is in top-k regions
        """
        attention_flat = attention_map.flatten()
        k = int(len(attention_flat) * top_k_percent)
        top_k_values = np.sort(attention_flat)[-k:]
        total_attention = np.sum(attention_flat)
        concentration = np.sum(top_k_values) / (total_attention + 1e-8)
        return concentration

    def analyze_attention_patterns(self, attention_data: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive analysis of attention patterns
        """
        analysis = {}

        for layer_name, attention in attention_data.items():
            if 'bam' in layer_name:
                attention_map = attention[0]
                if attention_map.ndim == 4:
                    attention_map = np.mean(attention_map, axis=0)
                elif attention_map.ndim == 3 and attention_map.shape[0] > 1:
                    attention_map = np.mean(attention_map, axis=0)

                analysis[layer_name] = {
                    'entropy': self.compute_attention_entropy(attention_map),
                    'concentration_10': self.compute_attention_concentration(attention_map, 0.1),
                    'concentration_5': self.compute_attention_concentration(attention_map, 0.05),
                    'mean_activation': np.mean(attention_map),
                    'std_activation': np.std(attention_map),
                    'sparsity': np.mean(np.abs(attention_map) < 0.01)
                }

        return analysis

    def compare_attention_patterns(self, attention_data_list: List[Dict[str, np.ndarray]],
                                 labels: List[str]) -> pd.DataFrame:
        """
        Compare attention patterns across multiple samples
        """
        import pandas as pd

        all_results = []

        for i, attention_data in enumerate(attention_data_list):
            analysis = self.analyze_attention_patterns(attention_data)
            for layer_name, metrics in analysis.items():
                metrics['sample'] = labels[i]
                metrics['layer'] = layer_name
                all_results.append(metrics)

        return pd.DataFrame(all_results)


def main():
    """
    Main function to run attention visualization
    """
    print("Initializing Attention Visualization...")

    # Create model and extractor
    model = DAE_KAN_Attention()
    extractor = AttentionExtractor(model, device="cuda")
    visualizer = AttentionVisualizer(extractor)
    analyzer = AttentionAnalyzer(extractor)

    # Create dataset
    train_ds = ImageDataset(*create_dataset('train'))
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)

    # Analyze a few samples
    print("Extracting attention patterns...")
    results = []

    for i, batch in enumerate(dataloader):
        if i >= 5:  # Analyze 5 samples
            break

        x, y = batch

        # Extract different types of attention
        bam_attention = extractor.extract_bam_attention(x)
        kan_activations = extractor.extract_kan_activations(x)
        features = extractor.extract_layer_features(x)

        # Create visualizations
        print(f"Visualizing sample {i+1}/5...")

        # Visualize BAM attention
        visualizer.plot_bam_attention(
            bam_attention,
            x.numpy(),
            save_path=f"../analysis/visualizations/attention_sample_{i+1}.png"
        )

        # Visualize KAN activations
        visualizer.plot_kan_activations(
            kan_activations,
            save_path=f"../analysis/visualizations/kan_activations_sample_{i+1}.png"
        )

        # Visualize feature evolution
        visualizer.plot_feature_evolution(
            features,
            save_path=f"../analysis/visualizations/feature_evolution_sample_{i+1}.png"
        )

        # Analyze attention patterns
        attention_analysis = analyzer.analyze_attention_patterns(bam_attention)
        results.append({
            'sample_id': i,
            'analysis': attention_analysis
        })

        # Create interactive plot for first sample
        if i == 0:
            visualizer.create_interactive_attention_plot(
                bam_attention,
                x.numpy(),
                save_path="../analysis/visualizations/interactive_attention.html"
            )

    print("\nAttention Visualization Complete!")
    print("Results saved to ../analysis/visualizations/")

    return visualizer, results


if __name__ == "__main__":
    visualizer, results = main()