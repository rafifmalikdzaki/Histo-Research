#!/usr/bin/env python3
"""
Extract DAE-KAN Embeddings from Histopathology Tiles

This script extracts z-value embeddings from pre-trained DAE-KAN models
for all histopathology tiles, enabling downstream analysis and clustering.

Usage:
    # Extract embeddings from all tiles
    python extract_tile_embeddings.py --model-path path/to/model.pth --data-dir data/processed/HeparUnifiedPNG/tiles

    # Extract with specific batch size and model
    python extract_tile_embeddings.py \
        --model-path checkpoints/best_model.pth \
        --data-dir data/processed/HeparUnifiedPNG/tiles \
        --model-name dae_kan_attention \
        --batch-size 32 \
        --output-dir embeddings_analysis

    # Extract with t-SNE/UMAP visualization
    python extract_tile_embeddings.py \
        --model-path path/to/model.pth \
        --data-dir data/processed/HeparUnifiedPNG/tiles \
        --create-visualizations \
        --sample-size 5000
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import glob
import argparse
import time
from datetime import datetime
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.model import DAE_KAN_Attention
    from models.factory import get_model
    MODEL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"⚠ Model import error: {e}")
    MODEL_IMPORTS_AVAILABLE = False


class TileEmbeddingExtractor:
    """Extract embeddings from histopathology tiles using DAE-KAN models"""

    def __init__(self, model_name: str = "dae_kan_attention", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.transform = None

    def load_model(self, model_path: str = None):
        """Load trained DAE-KAN model"""
        if not MODEL_IMPORTS_AVAILABLE:
            raise ImportError("Required model imports not available")

        try:
            # Initialize model
            self.model = get_model(self.model_name)()

            # Load weights if provided
            if model_path and os.path.exists(model_path):
                print(f"Loading model from: {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)

                # Handle different checkpoint formats
                if 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                else:
                    self.model.load_state_dict(checkpoint)

                print("✅ Model loaded successfully")
            else:
                print("⚠ No model path provided, using randomly initialized model")

            self.model.to(self.device)
            self.model.eval()

            # Setup image transforms
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ])

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False

    def extract_embeddings_from_directory(self, data_dir: str, output_dir: str,
                                        batch_size: int = 32, max_tiles: int = None,
                                        create_visualizations: bool = True):
        """Extract embeddings from all tiles in a directory"""

        print(f"🔍 Scanning directory: {data_dir}")

        # Get all tile files
        tile_files = glob.glob(os.path.join(data_dir, "*.png"))
        if not tile_files:
            tile_files = glob.glob(os.path.join(data_dir, "*.jpg"))
        if not tile_files:
            tile_files = glob.glob(os.path.join(data_dir, "*.jpeg"))

        if not tile_files:
            raise ValueError(f"No image files found in {data_dir}")

        # Limit number of tiles if specified
        if max_tiles and max_tiles < len(tile_files):
            tile_files = tile_files[:max_tiles]
            print(f"📊 Limiting to {max_tiles} tiles")

        print(f"📊 Found {len(tile_files)} tiles to process")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Extract embeddings
        embeddings, metadata = self._extract_embeddings_batch(
            tile_files, batch_size, create_visualizations
        )

        # Save results
        self._save_embeddings(embeddings, metadata, output_dir, create_visualizations)

        return embeddings, metadata

    def _extract_embeddings_batch(self, tile_files: list, batch_size: int,
                                 create_visualizations: bool):
        """Extract embeddings in batches"""

        embeddings_list = []
        metadata_list = []

        print("🚀 Starting embedding extraction...")

        # Process tiles in batches
        for i in tqdm(range(0, len(tile_files), batch_size), desc="Processing batches"):
            batch_files = tile_files[i:i+batch_size]

            # Load and preprocess batch
            batch_images = []
            batch_metadata = []

            for file_path in batch_files:
                try:
                    # Load image
                    image = Image.open(file_path).convert('RGB')

                    # Extract metadata
                    tile_name = os.path.basename(file_path)

                    # Parse tile information from filename
                    metadata = {
                        'tile_path': file_path,
                        'tile_name': tile_name,
                        'file_size': os.path.getsize(file_path),
                        'processing_time': datetime.now().isoformat()
                    }

                    # Extract experimental info from filename
                    if '_' in tile_name and 'T' in tile_name:
                        parts = tile_name.replace('.png', '').replace('.jpg', '').replace('.jpeg', '').split('_')
                        if len(parts) >= 3:
                            metadata['experiment'] = parts[0]
                            metadata['treatment'] = parts[1]
                            metadata['tile_id'] = parts[2]

                    # Transform image
                    image_tensor = self.transform(image)
                    batch_images.append(image_tensor)
                    batch_metadata.append(metadata)

                except Exception as e:
                    print(f"⚠ Failed to load {file_path}: {e}")
                    continue

            if not batch_images:
                continue

            # Stack images
            batch_tensor = torch.stack(batch_images).to(self.device)

            # Extract embeddings
            with torch.no_grad():
                start_time = time.time()
                encoded, decoded, z = self.model(batch_tensor)
                extraction_time = time.time() - start_time

                # Process embeddings
                for j, z_embedding in enumerate(z):
                    embedding_flat = z_embedding.cpu().numpy().flatten()

                    # Add extraction metadata
                    batch_metadata[j]['extraction_time_ms'] = extraction_time * 1000 / len(z)
                    batch_metadata[j]['embedding_shape'] = z_embedding.shape
                    batch_metadata[j]['embedding_norm'] = float(np.linalg.norm(embedding_flat))
                    batch_metadata[j]['embedding_mean'] = float(np.mean(embedding_flat))
                    batch_metadata[j]['embedding_std'] = float(np.std(embedding_flat))

                    embeddings_list.append(embedding_flat)
                    metadata_list.append(batch_metadata[j])

        print(f"✅ Extracted embeddings for {len(embeddings_list)} tiles")

        return np.array(embeddings_list), pd.DataFrame(metadata_list)

    def _save_embeddings(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                        output_dir: str, create_visualizations: bool):
        """Save embeddings and create analysis"""

        print("💾 Saving embeddings...")

        # Save raw embeddings
        embeddings_file = os.path.join(output_dir, 'tile_embeddings.npy')
        np.save(embeddings_file, embeddings)
        print(f"✅ Saved embeddings to {embeddings_file}")

        # Save metadata
        metadata_file = os.path.join(output_dir, 'tile_metadata.csv')
        metadata.to_csv(metadata_file, index=False)
        print(f"✅ Saved metadata to {metadata_file}")

        # Save summary statistics
        summary = {
            'total_tiles': len(embeddings),
            'embedding_dimensions': embeddings.shape[1],
            'extraction_date': datetime.now().isoformat(),
            'model_name': self.model_name,
            'device': str(self.device),
            'embedding_stats': {
                'mean': float(np.mean(embeddings)),
                'std': float(np.std(embeddings)),
                'min': float(np.min(embeddings)),
                'max': float(np.max(embeddings)),
                'median': float(np.median(embeddings))
            }
        }

        summary_file = os.path.join(output_dir, 'extraction_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✅ Saved summary to {summary_file}")

        # Create visualizations if requested
        if create_visualizations:
            self._create_embedding_visualizations(embeddings, metadata, output_dir)

        # Create HTML report
        self._create_html_report(embeddings, metadata, output_dir, summary)

    def _create_embedding_visualizations(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                       output_dir: str):
        """Create embedding visualizations"""

        print("📊 Creating visualizations...")

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")

            # Create visualizations directory
            viz_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)

            # 1. Embedding statistics
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Tile Embedding Statistics', fontsize=16, fontweight='bold')

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
            stats_path = os.path.join(viz_dir, 'embedding_statistics.png')
            plt.savefig(stats_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            # 2. Dimensionality reduction (sample if too many)
            max_samples = 2000
            if len(embeddings) > max_samples:
                indices = np.random.choice(len(embeddings), max_samples, replace=False)
                embeddings_sample = embeddings[indices]
                metadata_sample = metadata.iloc[indices].reset_index(drop=True)
            else:
                embeddings_sample = embeddings
                metadata_sample = metadata.reset_index(drop=True)

            # Standardize for better results
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_sample)

            # UMAP
            try:
                import umap
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                fig.suptitle('Dimensionality Reduction of Tile Embeddings', fontsize=16, fontweight='bold')

                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                embedding_2d = reducer.fit_transform(embeddings_scaled)

                # Color by experiment if available
                if 'experiment' in metadata_sample.columns:
                    experiments = metadata_sample['experiment']
                    unique_experiments = experiments.unique()
                    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_experiments)))

                    for i, exp in enumerate(unique_experiments):
                        mask = experiments == exp
                        axes[0].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                                      c=[colors[i]], label=exp, alpha=0.7, s=30)

                    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6, s=30)

                axes[0].set_title('UMAP Projection')
                axes[0].set_xlabel('UMAP 1')
                axes[0].set_ylabel('UMAP 2')
                axes[0].grid(True, alpha=0.3)

                # t-SNE
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)//4))
                embedding_2d = tsne.fit_transform(embeddings_scaled)

                if 'experiment' in metadata_sample.columns:
                    for i, exp in enumerate(unique_experiments):
                        mask = experiments == exp
                        axes[1].scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                                      c=[colors[i]], label=exp, alpha=0.7, s=30)

                    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                else:
                    axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6, s=30)

                axes[1].set_title('t-SNE Projection')
                axes[1].set_xlabel('t-SNE 1')
                axes[1].set_ylabel('t-SNE 2')
                axes[1].grid(True, alpha=0.3)

                plt.tight_layout()
                dimred_path = os.path.join(viz_dir, 'dimensionality_reduction.png')
                plt.savefig(dimred_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()

            except ImportError:
                print("⚠ UMAP not available. Install with: pip install umap-learn")
            except Exception as e:
                print(f"⚠ Dimensionality reduction failed: {e}")

            print(f"✅ Visualizations saved to {viz_dir}")

        except ImportError:
            print("⚠ Matplotlib not available for visualizations")
        except Exception as e:
            print(f"⚠ Visualization creation failed: {e}")

    def _create_html_report(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                           output_dir: str, summary: dict):
        """Create HTML report"""

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tile Embedding Analysis Report</title>
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
        .viz-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .viz-item {{
            background-color: #fafafa;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .viz-item img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .viz-item h3 {{
            margin-top: 10px;
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧬 Tile Embedding Analysis Report</h1>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{summary['total_tiles']:,}</div>
                <div class="stat-label">Total Tiles Processed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{summary['embedding_dimensions']:,}</div>
                <div class="stat-label">Embedding Dimensions</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{summary['embedding_stats']['mean']:.4f}</div>
                <div class="stat-label">Mean Embedding Value</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{summary['embedding_stats']['std']:.4f}</div>
                <div class="stat-label">Embedding Std Dev</div>
            </div>
        </div>

        <div class="section">
            <h2>📋 Generated Files</h2>
            <div class="file-list">
                <ul>
                    <li><a href="tile_embeddings.npy" target="_blank">tile_embeddings.npy</a> - Raw embedding vectors</li>
                    <li><a href="tile_metadata.csv" target="_blank">tile_metadata.csv</a> - Tile metadata</li>
                    <li><a href="extraction_summary.json" target="_blank">extraction_summary.json</a> - Extraction summary</li>
                </ul>
            </div>
        </div>

        <div class="section">
            <h2>📊 Visualizations</h2>
            <div class="viz-grid">
                <div class="viz-item">
                    <img src="visualizations/embedding_statistics.png" alt="Embedding Statistics" loading="lazy">
                    <h3>Embedding Statistics</h3>
                </div>
                <div class="viz-item">
                    <img src="visualizations/dimensionality_reduction.png" alt="Dimensionality Reduction" loading="lazy">
                    <h3>UMAP/t-SNE Projection</h3>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>🔬 Extraction Details</h2>
            <ul>
                <li><strong>Model:</strong> {summary['model_name']}</li>
                <li><strong>Device:</strong> {summary['device']}</li>
                <li><strong>Extraction Date:</strong> {summary['extraction_date']}</li>
                <li><strong>Embedding Range:</strong> [{summary['embedding_stats']['min']:.4f}, {summary['embedding_stats']['max']:.4f}]</li>
                <li><strong>Embedding Median:</strong> {summary['embedding_stats']['median']:.4f}</li>
            </ul>
        </div>

        <div style="text-align: center; margin-top: 30px; color: #6c757d; font-style: italic;">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><em>This report contains embeddings extracted from histopathology tiles using DAE-KAN</em></p>
        </div>
    </div>
</body>
</html>
        """

        html_path = os.path.join(output_dir, 'embedding_analysis_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ HTML report saved to {html_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract DAE-KAN embeddings from histopathology tiles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  python extract_tile_embeddings.py --data-dir data/processed/HeparUnifiedPNG/tiles

  # With trained model
  python extract_tile_embeddings.py \\
    --model-path checkpoints/best_model.pth \\
    --data-dir data/processed/HeparUnifiedPNG/tiles \\
    --batch-size 32

  # With visualizations and sampling
  python extract_tile_embeddings.py \\
    --model-path checkpoints/best_model.pth \\
    --data-dir data/processed/HeparUnifiedPNG/tiles \\
    --create-visualizations \\
    --sample-size 5000 \\
    --output-dir embeddings_analysis
        """
    )

    # Model arguments
    parser.add_argument('--model-path', type=str,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model-name', type=str, default='dae_kan_attention',
                       choices=['dae_kan_attention', 'kan_conv', 'no_kan'],
                       help='Model architecture to use')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for extraction')

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing histopathology tiles')
    parser.add_argument('--output-dir', type=str, default='tile_embeddings',
                       help='Output directory for embeddings and analysis')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--max-tiles', type=int,
                       help='Maximum number of tiles to process')
    parser.add_argument('--sample-size', type=int,
                       help='Number of tiles to sample for visualizations')

    # Processing arguments
    parser.add_argument('--create-visualizations', action='store_true',
                       help='Create UMAP/t-SNE visualizations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("🧬 DAE-KAN Tile Embedding Extractor")
    print("=" * 50)

    # Initialize extractor
    extractor = TileEmbeddingExtractor(model_name=args.model_name, device=args.device)

    # Load model
    if not extractor.load_model(args.model_path):
        print("❌ Failed to load model. Exiting.")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Extract embeddings
    try:
        embeddings, metadata = extractor.extract_embeddings_from_directory(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_tiles=args.max_tiles or args.sample_size,
            create_visualizations=args.create_visualizations
        )

        print(f"\n🎉 Embedding extraction completed successfully!")
        print(f"📁 Results saved to: {args.output_dir}")
        print(f"📊 Total embeddings: {len(embeddings)}")
        print(f"📏 Embedding dimensions: {embeddings.shape[1]}")
        print(f"📄 Open {os.path.join(args.output_dir, 'embedding_analysis_report.html')} for detailed analysis")

    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()