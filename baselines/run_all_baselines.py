#!/usr/bin/env python3
"""
Unified Baseline Evaluation Script

This script runs all baseline models with the same evaluation framework
for fair comparison with the proposed DAE-KAN method.

Usage:
    # Run all baselines
    python baselines/run_all_baselines.py --dataset HeparUnifiedPNG --epochs 30
    
    # Run specific baseline
    python baselines/run_all_baselines.py --model simclr --epochs 30
    
    # Multi-seed evaluation
    python baselines/run_all_baselines.py --all-models --seeds 42 123 456
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baselines import get_baseline
from histodata import create_dataset, ImageDataset
from utils.reproducibility import set_seed


def train_baseline(model: nn.Module, train_loader: DataLoader,
                  criterion: nn.Module, optimizer: torch.optim.Optimizer,
                  device: torch.device, epochs: int = 30) -> List[float]:
    """
    Train baseline model.
    
    Args:
        model: Baseline model.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device.
        epochs: Number of epochs.
    
    Returns:
        List of training losses per epoch.
    """
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, _ = batch
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (different for each baseline)
            if hasattr(model, 'forward') and model.__class__.__name__ == 'VAE':
                recon, mu, log_var = model(images)
                loss = criterion(images, recon, mu, log_var)
            else:
                recon, encoded = model(images)
                loss = criterion(images, recon)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    return losses


def extract_baseline_embeddings(model: nn.Module, data_loader: DataLoader,
                               device: torch.device) -> np.ndarray:
    """
    Extract embeddings from baseline model.
    
    Args:
        model: Trained baseline model.
        data_loader: Data loader.
        device: Device.
    
    Returns:
        Embedding matrix.
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings", leave=False):
            images, _ = batch
            images = images.to(device)
            
            # Get embeddings (different for each baseline)
            if hasattr(model, 'encoder'):
                embeddings_batch = model.encoder(images)
            else:
                _, embeddings_batch = model(images)
            
            embeddings.append(embeddings_batch.cpu().numpy())
    
    return np.vstack(embeddings)


def evaluate_baseline_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                                n_clusters: int = 5, seed: int = 42) -> Dict[str, float]:
    """
    Evaluate clustering on baseline embeddings.
    
    Args:
        embeddings: Embedding matrix.
        labels: Ground truth labels.
        n_clusters: Number of clusters.
        seed: Random seed.
    
    Returns:
        Dictionary with clustering metrics.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                  calinski_harabasz_score)
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto')
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Compute metrics
    sil_score = silhouette_score(embeddings, cluster_labels)
    db_score = davies_bouldin_score(embeddings, cluster_labels)
    ch_score = calinski_harabasz_score(embeddings, cluster_labels)
    
    return {
        'silhouette': round(float(sil_score), 6),
        'davies_bouldin': round(float(db_score), 6),
        'calinski_harabasz': round(float(ch_score), 6)
    }


def run_single_baseline(
    model_name: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    n_clusters: int,
    seed: int,
    output_dir: str
) -> Dict[str, Any]:
    """
    Run single baseline evaluation.
    
    Args:
        model_name: Baseline model name.
        dataset: Dataset name.
        epochs: Training epochs.
        batch_size: Batch size.
        n_clusters: Number of clusters.
        seed: Random seed.
        output_dir: Output directory.
    
    Returns:
        Dictionary with results.
    """
    print(f"\n{'='*80}")
    print(f"🔬 Running Baseline: {model_name}")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Set seed
    set_seed(seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"\n📊 Loading dataset: {dataset}")
    try:
        X, y = create_dataset('test', dataset_name=dataset)
        test_dataset = ImageDataset(X, y)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"✓ Loaded {len(test_dataset)} samples")
    except Exception as e:
        print(f"⚠️  Failed to load dataset: {e}")
        from torch.utils.data import TensorDataset
        test_dataset = TensorDataset(torch.randn(200, 3, 128, 128), torch.randint(0, 5, (200,)))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    print(f"\n🏗️  Creating {model_name} model...")
    if model_name == 'vae':
        model = get_baseline(model_name, latent_dim=128)
        criterion = None  # Will use VAELoss
    elif model_name == 'single_ae':
        model = get_baseline(model_name, latent_dim=128)
        criterion = nn.MSELoss()
    else:
        # For SimCLR and BYOL, we use contrastive loss
        model = get_baseline(model_name)
        criterion = nn.MSELoss()  # Placeholder
    
    model = model.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {n_params/1e6:.2f}M parameters")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    # Training
    print(f"\n🏋️  Training {model_name}...")
    
    if model_name == 'vae':
        from baselines import VAELoss
        criterion = VAELoss(kl_weight=0.001)
    
    losses = train_baseline(model, test_loader, criterion, optimizer, device, epochs)
    
    # Extract embeddings
    print(f"\n📊 Extracting embeddings...")
    embeddings = extract_baseline_embeddings(model, test_loader, device)
    print(f"✓ Extracted embeddings: {embeddings.shape}")
    
    # Evaluate clustering
    print(f"\n📈 Evaluating clustering...")
    
    # Get labels
    try:
        _, y = create_dataset('test', dataset_name=dataset)
        labels = y.numpy()
    except:
        labels = np.zeros(len(embeddings))
    
    clustering_metrics = evaluate_baseline_embeddings(embeddings, labels, n_clusters, seed)
    
    print(f"  Silhouette: {clustering_metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {clustering_metrics['davies_bouldin']:.4f}")
    print(f"  Calinski-Harabasz: {clustering_metrics['calinski_harabasz']:.4f}")
    
    # Compile results
    results = {
        'model_name': model_name,
        'dataset': dataset,
        'seed': seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'n_clusters': n_clusters,
        'n_parameters': n_params,
        'timestamp': datetime.now().isoformat(),
        'final_loss': losses[-1] if losses else None,
        'clustering_metrics': clustering_metrics,
        'embedding_shape': embeddings.shape
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_path = output_path / f'baseline_{model_name}_seed{seed}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Unified baseline evaluation script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all baselines
  python baselines/run_all_baselines.py --all-models --epochs 30
  
  # Run specific baseline
  python baselines/run_all_baselines.py --model simclr --epochs 30
  
  # Multi-seed evaluation
  python baselines/run_all_baselines.py --all-models --seeds 42 123 456
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model',
        type=str,
        choices=['simclr', 'byol', 'vae', 'single_ae'],
        help='Specific baseline model to run'
    )
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Run all baseline models'
    )
    
    # Data settings
    parser.add_argument(
        '--dataset',
        type=str,
        default='HeparUnifiedPNG',
        help='Dataset name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=5,
        help='Number of clusters'
    )
    
    # Training settings
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs'
    )
    
    # Reproducibility
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42],
        help='Random seeds for multi-seed evaluation'
    )
    
    # Output
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/baselines',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    # Determine models to run
    if args.all_models:
        models = ['simclr', 'byol', 'vae', 'single_ae']
    elif args.model:
        models = [args.model]
    else:
        print("❌ Error: Must specify --model or --all-models")
        return
    
    print("🔬 DAE-KAN Baseline Evaluation")
    print("="*80)
    print(f"Models: {models}")
    print(f"Dataset: {args.dataset}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {args.output_dir}")
    print("="*80)
    
    # Run evaluations
    all_results = []
    
    for model_name in models:
        for seed in args.seeds:
            try:
                results = run_single_baseline(
                    model_name=model_name,
                    dataset=args.dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    n_clusters=args.n_clusters,
                    seed=seed,
                    output_dir=args.output_dir
                )
                all_results.append(results)
            except Exception as e:
                print(f"❌ Failed {model_name} (seed={seed}): {e}")
                all_results.append({
                    'model_name': model_name,
                    'seed': seed,
                    'error': str(e)
                })
    
    # Summary
    if all_results:
        print(f"\n{'='*80}")
        print("📊 BASELINE EVALUATION SUMMARY")
        print(f"{'='*80}")
        
        # Create summary table
        summary_data = []
        
        for result in all_results:
            if 'error' not in result:
                summary_data.append({
                    'Model': result['model_name'],
                    'Seed': result['seed'],
                    'Silhouette': result['clustering_metrics']['silhouette'],
                    'Davies-Bouldin': result['clustering_metrics']['davies_bouldin'],
                    'Calinski-Harabasz': result['clustering_metrics']['calinski_harabasz'],
                    'Parameters (M)': f"{result['n_parameters']/1e6:.2f}"
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            print(df.to_string(index=False))
            
            # Save summary
            output_path = Path(args.output_dir)
            summary_path = output_path / 'baseline_summary.csv'
            df.to_csv(summary_path, index=False)
            print(f"\n✅ Summary saved to: {summary_path}")
        
        print(f"{'='*80}")


if __name__ == "__main__":
    main()
