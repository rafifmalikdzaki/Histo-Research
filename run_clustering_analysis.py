#!/usr/bin/env python3
"""
Clustering Analysis Script

Loads embeddings from experiment directories, runs clustering algorithms (Bisecting K-Means, K-Means, GMM),
and computes clustering validation metrics.

Usage:
    python run_clustering_analysis.py --base-dir auto_analysis --k 6 --output results/raw_clustering_metrics.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, BisectingKMeans, GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


def find_experiment_dirs(base_dir: str, model_filter: Optional[str] = None) -> List[Path]:
    """Find all experiment directories."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return []
    
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if model_filter:
        exp_dirs = [d for d in exp_dirs if model_filter in d.name.lower()]
    
    return sorted(exp_dirs)


def extract_model_name(dir_name: str) -> str:
    """Extract model name from directory name."""
    dir_lower = dir_name.lower()
    
    if 'dae_kan_attention' in dir_lower or 'daekan' in dir_lower:
        return 'dae_kan_attention'
    elif 'bam_only' in dir_lower:
        return 'bam_only'
    elif 'kan_only' in dir_lower:
        return 'kan_only'
    elif 'no_bam' in dir_lower:
        return 'no_bam'
    elif 'no_eka' in dir_lower:
        return 'no_eka'
    elif 'no_kan' in dir_lower:
        return 'no_kan'
    elif 'baseline' in dir_lower:
        return 'baseline'
    else:
        return 'unknown'


def extract_seed(dir_name: str) -> Optional[int]:
    """Extract seed from directory name."""
    import re
    match = re.search(r'seed(\d+)', dir_name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def load_embeddings(exp_dir: Path) -> Optional[np.ndarray]:
    """Load embeddings from experiment directory."""
    embeddings_dir = exp_dir / 'embeddings'
    
    if not embeddings_dir.exists():
        print(f"⚠️  No embeddings directory in {exp_dir.name}")
        return None
    
    # Look for embedding files
    npy_files = list(embeddings_dir.glob('*.npy'))
    
    if not npy_files:
        # Try epoch subdirectories
        epoch_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir() and d.name.startswith('epoch_')]
        if epoch_dirs:
            epoch_dirs.sort()
            npy_files = list(epoch_dirs[-1].glob('*.npy'))
    
    if not npy_files:
        print(f"⚠️  No .npy files found in {exp_dir.name}")
        return None
    
    # Load the first embedding file found
    try:
        embeddings = np.load(npy_files[0])
        print(f"✓ Loaded embeddings from {exp_dir.name}: shape={embeddings.shape}")
        return embeddings
    except Exception as e:
        print(f"⚠️  Error loading embeddings from {exp_dir.name}: {e}")
        return None


def run_clustering(embeddings: np.ndarray, method: str, k: int = 6) -> np.ndarray:
    """Run clustering algorithm on embeddings."""
    if method == 'bisecting_kmeans':
        model = BisectingKMeans(n_clusters=k, random_state=42, n_init=10)
    elif method == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    elif method == 'gmm':
        model = GaussianMixture(n_components=k, random_state=42, n_init=5)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    labels = model.fit_predict(embeddings)
    return labels


def compute_clustering_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute clustering validation metrics."""
    metrics = {}
    
    try:
        metrics['silhouette'] = silhouette_score(embeddings, labels)
    except Exception as e:
        metrics['silhouette'] = np.nan
    
    try:
        metrics['davies_bouldin'] = davies_bouldin_score(embeddings, labels)
    except Exception as e:
        metrics['davies_bouldin'] = np.nan
    
    try:
        metrics['calinski_harabasz'] = calinski_harabasz_score(embeddings, labels)
    except Exception as e:
        metrics['calinski_harabasz'] = np.nan
    
    try:
        metrics['xie_beni'] = compute_xie_beni(embeddings, labels)
    except Exception as e:
        metrics['xie_beni'] = np.nan
    
    return metrics


def compute_xie_beni(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Compute Xie-Beni clustering validity index."""
    k = len(np.unique(labels))
    n = len(embeddings)
    
    centers = np.array([embeddings[labels == i].mean(axis=0) for i in range(k)])
    
    compactness = 0
    for i in range(k):
        cluster_points = embeddings[labels == i]
        compactness += np.sum(np.linalg.norm(cluster_points - centers[i], axis=1) ** 2)
    compactness /= n
    
    separation = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            dist = np.linalg.norm(centers[i] - centers[j]) ** 2
            separation = min(separation, dist)
    
    if separation == 0:
        return float('inf')
    
    return compactness / separation


def process_experiment(exp_dir: Path, k: int, methods: List[str]) -> List[Dict]:
    """Process a single experiment directory."""
    model_name = extract_model_name(exp_dir.name)
    seed = extract_seed(exp_dir.name)
    
    if model_name == 'unknown':
        print(f"⚠️  Skipping unknown model: {exp_dir.name}")
        return []
    
    embeddings = load_embeddings(exp_dir)
    if embeddings is None:
        return []
    
    results = []
    for method in methods:
        print(f"  Running {method} clustering (k={k})...")
        
        try:
            labels = run_clustering(embeddings, method, k)
            metrics = compute_clustering_metrics(embeddings, labels)
            
            result = {
                'model': model_name,
                'seed': seed if seed is not None else -1,
                'experiment_dir': str(exp_dir),
                'clustering_method': method,
                'k': k,
                'n_samples': len(embeddings),
                'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 1,
            }
            result.update(metrics)
            results.append(result)
            
            print(f"    ✓ Silhouette: {metrics.get('silhouette', 'N/A'):.4f}")
            
        except Exception as e:
            print(f"  ❌ Error in {method}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run clustering analysis on experiment embeddings')
    
    parser.add_argument('--base-dir', type=str, default='auto_analysis',
                        help='Base directory containing experiment directories')
    parser.add_argument('--model-filter', type=str, default=None,
                        help='Filter experiments by model name (optional)')
    parser.add_argument('--k', type=int, default=6,
                        help='Number of clusters (default: 6)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['bisecting_kmeans', 'kmeans', 'gmm'],
                        help='Clustering methods to use')
    parser.add_argument('--output', type=str, default='results/raw_clustering_metrics.csv',
                        help='Output CSV file path')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 CLUSTERING ANALYSIS")
    print("="*80)
    print(f"Base directory: {args.base_dir}")
    print(f"Number of clusters (k): {args.k}")
    print(f"Clustering methods: {', '.join(args.methods)}")
    print("="*80)
    
    exp_dirs = find_experiment_dirs(args.base_dir, args.model_filter)
    
    if not exp_dirs:
        print("\n❌ No experiment directories found!")
        return 1
    
    print(f"\n📁 Found {len(exp_dirs)} experiment directories")
    
    all_results = []
    for i, exp_dir in enumerate(exp_dirs, 1):
        print(f"\n[{i}/{len(exp_dirs)}] Processing: {exp_dir.name}")
        results = process_experiment(exp_dir, args.k, args.methods)
        all_results.extend(results)
    
    if not all_results:
        print("\n❌ No results generated!")
        return 1
    
    df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("✅ CLUSTERING ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total experiments processed: {len(exp_dirs)}")
    print(f"Total results: {len(df)}")
    print(f"Results saved to: {output_path}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
