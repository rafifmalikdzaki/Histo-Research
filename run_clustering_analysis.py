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
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture

sys.path.insert(0, str(Path(__file__).parent / "src"))
from clustering_metrics import (  # noqa: E402
    normalize_embeddings, internal_metrics, external_metrics, NORMALIZERS,
)

# Categorical metadata columns used for external validation, and the namespace
# prefix each is reported under. Magnification is included as a CONFOUND check:
# we want clusters to track treatment_group/diet, NOT magnification.
EXTERNAL_LABEL_COLUMNS = {
    "treatment_group": "treatment_",
    "diet": "diet_",
    "magnification": "mag_",
}


def find_experiment_dirs(base_dir: str, model_filter: Optional[str] = None, model_exclude: Optional[List[str]] = None) -> List[Path]:
    """Find all experiment directories."""
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"❌ Base directory not found: {base_dir}")
        return []
    
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if model_filter:
        exp_dirs = [d for d in exp_dirs if model_filter in d.name.lower()]
    
    if model_exclude:
        for exclude_str in model_exclude:
            exp_dirs = [d for d in exp_dirs if exclude_str.lower() not in d.name.lower()]
    
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


def load_embeddings(exp_dir: Path) -> Optional[Tuple[np.ndarray, Optional[pd.DataFrame]]]:
    """Load embeddings (and aligned biology metadata, if present) from an
    experiment directory.

    Prefers a `test_embeddings.npy` (the held-out, leak-free split) over
    `train_embeddings.npy`; falls back to the first .npy otherwise. When a
    sibling `<phase>_metadata.csv` exists with the same row count, it is returned
    so external validation against treatment_group/diet/magnification is possible.
    """
    embeddings_dir = exp_dir / 'embeddings'
    if not embeddings_dir.exists():
        print(f"⚠️  No embeddings directory in {exp_dir.name}")
        return None

    npy_files = list(embeddings_dir.glob('*.npy'))
    if not npy_files:
        epoch_dirs = [d for d in embeddings_dir.iterdir() if d.is_dir() and d.name.startswith('epoch_')]
        if epoch_dirs:
            epoch_dirs.sort()
            npy_files = list(epoch_dirs[-1].glob('*.npy'))
    if not npy_files:
        print(f"⚠️  No .npy files found in {exp_dir.name}")
        return None

    # Deterministic preference: test > val > train > first.
    def _rank(p: Path) -> int:
        n = p.name.lower()
        return 0 if 'test' in n else 1 if 'val' in n else 2 if 'train' in n else 3
    npy_files.sort(key=lambda p: (_rank(p), p.name))
    emb_file = npy_files[0]

    try:
        embeddings = np.load(emb_file)
    except Exception as e:
        print(f"⚠️  Error loading embeddings from {exp_dir.name}: {e}")
        return None

    meta = None
    meta_file = emb_file.with_name(emb_file.name.replace('embeddings', 'metadata').replace('.npy', '.csv'))
    if meta_file.exists():
        try:
            m = pd.read_csv(meta_file)
            if len(m) == len(embeddings):
                meta = m
            else:
                print(f"⚠️  metadata/embeddings length mismatch in {exp_dir.name} "
                      f"({len(m)} vs {len(embeddings)}); skipping external validation")
        except Exception as e:
            print(f"⚠️  Error loading metadata for {exp_dir.name}: {e}")

    has_bio = meta is not None and any(c in meta.columns for c in EXTERNAL_LABEL_COLUMNS)
    print(f"✓ Loaded {emb_file.name} from {exp_dir.name}: shape={embeddings.shape}"
          f"{' (+biology metadata)' if has_bio else ''}")
    return embeddings, meta


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


def process_experiment(exp_dir: Path, k: int, methods: List[str],
                        normalize: str = "standardize_l2") -> List[Dict]:
    """Process a single experiment directory.

    Embeddings are normalized (see clustering_metrics.normalize_embeddings) before
    clustering. Internal validity indices are robust to degenerate clusters; when
    biology metadata is available, external validity (AMI/ARI/NMI/purity) is
    computed against treatment_group, diet, and magnification (confound check).
    """
    model_name = extract_model_name(exp_dir.name)
    seed = extract_seed(exp_dir.name)

    if model_name == 'unknown':
        print(f"⚠️  Skipping unknown model: {exp_dir.name}")
        return []

    loaded = load_embeddings(exp_dir)
    if loaded is None:
        return []
    raw_embeddings, meta = loaded

    embeddings = normalize_embeddings(raw_embeddings, normalize)

    results = []
    for method in methods:
        print(f"  Running {method} clustering (k={k}, normalize={normalize})...")
        try:
            labels = run_clustering(embeddings, method, k)
            metrics = internal_metrics(embeddings, labels)

            result = {
                'model': model_name,
                'seed': seed if seed is not None else -1,
                'experiment_dir': str(exp_dir),
                'clustering_method': method,
                'normalize': normalize,
                'k': k,
                'n_samples': len(embeddings),
                'embedding_dim': embeddings.shape[1] if len(embeddings.shape) > 1 else 1,
            }
            result.update(metrics)

            # External validation against any available biology columns.
            if meta is not None:
                for col, prefix in EXTERNAL_LABEL_COLUMNS.items():
                    if col in meta.columns and meta[col].notna().any():
                        result.update(external_metrics(meta[col].to_numpy(), labels, prefix=prefix))

            results.append(result)
            sil = metrics.get('silhouette', float('nan'))
            print(f"    ✓ Silhouette: {sil:.4f}" if sil == sil else "    ✓ Silhouette: NaN")
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
    parser.add_argument('--model-exclude', type=str, nargs='+', default=None,
                        help='Exclude models containing these strings from retraining (optional)')
    parser.add_argument('--preserve-from', type=str, default=None,
                        help='Preserve results for excluded models from this CSV file (optional)')
    parser.add_argument('--k', type=int, default=6,
                        help='Number of clusters (default: 6)')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['bisecting_kmeans', 'kmeans', 'gmm'],
                        help='Clustering methods to use')
    parser.add_argument('--output', type=str, default='results/raw_clustering_metrics.csv',
                        help='Output CSV file path')
    parser.add_argument('--normalize', type=str, default='standardize_l2',
                        choices=list(NORMALIZERS),
                        help='Embedding normalization before clustering (default: standardize_l2)')

    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🔬 CLUSTERING ANALYSIS")
    print("="*80)
    print(f"Base directory: {args.base_dir}")
    print(f"Number of clusters (k): {args.k}")
    if args.model_exclude:
        print(f"Excluding from retraining: {', '.join(args.model_exclude)}")
    if args.preserve_from:
        print(f"Preserving results from: {args.preserve_from}")
    print(f"Clustering methods: {', '.join(args.methods)}")
    print("="*80)
    
    # Load existing results if preservation is requested
    preserve_df = None
    if args.preserve_from:
        preserve_path = Path(args.preserve_from)
        if preserve_path.exists():
            try:
                preserve_df = pd.read_csv(preserve_path)
                print(f"✓ Loaded {len(preserve_df)} records for preservation")
            except Exception as e:
                print(f"⚠️  Error loading preservation file: {e}")
        else:
            print(f"⚠️  Preservation file not found: {args.preserve_from}")

    # Find ALL experiment dirs first (don't filter yet, we handle exclusion during processing)
    exp_dirs = find_experiment_dirs(args.base_dir, args.model_filter, None)
    
    if not exp_dirs:
        print("\n❌ No experiment directories found!")
        return 1
    
    print(f"\n📁 Found {len(exp_dirs)} experiment directories")
    
    all_results = []
    for i, exp_dir in enumerate(exp_dirs, 1):
        model_name = extract_model_name(exp_dir.name)
        
        # Check if we should exclude this model from retraining
        should_exclude = False
        if args.model_exclude:
            for exclude_str in args.model_exclude:
                if exclude_str.lower() in model_name.lower():
                    should_exclude = True
                    break
        
        if should_exclude:
            if preserve_df is not None:
                # Try to find existing results for this experiment dir
                matches = preserve_df[preserve_df['experiment_dir'].str.contains(exp_dir.name)]
                
                if not matches.empty:
                    print(f"\n[{i}/{len(exp_dirs)}] Preserving existing results for: {exp_dir.name}")
                    all_results.extend(matches.to_dict('records'))
                    continue
                else:
                    print(f"\n[{i}/{len(exp_dirs)}] No existing results found for: {exp_dir.name} - Skipping")
                    continue
            else:
                print(f"\n[{i}/{len(exp_dirs)}] Skipping excluded model: {exp_dir.name}")
                continue

        print(f"\n[{i}/{len(exp_dirs)}] Processing: {exp_dir.name}")
        results = process_experiment(exp_dir, args.k, args.methods, args.normalize)
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
