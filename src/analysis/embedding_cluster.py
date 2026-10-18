"""Embedding clustering + bootstrap stability analysis (mixin)."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class EmbeddingClusterMixin:
    """Clustering analysis; see AutomaticAnalyzer."""

    def analyze_embedding_clustering(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                     phase: str, embeddings_dir: str, k: int = 6):
        """
        Perform and visualize clustering on embeddings using various algorithms.

        Enhanced with additional metrics (Reviewer 1 #6):
        - Adjusted Mutual Information (AMI) when labels available
        - Bootstrap stability analysis

        Args:
            embeddings (np.ndarray): The embeddings to cluster.
            metadata (pd.DataFrame): Metadata associated with the embeddings.
            phase (str): The phase of training (e.g., 'train', 'val').
            embeddings_dir (str): Directory to save the analysis plots.
            k (int): The number of clusters.

        Returns:
            dict: Clustering metrics for all algorithms
        """
        clustering_results = {}

        try:
            from sklearn.cluster import BisectingKMeans, KMeans
            from sklearn.metrics import (
                adjusted_mutual_info_score,
                adjusted_rand_score,
                calinski_harabasz_score,
                davies_bouldin_score,
                silhouette_score,
            )
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler

            if not UMAP_AVAILABLE:
                print("⚠ UMAP not available, skipping clustering visualization. Please install umap-learn.")
                return clustering_results

            # Sample if too many embeddings
            max_samples = 2000
            if len(embeddings) > max_samples:
                indices = np.random.choice(len(embeddings), max_samples, replace=False)
                embeddings_sample = embeddings[indices]
                metadata_sample = metadata.iloc[indices].reset_index(drop=True) if metadata is not None else None
            else:
                embeddings_sample = embeddings
                metadata_sample = metadata

            if len(embeddings_sample) < k * 2:
                print(f"⚠ Not enough embeddings ({len(embeddings_sample)}) for clustering with k={k}.")
                return clustering_results

            # Standardize
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_sample)

            # Dimensionality reduction for visualization
            reducer = UMAP(n_neighbors=min(15, len(embeddings_sample)-1), min_dist=0.1, random_state=42)
            embedding_2d = reducer.fit_transform(embeddings_scaled)

            # Clustering models
            clustering_models = {
                'KMeans': KMeans(n_clusters=k, random_state=42, n_init='auto'),
                'Gaussian Mixture': GaussianMixture(n_components=k, random_state=42),
                'Bisecting KMeans': BisectingKMeans(n_clusters=k, random_state=42)
            }

            # Check if ground truth labels are available
            true_labels = None
            if metadata_sample is not None and 'label' in metadata_sample.columns:
                true_labels = metadata_sample['label'].values

            fig, axes = plt.subplots(len(clustering_models), 3, figsize=(20, 5 * len(clustering_models)), squeeze=False)
            fig.suptitle(f'{phase.capitalize()} Embedding Clustering Analysis (k={k})', fontsize=24, fontweight='bold')

            for i, (name, model) in enumerate(clustering_models.items()):
                # Fit model and get labels
                cluster_labels = model.fit_predict(embeddings_scaled)

                # Calculate scores
                if len(np.unique(cluster_labels)) > 1:
                    sil_score = silhouette_score(embeddings_scaled, cluster_labels)
                    db_score = davies_bouldin_score(embeddings_scaled, cluster_labels)
                    ch_score = calinski_harabasz_score(embeddings_scaled, cluster_labels)

                    # Bootstrap stability analysis (Reviewer 1 #6)
                    bootstrap_scores = self._compute_bootstrap_stability(
                        embeddings_scaled, model, k, n_resamples=10, resample_ratio=0.8
                    )
                    bootstrap_std = np.std(bootstrap_scores)

                    # AMI and ARI if labels available (Reviewer 1 #6)
                    ami_score = None
                    ari_score = None
                    if true_labels is not None:
                        ami_score = adjusted_mutual_info_score(true_labels, cluster_labels)
                        ari_score = adjusted_rand_score(true_labels, cluster_labels)

                    # Store results
                    clustering_results[name] = {
                        'silhouette': sil_score,
                        'davies_bouldin': db_score,
                        'calinski_harabasz': ch_score,
                        'bootstrap_std': bootstrap_std,
                        'ami': ami_score,
                        'ari': ari_score,
                        'n_samples': len(embeddings_sample)
                    }
                else:
                    sil_score = -1
                    db_score = -1
                    ch_score = -1
                    bootstrap_std = None
                    ami_score = None
                    ari_score = None

                # 1. Scatter plot of clusters
                ax = axes[i, 0]
                scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=cluster_labels, cmap='viridis', s=25, alpha=0.7)
                ax.set_title(f'{name}: UMAP Projection', fontsize=22, fontweight='bold')
                ax.set_xlabel('UMAP 1', fontsize=20)
                ax.set_ylabel('UMAP 2', fontsize=20)
                ax.grid(True, alpha=0.3)
                if len(np.unique(cluster_labels)) > 1:
                    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters", fontsize=18)
                    ax.add_artist(legend1)

                # 2. Cluster distribution
                ax = axes[i, 1]
                ax.hist(cluster_labels, bins=np.arange(k + 1) - 0.5, rwidth=0.8, alpha=0.7)
                ax.set_title(f'{name}: Cluster Distribution', fontsize=18, fontweight='bold')
                ax.set_xlabel('Cluster ID', fontsize=16)
                ax.set_ylabel('Count', fontsize=16)
                ax.set_xticks(range(k))
                ax.grid(True, alpha=0.3)

                # 3. Clustering scores (enhanced with bootstrap stability)
                ax = axes[i, 2]
                scores = {
                    'Silhouette': sil_score,
                    'Davies-Bouldin': db_score,
                    'Calinski-Harabasz': ch_score
                }
                if ami_score is not None:
                    scores['AMI'] = ami_score
                if bootstrap_std is not None:
                    scores['Bootstrap\nStability'] = 1.0 - bootstrap_std  # Higher is better

                bars = ax.bar(scores.keys(), scores.values(), color=['coral', 'skyblue', 'lightgreen', 'gold', 'purple'][:len(scores)], alpha=0.7)
                ax.set_title(f'{name}: Clustering Scores', fontsize=18, fontweight='bold')
                ax.set_ylabel('Score', fontsize=16)
                ax.grid(True, axis='y', alpha=0.3)
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom' if yval >= 0 else 'top', fontsize=12)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            viz_path = os.path.join(embeddings_dir, f'{phase}_clustering_analysis.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✓ Clustering analysis saved to {viz_path}")
            if clustering_results:
                print(f"✓ Clustering metrics computed for {len(clustering_results)} algorithms")

        except ImportError as e:
            print(f"⚠ Clustering analysis failed due to missing package: {e}. Please install umap-learn, scikit-learn.")
        except Exception as e:
            print(f"⚠ Failed to create clustering analysis: {e}")

        return clustering_results
    def _compute_bootstrap_stability(self, embeddings: np.ndarray, clustering_model,
                                     k: int, n_resamples: int = 10, resample_ratio: float = 0.8) -> list[float]:
        """
        Compute clustering stability via bootstrapping (Reviewer 1 #6).

        This function resamples the data multiple times, refits the clustering model,
        and computes the Davies-Bouldin Index for each resample. Lower variance
        indicates more stable clustering.

        Args:
            embeddings (np.ndarray): Embedding matrix (n_samples, n_features).
            clustering_model: Clustering model instance (e.g., KMeans, GMM).
            k (int): Number of clusters.
            n_resamples (int): Number of bootstrap resamples.
            resample_ratio (float): Fraction of data to use in each resample.

        Returns:
            List[float]: Davies-Bouldin Index for each bootstrap resample.

        Example:
            >>> kmeans = KMeans(n_clusters=5, random_state=42)
            >>> bootstrap_scores = analyzer._compute_bootstrap_stability(embeddings, kmeans, k=5)
            >>> print(f"Bootstrap std: {np.std(bootstrap_scores):.4f}")
        """
        from sklearn.metrics import davies_bouldin_score

        n_samples = len(embeddings)
        resample_size = int(n_samples * resample_ratio)
        bootstrap_scores = []

        for i in range(n_resamples):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=resample_size, replace=True)
            embeddings_resampled = embeddings[indices]

            # Fit clustering model
            try:
                # Clone the model to reset state
                from sklearn.base import clone
                model_clone = clone(clustering_model)
                model_clone.fit(embeddings_resampled)

                # Get cluster labels
                if hasattr(model_clone, 'predict'):
                    labels = model_clone.predict(embeddings_resampled)
                else:
                    labels = model_clone.labels_

                # Compute Davies-Bouldin Index
                if len(np.unique(labels)) > 1:
                    db_score = davies_bouldin_score(embeddings_resampled, labels)
                    bootstrap_scores.append(db_score)

            except Exception as e:
                print(f"⚠ Bootstrap resample {i+1}/{n_resamples} failed: {e}")
                continue

        return bootstrap_scores if bootstrap_scores else [float('nan')]
