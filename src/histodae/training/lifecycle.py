"""Epoch/end-of-training lifecycle hooks (mixin for the Lightning module).

Batch/epoch/end callbacks plus the epoch embedding-analysis drivers. Runs on
``self`` state owned by the Lightning module and the sibling mixins — see the
method bodies for the attributes consumed.
"""

import os

import numpy as np
import pandas as pd
import torch


class LifecycleMixin:
    """Training lifecycle hooks; see the Lightning module."""

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Optimize memory after each batch"""
        # Periodic memory cleanup (less frequent than before)
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    def _finalize_embedding_analysis(self):
        """Save remaining embeddings and create comprehensive embedding analysis"""

        # Save any remaining embeddings (cheap; always do unless fully off).
        if self.epoch_viz != "off":
            for phase in ['train', 'val']:
                self._save_embeddings_to_disk(phase)

        # Rendering (t-SNE/UMAP, evolution plots) only in 'render' mode. In 'dump'
        # mode the saved .npy + metadata are sufficient to regenerate everything
        # offline with regenerate_epoch_viz.py.
        if self.epoch_viz == "render":
            self._create_embedding_analysis()
            self._create_epoch_evolution_analysis()

    def _create_embedding_analysis(self):
        """Create comprehensive embedding analysis with visualizations"""

        if not hasattr(self, 'auto_analyzer') or not self.auto_analyzer:
            return

        try:
            # Create embeddings directory
            embeddings_dir = os.path.join(self.auto_analyzer.save_dir, 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)

            # Load embeddings if they exist
            train_embeddings_file = os.path.join(embeddings_dir, 'train_embeddings.npy')
            val_embeddings_file = os.path.join(embeddings_dir, 'val_embeddings.npy')

            if os.path.exists(train_embeddings_file):
                train_embeddings = np.load(train_embeddings_file)
                train_metadata = pd.read_csv(os.path.join(embeddings_dir, 'train_metadata.csv'))

                # Create comprehensive analysis
                self._create_embedding_visualizations(train_embeddings, train_metadata, 'train', embeddings_dir)

            if os.path.exists(val_embeddings_file):
                val_embeddings = np.load(val_embeddings_file)
                val_metadata = pd.read_csv(os.path.join(embeddings_dir, 'val_metadata.csv'))

                # Create comprehensive analysis
                self._create_embedding_visualizations(val_embeddings, val_metadata, 'val', embeddings_dir)

        except Exception as e:
            print(f"⚠ Failed to create embedding analysis: {e}")

    def on_train_epoch_end(self):
        """Called at the end of each training epoch - save collected embeddings and run analysis."""

        self._training_epoch += 1  # Increment epoch counter

        if self.epoch_viz == "off":
            for phase in ['train', 'val']:
                self.epoch_embeddings_collection[phase] = {
                    'embeddings': [], 'metadata': [], 'epoch_indices': []
                }
            return

        # Save embeddings collected during this epoch (cheap: just .npy + csv).
        for phase in ['train', 'val']:
            if self.epoch_embeddings_collection[phase]['embeddings']:
                embeddings_file, metadata_file = self._save_epoch_embeddings(phase)
                print(f"✅ Epoch {self._training_epoch-1}: Saved {len(self.epoch_embeddings_collection[phase]['embeddings'])} {phase} embeddings")

                # Render per-epoch visualisations only in 'render' mode; in 'dump'
                # mode the .npy + metadata above are enough to rebuild them offline
                # via regenerate_epoch_viz.py (skips the expensive t-SNE/UMAP/PNG work).
                if self.epoch_viz == "render" and embeddings_file and metadata_file:
                    self._run_epoch_embedding_analysis(phase, embeddings_file, metadata_file)

        # Clear epoch collection for next epoch
        for phase in ['train', 'val']:
            self.epoch_embeddings_collection[phase] = {
                'embeddings': [], 'metadata': [], 'epoch_indices': []
            }

    def _run_epoch_embedding_analysis(self, phase, embeddings_file, metadata_file):
        """Run analysis on embeddings from a single epoch."""
        if not self.auto_analyzer:
            return

        try:
            embeddings = np.load(embeddings_file)
            metadata = pd.read_csv(metadata_file)

            epoch_num = self._training_epoch - 1
            epoch_analysis_dir = os.path.join(self.auto_analyzer.get_subdir_path('embeddings'), f'epoch_{epoch_num:03d}')
            os.makedirs(epoch_analysis_dir, exist_ok=True)

            # Run the same visualizations as at the end of training
            self._create_embedding_visualizations(embeddings, metadata, f"{phase}_epoch_{epoch_num}", epoch_analysis_dir)
            print(f"✓ Epoch {epoch_num}: Embedding analysis complete for {phase} phase.")

        except Exception as e:
            print(f"⚠ Failed to run epoch embedding analysis for {phase} phase, epoch {epoch_num}: {e}")

    def on_train_end(self):
        # Final cleanup and summary
        if self.auto_analyzer:
            try:
                # Save final comprehensive metrics
                final_metrics = self.auto_analyzer.get_batch_summary()

                # Create final summary report
                self.create_final_summary_report(final_metrics)

                # Save all data
                self.auto_analyzer.save_metrics("final_training_metrics.json")

                # Save remaining embeddings and create analysis
                self._finalize_embedding_analysis()

                print("✓ Final analysis completed")
                print(f"✓ Embedding analysis saved to {self.auto_analyzer.save_dir}/embeddings/")

            except Exception as e:
                print(f"⚠ Final analysis failed: {e}")
            finally:
                # Cleanup
                self.auto_analyzer.cleanup()

