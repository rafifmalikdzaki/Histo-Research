"""Embedding dump/collect helpers (mixin for the Lightning module)."""

import os
import time

import numpy as np
import torch


class EmbeddingIOMixin:
    """Embedding dump/collect helpers; see the Lightning module."""

    def _store_embeddings(self, batch_idx: int, input_tensor: torch.Tensor,
                          z_embeddings: torch.Tensor, phase: str):
        """Store embeddings and metadata for later analysis"""

        # Convert to numpy and move to CPU
        z_np = z_embeddings.detach().cpu().numpy()
        input_np = input_tensor.detach().cpu().numpy()

        # Store each sample in the batch
        for i in range(z_np.shape[0]):
            # Flatten embedding for easier analysis
            embedding_flat = z_np[i].flatten()

            # Create metadata
            metadata = {
                'batch_idx': batch_idx,
                'sample_idx': i,
                'phase': phase,
                'embedding_shape': z_np[i].shape,
                'timestamp': time.time(),
                'input_shape': input_np[i].shape,
                'input_mean': float(input_np[i].mean()),
                'input_std': float(input_np[i].std())
            }

            # Add to storage (with memory limit)
            if len(self.embeddings_storage[phase]['embeddings']) < self.max_embeddings_to_store:
                self.embeddings_storage[phase]['embeddings'].append(embedding_flat)
                self.embeddings_storage[phase]['metadata'].append(metadata)
                self.embeddings_storage[phase]['batch_indices'].append(batch_idx)
            elif len(self.embeddings_storage[phase]['embeddings']) % 100 == 0:
                # Periodically save embeddings to disk to free memory
                self._save_embeddings_to_disk(phase)
    def _save_embeddings_to_disk(self, phase: str):
        """Save embeddings to disk to free memory"""
        if not self.embeddings_storage[phase]['embeddings']:
            return

        try:
            import pandas as pd

            # Create embeddings directory
            if hasattr(self, 'auto_analyzer') and self.auto_analyzer:
                embeddings_dir = os.path.join(self.auto_analyzer.save_dir, 'embeddings')
            else:
                embeddings_dir = 'embeddings'

            os.makedirs(embeddings_dir, exist_ok=True)

            # Convert to arrays
            embeddings_array = np.array(self.embeddings_storage[phase]['embeddings'])
            metadata_list = self.embeddings_storage[phase]['metadata']

            # Save embeddings
            embeddings_file = os.path.join(embeddings_dir, f'{phase}_embeddings.npy')
            np.save(embeddings_file, embeddings_array)

            # Save metadata
            metadata_df = pd.DataFrame(metadata_list)
            metadata_file = os.path.join(embeddings_dir, f'{phase}_metadata.csv')
            metadata_df.to_csv(metadata_file, index=False)

            print(f"✓ Saved {len(embeddings_array)} {phase} embeddings to {embeddings_dir}")

            # Clear storage to free memory
            self.embeddings_storage[phase] = {'embeddings': [], 'metadata': [], 'batch_indices': []}

        except Exception as e:
            print(f"⚠ Failed to save {phase} embeddings to disk: {e}")
    def _collect_epoch_embeddings(self, batch_idx: int, input_tensor: torch.Tensor,
                                 z_embeddings: torch.Tensor, phase: str):
        """Collect embeddings during epoch for end-of-epoch analysis"""

        # Convert to numpy and move to CPU
        z_np = z_embeddings.detach().cpu().numpy()
        input_np = input_tensor.detach().cpu().numpy()

        # Calculate current epoch
        current_epoch = self._training_epoch

        # Sample embeddings to avoid collecting too many
        max_per_batch = max(1, self.embeddings_per_epoch // 100)  # Rough estimate of batches per epoch

        for i in range(min(z_np.shape[0], max_per_batch)):
            # Flatten embedding for easier analysis
            embedding_flat = z_np[i].flatten()

            # Create epoch-specific metadata
            metadata = {
                'epoch': current_epoch,
                'batch_idx': batch_idx,
                'sample_idx': i,
                'phase': phase,
                'embedding_shape': z_np[i].shape,
                'timestamp': time.time(),
                'input_shape': input_np[i].shape,
                'input_mean': float(input_np[i].mean()),
                'input_std': float(input_np[i].std())
            }

            # Add to epoch collection
            if len(self.epoch_embeddings_collection[phase]['embeddings']) < self.embeddings_per_epoch:
                self.epoch_embeddings_collection[phase]['embeddings'].append(embedding_flat)
                self.epoch_embeddings_collection[phase]['metadata'].append(metadata)
                self.epoch_embeddings_collection[phase]['epoch_indices'].append(current_epoch)
    def _save_epoch_embeddings(self, phase: str):
        """Save embeddings collected during current epoch"""

        if not self.epoch_embeddings_collection[phase]['embeddings']:
            return None, None

        try:
            import pandas as pd

            # Create embeddings directory
            if hasattr(self, 'auto_analyzer') and self.auto_analyzer:
                embeddings_dir = os.path.join(self.auto_analyzer.save_dir, 'embeddings')
            else:
                embeddings_dir = 'embeddings'

            os.makedirs(embeddings_dir, exist_ok=True)

            # Convert to arrays
            embeddings_array = np.array(self.epoch_embeddings_collection[phase]['embeddings'])
            metadata_list = self.epoch_embeddings_collection[phase]['metadata']

            # Save epoch-specific embeddings
            epoch_num = self._training_epoch - 1
            embeddings_file = os.path.join(embeddings_dir, f'{phase}_embeddings_epoch_{epoch_num:03d}.npy')
            np.save(embeddings_file, embeddings_array)

            # Save metadata
            metadata_df = pd.DataFrame(metadata_list)
            metadata_file = os.path.join(embeddings_dir, f'{phase}_metadata_epoch_{epoch_num:03d}.csv')
            metadata_df.to_csv(metadata_file, index=False)

            # Also append to overall storage for analysis
            if len(self.embeddings_storage[phase]['embeddings']) < self.max_embeddings_to_store:
                self.embeddings_storage[phase]['embeddings'].extend(embeddings_array.tolist())
                self.embeddings_storage[phase]['metadata'].extend(metadata_list)
                self.embeddings_storage[phase]['batch_indices'].extend([epoch_num] * len(embeddings_array))

            return embeddings_file, metadata_file

        except Exception as e:
            print(f"⚠ Failed to save {phase} embeddings for epoch {epoch_num}: {e}")
            return None, None
