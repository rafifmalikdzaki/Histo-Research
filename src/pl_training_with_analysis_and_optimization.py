import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)
from models.factory import get_model
from histodata import *
from torch.utils.data import Dataset, DataLoader
import wandb
import gc
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import psutil
import json
import warnings
warnings.filterwarnings('ignore')

# Automatic analysis module
from analysis.auto_analysis import AutomaticAnalyzer


class OptimizedMainModelWithAnalysis(pl.LightningModule):
    """Optimized Lightning module with better memory management and GPU utilization"""

    def __init__(self, model_name: str = "dae_kan_attention", batch_size=8, analysis_frequency=100,
                 attention_components_frequency=400, max_embeddings_to_store=1000, embeddings_per_epoch=500):
        super(OptimizedMainModelWithAnalysis, self).__init__()
        self.model = get_model(model_name)()
        self.batch_size = batch_size
        self.analysis_frequency = analysis_frequency  # Run full analysis every N batches
        self.attention_components_frequency = attention_components_frequency  # Save detailed components every N batches
        self.max_embeddings_to_store = max_embeddings_to_store  # Limit embedding storage
        self.embeddings_per_epoch = embeddings_per_epoch  # Number of embeddings to collect per epoch

        # Initialize automatic analyzer
        self.auto_analyzer = None

        # Store basic metrics
        self.batch_losses = []
        self.reconstruction_metrics = []

        # Initialize embedding storage
        self.embeddings_storage = {
            'train': {'embeddings': [], 'metadata': [], 'batch_indices': []},
            'val': {'embeddings': [], 'metadata': [], 'batch_indices': []}
        }
        self.max_embeddings_to_store = 1000  # Limit memory usage

        # Epoch-based embedding collection
        self.epoch_embeddings_collection = {
            'train': {'embeddings': [], 'metadata': [], 'epoch_indices': []},
            'val': {'embeddings': [], 'metadata': [], 'epoch_indices': []}
        }
        self._training_epoch = 0

        # Training timing and performance tracking
        self.training_times = []
        self.inference_times = []
        self.memory_usage = []

    def on_fit_start(self):
        """Initialize automatic analyzer after model is on correct device"""
        device = self.device

        # Create analysis directory and run ID
        base_analysis_dir = "auto_analysis"
        run_id = self.logger.experiment.name if hasattr(self, 'logger') and self.logger.experiment else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Prepare experiment configuration for ablation study tracking
        experiment_config = {
            'model_name': self.model_name if hasattr(self, 'model_name') else 'unknown',
            'batch_size': self.batch_size,
            'analysis_frequency': self.analysis_frequency,
            'learning_rate': 0.002,  # Default from configure_optimizers
            'max_epochs': 30 if hasattr(self.trainer, 'max_epochs') else 'unknown',
            'precision': str(self.trainer.precision) if hasattr(self.trainer, 'precision') else 'unknown',
            'device': str(device),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }

        # Add ablation study configuration if available
        if hasattr(self, 'ablation_config'):
            experiment_config.update({
                'ablation_mode': self.ablation_config.get('ablation_mode', 'unknown'),
                'ablation_description': self.ablation_config.get('description', ''),
                'ablation_tags': self.ablation_config.get('tags', []),
                'wandb_group': self.ablation_config.get('wandb_group', ''),
                'wandb_run_name': self.ablation_config.get('run_name', ''),
                'notes': self.ablation_config.get('notes', '')
            })

        # Add any hyperparameters from logger
        if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            if hasattr(self.logger.experiment, 'config'):
                experiment_config.update(self.logger.experiment.config)

        # Initialize automatic analyzer with run ID and config
        self.auto_analyzer = AutomaticAnalyzer(
            model=self.model,
            device=str(device),
            save_dir=base_analysis_dir,
            run_id=run_id,
            experiment_config=experiment_config,
            logger=self.logger  # Pass W&B logger for direct logging
        )

        # Update W&B logging frequencies if specified in args
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'optimizer') and hasattr(self.trainer.optimizers[0], 'param_groups'):
            # This is a bit of a hack, but we need to access the args from the main script
            # We'll update the frequencies when they're passed through the training loop
            pass

        # Update checkpoint directory to be within analysis directory
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'callbacks'):
            for callback in self.trainer.callbacks:
                if hasattr(callback, 'dirpath') and 'checkpoint' in callback.dirpath:
                    callback.dirpath = os.path.join(self.auto_analyzer.save_dir, 'checkpoints')
                    os.makedirs(callback.dirpath, exist_ok=True)
                    print(f"✓ Checkpoints will be saved to: {callback.dirpath}")

        print(f"✓ Automatic analyzer initialized for run: {run_id}")
        print(f"✓ Analysis will be saved to: {self.auto_analyzer.save_dir}")
        print(f"✓ Experiment config includes {len(experiment_config)} parameters")

    def forward(self, x):
        encoded, decoded, z = self.model(x)
        return encoded, decoded, z

    def training_step(self, batch, batch_idx):
        # Track training time
        training_start_time = time.perf_counter()

        x, _ = batch
        x = x.to(self.device, non_blocking=True)

        # Track memory before forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(self.device)
        else:
            memory_before = 0

        # Track inference time
        inference_start_time = time.perf_counter()
        encoded, decoded, z = self.forward(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Collect embeddings for epoch-end analysis (sample from batches)
        self._collect_epoch_embeddings(batch_idx, x, z, "train")

        inference_time = (time.perf_counter() - inference_start_time) * 1000  # ms

        # Compute loss
        mse_loss = nn.functional.mse_loss(x, decoded)

        # Store basic metrics
        self.batch_losses.append(mse_loss.item())

        # Calculate additional metrics
        ssim_score = self.calculate_ssim(x, decoded)
        reconstruction_error = torch.mean((x - decoded) ** 2, dim=[1, 2, 3])

        self.reconstruction_metrics.append({
            'mse': mse_loss.item(),
            'ssim': ssim_score.item(),
            'error_std': reconstruction_error.std().item(),
            'error_mean': reconstruction_error.mean().item()
        })

        # Track timing and memory
        training_time = (time.perf_counter() - training_start_time) * 1000  # ms
        self.training_times.append(training_time)
        self.inference_times.append(inference_time)

        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated(self.device)
            memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
            self.memory_usage.append(memory_used)

        # Log basic metrics
        self.log("train/loss", mse_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/ssim", ssim_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/reconstruction_error", reconstruction_error.mean(), on_epoch=True, sync_dist=True)

        # Log timing metrics
        self.log("train/training_time_ms", training_time, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/inference_time_ms", inference_time, on_epoch=True, prog_bar=False, sync_dist=True)
        if torch.cuda.is_available():
            self.log("train/memory_mb", memory_used, on_epoch=True, prog_bar=False, sync_dist=True)

        # Run automatic analysis every batch
        if self.auto_analyzer:
            try:
                # Analyze current batch
                batch_metrics = self.auto_analyzer.analyze_batch(
                    batch_idx=batch_idx,
                    input_tensor=x,
                    output_tensor=decoded,
                    loss=mse_loss.item(),
                    phase="train",
                    global_step=self.global_step
                )

                # Log key attention metrics to W&B
                if batch_idx % 20 == 0:  # Log detailed metrics every 20 batches
                    for metric_name, value in batch_metrics.items():
                        if 'attention_' in metric_name and isinstance(value, (int, float)):
                            self.log(f"train/{metric_name}", value, on_step=True, on_epoch=False, sync_dist=False)

                # Create visualization every N batches
                if batch_idx % self.analysis_frequency == 0 and batch_idx > 0:
                    viz_path = self.auto_analyzer.create_batch_visualization(
                        batch_idx=batch_idx,
                        input_tensor=x,
                        output_tensor=decoded,
                        phase="train",
                        global_step=self.global_step
                    )

                    # Log visualization to W&B (also logged by AutoAnalyzer now)
                    if hasattr(self, 'logger') and self.logger is not None:
                        self.logger.experiment.log({
                            f'train/batch_visualization_{self.global_step}': wandb.Image(viz_path)
                        })

                # Save detailed individual components every N batches (less frequent)
                if batch_idx % self.attention_components_frequency == 0 and batch_idx > 0:
                    component_files = self.auto_analyzer.save_individual_components(
                        batch_idx=batch_idx,
                        input_tensor=x,
                        output_tensor=decoded,
                        phase="train",
                        global_step=self.global_step
                    )

                    # Log component paths to console for debugging
                    if component_files:
                        print(f"✓ Saved {len(component_files)} individual attention components for batch {batch_idx}")
                        print(f"  Files: {list(component_files.keys())[:3]}...")  # Show first 3 file types

                    # Create comprehensive paper dashboard
                    if batch_idx % self.auto_analyzer.dashboard_frequency == 0:
                        try:
                            paper_dashboard = self.auto_analyzer.create_comprehensive_paper_dashboard(
                                batch_idx=batch_idx,
                                input_tensor=x,
                                output_tensor=decoded,
                                phase="train",
                                global_step=self.global_step
                            )
                            print(f"✓ Paper dashboard created for batch {batch_idx}")
                        except Exception as e:
                            print(f"⚠ Paper dashboard creation failed for batch {batch_idx}: {e}")

                    # Save metrics
                    self.auto_analyzer.save_metrics()

                    print(f"✓ Analysis completed for batch {batch_idx}")

            except Exception as e:
                error_msg = f"⚠ Analysis failed for batch {batch_idx}: {e}"
                print(error_msg)

                # Log error to W&B
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.experiment.log({
                        'train/analysis_error_count': 1,
                        'train/analysis_error_message': str(e)[:500]  # Limit length
                    })

                # Don't crash training if analysis fails
                pass

        return mse_loss

    def validation_step(self, batch, batch_idx):
        # Track validation time
        validation_start_time = time.perf_counter()

        x, _ = batch
        x = x.to(self.device, non_blocking=True)

        # Track memory before forward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(self.device)
        else:
            memory_before = 0

        # Track inference time
        inference_start_time = time.perf_counter()
        encoded, decoded, z = self.forward(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Collect embeddings for epoch-end analysis (limited validation collection)
        if batch_idx < 20:  # Only collect from first 20 validation batches
            self._collect_epoch_embeddings(batch_idx, x, z, "val")

        inference_time = (time.perf_counter() - inference_start_time) * 1000  # ms

        mse_loss = nn.functional.mse_loss(x, decoded)
        ssim_score = self.calculate_ssim(x, decoded)

        validation_time = (time.perf_counter() - validation_start_time) * 1000  # ms

        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated(self.device)
            memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
        else:
            memory_used = 0

        # Log validation metrics
        self.log("val/loss", mse_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ssim", ssim_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/validation_time_ms", validation_time, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/inference_time_ms", inference_time, on_epoch=True, prog_bar=False, sync_dist=True)
        if torch.cuda.is_available():
            self.log("val/memory_mb", memory_used, on_epoch=True, prog_bar=False, sync_dist=True)

        # Run automatic analysis on validation samples
        if self.auto_analyzer and batch_idx < 5:  # Analyze first few validation batches
            try:
                # Analyze current batch
                batch_metrics = self.auto_analyzer.analyze_batch(
                    batch_idx=batch_idx,
                    input_tensor=x,
                    output_tensor=decoded,
                    loss=mse_loss.item(),
                    phase="val",
                    global_step=self.global_step
                )

                # Log key attention metrics to W&B
                for metric_name, value in batch_metrics.items():
                    if 'attention_' in metric_name and isinstance(value, (int, float)):
                        self.log(f"val/{metric_name}", value, on_step=True, on_epoch=False, sync_dist=False)

                # Create visualization for validation
                viz_path = self.auto_analyzer.create_batch_visualization(
                    batch_idx=batch_idx,
                    input_tensor=x,
                    output_tensor=decoded,
                    phase="val",
                    global_step=self.global_step
                )

                # Log visualization to W&B (also logged by AutoAnalyzer now)
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.experiment.log({
                        f'val/batch_visualization_{self.global_step}': wandb.Image(viz_path)
                    })

                # Save individual components for validation (first 3 batches only)
                if batch_idx < 3:
                    component_files = self.auto_analyzer.save_individual_components(
                        batch_idx=batch_idx,
                        input_tensor=x,
                        output_tensor=decoded,
                        phase="val",
                        global_step=self.global_step
                    )

                    # Log component paths to console for debugging
                    if component_files:
                        print(f"✓ Saved {len(component_files)} validation attention components for batch {batch_idx}")
                        print(f"  Files: {list(component_files.keys())[:3]}...")  # Show first 3 file types

                # Create paper dashboard for validation (less frequent)
                if batch_idx % 2 == 0:  # Every 2nd validation batch
                    try:
                        paper_dashboard = self.auto_analyzer.create_comprehensive_paper_dashboard(
                            batch_idx=batch_idx,
                            input_tensor=x,
                            output_tensor=decoded,
                            phase="val",
                            global_step=self.global_step
                        )
                        print(f"✓ Validation paper dashboard created for batch {batch_idx}")
                    except Exception as e:
                        print(f"⚠ Validation paper dashboard creation failed for batch {batch_idx}: {e}")

                print(f"✓ Validation analysis completed for batch {batch_idx}")

            except Exception as e:
                error_msg = f"⚠ Validation analysis failed for batch {batch_idx}: {e}"
                print(error_msg)

                # Log error to W&B
                if hasattr(self, 'logger') and self.logger is not None:
                    self.logger.experiment.log({
                        'val/analysis_error_count': 1,
                        'val/analysis_error_message': str(e)[:500]  # Limit length
                    })

                # Don't crash validation if analysis fails
                pass

        return mse_loss

    def calculate_ssim(self, img1, img2):
        """Calculate SSIM between two images"""
        # Simple SSIM calculation
        mu1 = torch.mean(img1, dim=[1, 2, 3])
        mu2 = torch.mean(img2, dim=[1, 2, 3])

        sigma1_sq = torch.var(img1, dim=[1, 2, 3], unbiased=False)
        sigma2_sq = torch.var(img2, dim=[1, 2, 3], unbiased=False)
        sigma12 = torch.mean((img1 - mu1.unsqueeze(1).unsqueeze(2).unsqueeze(3)) *
                              (img2 - mu2.unsqueeze(1).unsqueeze(2).unsqueeze(3)),
                              dim=[1, 2, 3])

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_num = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2))
        ssim_den = ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))

        ssim = ssim_num / ssim_den
        return torch.mean(ssim)

    def configure_optimizers(self):
        # Use Adam with faster convergence
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.002,  # Higher learning rate for faster training
            weight_decay=1e-5,
        )

        # Simple step scheduler for faster training
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=2, min_lr=1e-5
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }

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

    def _create_epoch_evolution_analysis(self):
        """Create analysis of how embeddings evolve across epochs"""

        if not hasattr(self, 'auto_analyzer') or not self.auto_analyzer:
            return

        try:
            import glob
            # Create embeddings directory
            embeddings_dir = os.path.join(self.auto_analyzer.save_dir, 'embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)

            # Find all epoch embedding files
            train_epoch_files = sorted(glob.glob(os.path.join(embeddings_dir, 'train_embeddings_epoch_*.npy')))
            val_epoch_files = sorted(glob.glob(os.path.join(embeddings_dir, 'val_embeddings_epoch_*.npy')))

            if not train_epoch_files:
                print("⚠ No epoch embedding files found for evolution analysis")
                return

            # Load all epoch embeddings
            train_embeddings_by_epoch = []
            val_embeddings_by_epoch = []

            for file_path in train_epoch_files:
                epoch_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
                embeddings = np.load(file_path)
                metadata = pd.read_csv(file_path.replace('_embeddings_epoch_', '_metadata_epoch_').replace('.npy', '.csv'))
                train_embeddings_by_epoch.append({
                    'epoch': epoch_num,
                    'embeddings': embeddings,
                    'metadata': metadata,
                    'file_path': file_path
                })

            for file_path in val_epoch_files:
                epoch_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
                embeddings = np.load(file_path)
                metadata = pd.read_csv(file_path.replace('_embeddings_epoch_', '_metadata_epoch_').replace('.npy', '.csv'))
                val_embeddings_by_epoch.append({
                    'epoch': epoch_num,
                    'embeddings': embeddings,
                    'metadata': metadata,
                    'file_path': file_path
                })

            print(f"Found {len(train_embeddings_by_epoch)} training epochs and {len(val_embeddings_by_epoch)} validation epochs")

            # Create epoch evolution visualizations
            self._create_epoch_evolution_visualizations(train_embeddings_by_epoch, val_embeddings_by_epoch, embeddings_dir)

            # Create epoch evolution HTML report
            self._create_epoch_evolution_html_report(train_embeddings_by_epoch, val_embeddings_by_epoch, embeddings_dir)

        except Exception as e:
            print(f"⚠ Failed to create epoch evolution analysis: {e}")

    def _create_epoch_evolution_visualizations(self, train_epochs: list, val_epochs: list, embeddings_dir: str):
        """Create visualizations showing how embeddings evolve across epochs"""

        try:
            import matplotlib.pyplot as plt

            # Create evolution plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Embedding Evolution Across Training Epochs', fontsize=16, fontweight='bold')

            # 1. Embedding statistics evolution (training)
            ax1 = axes[0, 0]
            epochs = [epoch['epoch'] for epoch in train_epochs]
            means = [np.mean(epoch['embeddings']) for epoch in train_epochs]
            stds = [np.std(epoch['embeddings']) for epoch in train_epochs]
            norms = [np.mean(np.linalg.norm(epoch['embeddings'], axis=1)) for epoch in train_epochs]

            ax1_twin = ax1.twinx()
            line1 = ax1.plot(epochs, means, 'b-o', label='Mean', linewidth=2, markersize=6)
            line2 = ax1_twin.plot(epochs, stds, 'r-s', label='Std Dev', linewidth=2, markersize=6)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Mean Value', color='b')
            ax1_twin.set_ylabel('Standard Deviation', color='r')
            ax1.tick_params(axis='y', labelcolor='b')
            ax1_twin.tick_params(axis='y', labelcolor='r')
            ax1.set_title('Training Embedding Statistics Evolution')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')

            # 2. Embedding norms evolution (training)
            ax2 = axes[0, 1]
            ax2.plot(epochs, norms, 'g-^', linewidth=2, markersize=6)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Mean L2 Norm')
            ax2.set_title('Training Embedding Norm Evolution')
            ax2.grid(True, alpha=0.3)

            # 3. Embedding sparsity evolution (training)
            ax3 = axes[0, 2]
            sparsities = [np.mean(epoch['embeddings'] == 0) for epoch in train_epochs]
            ax3.plot(epochs, sparsities, 'm-d', linewidth=2, markersize=6)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Sparsity (fraction of zeros)')
            ax3.set_title('Training Embedding Sparsity Evolution')
            ax3.grid(True, alpha=0.3)

            # 4-6. Validation evolution plots (if available)
            if val_epochs:
                val_epochs_list = [epoch['epoch'] for epoch in val_epochs]
                val_means = [np.mean(epoch['embeddings']) for epoch in val_epochs]
                val_stds = [np.std(epoch['embeddings']) for epoch in val_epochs]
                val_norms = [np.mean(np.linalg.norm(epoch['embeddings'], axis=1)) for epoch in val_epochs]
                val_sparsities = [np.mean(epoch['embeddings'] == 0) for epoch in val_epochs]

                # Validation statistics
                ax4 = axes[1, 0]
                ax4_twin = ax4.twinx()
                ax4.plot(val_epochs_list, val_means, 'b-o', label='Mean', linewidth=2, markersize=6)
                ax4_twin.plot(val_epochs_list, val_stds, 'r-s', label='Std Dev', linewidth=2, markersize=6)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Mean Value', color='b')
                ax4_twin.set_ylabel('Standard Deviation', color='r')
                ax4.tick_params(axis='y', labelcolor='b')
                ax4_twin.tick_params(axis='y', labelcolor='r')
                ax4.set_title('Validation Embedding Statistics Evolution')
                ax4.grid(True, alpha=0.3)
                ax4.legend(loc='upper left')

                # Validation norms
                ax5 = axes[1, 1]
                ax5.plot(val_epochs_list, val_norms, 'g-^', linewidth=2, markersize=6)
                ax5.set_xlabel('Epoch')
                ax5.set_ylabel('Mean L2 Norm')
                ax5.set_title('Validation Embedding Norm Evolution')
                ax5.grid(True, alpha=0.3)

                # Validation sparsity
                ax6 = axes[1, 2]
                ax6.plot(val_epochs_list, val_sparsities, 'm-d', linewidth=2, markersize=6)
                ax6.set_xlabel('Epoch')
                ax6.set_ylabel('Sparsity (fraction of zeros)')
                ax6.set_title('Validation Embedding Sparsity Evolution')
                ax6.grid(True, alpha=0.3)
            else:
                # Hide validation plots if not available
                for ax in axes[1, :]:
                    ax.text(0.5, 0.5, 'No validation\nembeddings\navailable',
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Validation Data (Not Available)')

            plt.tight_layout()
            evolution_path = os.path.join(embeddings_dir, 'embedding_evolution_analysis.png')
            plt.savefig(evolution_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"✅ Epoch evolution visualizations saved to {embeddings_dir}")

        except ImportError:
            print("⚠ Matplotlib not available for evolution visualizations")
        except Exception as e:
            print(f"⚠ Evolution visualization failed: {e}")

    def _create_epoch_evolution_html_report(self, train_epochs: list, val_epochs: list, embeddings_dir: str):
        """Create HTML report for epoch evolution analysis"""

        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Embedding Evolution Analysis</title>
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
        .epoch-list {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 5px;
            padding: 15px;
        }}
        .epoch-item {{
            padding: 10px;
            border-bottom: 1px solid #e9ecef;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .epoch-item:last-child {{
            border-bottom: none;
        }}
        .file-link {{
            color: #007bff;
            text-decoration: none;
            padding: 5px 10px;
            background-color: #e9ecef;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        .file-link:hover {{
            background-color: #dee2e6;
            text-decoration: none;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔄 Embedding Evolution Analysis</h1>

        <div class="stats">
            <div class="stat-item">
                <div class="stat-number">{len(train_epochs)}</div>
                <div class="stat-label">Training Epochs Analyzed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{len(val_epochs) if val_epochs else 0}</div>
                <div class="stat-label">Validation Epochs Analyzed</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{train_epochs[0]['embeddings'].shape[0] if train_epochs else 0}</div>
                <div class="stat-label">Embeddings per Epoch</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{train_epochs[0]['embeddings'].shape[1] if train_epochs else 0}</div>
                <div class="stat-label">Embedding Dimensions</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Evolution Visualizations</h2>
            <div class="image-container">
                <img src="embedding_evolution_analysis.png" alt="Embedding Evolution Analysis" loading="lazy">
            </div>
        </div>

        <div class="section">
            <h2>📁 Training Epoch Files</h2>
            <div class="epoch-list">
                {''.join([f'''
                <div class="epoch-item">
                    <span>Epoch {epoch['epoch']}</span>
                    <a href="{os.path.basename(epoch['file_path'])}" class="file-link" download>Embeddings</a>
                    <a href="{os.path.basename(epoch['file_path']).replace('_embeddings_epoch_', '_metadata_epoch_').replace('.npy', '.csv')}" class="file-link" download>Metadata</a>
                </div>''' for epoch in train_epochs])}
            </div>
        </div>

        <div class="section">
            <h2>📁 Validation Epoch Files</h2>
            <div class="epoch-list">
                {''.join([f'''
                <div class="epoch-item">
                    <span>Epoch {epoch['epoch']}</span>
                    <a href="{os.path.basename(epoch['file_path'])}" class="file-link" download>Embeddings</a>
                    <a href="{os.path.basename(epoch['file_path']).replace('_embeddings_epoch_', '_metadata_epoch_').replace('.npy', '.csv')}" class="file-link" download>Metadata</a>
                </div>''' for epoch in val_epochs]) if val_epochs else '<p>No validation epoch files available</p>'}
            </div>
        </div>

        <div style="text-align: center; margin-top: 30px; color: #6c757d; font-style: italic;">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><em>This report shows how DAE-KAN embeddings evolve across training epochs</em></p>
        </div>
    </div>
</body>
</html>
        """

        html_path = os.path.join(embeddings_dir, 'embedding_evolution_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Epoch evolution HTML report saved to {html_path}")

    def on_train_epoch_start(self):
        """Enable optimization at the start of each epoch"""
        # Enable mixed precision training
        torch.set_float32_matmul_precision('high')

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Optimize memory after each batch"""
        # Periodic memory cleanup (less frequent than before)
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    def _finalize_embedding_analysis(self):
        """Save remaining embeddings and create comprehensive embedding analysis"""

        # Save any remaining embeddings
        for phase in ['train', 'val']:
            self._save_embeddings_to_disk(phase)

        # Create embedding analysis and visualizations
        self._create_embedding_analysis()

        # Create epoch evolution analysis
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

    def on_train_epoch_end(self):
        """Called at the end of each training epoch - save collected embeddings and run analysis."""

        self._training_epoch += 1  # Increment epoch counter

        # Save embeddings collected during this epoch
        for phase in ['train', 'val']:
            if self.epoch_embeddings_collection[phase]['embeddings']:
                embeddings_file, metadata_file = self._save_epoch_embeddings(phase)
                print(f"✅ Epoch {self._training_epoch-1}: Saved {len(self.epoch_embeddings_collection[phase]['embeddings'])} {phase} embeddings")
                
                # Run analysis on epoch embeddings
                if embeddings_file and metadata_file:
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

    def _create_embedding_visualizations(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                       phase: str, embeddings_dir: str):
        """Create various embedding visualizations and analysis"""

        if len(embeddings) == 0:
            return

        print(f"Creating embedding visualizations for {len(embeddings)} {phase} embeddings...")

        # 1. Basic embedding statistics
        self._create_embedding_statistics(embeddings, metadata, phase, embeddings_dir)

        # 2. Dimensionality reduction (UMAP/t-SNE)
        self._create_dimensionality_reduction_plots(embeddings, metadata, phase, embeddings_dir)

        # 3. Embedding clustering analysis
        if self.auto_analyzer:
            self.auto_analyzer.analyze_embedding_clustering(embeddings, metadata, phase, embeddings_dir)

        # 4. Embedding quality metrics
        self._create_embedding_quality_analysis(embeddings, metadata, phase, embeddings_dir)

        # 5. Create HTML summary
        self._create_embedding_html_summary(embeddings, metadata, phase, embeddings_dir)

    def _create_embedding_statistics(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                   phase: str, embeddings_dir: str):
        """Create basic embedding statistics visualization"""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Embedding Statistics - {phase.upper()} Set', fontsize=16, fontweight='bold')

        # 1. Embedding dimension distribution
        ax1 = axes[0, 0]
        embedding_means = np.mean(embeddings, axis=1)
        ax1.hist(embedding_means, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('Distribution of Embedding Means')
        ax1.set_xlabel('Mean Value')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # 2. Embedding variance distribution
        ax2 = axes[0, 1]
        embedding_vars = np.var(embeddings, axis=1)
        ax2.hist(embedding_vars, bins=50, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_title('Distribution of Embedding Variances')
        ax2.set_xlabel('Variance')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # 3. Embedding sparsity
        ax3 = axes[1, 0]
        sparsity = np.mean(embeddings == 0, axis=1)
        ax3.hist(sparsity, bins=50, alpha=0.7, color='mediumseagreen', edgecolor='black')
        ax3.set_title('Embedding Sparsity Distribution')
        ax3.set_xlabel('Sparsity (fraction of zeros)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)

        # 4. Batch progression
        ax4 = axes[1, 1]
        batch_means = []
        for batch_idx in sorted(metadata['batch_idx'].unique()):
            batch_mask = metadata['batch_idx'] == batch_idx
            batch_means.append(np.mean(embeddings[batch_mask]))

        ax4.plot(batch_means, 'o-', color='purple', linewidth=2, markersize=6)
        ax4.set_title('Embedding Means vs Batch Progress')
        ax4.set_xlabel('Batch Index')
        ax4.set_ylabel('Mean Embedding Value')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        stats_path = os.path.join(embeddings_dir, f'{phase}_embedding_statistics.png')
        plt.savefig(stats_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

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

    def create_final_summary_report(self, final_metrics):
        """Create final summary report"""
        if not final_metrics:
            return

        save_dir = self.auto_analyzer.save_dir if self.auto_analyzer else "final_analysis"
        os.makedirs(save_dir, exist_ok=True)

        # Create text report
        report = []
        report.append("# DAE-KAN Training Final Summary\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Training Summary\n")
        report.append(f"- **Total Batches**: {final_metrics.get('total_batches', 'N/A')}")
        report.append(f"- **Final Batch**: {final_metrics.get('latest_batch', 'N/A')}")

        if 'latest_metrics' in final_metrics:
            latest = final_metrics['latest_metrics']
            report.append(f"- **Final Loss**: {latest.get('loss', 'N/A'):.6f}")
            report.append(f"- **Final MSE**: {latest.get('mse', 'N/A'):.6f}")
            report.append(f"- **Final SSIM**: {latest.get('ssim', 'N/A'):.4f}")

        report.append("\n## Attention Analysis Summary\n")
        attention_metrics = [k for k in final_metrics.keys() if 'attention_' in k and 'mean' in k]
        if attention_metrics:
            report.append("Key attention metrics tracked:")
            for metric in sorted(attention_metrics)[:10]:  # Top 10 metrics
                clean_name = metric.replace('attention_', '').replace('_mean', '')
                value = final_metrics[metric]
                report.append(f"- **{clean_name}**: {value:.4f}")

        report.append(f"\n## Files Generated\n")
        report.append(f"- **Analysis Directory**: {save_dir}")
        report.append(f"- **Metrics JSON**: final_training_metrics.json")
        report.append(f"- **Visualizations**: Batch analysis images")

        if self.training_times and self.inference_times:
            report.append(f"\n## Performance Summary\n")
            report.append(f"- **Avg Training Time**: {np.mean(self.training_times):.2f} ms")
            report.append(f"- **Avg Inference Time**: {np.mean(self.inference_times):.2f} ms")
            report.append(f"- **Throughput**: {1000/np.mean(self.inference_times):.1f} images/sec")

            if self.memory_usage:
                report.append(f"- **Avg Memory Usage**: {np.mean(self.memory_usage):.1f} MB")

        report_text = "\n".join(report)

        # Save report
        report_path = os.path.join(save_dir, "final_summary_report.md")
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"✓ Final summary report saved: {report_path}")

    def _create_dimensionality_reduction_plots(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                             phase: str, embeddings_dir: str):
        """Create UMAP/t-SNE visualizations"""
        try:
            import matplotlib.pyplot as plt
            from sklearn.preprocessing import StandardScaler

            # Sample if too many embeddings
            max_samples = 2000
            if len(embeddings) > max_samples:
                indices = np.random.choice(len(embeddings), max_samples, replace=False)
                embeddings_sample = embeddings[indices]
                metadata_sample = metadata.iloc[indices].reset_index(drop=True)
            else:
                embeddings_sample = embeddings
                metadata_sample = metadata.reset_index(drop=True)

            # Standardize
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_sample)

            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'{phase.capitalize()} Embedding Dimensionality Reduction', fontsize=16, fontweight='bold')

            # Try UMAP first
            try:
                import umap
                reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
                embedding_2d = reducer.fit_transform(embeddings_scaled)

                axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6, s=30)
                axes[0].set_title('UMAP Projection')
                axes[0].set_xlabel('UMAP 1')
                axes[0].set_ylabel('UMAP 2')
                axes[0].grid(True, alpha=0.3)

            except ImportError:
                axes[0].text(0.5, 0.5, 'UMAP not available\n(pip install umap-learn)',
                           ha='center', va='center', transform=axes[0].transAxes)
                axes[0].set_title('UMAP Projection')

            # t-SNE
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_sample)-1))
            embedding_2d = tsne.fit_transform(embeddings_scaled)

            axes[1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], alpha=0.6, s=30)
            axes[1].set_title('t-SNE Projection')
            axes[1].set_xlabel('t-SNE 1')
            axes[1].set_ylabel('t-SNE 2')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            viz_path = os.path.join(embeddings_dir, f'{phase}_dimensionality_reduction.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        except Exception as e:
            print(f"⚠ Failed to create dimensionality reduction plots: {e}")



    def _create_embedding_quality_analysis(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                         phase: str, embeddings_dir: str):
        """Create embedding quality metrics analysis"""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{phase.capitalize()} Embedding Quality Analysis', fontsize=16, fontweight='bold')

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
            viz_path = os.path.join(embeddings_dir, f'{phase}_quality_analysis.png')
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

        except Exception as e:
            print(f"⚠ Failed to create quality analysis: {e}")

    def _create_embedding_html_summary(self, embeddings: np.ndarray, metadata: pd.DataFrame,
                                     phase: str, embeddings_dir: str):
        """Create HTML summary for embedding analysis"""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{phase.capitalize()} Embedding Analysis</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    .header {{ text-align: center; color: #333; }}
                    .stats {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
                    .stats th {{ text-align: left; padding: 5px; }}
                    .stats td {{ padding: 5px; }}
                    .viz-container {{ text-align: center; margin: 20px 0; }}
                    .viz-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{phase.capitalize()} Embedding Analysis</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>

                <div class="stats">
                    <h2>Embedding Statistics</h2>
                    <table>
                        <tr><th>Total Embeddings</th><td>{len(embeddings):,}</td></tr>
                        <tr><th>Embedding Dimension</th><td>{embeddings.shape[1]}</td></tr>
                        <tr><th>Average Norm</th><td>{np.mean(np.linalg.norm(embeddings, axis=1)):.4f}</td></tr>
                        <tr><th>Average Mean</th><td>{np.mean(np.mean(embeddings, axis=1)):.4f}</td></tr>
                        <tr><th>Average Variance</th><td>{np.mean(np.var(embeddings, axis=1)):.4f}</td></tr>
                        <tr><th>Sparsity</th><td>{np.mean(np.mean(embeddings == 0, axis=1)):.4f}</td></tr>
                    </table>
                </div>

                <div class="viz-container">
                    <h2>Visualizations</h2>
                    <img src="{phase}_dimensionality_reduction.png" alt="Dimensionality Reduction">
                    <img src="{phase}_clustering_analysis.png" alt="Clustering Analysis">
                    <img src="{phase}_quality_analysis.png" alt="Quality Analysis">
                </div>
            </body>
            </html>
            """

            html_path = os.path.join(embeddings_dir, f'{phase}_embedding_analysis.html')
            with open(html_path, 'w') as f:
                f.write(html_content)

        except Exception as e:
            print(f"⚠ Failed to create HTML summary: {e}")


def get_optimal_batch_size(model, device, input_size=(3, 128, 128)):
    """
    Dynamically determine optimal batch size based on available GPU memory
    """
    model.eval()
    torch.cuda.empty_cache()

    # Get available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = gpu_memory - reserved_memory - allocated_memory

        # Estimate memory per sample (rough heuristic)
        sample_size = torch.prod(torch.tensor(input_size)).item() * 4  # 4 bytes per float32

        # Start with batch size and increase until memory is full
        batch_size = 4
        max_batch_size = 32  # Upper limit

        while batch_size <= max_batch_size:
            try:
                # Test with current batch size
                test_input = torch.randn(batch_size, *input_size).to(device)
                with torch.no_grad():
                    _ = model(test_input)

                # If successful, try larger batch
                del test_input
                batch_size *= 2

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e

        # Return the largest successful batch size divided by 2 for safety
        return max(4, batch_size // 2)
    else:
        return 4  # Default for CPU


if __name__ == '__main__':
    import argparse
    import pandas as pd

    # Optimized CUDA settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async CUDA operations

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train optimized DAE-KAN model with analysis')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA and use CPU')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto-detected if not specified)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--analysis-freq', type=int, default=100, help='Run analysis every N batches')
    parser.add_argument('--dashboard-freq', type=int, default=200, help='Run comprehensive dashboard every N batches')
    parser.add_argument('--wandb-metrics-freq', type=int, default=10, help='Log detailed metrics to W&B every N batches')
    parser.add_argument('--wandb-viz-freq', type=int, default=100, help='Log visualizations to W&B every N batches')
    parser.add_argument('--wandb-paper-freq', type=int, default=200, help='Log paper figures to W&B every N batches')
    parser.add_argument('--attention-components-freq', type=int, default=400, help='Save detailed attention components every N batches')
    parser.add_argument('--embeddings-per-epoch', type=int, default=500, help='Number of embeddings to collect and save at each epoch end')
    parser.add_argument('--max-embeddings-to-store', type=int, default=1000, help='Maximum number of embeddings to store in memory for final analysis')
    parser.add_argument('--save-epoch-embeddings', action='store_true', default=True, help='Save embeddings at the end of each epoch')
    parser.add_argument('--project-name', type=str, default='histopath-kan-optimized-analysis', help='W&B project name')
    parser.add_argument('--model-name', type=str, default="dae_kan_attention", help='Name of the model to use')

    # Ablation study organization arguments
    parser.add_argument('--ablation-mode', type=str, default='baseline',
                       choices=['baseline', 'no-attention', 'different-batch-size', 'different-lr', 'different-architecture', 'custom'],
                       help='Ablation study mode for organization')
    parser.add_argument('--ablation-group', type=str, help='W&B group for organizing related experiments')
    parser.add_argument('--experiment-name', type=str, help='Custom experiment name for W&B')
    parser.add_argument('--ablation-tags', nargs='+', default=[], help='Additional tags for W&B organization')
    parser.add_argument('--run-description', type=str, help='Description of this ablation run')

    # Training duration control
    parser.add_argument('--max-epochs', type=int, default=1, help='Maximum number of training epochs (default: 1)')
    parser.add_argument('--min-epochs', type=int, default=1, help='Minimum number of training epochs (default: 1)')
    parser.add_argument('--early-stopping', action='store_true', help='Enable early stopping based on validation loss')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping (epochs)')
    parser.add_argument('--fast-mode', action='store_true', help='Fast mode: 1 epoch, minimal analysis, quick training')

    args = parser.parse_args()

    # Fast mode configuration
    if args.fast_mode:
        args.max_epochs = 1
        args.min_epochs = 1
        args.analysis_freq = max(20, args.analysis_freq)  # Less frequent analysis
        args.wandb_metrics_freq = max(5, args.wandb_metrics_freq)
        args.wandb_viz_freq = max(50, args.wandb_viz_freq)
        args.wandb_paper_freq = max(100, args.wandb_paper_freq)
        print("🚀 Fast mode enabled: 1 epoch, reduced analysis frequency")

    # Set device
    if args.no_cuda:
        device = torch.device('cpu')
        gpu_id = None
        num_workers = 2
    else:
        if torch.cuda.is_available():
            gpu_id = args.gpu
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}")
            num_workers = args.num_workers
            # Enable cuDNN benchmark for consistent input sizes
            torch.backends.cudnn.benchmark = True
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
            gpu_id = None
            num_workers = 2

    # Create datasets
    train_ds = ImageDataset(*create_dataset('train'))
    test_ds = ImageDataset(*create_dataset('test'))

    # Determine optimal batch size
    if args.batch_size is None:
        temp_model = get_model(args.model_name)()
        temp_model = temp_model.to(device)
        batch_size = get_optimal_batch_size(temp_model, device)
        del temp_model
        torch.cuda.empty_cache()
        print(f"Auto-detected optimal batch size: {batch_size}")
    else:
        batch_size = args.batch_size

    # Create optimized data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # For consistent batch sizes
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )


def generate_ablation_config(args, batch_size, device):
    """Generate comprehensive ablation study configuration"""

    # Base configuration
    config = {
        'ablation_mode': args.ablation_mode,
        'model_name': args.model_name,
        'batch_size': batch_size,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }

    # Mode-specific configurations
    mode_configs = {
        'baseline': {
            'description': 'Baseline DAE-KAN with attention mechanisms',
            'tags': ['baseline', 'attention', 'complete-model'],
            'group_name': 'baseline-experiments'
        },
        'no-attention': {
            'description': 'DAE-KAN without attention mechanisms',
            'tags': ['no-attention', 'ablation', 'attention-removed'],
            'group_name': 'attention-ablation'
        },
        'different-batch-size': {
            'description': f'DAE-KAN with batch size {batch_size}',
            'tags': ['batch-size', 'ablation', f'bs-{batch_size}'],
            'group_name': 'batch-size-ablation'
        },
        'different-lr': {
            'description': 'DAE-KAN with modified learning rate',
            'tags': ['learning-rate', 'ablation', 'lr-modified'],
            'group_name': 'learning-rate-ablation'
        },
        'different-architecture': {
            'description': f'DAE-KAN with {args.model_name} architecture',
            'tags': ['architecture', 'ablation', args.model_name],
            'group_name': 'architecture-ablation'
        },
        'custom': {
            'description': args.run_description or 'Custom ablation experiment',
            'tags': ['custom', 'ablation'] + args.ablation_tags,
            'group_name': args.ablation_group or 'custom-ablation'
        }
    }

    mode_config = mode_configs.get(args.ablation_mode, mode_configs['baseline'])

    # Generate experiment name
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f"{args.ablation_mode}_{args.model_name}_{timestamp}"

    # Generate W&B group
    if args.ablation_group:
        wandb_group = args.ablation_group
    else:
        wandb_group = mode_config['group_name']

    # Generate tags
    base_tags = ['dae-kan', 'optimized', 'analysis', 'histopathology']
    mode_tags = mode_config['tags']
    user_tags = args.ablation_tags

    all_tags = list(set(base_tags + mode_tags + user_tags))

    # Combine everything
    ablation_config = {
        **config,
        **mode_config,
        'experiment_name': experiment_name,
        'wandb_group': wandb_group,
        'tags': all_tags,
        'run_name': experiment_name,
        'notes': args.run_description or mode_config['description']
    }

    return ablation_config


if __name__ == "__main__":
    # Test data loading
    x, y = next(iter(train_loader))
    print(f"Data loading successful. Batch shape: {x.shape}")

    # Generate ablation configuration
    ablation_config = generate_ablation_config(args, batch_size, device)
    print(f"✓ Ablation configuration generated:")
    print(f"  - Mode: {ablation_config['ablation_mode']}")
    print(f"  - Experiment Name: {ablation_config['experiment_name']}")
    print(f"  - W&B Group: {ablation_config['wandb_group']}")
    print(f"  - Tags: {', '.join(ablation_config['tags'][:5])}...")
    print(f"  - Description: {ablation_config['notes']}")

    # Initialize optimized model with configurable analysis frequencies
    model = OptimizedMainModelWithAnalysis(
        model_name=args.model_name,
        batch_size=batch_size,
        analysis_frequency=args.analysis_freq,
        attention_components_frequency=args.attention_components_freq,
        max_embeddings_to_store=args.max_embeddings_to_store,
        embeddings_per_epoch=args.embeddings_per_epoch
    )

    # Pass ablation configuration to model for local storage
    model.ablation_config = ablation_config
    model = model.to(device)

    # Setup logging with ablation configuration
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=ablation_config['run_name'],
        group=ablation_config['wandb_group'],
        tags=ablation_config['tags'],
        notes=ablation_config['notes'],
        config={
            **vars(args),
            **ablation_config,
            'analysis_config': {
                'analysis_frequency': args.analysis_freq,
                'dashboard_frequency': args.dashboard_freq,
                'wandb_metrics_frequency': args.wandb_metrics_freq,
                'wandb_visualization_frequency': args.wandb_viz_freq,
                'wandb_paper_frequency': args.wandb_paper_freq
            }
        }
    )
    wandb_logger.watch(model, log='all', log_freq=10)

    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create temporary checkpoint directory (will be moved to analysis directory in on_fit_start)
    temp_checkpoint_dir = './temp_checkpoints'
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath=temp_checkpoint_dir,
        filename='best-checkpoint-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )


    # Create callbacks for early stopping if enabled
    callbacks = [checkpoint_callback, lr_monitor]

    if args.early_stopping:
        from pytorch_lightning.callbacks import EarlyStopping
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            patience=args.patience,
            mode='min',
            min_delta=1e-6,
            verbose=True
        )
        callbacks.append(early_stop_callback)
        print(f"✓ Early stopping enabled with patience: {args.patience}")

    # Create optimized trainer with configurable epochs
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator='gpu' if gpu_id is not None else 'cpu',
        devices=[gpu_id] if gpu_id is not None else 1,
        log_every_n_steps=20,
        precision='16-mixed',  # Mixed precision training
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        # Optimized training settings
        deterministic=False,
        benchmark=True,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        # Faster logging
        enable_checkpointing=True,
        enable_progress_bar=True,
        # Fast mode optimizations
        fast_dev_run=args.fast_mode,
    )

    print(f"✓ Training configured:")
    print(f"  - Max epochs: {args.max_epochs}")
    print(f"  - Min epochs: {args.min_epochs}")
    print(f"  - Early stopping: {args.early_stopping}")
    if args.early_stopping:
        print(f"  - Patience: {args.patience}")
    print(f"  - Fast mode: {args.fast_mode}")

    # Configure analysis frequencies from command line arguments
    if hasattr(model, 'auto_analyzer') and model.auto_analyzer:
        model.auto_analyzer.dashboard_frequency = args.dashboard_freq
        model.auto_analyzer.wandb_log_frequencies.update({
            'metrics': args.wandb_metrics_freq,
            'visualizations': args.wandb_viz_freq,
            'paper_figures': args.wandb_paper_freq,
            'individual_components': max(args.wandb_paper_freq * 2, 500)
        })

        print(f"✓ Analysis frequencies configured:")
        print(f"  - Basic analysis: every {args.analysis_freq} batches")
        print(f"  - Dashboard: every {args.dashboard_freq} batches")
        print(f"  - W&B metrics: every {args.wandb_metrics_freq} batches")
        print(f"  - W&B visualizations: every {args.wandb_viz_freq} batches")
        print(f"  - W&B paper figures: every {args.wandb_paper_freq} batches")

    # Start training
    print(f"Starting training with batch size {batch_size} on {device}")
    trainer.fit(model, train_loader, test_loader)

    # Cleanup
    wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()
