import gc
import os
import time
import warnings
from datetime import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader

import wandb
from clustering_objective import DECClusteringLoss, LatentProjectionHead
from histodata import ImageDataset, create_dataset
from models.factory import get_model

warnings.filterwarnings('ignore')

# Automatic analysis module
from analysis.auto_analysis import AutomaticAnalyzer
from training.embedding_io import EmbeddingIOMixin
from training.hparams import generate_ablation_config, get_optimal_batch_size
from training.lifecycle import LifecycleMixin
from training.reports import ReportsMixin


class OptimizedMainModelWithAnalysis(pl.LightningModule,
                                     EmbeddingIOMixin,
                                     LifecycleMixin,
                                     ReportsMixin):
    """Optimized Lightning module with better memory management and GPU utilization"""

    def __init__(self, model_name: str = "dae_kan_attention", batch_size=8, analysis_frequency=100,
                 attention_components_frequency=400, max_embeddings_to_store=1000, embeddings_per_epoch=500,
                 cluster_weight: float = 0.0, recon_weight: float = 1.0,
                 latent_dim: int = 128, n_clusters: int = 5,
                 cluster_warmup_epochs: int = 5, learning_rate: float = 0.002,
                 epoch_viz: str = "render"):
        super().__init__()
        self.model_name = model_name
        self.model = get_model(model_name)()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # Epoch-embedding visualisation mode (a major training-time cost):
        #   render -> save per-epoch .npy + metadata AND render t-SNE/UMAP/clustering
        #             plots every epoch (original behaviour).
        #   dump   -> save per-epoch .npy + metadata only; skip rendering. The dumps
        #             carry all information; regenerate plots offline with
        #             regenerate_epoch_viz.py (no GPU). ~40% faster epochs.
        #   off    -> no per-epoch embedding dumps or rendering at all.
        self.epoch_viz = epoch_viz if epoch_viz in ("render", "dump", "off") else "render"

        # --- Clustering objective (opt-in; default OFF keeps training identical) ---
        # With cluster_weight == 0 nothing below is built and the loss is pure MSE,
        # so existing reconstruction-only runs are byte-for-byte unchanged.
        self.cluster_weight = float(cluster_weight)
        self.recon_weight = float(recon_weight)
        self.latent_dim = int(latent_dim)
        self.n_clusters = int(n_clusters)
        self.cluster_warmup_epochs = int(cluster_warmup_epochs)
        self.latent_head = None
        self.dec_loss = None
        self._latent_pool = []          # rolling buffer for KMeans centroid init
        self._latent_pool_cap = 2000
        self._head_optimised = False    # set once head params join the optimizer
        if self.cluster_weight > 0:
            self.latent_head = LatentProjectionHead(self.latent_dim)
            self.dec_loss = DECClusteringLoss(self.n_clusters, self.latent_dim)
            # Eagerly build the projection head's Linear so its parameters are
            # registered BEFORE configure_optimizers() runs (a lazily-built layer
            # would be silently excluded from the optimizer).
            try:
                self.model.eval()
                with torch.no_grad():
                    _, _, z = self.model(torch.zeros(1, 3, 128, 128))
                self.latent_head(z)  # triggers lazy _build with the real z dim
                self.model.train()
                print(f"✓ Clustering objective ON: latent_dim={self.latent_dim}, "
                      f"n_clusters={self.n_clusters}, cluster_weight={self.cluster_weight}, "
                      f"warmup={self.cluster_warmup_epochs} epochs")
            except Exception as e:
                # Lazy fallback: the head builds on its first real forward and
                # _ensure_head_optimised() registers it with the optimizer then.
                print(f"⚠ Could not pre-build latent head ({e}); will build lazily "
                      f"and register via the training-step safety net.")
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

        # Create analysis directory and run ID. In offline / non-wandb modes
        # `experiment.name` can be a bound method rather than a string, so coerce
        # and fall back to a timestamped id if it is not a usable string.
        base_analysis_dir = "auto_analysis"
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if hasattr(self, 'logger') and getattr(self, 'logger', None) and getattr(self.logger, 'experiment', None):
            candidate = getattr(self.logger.experiment, 'name', None)
            if isinstance(candidate, str) and candidate:
                run_id = candidate

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

        # Add any hyperparameters from logger. In offline / non-wandb modes
        # `experiment.config` may be a bound method or a non-mapping; only merge
        # it when it is actually dict-like, otherwise skip (it is non-essential).
        if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            cfg = getattr(self.logger.experiment, 'config', None)
            if isinstance(cfg, dict):
                experiment_config.update(cfg)
            elif cfg is not None and hasattr(cfg, 'items') and not callable(cfg):
                try:
                    experiment_config.update(dict(cfg))
                except Exception:
                    pass

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

    def _ensure_head_optimised(self) -> None:
        """Register lazily-built head params with the optimizer (safety net).

        The projection head is normally pre-built in ``__init__`` (probed with
        a zeros batch) before ``configure_optimizers()`` runs, making this a
        no-op. If pre-build failed, the head materialises on its first real
        forward — after the optimizer exists — and without this its weights
        would silently never train. Retried every step until registered, so a
        missing trainer (e.g. manual calls) simply defers.
        """
        if self._head_optimised or self.latent_head is None:
            return
        if not list(self.latent_head.parameters()):
            return  # head not built yet; nothing to register
        trainer = getattr(self, "trainer", None)
        optimizers = list(getattr(trainer, "optimizers", None) or [])
        if not optimizers:
            return
        for opt in optimizers:
            known = {id(p) for group in opt.param_groups for p in group["params"]}
            missing = [p for p in self.latent_head.parameters() if id(p) not in known]
            if missing:
                opt.add_param_group({"params": missing})
        self._head_optimised = True

    def training_step(self, batch, batch_idx):
        # Track training time
        training_start_time = time.perf_counter()

        x, _ = batch
        x = x.to(self.device, non_blocking=True)

        # Per-batch timing/SSIM/memory is instrumentation only and each
        # torch.cuda.synchronize() serialises the GPU pipeline (kills throughput).
        # Do it only on "profile" batches; the model/loss path runs every batch
        # unchanged, so results are identical — only timing-log resolution drops.
        profile = (batch_idx % max(1, self.analysis_frequency) == 0)

        memory_before = 0
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated(self.device)

        inference_start_time = time.perf_counter()
        encoded, decoded, z = self.forward(x)
        if profile and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Collect embeddings for epoch-end analysis (sample from batches)
        self._collect_epoch_embeddings(batch_idx, x, z, "train")

        inference_time = (time.perf_counter() - inference_start_time) * 1000  # ms

        # Compute loss
        mse_loss = nn.functional.mse_loss(x, decoded)

        # --- Composite clustering objective (opt-in) ---------------------------
        # loss = recon_weight * MSE + cluster_weight * DEC-KL (after a warmup,
        # once centroids are KMeans-initialised from a pool of latents).
        loss = self.recon_weight * mse_loss
        cluster_loss_val = 0.0
        if self.cluster_weight > 0 and self.latent_head is not None:
            self._ensure_head_optimised()
            latent = self.latent_head(z)
            # Maintain a rolling latent pool for centroid initialisation.
            if not self.dec_loss.initialised:
                self._latent_pool.append(latent.detach().cpu())
                pooled = sum(t.size(0) for t in self._latent_pool)
                if pooled > self._latent_pool_cap:
                    self._latent_pool.pop(0)
                past_warmup = self.current_epoch >= self.cluster_warmup_epochs
                enough = pooled >= max(10 * self.n_clusters, self.n_clusters + 1)
                if past_warmup and enough:
                    self.dec_loss.init_centers(torch.cat(self._latent_pool, 0))
                    self._latent_pool = []
                    print(f"✓ DEC centroids initialised at epoch {self.current_epoch}")
            if self.dec_loss.initialised:
                cluster_loss = self.dec_loss(latent)
                loss = loss + self.cluster_weight * cluster_loss
                cluster_loss_val = cluster_loss.item()

        # Loss is logged every batch (Lightning detaches internally — no extra sync).
        self.log("train/loss", mse_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.cluster_weight > 0:
            self.log("train/total_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/cluster_loss", cluster_loss_val, on_epoch=True, sync_dist=True)

        # Everything below is instrumentation (each .item()/SSIM/synchronize forces
        # a GPU stall) — restrict to profile batches to keep throughput high.
        if profile:
            self.batch_losses.append(mse_loss.item())
            ssim_score = self.calculate_ssim(x, decoded)
            reconstruction_error = torch.mean((x - decoded) ** 2, dim=[1, 2, 3])
            self.reconstruction_metrics.append({
                'mse': mse_loss.item(),
                'ssim': ssim_score.item(),
                'error_std': reconstruction_error.std().item(),
                'error_mean': reconstruction_error.mean().item()
            })
            training_time = (time.perf_counter() - training_start_time) * 1000  # ms
            self.training_times.append(training_time)
            self.inference_times.append(inference_time)
            self.log("train/ssim", ssim_score, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/reconstruction_error", reconstruction_error.mean(), on_epoch=True, sync_dist=True)
            self.log("train/training_time_ms", training_time, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/inference_time_ms", inference_time, on_epoch=True, prog_bar=False, sync_dist=True)
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.device)
                memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
                self.memory_usage.append(memory_used)
                self.log("train/memory_mb", memory_used, on_epoch=True, prog_bar=False, sync_dist=True)

        # Run automatic analysis periodically. Previously this ran on EVERY
        # batch, which dominated epoch time (~27 min/epoch) and is unnecessary —
        # embeddings for the paper are re-extracted post-hoc. Gate on
        # analysis_frequency (and always include batch 0 for a first look).
        run_analysis = self.auto_analyzer and (
            batch_idx % max(1, self.analysis_frequency) == 0)
        if run_analysis:
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
                        if 'attention_' in metric_name and isinstance(value, int | float):
                            self.log(f"train/{metric_name}", value, on_step=True, on_epoch=False, sync_dist=False)

                # Create visualization every N batches
                if batch_idx % self.analysis_frequency == 0 and batch_idx > 0:
                    self.auto_analyzer.create_batch_visualization(
                        batch_idx=batch_idx,
                        input_tensor=x,
                        output_tensor=decoded,
                        phase="train",
                        global_step=self.global_step
                    )
                    # Note: W&B logging is handled by AutoAnalyzer.create_batch_visualization()

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
                            self.auto_analyzer.create_comprehensive_paper_dashboard(
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

        return loss

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
                    if 'attention_' in metric_name and isinstance(value, int | float):
                        self.log(f"val/{metric_name}", value, on_step=True, on_epoch=False, sync_dist=False)

                # Create visualization for validation (first batch only to avoid excessive logging)
                if batch_idx == 0:
                    self.auto_analyzer.create_batch_visualization(
                        batch_idx=batch_idx,
                        input_tensor=x,
                        output_tensor=decoded,
                        phase="val",
                        global_step=self.global_step
                    )
                    # Note: W&B logging is handled by AutoAnalyzer.create_batch_visualization()

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
                        self.auto_analyzer.create_comprehensive_paper_dashboard(
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

    def configure_optimizers(self):
        # Use Adam with faster convergence
        optimizer = torch.optim.Adam(
            self.parameters(),  # includes latent_head + dec_loss centers when enabled
            lr=self.learning_rate,
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

    def on_train_epoch_start(self):
        """Enable optimization at the start of each epoch"""
        # Enable mixed precision training
        torch.set_float32_matmul_precision('high')

def main():
    """Train entrypoint (also installed as `histodae-train`)."""
    import argparse

    from utils.reproducibility import create_experiment_config, set_seed

    # Optimized CUDA settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async CUDA operations

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train optimized DAE-KAN model with analysis')

    # Reproducibility arguments (Task 1)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--config-path', type=str, default='config/experiment_config.yaml', help='Path to experiment configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Base output directory for experiments')

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
    parser.add_argument('--project-name', type=str, default='histopath-kan-pannuke-analysis', help='W&B project name')
    parser.add_argument('--model-name', type=str, default="dae_kan_attention", help='Name of the model to use')
    parser.add_argument('--dataset', type=str, default='pannuke', choices=['pannuke', 'hepar'],
                       help='Dataset to use for training (default: pannuke)')

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

    # Clustering objective (opt-in; default OFF == pure reconstruction)
    parser.add_argument('--cluster-weight', type=float, default=0.0,
                       help='Weight of the DEC/IDEC clustering loss. 0 (default) = pure '
                            'reconstruction (unchanged behaviour); >0 enables the latent '
                            'bottleneck + clustering objective.')
    parser.add_argument('--recon-weight', type=float, default=1.0,
                       help='Weight of the reconstruction (MSE) loss term.')
    parser.add_argument('--latent-dim', type=int, default=128,
                       help='Projected latent dimension used for clustering (when enabled).')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Number of clusters for the DEC objective (when enabled).')
    parser.add_argument('--cluster-warmup-epochs', type=int, default=5,
                       help='Reconstruction-only warmup before the clustering loss kicks in.')
    parser.add_argument('--learning-rate', type=float, default=0.002,
                       help='Adam learning rate (default: 0.002).')
    parser.add_argument('--epoch-viz', type=str, default='render',
                       choices=['render', 'dump', 'off'],
                       help="Per-epoch embedding viz: 'render' (save .npy+meta AND plot, "
                            "original), 'dump' (save .npy+meta only, ~40%% faster epochs; "
                            "rebuild plots offline with regenerate_epoch_viz.py), "
                            "'off' (nothing). Default: render.")

    args = parser.parse_args()

    # Map dataset names to directory names and update project name
    dataset_mapping = {
        'pannuke': 'PANnuke',
        'hepar': 'HeparUnifiedPNG'
    }

    args.dataset_dir = dataset_mapping[args.dataset]

    # Update project name based on dataset
    if args.project_name == 'histopath-kan-pannuke-analysis':  # Only update if default
        args.project_name = f'histopath-kan-{args.dataset}-analysis'

    # Fast mode configuration
    if args.fast_mode:
        args.max_epochs = 1
        args.min_epochs = 1
        args.analysis_freq = max(20, args.analysis_freq)  # Less frequent analysis
        args.wandb_metrics_freq = max(5, args.wandb_metrics_freq)
        args.wandb_viz_freq = max(50, args.wandb_viz_freq)
        args.wandb_paper_freq = max(100, args.wandb_paper_freq)
        print("🚀 Fast mode enabled: 1 epoch, reduced analysis frequency")

    # Set random seed for reproducibility (Task 1 - Reviewer 1 #2, Reviewer 2)
    print(f"\n🔬 Setting random seed: {args.seed}")
    set_seed(args.seed)

    # Create experiment directory and save configuration
    experiment_name = f"{args.model_name}_seed{args.seed}"
    if hasattr(args, 'experiment_name') and args.experiment_name:
        experiment_name = args.experiment_name

    exp_dir, config = create_experiment_config(
        experiment_name=experiment_name,
        custom_params={
            'seed': args.seed,
            'model_name': args.model_name,
            'dataset': args.dataset,
            'batch_size': args.batch_size,
            'max_epochs': args.max_epochs,
            'analysis_frequency': args.analysis_freq,
        },
        base_dir=args.output_dir
    )
    print(f"✓ Experiment directory: {exp_dir}")

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
    train_ds = ImageDataset(*create_dataset('train', dataset_name=args.dataset_dir))
    test_ds = ImageDataset(*create_dataset('test', dataset_name=args.dataset_dir))

    print(f"✅ Using dataset: {args.dataset} (directory: {args.dataset_dir})")
    print(f"✅ Training samples: {len(train_ds)}")
    print(f"✅ Test samples: {len(test_ds)}")

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

    # Test data loading
    x, y = next(iter(train_loader))
    print(f"Data loading successful. Batch shape: {x.shape}")

    # Generate ablation configuration
    ablation_config = generate_ablation_config(args, batch_size, device)
    print("✓ Ablation configuration generated:")
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
        embeddings_per_epoch=args.embeddings_per_epoch,
        cluster_weight=args.cluster_weight,
        recon_weight=args.recon_weight,
        latent_dim=args.latent_dim,
        n_clusters=args.n_clusters,
        cluster_warmup_epochs=args.cluster_warmup_epochs,
        learning_rate=args.learning_rate,
        epoch_viz=args.epoch_viz,
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

    print("✓ Training configured:")
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

        print("✓ Analysis frequencies configured:")
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


if __name__ == "__main__":
    main()
