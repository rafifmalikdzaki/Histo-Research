"""The training Lightning module.

Responsibilities kept here: model construction, the forward pass, the
train/validation steps, and the bookkeeping of metrics and embedding
buffers. Everything that can be expressed without Lightning state lives in
sibling modules:

* :mod:`histodae.training.losses`      — objective computation
* :mod:`histodae.training.optim`       — optimizer + head-registration safety net
* :mod:`histodae.training.analysis_hook` — periodic in-training analysis
* :mod:`histodae.training.embedding_io` — embedding collection and dumping
* :mod:`histodae.training.lifecycle`    — epoch/end-of-training hooks
* :mod:`histodae.training.reports`      — report and figure generation
"""

import os
import time
from datetime import datetime

import pytorch_lightning as pl
import torch

from histodae.analysis.auto_analysis import AutomaticAnalyzer
from histodae.clustering.dec import DECClusteringLoss, LatentProjectionHead
from histodae.models.factory import get_model
from histodae.training.analysis_hook import run_training_analysis, run_validation_analysis
from histodae.training.embedding_io import EmbeddingIOMixin
from histodae.training.lifecycle import LifecycleMixin
from histodae.training.losses import compute_training_loss, compute_validation_loss
from histodae.training.optim import ensure_head_registered, optimizer_config
from histodae.training.reports import ReportsMixin

__all__ = ["OptimizedMainModelWithAnalysis"]


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

        Thin wrapper over :func:`histodae.training.optim.ensure_head_registered`
        so the method stays reachable as ``self._ensure_head_optimised()``.
        """
        ensure_head_registered(self)

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
        loss, mse_loss, cluster_loss_val = compute_training_loss(self, x, decoded, z)

        # Loss is logged every batch (Lightning detaches internally — no extra sync).
        self.log("train/loss", mse_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        if self.cluster_weight > 0:
            self.log("train/total_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/cluster_loss", cluster_loss_val, on_epoch=True, prog_bar=True, sync_dist=True)

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
            self.log("train/reconstruction_error", reconstruction_error.mean(), on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("train/training_time_ms", training_time, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/inference_time_ms", inference_time, on_epoch=True, prog_bar=False, sync_dist=True)
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated(self.device)
                memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
                self.memory_usage.append(memory_used)
                self.log("train/memory_mb", memory_used, on_epoch=True, prog_bar=False, sync_dist=True)

        # Run automatic analysis periodically (see analysis_hook for the gating).
        run_training_analysis(self, batch_idx, x, decoded, mse_loss)

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

        mse_loss = compute_validation_loss(self, x, decoded)
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
        run_validation_analysis(self, batch_idx, x, decoded, mse_loss)

        return mse_loss

    def configure_optimizers(self):
        return optimizer_config(self)

    def on_train_epoch_start(self):
        """Enable optimization at the start of each epoch"""
        # Enable mixed precision training
        torch.set_float32_matmul_precision('high')
