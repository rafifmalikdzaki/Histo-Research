#!/usr/bin/env python3
"""Training entry point (also installed as the ``histodae-train`` console script).

This module owns the CLI surface only: argument parsing, device/dataset
setup, trainer construction. The Lightning module itself lives in
:mod:`histodae.training.module`, with its objective, optimizer and analysis
hook in the sibling modules of :mod:`histodae.training`.
"""

import gc
import os
import warnings

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb

warnings.filterwarnings('ignore')

# Late imports are intentional: the warnings filter (and the matplotlib
# backend selection inside the analysis package) must run before the heavy
# third-party imports below them.
from histodae.data.datasets import ImageDataset, create_dataset
from histodae.models.factory import get_model
from histodae.training.hparams import generate_ablation_config, get_optimal_batch_size
from histodae.training.module import OptimizedMainModelWithAnalysis


def main(argv: list[str] | None = None):
    """Train entrypoint (also installed as `histodae-train`)."""
    import argparse

    from histodae.training.config import config_defaults
    from histodae.utils.reproducibility import create_experiment_config, set_seed

    # Optimized CUDA settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async CUDA operations

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train optimized DAE-KAN model with analysis')

    # Reproducibility arguments (Task 1)
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--config-path', type=str, default=None,
                        help='Experiment YAML config. When given, its values become the '
                             'defaults for the flags below (explicit CLI flags win). '
                             'Omit to use the built-in defaults.')
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

    # An explicit --config-path turns the YAML into the source of the flag
    # DEFAULTS; anything passed explicitly on the command line still wins
    # (argparse applies set_defaults before parse_args). Omitting the flag
    # keeps the historical built-in defaults untouched.
    pre, _ = parser.parse_known_args(argv)
    if pre.config_path:
        parser.set_defaults(**config_defaults(pre.config_path))

    args = parser.parse_args(argv)

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
