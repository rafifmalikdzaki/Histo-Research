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


class OptimizedMainModel(pl.LightningModule):
    """Optimized Lightning module with better memory management and GPU utilization"""

    def __init__(self, model_name: str = "dae_kan_attention", batch_size=8):
        super(OptimizedMainModel, self).__init__()
        self.model = get_model(model_name)()
        self.batch_size = batch_size

    def forward(self, x):
        encoded, decoded, z = self.model(x)
        return encoded, decoded, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        # Use non-blocking transfer for better GPU utilization
        x = x.to(self.device, non_blocking=True)

        encoded, decoded, z = self.forward(x)

        # Compute loss
        mse_loss = nn.functional.mse_loss(x, decoded)

        # Simplified loss computation for speed
        self.log("train/loss", mse_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.to(self.device, non_blocking=True)
        encoded, decoded, z = self.forward(x)

        mse_loss = nn.functional.mse_loss(x, decoded)

        self.log("val/loss", mse_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        return mse_loss

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

    def on_train_epoch_start(self):
        """Enable optimization at the start of each epoch"""
        # Enable mixed precision training
        torch.set_float32_matmul_precision('high')

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Optimize memory after each batch"""
        # Periodic memory cleanup (less frequent than before)
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()


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
    import os

    # Optimized CUDA settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Enable async CUDA operations

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train optimized DAE-KAN model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA and use CPU')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto-detected if not specified)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--model-name', type=str, default="dae_kan_attention", help='Name of the model to use')
    args = parser.parse_args()

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

    # Test data loading
    x, y = next(iter(train_loader))
    print(f"Data loading successful. Batch shape: {x.shape}")

    # Initialize optimized model
    model = OptimizedMainModel(model_name=args.model_name, batch_size=batch_size)
    model = model.to(device)

    # Setup logging
    wandb_logger = WandbLogger(
        project='histopath-ablation',
        group=args.model_name,
        tags=['optimized', args.model_name],
        config=args
    )
    wandb_logger.watch(model, log='all', log_freq=10)

    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='./checkpoints_optimized',
        filename='best-checkpoint-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )

    
    # Create optimized trainer
    trainer = pl.Trainer(
        max_epochs=30,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
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
    )

    # Start training
    print(f"Starting training with batch size {batch_size} on {device}")
    trainer.fit(model, train_loader, test_loader)

    # Cleanup
    wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()
