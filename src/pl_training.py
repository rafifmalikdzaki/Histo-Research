import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from models.factory import get_model
from histodata import *
from torch.utils.data import Dataset, DataLoader
import wandb
import time
import numpy as np
from thop import profile

class MainModel(pl.LightningModule):
    def __init__(self, model_name: str = "dae_kan_attention"):
        super(MainModel, self).__init__()
        self.model = get_model(model_name)()

    def forward(self, x):
        encoded, decoded, z = self.model(x)
        return encoded, decoded, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.to(self.device, non_blocking=True)  # Move to GPU in training step
        encoded, decoded, z = self.forward(x)

        mse_loss = nn.functional.mse_loss(x, decoded)

        # Only free memory periodically, not every step
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        self.log("train/loss", mse_loss, on_epoch=True, prog_bar=True)
        return mse_loss

    def validation_step(self, batch ,batch_idx):
        x, _ = batch
        x = x.to(self.device, non_blocking=True)  # Move to GPU in validation step
        encoded, decoded, z = self.forward(x)

        mse_loss = nn.functional.mse_loss(x, decoded)

        self.log("val/loss", mse_loss, on_epoch=True, prog_bar=True)
        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.2, patience=5, min_lr=1e-6)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


class AnalysisModel(pl.LightningModule):
    def __init__(self, model_name: str = "dae_kan_attention"):
        super(AnalysisModel, self).__init__()
        self.model = get_model(model_name)()
        self.training_times = []
        self.inference_times = []
        self.model_complexity = {}

    def forward(self, x):
        encoded, decoded, z = self.model(x)
        return encoded, decoded, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.to(self.device, non_blocking=True)
        
        start_time = time.time()
        encoded, decoded, z = self.forward(x)
        self.inference_times.append(time.time() - start_time)
        
        mse_loss = nn.functional.mse_loss(x, decoded)

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        self.log("train/loss", mse_loss, on_epoch=True, prog_bar=True)
        return mse_loss

    def on_train_epoch_start(self):
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = time.time() - self.epoch_start_time
        self.training_times.append(epoch_time)
        self.log('train/epoch_time', epoch_time)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.to(self.device, non_blocking=True)
        
        start_time = time.time()
        encoded, decoded, z = self.forward(x)
        self.log('val/inference_time', time.time() - start_time)
        
        mse_loss = nn.functional.mse_loss(x, decoded)

        self.log("val/loss", mse_loss, on_epoch=True, prog_bar=True)
        return mse_loss

    def on_train_end(self):
        self.log_analysis()

    def log_analysis(self):
        # Training time analysis
        avg_training_time = np.mean(self.training_times)
        self.logger.experiment.summary["avg_training_time_per_epoch"] = avg_training_time

        # Inference time analysis
        avg_inference_time = np.mean(self.inference_times)
        self.logger.experiment.summary["avg_inference_time_per_batch"] = avg_inference_time

        # Model complexity
        input_sample = torch.randn(1, 3, 128, 128).to(self.device)
        macs, params = profile(self.model, inputs=(input_sample,))
        self.logger.experiment.summary["model_macs"] = macs
        self.logger.experiment.summary["model_params"] = params

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.2, patience=5, min_lr=1e-6)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


if __name__ == '__main__':
    import argparse
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DAE-KAN model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA and use CPU')
    parser.add_argument('--analysis', action='store_true', help='Run with analysis model')
    parser.add_argument('--model-name', type=str, default="dae_kan_attention", help='Name of the model to use')
    args = parser.parse_args()

    # Set device based on arguments
    if args.no_cuda:
        device = torch.device('cpu')
        gpu_id = None
    else:
        if torch.cuda.is_available():
            gpu_id = args.gpu
            device = torch.device(f'cuda:{gpu_id}')
            print(f"Using GPU {gpu_id}")
        else:
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
            gpu_id = None

    train_ds = ImageDataset(*create_dataset('train'))
    test_ds = ImageDataset(*create_dataset('test'))

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, persistent_workers=False, prefetch_factor=2)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2, persistent_workers=False, prefetch_factor=2)
    x, y = next(iter(train_loader))

    if args.analysis:
        model = AnalysisModel(model_name=args.model_name)
        project_name = 'histopath-ablation'
        tags = ['analysis', args.model_name]
    else:
        model = MainModel(model_name=args.model_name)
        project_name = 'histopath-ablation'
        tags = ['training', args.model_name]

    torch.cuda.empty_cache()
    model = model.to(device)
    wandb_logger = WandbLogger(
        project=project_name, 
        group=args.model_name, 
        tags=tags,
        config=args
    )
    wandb_logger.watch(model, log='all', log_freq=10)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='./checkpoints',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=30,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu' if gpu_id is not None else 'cpu',
        devices=[gpu_id] if gpu_id is not None else 1,
        log_every_n_steps=10,
        precision='16-mixed',
        accumulate_grad_batches=4,
        gradient_clip_val=1.0,
    )

    trainer.fit(model, train_loader, test_loader)
    wandb.finish()

