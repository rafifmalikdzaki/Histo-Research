import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from models.model import DAE_KAN_Attention
# from models.modelreg.modelv1 import DAE_KAN_Attention
from histodata import *
from torch.utils.data import Dataset, DataLoader
import wandb


class MainModel(pl.LightningModule):
    def __init__(self):
        super(MainModel, self).__init__()
        self.model = DAE_KAN_Attention()

    def forward(self, x):
        encoded, decoded, z = self.model(x)
        return encoded, decoded, z

    def training_step(self, batch, batch_idx):
        x, _ = batch
        encoded, decoded, z = self.forward(x)

        mse_loss = nn.functional.mse_loss(x, decoded)

        self.log("train/loss", mse_loss, on_epoch=True, prog_bar=True)
        return mse_loss

    def validation_step(self, batch ,batch_idx):
        x, _ = batch
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


if __name__ == '__main__':
    import argparse
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DAE-KAN model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA and use CPU')
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

    train_ds = ImageDataset(*create_dataset('train'), device)
    test_ds = ImageDataset(*create_dataset('test'), device)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)
    x, y = next(iter(train_loader))

    model = MainModel().to(device)
    wandb_logger = WandbLogger(project='histopath')
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
        accumulate_grad_batches=3,
    )

    trainer.fit(model, train_loader, test_loader)
    wandb.finish()

