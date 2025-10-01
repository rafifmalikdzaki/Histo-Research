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
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.2, patience=5, min_lr=1e-6)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/loss'
            }
        }


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = ImageDataset(*create_dataset('train'), device)
    test_ds = ImageDataset(*create_dataset('test'), device)

    train_loader = DataLoader(train_ds, batch_size=12, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=12, shuffle=False)
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
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader, test_loader)
    wandb.finish()

