#!/usr/bin/env python3
"""
Baseline Models for DAE-KAN Comparison

This module implements baseline models for comparison with the proposed DAE-KAN architecture:
- SimCLR (contrastive learning)
- BYOL (self-supervised learning)
- VAE (variational autoencoder)
- Single AE (ablation baseline)

Addresses Reviewer 1 #2 and Reviewer 2 requests for:
- Comparison against state-of-the-art self-supervised methods
- Proper ablation against standard autoencoder

Usage:
    # Run all baselines
    python baselines/run_all_baselines.py --dataset HeparUnifiedPNG
    
    # Run specific baseline
    python baselines/simclr_baseline.py --epochs 30 --batch-size 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================================
# SHARED ENCODER BACKBONE
# ============================================================================

class ResNetEncoder(nn.Module):
    """
    ResNet-style encoder for contrastive learning baselines.
    Simpler than DAE-KAN but standard for self-supervised learning.
    """
    def __init__(self, input_dim: int = 3, hidden_dims: list = [64, 128, 256, 512]):
        super().__init__()
        
        layers = []
        in_channels = input_dim
        
        for out_channels in hidden_dims:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


# ============================================================================
# SIMCLR BASELINE
# ============================================================================

class SimCLRModel(nn.Module):
    """
    SimCLR (Simple Framework for Contrastive Learning) baseline.
    
    Reference: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020.
    
    Architecture:
    - Encoder: ResNet-style CNN
    - Projector: MLP for contrastive loss
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 512, proj_dim: int = 128):
        super().__init__()
        self.encoder = ResNetEncoder(input_dim=input_dim, hidden_dims=[64, 128, 256, hidden_dim])
        
        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.out_dim = proj_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        projection = self.projector(features)
        return F.normalize(projection, dim=1)


class SimCLRLoss(nn.Module):
    """
    NT-Xent contrastive loss for SimCLR.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Contrastive features (2N, D) where N is batch size
        """
        batch_size = features.shape[0] // 2
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Positive pairs are (i, i+N) and (i+N, i)
        labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(features.device)
        
        # Mask out self-similarity
        mask = torch.eye(batch_size * 2, dtype=bool).to(features.device)
        labels = labels & ~mask
        
        # Compute loss
        log_softmax = F.log_softmax(similarity_matrix, dim=1)
        loss = -torch.sum(labels * log_softmax) / (2 * batch_size)
        
        return loss


# ============================================================================
# BYOL BASELINE
# ============================================================================

class BYOLModel(nn.Module):
    """
    BYOL (Bootstrap Your Own Latent) baseline.
    
    Reference: Grill et al., "Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning", NeurIPS 2020.
    
    Architecture:
    - Online network: Encoder + Projector + Predictor
    - Target network: EMA of online network
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 512, proj_dim: int = 256, pred_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder = ResNetEncoder(input_dim=input_dim, hidden_dims=[64, 128, 256, hidden_dim])
        
        # Projector
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pred_dim, proj_dim)
        )
        
        # Target network (initialized as copy of online)
        self.target_encoder = ResNetEncoder(input_dim=input_dim, hidden_dims=[64, 128, 256, hidden_dim])
        self.target_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Initialize target as copy of online
        self._initialize_target()
    
    def _initialize_target(self):
        """Initialize target network weights from online network."""
        for target_param, online_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data.copy_(online_param.data)
        
        for target_param, online_param in zip(self.target_projector.parameters(), self.projector.parameters()):
            target_param.data.copy_(online_param.data)
    
    @torch.no_grad()
    def _momentum_update(self, momentum: float = 0.99):
        """Update target network with momentum."""
        for target_param, online_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
        
        for target_param, online_param in zip(self.target_projector.parameters(), self.projector.parameters()):
            target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
    
    def forward_online(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        projected = self.projector(features)
        prediction = self.predictor(projected)
        return F.normalize(prediction, dim=1)
    
    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.target_encoder(x)
            projected = self.target_projector(features)
        return F.normalize(projected, dim=1)


class BYOLLoss(nn.Module):
    """
    MSE loss for BYOL (after normalization).
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, online_pred: torch.Tensor, target_pred: torch.Tensor) -> torch.Tensor:
        return 2 - 2 * (online_pred * target_pred).sum(dim=-1).mean()


# ============================================================================
# VAE BASELINE
# ============================================================================

class VAE(nn.Module):
    """
    Variational Autoencoder baseline.
    
    Reference: Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014.
    
    Architecture:
    - Encoder: CNN -> latent distribution (mu, log_var)
    - Decoder: Latent -> reconstruction
    """
    def __init__(self, input_dim: int = 3, latent_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_log_var = nn.Linear(256 * 16 * 16, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )
        
        self.latent_dim = latent_dim
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var


class VAELoss(nn.Module):
    """
    VAE loss = Reconstruction loss + KL divergence.
    """
    def __init__(self, kl_weight: float = 0.001):
        super().__init__()
        self.kl_weight = kl_weight
        self.mse = nn.MSELoss()
    
    def forward(self, x: torch.Tensor, reconstruction: torch.Tensor,
                mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        recon_loss = self.mse(reconstruction, x)
        
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        return recon_loss + self.kl_weight * kl_loss


# ============================================================================
# SINGLE AE BASELINE (Ablation)
# ============================================================================

class SingleAutoencoder(nn.Module):
    """
    Single Autoencoder baseline (ablation study).
    
    This is a simplified version without:
    - Dual encoder architecture
    - KAN layers
    - BAM attention
    - ECA attention
    
    Used to justify the proposed architectural choices.
    """
    def __init__(self, input_dim: int = 3, latent_dim: int = 128):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


# ============================================================================
# BASELINE FACTORY
# ============================================================================

BASELINE_MODELS = {
    'simclr': SimCLRModel,
    'byol': BYOLModel,
    'vae': VAE,
    'single_ae': SingleAutoencoder
}


def get_baseline(model_name: str, **kwargs):
    """
    Factory function to get baseline models.
    
    Args:
        model_name: Name of baseline model.
        **kwargs: Model-specific arguments.
    
    Returns:
        Baseline model instance.
    """
    if model_name not in BASELINE_MODELS:
        raise ValueError(f"Unknown baseline: {model_name}. Available: {list(BASELINE_MODELS.keys())}")
    
    return BASELINE_MODELS[model_name](**kwargs)


if __name__ == "__main__":
    # Test baseline models
    print("Testing baseline models...")
    
    x = torch.randn(4, 3, 128, 128)
    
    # SimCLR
    print("\n1. SimCLR:")
    simclr = get_baseline('simclr')
    proj = simclr(x)
    print(f"   Projection shape: {proj.shape}")
    
    # BYOL
    print("\n2. BYOL:")
    byol = get_baseline('byol')
    online_pred = byol.forward_online(x)
    target_pred = byol.forward_target(x)
    print(f"   Online prediction shape: {online_pred.shape}")
    print(f"   Target prediction shape: {target_pred.shape}")
    
    # VAE
    print("\n3. VAE:")
    vae = get_baseline('vae')
    recon, mu, log_var = vae(x)
    print(f"   Reconstruction shape: {recon.shape}")
    print(f"   Latent mu shape: {mu.shape}")
    
    # Single AE
    print("\n4. Single Autoencoder:")
    ae = get_baseline('single_ae')
    recon, encoded = ae(x)
    print(f"   Reconstruction shape: {recon.shape}")
    print(f"   Encoded shape: {encoded.shape}")
    
    print("\n✅ All baseline models tested successfully!")
