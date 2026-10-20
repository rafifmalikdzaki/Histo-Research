"""Per-batch analysis: reconstruction quality and attention summaries."""

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

__all__ = ["BatchAnalysisMixin"]


class BatchAnalysisMixin:
    """Run one forward pass, summarise it, and accumulate the metrics."""

    def analyze_batch(self, batch_idx: int, input_tensor: torch.Tensor,
                     output_tensor: torch.Tensor, loss: float,
                     phase: str = "train", global_step: int = None) -> dict:
        """Analyze a single batch and return metrics"""

        # Clear attention data for this batch
        self.attention_data.clear()

        # Run forward pass to collect attention data
        with torch.no_grad():
            # Always call the actual DAE_KAN_Attention model
            actual_model = self.model.model if hasattr(self.model, 'model') else self.model
            _ = actual_model(input_tensor)

        # Compute basic metrics
        metrics = {
            'batch_idx': batch_idx,
            'phase': phase,
            'loss': loss,
            'timestamp': time.time(),
            'attention_layers': list(self.attention_data.keys())
        }

        # Add reconstruction metrics
        mse_loss = nn.functional.mse_loss(input_tensor, output_tensor)
        ssim_score = self._calculate_ssim(input_tensor, output_tensor)
        metrics['mse'] = float(mse_loss.item())
        metrics['ssim'] = float(ssim_score)

        # Analyze attention patterns
        for layer_name, attention_info in self.attention_data.items():
            if 'bam' in layer_name:  # Focus on BAM attention
                attention_map = attention_info['data']
                attention_metrics = self.compute_attention_metrics(attention_map)

                # Add layer-specific metrics
                for metric_name, value in attention_metrics.items():
                    metrics[f'attention_{layer_name}_{metric_name}'] = value

        # Store batch metrics
        self.batch_metrics.append(metrics)

        # Log to W&B if enabled and at the right frequency
        if self.should_log_to_wandb('metrics', batch_idx):
            wandb_metrics = {
                'loss': loss,
                'mse': metrics['mse'],
                'ssim': metrics['ssim']
            }

            # Add attention metrics
            for key, value in metrics.items():
                if 'attention_' in key and isinstance(value, int | float):
                    wandb_metrics[key] = value

            log_step = global_step if global_step is not None else batch_idx
            self.log_to_wandb_safe(wandb_metrics, f"{phase}/analysis", step=log_step)

        return metrics

    def _calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images"""
        # Ensure tensors are detached before computing
        img1 = img1.detach()
        img2 = img2.detach()

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

        ssim = ssim_num / (ssim_den + 1e-8)
        return float(torch.mean(ssim).item())

    def _estimate_epoch(self, batch_idx):
        """Estimate current epoch from batch index"""
        # Rough estimate assuming 4233 batches per epoch (from training data)
        batches_per_epoch = 4233
        epoch = batch_idx // batches_per_epoch + 1
        return epoch

    def get_batch_summary(self) -> dict:
        """Get summary of all batch metrics"""
        if not self.batch_metrics:
            return {}

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.batch_metrics)

        summary = {
            'total_batches': len(self.batch_metrics),
            'latest_batch': self.batch_metrics[-1]['batch_idx'],
            'latest_metrics': self.batch_metrics[-1],
        }

        # Compute statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['batch_idx', 'timestamp']:
                summary[f'{col}_mean'] = float(df[col].mean())
                summary[f'{col}_std'] = float(df[col].std())
                summary[f'{col}_min'] = float(df[col].min())
                summary[f'{col}_max'] = float(df[col].max())

        return summary

    def save_metrics(self, filename: str = None):
        """Save all metrics to file"""
        if filename is None:
            filename = f'metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        save_path = os.path.join(self.get_subdir_path('metrics'), filename)

        # Prepare data for JSON serialization
        data = {
            'batch_metrics': self.batch_metrics,
            'summary': self.get_batch_summary(),
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'total_batches': len(self.batch_metrics),
                'device': str(self.device)
            }
        }

        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        return save_path
