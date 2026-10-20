"""Safe W&B logging helpers for the in-training analyzer.

Every call is guarded: a W&B failure (offline mode, auth issues, a stale
step) must never take the training loop down.
"""

import os

import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not available - W&B logging will be disabled")

__all__ = ["WANDB_AVAILABLE", "WandbLoggingMixin"]


class WandbLoggingMixin:
    """Frequency-aware, exception-safe W&B logging."""

    def log_to_wandb_safe(self, data: dict, prefix: str = "", step: int = None):
        """Safely log data to W&B with error handling"""
        if not self.log_to_wandb:
            return

        try:
            # Add prefix to all keys
            if prefix:
                prefixed_data = {f"{prefix}/{k}": v for k, v in data.items()}
            else:
                prefixed_data = data

            # Log to W&B (skip if step is provided and might be outdated to prevent warnings)
            if step is not None and step < 0:
                return
            wandb.log(prefixed_data, step=step)
        except Exception as e:
            print(f"⚠ Failed to log to W&B: {e}")

    def log_image_to_wandb_safe(self, image_path: str, caption: str, step: int = None):
        """Safely log image to W&B with error handling"""
        if not self.log_to_wandb or not os.path.exists(image_path):
            return

        try:
            # Skip if step is provided and might be outdated to prevent warnings
            if step is not None and step < 0:
                return
            wandb.log({caption: wandb.Image(image_path)}, step=step)
        except Exception as e:
            print(f"⚠ Failed to log image {caption} to W&B: {e}")

    def log_table_to_wandb_safe(self, df: pd.DataFrame, table_name: str, step: int = None):
        """Safely log table to W&B with error handling"""
        if not self.log_to_wandb:
            return

        try:
            # Skip if step is provided and might be outdated to prevent warnings
            if step is not None and step < 0:
                return
            table = wandb.Table(dataframe=df)
            wandb.log({table_name: table}, step=step)
        except Exception as e:
            print(f"⚠ Failed to log table {table_name} to W&B: {e}")

    def should_log_to_wandb(self, log_type: str, batch_idx: int) -> bool:
        """Check if should log to W&B based on frequency"""
        if not self.log_to_wandb:
            return False

        frequency = self.wandb_log_frequencies.get(log_type, self.analysis_frequency)
        return batch_idx % frequency == 0
