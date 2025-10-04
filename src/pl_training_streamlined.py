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
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import psutil
import json
import warnings
warnings.filterwarnings('ignore')

# Automatic analysis module
from analysis.auto_analysis import AutomaticAnalyzer


class AnalysisEnabledMainModel(pl.LightningModule):
    """
    Lightning module with automatic analysis integration
    """

    def __init__(self, model_name: str = "dae_kan_attention", batch_size=8, analysis_frequency=100):
        super(AnalysisEnabledMainModel, self).__init__()
        self.model = get_model(model_name)()
        self.batch_size = batch_size
        self.analysis_frequency = analysis_frequency  # Run full analysis every N batches

        # Initialize automatic analyzer
        self.auto_analyzer = None

        # Store basic metrics
        self.batch_losses = []
        self.reconstruction_metrics = []

        # Training timing and performance tracking
        self.training_times = []
        self.inference_times = []
        self.memory_usage = []

    def on_fit_start(self):
        """Initialize automatic analyzer after model is on correct device"""
        device = self.device

        # Create analysis directory (will be updated after logger is set)
        analysis_dir = "auto_analysis"

        # Initialize automatic analyzer
        self.auto_analyzer = AutomaticAnalyzer(
            model=self.model,
            device=str(device),
            save_dir=analysis_dir
        )

        print(f"Automatic analyzer initialized. Saving to: {analysis_dir}")

    def setup_analysis_dir_after_logger(self):
        """Update analysis directory after logger is initialized"""
        if hasattr(self, 'logger') and self.logger is not None and hasattr(self.logger, 'experiment'):
            run_name = self.logger.experiment.name
            analysis_dir = f"auto_analysis/{run_name}"
            os.makedirs(analysis_dir, exist_ok=True)

            # Move any existing files from auto_analysis to run-specific directory
            import shutil
            temp_dir = "auto_analysis_temp"
            if os.path.exists("auto_analysis") and not os.path.exists(analysis_dir):
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                # Rename auto_analysis to temp
                os.rename("auto_analysis", temp_dir)
                # Create new auto_analysis directory
                os.makedirs("auto_analysis", exist_ok=True)
                # Move contents to run-specific directory
                if os.path.exists(temp_dir):
                    for item in os.listdir(temp_dir):
                        shutil.move(os.path.join(temp_dir, item), analysis_dir)
                    # Remove temp directory
                    shutil.rmtree(temp_dir)

            # Update the analyzer save directory
            self.auto_analyzer.save_dir = analysis_dir
            print(f"Updated analysis directory to: {analysis_dir}")

            # Create paper-ready directory structure
            paper_dir = os.path.join(analysis_dir, "paper_figures")
            os.makedirs(paper_dir, exist_ok=True)
            print(f"Paper figures directory: {paper_dir}")

    def forward(self, x):
        encoded, decoded, z = self.model(x)
        return encoded, decoded, z

    def training_step(self, batch, batch_idx):
        # Track training time
        training_start_time = time.perf_counter()

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
        inference_time = (time.perf_counter() - inference_start_time) * 1000  # ms

        # Compute loss
        mse_loss = nn.functional.mse_loss(x, decoded)

        # Store basic metrics
        self.batch_losses.append(mse_loss.item())

        # Calculate additional metrics
        ssim_score = self.calculate_ssim(x, decoded)
        reconstruction_error = torch.mean((x - decoded) ** 2, dim=[1, 2, 3])

        self.reconstruction_metrics.append({
            'mse': mse_loss.item(),
            'ssim': ssim_score if isinstance(ssim_score, float) else ssim_score.item(),
            'error_std': reconstruction_error.std().item(),
            'error_mean': reconstruction_error.mean().item()
        })

        # Track timing and memory
        training_time = (time.perf_counter() - training_start_time) * 1000  # ms
        self.training_times.append(training_time)
        self.inference_times.append(inference_time)

        if torch.cuda.is_available():
            memory_after = torch.cuda.memory_allocated(self.device)
            memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
            self.memory_usage.append(memory_used)

        # Log basic metrics
        self.log("train/loss", mse_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/ssim", ssim_score, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/reconstruction_error", reconstruction_error.mean(), on_epoch=True, sync_dist=True)

        # Log timing metrics
        self.log("train/training_time_ms", training_time, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/inference_time_ms", inference_time, on_epoch=True, prog_bar=False, sync_dist=True)
        if torch.cuda.is_available():
            self.log("train/memory_mb", memory_used, on_epoch=True, prog_bar=False, sync_dist=True)

        # Run automatic analysis every batch
        if self.auto_analyzer:
            try:
                # Analyze current batch
                batch_metrics = self.auto_analyzer.analyze_batch(
                    batch_idx=batch_idx,
                    input_tensor=x,
                    output_tensor=decoded,
                    loss=mse_loss.item(),
                    phase="train"
                )

                # Log key attention metrics to W&B
                if batch_idx % 20 == 0:  # Log detailed metrics every 20 batches
                    for metric_name, value in batch_metrics.items():
                        if 'attention_' in metric_name and isinstance(value, (int, float)):
                            self.log(f"train/{metric_name}", value, on_step=True, on_epoch=False, sync_dist=False)

                # Create visualization every N batches
                if batch_idx % self.analysis_frequency == 0 and batch_idx > 0:
                    # Create comprehensive paper-ready dashboard
                    dashboard_result = self.auto_analyzer.create_comprehensive_paper_dashboard(
                        batch_idx=batch_idx,
                        input_tensor=x,
                        output_tensor=decoded,
                        phase="train"
                    )

                    # Log individual paper figures to W&B
                    if hasattr(self, 'logger') and self.logger is not None:
                        for fig_name, fig_path in dashboard_result['individual_figures'].items():
                            self.logger.experiment.log({
                                f'train/{fig_name}_{batch_idx}': wandb.Image(fig_path)
                            })

                    # Save metrics
                    self.auto_analyzer.save_metrics()

                    print(f"✓ Individual figures analysis completed for batch {batch_idx}")
                    print(f"  📄 Individual figures: {len(dashboard_result['individual_figures'])} generated")
                    for fig_name, fig_path in dashboard_result['individual_figures'].items():
                        print(f"    - {fig_name}: {fig_path}")

            except Exception as e:
                print(f"⚠ Analysis failed for batch {batch_idx}: {e}")
                # Don't crash training if analysis fails
                pass

        # Log basic training metrics
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}: Loss={mse_loss.item():.4f}, SSIM={ssim_score:.4f}")

        return mse_loss

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
                    phase="val"
                )

                # Log key attention metrics to W&B
                for metric_name, value in batch_metrics.items():
                    if 'attention_' in metric_name and isinstance(value, (int, float)):
                        self.log(f"val/{metric_name}", value, on_step=True, on_epoch=False, sync_dist=False)

                # Create comprehensive visualization for validation
                dashboard_result = self.auto_analyzer.create_comprehensive_paper_dashboard(
                    batch_idx=batch_idx,
                    input_tensor=x,
                    output_tensor=decoded,
                    phase="val"
                )

                # Log individual paper figures to W&B for validation
                if hasattr(self, 'logger') and self.logger is not None:
                    for fig_name, fig_path in dashboard_result['individual_figures'].items():
                        self.logger.experiment.log({
                            f'val/{fig_name}_{batch_idx}': wandb.Image(fig_path)
                        })

                print(f"✓ Validation individual figures analysis completed for batch {batch_idx}")
                print(f"  📄 Individual figures: {len(dashboard_result['individual_figures'])} generated")
                for fig_name, fig_path in dashboard_result['individual_figures'].items():
                    print(f"    - {fig_name}: {fig_path}")

            except Exception as e:
                print(f"⚠ Validation analysis failed for batch {batch_idx}: {e}")
                # Don't crash validation if analysis fails
                pass

        # Log basic validation metrics
        if batch_idx == 0:
            print(f"Validation: Loss={mse_loss.item():.4f}, SSIM={ssim_score:.4f}")

        return mse_loss

    def calculate_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Calculate SSIM between two images"""
        # Simple SSIM calculation
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
        return float(torch.mean(ssim))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.002,
            weight_decay=1e-5,
        )

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
        torch.set_float32_matmul_precision('high')

    def on_train_epoch_end(self):
        # Save final metrics for the epoch
        if self.auto_analyzer:
            try:
                # Save metrics after each epoch
                save_path = self.auto_analyzer.save_metrics(f"epoch_{self.current_epoch}_metrics.json")

                # Log summary to W&B
                summary = self.auto_analyzer.get_batch_summary()
                if summary and hasattr(self, 'logger') and self.logger is not None:
                    for key, value in summary.items():
                        if isinstance(value, (int, float)):
                            self.log(f"epoch/{key}", value, on_epoch=True, sync_dist=False)

                print(f"✓ Epoch {self.current_epoch} metrics saved")

            except Exception as e:
                print(f"⚠ Failed to save epoch metrics: {e}")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

    def on_train_end(self):
        # Final cleanup and summary
        if self.auto_analyzer:
            try:
                # Save final comprehensive metrics
                final_metrics = self.auto_analyzer.get_batch_summary()

                # Create final summary report
                self.create_final_summary_report(final_metrics)

                # Save all data
                self.auto_analyzer.save_metrics("final_training_metrics.json")

                print("✓ Final analysis completed")

            except Exception as e:
                print(f"⚠ Final analysis failed: {e}")
            finally:
                # Cleanup
                self.auto_analyzer.cleanup()

    def create_final_summary_report(self, final_metrics):
        """Create final summary report"""
        if not final_metrics:
            return

        save_dir = self.auto_analyzer.save_dir if self.auto_analyzer else "final_analysis"
        os.makedirs(save_dir, exist_ok=True)

        # Create text report
        report = []
        report.append("# DAE-KAN Training Final Summary\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Training Summary\n")
        report.append(f"- **Total Batches**: {final_metrics.get('total_batches', 'N/A')}")
        report.append(f"- **Final Batch**: {final_metrics.get('latest_batch', 'N/A')}")

        if 'latest_metrics' in final_metrics:
            latest = final_metrics['latest_metrics']
            report.append(f"- **Final Loss**: {latest.get('loss', 'N/A'):.6f}")
            report.append(f"- **Final MSE**: {latest.get('mse', 'N/A'):.6f}")
            report.append(f"- **Final SSIM**: {latest.get('ssim', 'N/A'):.4f}")

        report.append("\n## Attention Analysis Summary\n")
        attention_metrics = [k for k in final_metrics.keys() if 'attention_' in k and 'mean' in k]
        if attention_metrics:
            report.append("Key attention metrics tracked:")
            for metric in sorted(attention_metrics)[:10]:  # Top 10 metrics
                clean_name = metric.replace('attention_', '').replace('_mean', '')
                value = final_metrics[metric]
                report.append(f"- **{clean_name}**: {value:.4f}")

        report.append(f"\n## Files Generated\n")
        report.append(f"- **Analysis Directory**: {save_dir}")
        report.append(f"- **Metrics JSON**: final_training_metrics.json")
        report.append(f"- **Visualizations**: Batch analysis images")

        if self.training_times and self.inference_times:
            report.append(f"\n## Performance Summary\n")
            report.append(f"- **Avg Training Time**: {np.mean(self.training_times):.2f} ms")
            report.append(f"- **Avg Inference Time**: {np.mean(self.inference_times):.2f} ms")
            report.append(f"- **Throughput**: {1000/np.mean(self.inference_times):.1f} images/sec")

            if self.memory_usage:
                report.append(f"- **Avg Memory Usage**: {np.mean(self.memory_usage):.1f} MB")

        report_text = "\n".join(report)

        # Save report
        report_path = os.path.join(save_dir, "final_summary_report.md")
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"✓ Final summary report saved: {report_path}")


def get_optimal_batch_size(model, device, input_size=(3, 128, 128)):
    """Dynamically determine optimal batch size based on available GPU memory"""
    model.eval()
    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)
        available_memory = gpu_memory - reserved_memory - allocated_memory

        sample_size = torch.prod(torch.tensor(input_size)).item() * 4
        batch_size = 4
        max_batch_size = 32

        while batch_size <= max_batch_size:
            try:
                test_input = torch.randn(batch_size, *input_size).to(device)
                with torch.no_grad():
                    _ = model(test_input)

                del test_input
                batch_size *= 2

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    break
                else:
                    raise e

        return max(4, batch_size // 2)
    else:
        return 4


if __name__ == '__main__':
    import argparse
    import pandas as pd

    # Optimized CUDA settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DAE-KAN with automatic analysis')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA and use CPU')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (auto-detected if not specified)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--analysis-freq', type=int, default=100, help='Run full analysis every N batches')
    parser.add_argument('--project-name', type=str, default='histopath-kan-auto', help='W&B project name')
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

    # Create data loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else 2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
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

    # Initialize model with automatic analysis
    model = AnalysisEnabledMainModel(model_name=args.model_name, batch_size=batch_size, analysis_frequency=args.analysis_freq)
    model = model.to(device)

    # Setup W&B logging with comprehensive configuration
    wandb_logger = WandbLogger(
        project=args.project_name,
        group=args.model_name,
        tags=['DAE-KAN', 'histopathology', 'autoencoder', 'automatic-analysis', args.model_name],
        config=args
    )
    wandb_logger.watch(model, log='all', log_freq=10)


    # Setup callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='./checkpoints_auto_analysis',
        filename='best-checkpoint-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=30,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator='gpu' if gpu_id is not None else 'cpu',
        devices=[gpu_id] if gpu_id is not None else 1,
        log_every_n_steps=20,
        precision='32-true',  # Use 32-bit precision to avoid mixed precision issues
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        deterministic=False,
        benchmark=True,
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    # Start training
    print(f"Starting training with automatic analysis")
    print(f"Batch size: {batch_size}, Analysis frequency: every {args.analysis_freq} batches")
    print(f"W&B project: {args.project_name}")

    trainer.fit(model, train_loader, test_loader)

    # Final cleanup
    print("Training completed. Automatic analysis results are saved in the analysis directory.")

    # Cleanup
    wandb.finish()
    gc.collect()
    torch.cuda.empty_cache()