#!/usr/bin/env python3
"""
Runtime and Memory Profiling Script for DAE-KAN Models

This script profiles the computational performance of DAE-KAN models, including:
- Training time per epoch
- Inference time (encoder forward pass only, and full pipeline)
- GPU memory usage (peak, allocated, reserved)
- Hardware specifications (GPU model, VRAM, etc.)
- MACs (Multiply-Accumulate Operations) and parameter count

Addresses Reviewer 1 #7: Runtime and memory analysis with hardware specifications

Usage:
    # Profile a single model
    python profile_runtime.py --model-name dae_kan_attention --batch-size 8
    
    # Compare multiple batch sizes
    python profile_runtime.py --model-name dae_kan_attention --batch-sizes 4 8 16
    
    # Profile all model variants
    python profile_runtime.py --all-models --output outputs/profiling/all_models_profile.csv
    
    # Include clustering overhead in timing
    python profile_runtime.py --model-name dae_kan_attention --profile-clustering
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.factory import get_model
from utils.reproducibility import set_seed


def get_gpu_specs() -> Dict[str, any]:
    """
    Get GPU hardware specifications.
    
    Returns:
        Dictionary containing GPU information.
    """
    specs = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_name': 'N/A',
        'gpu_memory_total_gb': 'N/A',
        'gpu_memory_allocated_gb': 0.0,
        'gpu_memory_reserved_gb': 0.0,
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
        'pytorch_version': torch.__version__,
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        specs['gpu_name'] = torch.cuda.get_device_name(device_id)
        
        # Get total memory
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        specs['gpu_memory_total_gb'] = round(total_memory / (1024**3), 2)
        
        # Get current memory usage
        specs['gpu_memory_allocated_gb'] = round(torch.cuda.memory_allocated(device_id) / (1024**3), 4)
        specs['gpu_memory_reserved_gb'] = round(torch.cuda.memory_reserved(device_id) / (1024**3), 4)
    
    return specs


def get_model_complexity(model: nn.Module, input_size: Tuple[int, ...] = (1, 3, 128, 128)) -> Dict[str, any]:
    """
    Get model complexity metrics (parameters and MACs).
    
    Args:
        model: PyTorch model to profile.
        input_size: Input tensor size (batch, channels, height, width).
    
    Returns:
        Dictionary containing parameter count and MACs.
    """
    try:
        from thop import profile
        
        # Create dummy input
        x = torch.randn(input_size)
        
        # Profile model
        macs, params = profile(model, inputs=(x,), verbose=False)
        
        return {
            'params_millions': round(params / 1e6, 2),
            'macs_giga': round(macs / 1e9, 2),
            'flops_giga': round(2 * macs / 1e9, 2)  # Approximate FLOPs = 2 * MACs
        }
    
    except ImportError:
        # Fallback: count parameters only
        params = sum(p.numel() for p in model.parameters())
        return {
            'params_millions': round(params / 1e6, 2),
            'macs_giga': 'N/A',
            'flops_giga': 'N/A'
        }
    except Exception as e:
        print(f"⚠️  Model complexity profiling failed: {e}")
        return {
            'params_millions': 'N/A',
            'macs_giga': 'N/A',
            'flops_giga': 'N/A'
        }


def profile_inference_time(model: nn.Module, input_tensor: torch.Tensor, 
                          n_runs: int = 100, warmup: int = 10) -> Dict[str, float]:
    """
    Profile inference time (encoder forward pass only).
    
    Args:
        model: PyTorch model to profile.
        input_tensor: Input tensor for inference.
        n_runs: Number of inference runs to average.
        warmup: Number of warmup runs.
    
    Returns:
        Dictionary containing timing statistics.
    """
    device = next(model.parameters()).device
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Profile
    times = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(input_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds
    
    times = np.array(times)
    
    return {
        'inference_time_ms_mean': round(float(np.mean(times)), 4),
        'inference_time_ms_std': round(float(np.std(times)), 4),
        'inference_time_ms_min': round(float(np.min(times)), 4),
        'inference_time_ms_max': round(float(np.max(times)), 4),
        'inference_fps': round(1000.0 / float(np.mean(times)), 2)  # Frames per second
    }


def profile_full_pipeline(model: nn.Module, input_tensor: torch.Tensor,
                         include_clustering: bool = False, n_clusters: int = 5) -> Dict[str, float]:
    """
    Profile full pipeline time (encoder + decoder + clustering if requested).
    
    Args:
        model: PyTorch model to profile.
        input_tensor: Input tensor.
        include_clustering: Whether to include clustering in timing.
        n_clusters: Number of clusters for clustering.
    
    Returns:
        Dictionary containing full pipeline timing.
    """
    device = next(model.parameters()).device
    model.eval()
    
    n_runs = 10
    times = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            # Forward pass
            encoded, decoded, z = model(input_tensor)
            
            # Optional clustering
            if include_clustering:
                from sklearn.cluster import KMeans
                z_flat = z.view(z.size(0), -1).cpu().numpy()
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                _ = kmeans.fit_predict(z_flat)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    times = np.array(times)
    
    return {
        'full_pipeline_ms_mean': round(float(np.mean(times)), 4),
        'full_pipeline_ms_std': round(float(np.std(times)), 4),
        'includes_clustering': include_clustering
    }


def profile_memory_usage(model: nn.Module, input_tensor: torch.Tensor, 
                        batch_size: int) -> Dict[str, float]:
    """
    Profile GPU memory usage.
    
    Args:
        model: PyTorch model.
        input_tensor: Input tensor.
        batch_size: Batch size used.
    
    Returns:
        Dictionary containing memory statistics.
    """
    if not torch.cuda.is_available():
        return {
            'memory_allocated_mb': 0.0,
            'memory_reserved_mb': 0.0,
            'memory_peak_mb': 0.0,
            'memory_per_sample_mb': 0.0
        }
    
    device = next(model.parameters()).device
    model.eval()
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Get memory stats
    memory_allocated = torch.cuda.memory_allocated(device) / (1024**2)  # MB
    memory_reserved = torch.cuda.memory_reserved(device) / (1024**2)  # MB
    memory_peak = torch.cuda.max_memory_allocated(device) / (1024**2)  # MB
    
    return {
        'memory_allocated_mb': round(float(memory_allocated), 2),
        'memory_reserved_mb': round(float(memory_reserved), 2),
        'memory_peak_mb': round(float(memory_peak), 2),
        'memory_per_sample_mb': round(float(memory_peak) / batch_size, 2)
    }


def profile_training_epoch(model: nn.Module, train_loader: DataLoader, 
                          criterion: nn.Module, optimizer: torch.optim.Optimizer,
                          device: torch.device) -> Dict[str, float]:
    """
    Profile training time for one epoch.
    
    Args:
        model: PyTorch model.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
    
    Returns:
        Dictionary containing training timing statistics.
    """
    model.train()
    
    times = []
    losses = []
    
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        # Forward pass
        optimizer.zero_grad()
        _, decoded, _ = model(x)
        loss = criterion(x, decoded)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # ms
        losses.append(loss.item())
    
    times = np.array(times)
    
    return {
        'epoch_time_ms_total': round(float(np.sum(times)), 2),
        'epoch_time_s': round(float(np.sum(times)) / 1000, 2),
        'batch_time_ms_mean': round(float(np.mean(times)), 4),
        'batch_time_ms_std': round(float(np.std(times)), 4),
        'batches_per_second': round(1000.0 / float(np.mean(times)), 2),
        'loss_mean': round(float(np.mean(losses)), 6)
    }


def run_full_profiling(model_name: str, batch_size: int, dataset: str = 'HeparUnifiedPNG',
                      include_clustering: bool = False, seed: int = 42) -> Dict[str, any]:
    """
    Run comprehensive profiling for a model.
    
    Args:
        model_name: Name of the model to profile.
        batch_size: Batch size for profiling.
        dataset: Dataset name.
        include_clustering: Whether to include clustering in timing.
        seed: Random seed.
    
    Returns:
        Dictionary containing all profiling results.
    """
    print(f"\n{'='*80}")
    print(f"🔬 Profiling Model: {model_name}")
    print(f"{'='*80}")
    print(f"Batch Size: {batch_size}")
    print(f"Dataset: {dataset}")
    print(f"Include Clustering: {include_clustering}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")
    
    # Set seed
    set_seed(seed)
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading model: {model_name}...")
    model = get_model(model_name)()
    model = model.to(device)
    model.eval()
    
    # Get GPU specs
    print("\n📊 GPU Specifications:")
    gpu_specs = get_gpu_specs()
    for key, value in gpu_specs.items():
        print(f"  {key}: {value}")
    
    # Get model complexity
    print("\n📈 Model Complexity:")
    input_size = (1, 3, 128, 128)
    complexity = get_model_complexity(model, input_size)
    for key, value in complexity.items():
        print(f"  {key}: {value}")
    
    # Create dummy data
    print(f"\n⏱️  Running inference profiling...")
    dummy_input = torch.randn(batch_size, 3, 128, 128).to(device)
    
    # Profile inference
    inference_stats = profile_inference_time(model, dummy_input, n_runs=100, warmup=10)
    print(f"  Inference time: {inference_stats['inference_time_ms_mean']:.4f} ± {inference_stats['inference_time_ms_std']:.4f} ms")
    print(f"  Throughput: {inference_stats['inference_fps']:.2f} FPS")
    
    # Profile full pipeline
    print(f"\n⏱️  Running full pipeline profiling...")
    pipeline_stats = profile_full_pipeline(model, dummy_input, include_clustering=include_clustering)
    print(f"  Full pipeline: {pipeline_stats['full_pipeline_ms_mean']:.4f} ± {pipeline_stats['full_pipeline_ms_std']:.4f} ms")
    
    # Profile memory
    print(f"\n💾 Profiling memory usage...")
    memory_stats = profile_memory_usage(model, dummy_input, batch_size)
    print(f"  Peak memory: {memory_stats['memory_peak_mb']:.2f} MB")
    print(f"  Memory per sample: {memory_stats['memory_per_sample_mb']:.2f} MB")
    
    # Profile training (single epoch)
    print(f"\n⏱️  Profiling training (1 epoch)...")
    model.train()
    train_data = TensorDataset(torch.randn(100, 3, 128, 128), torch.randint(0, 5, (100,)))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    
    training_stats = profile_training_epoch(model, train_loader, criterion, optimizer, device)
    print(f"  Epoch time: {training_stats['epoch_time_s']:.2f} seconds")
    print(f"  Batches/sec: {training_stats['batches_per_second']:.2f}")
    print(f"  Mean loss: {training_stats['loss_mean']:.6f}")
    
    # Compile results
    results = {
        'model_name': model_name,
        'batch_size': batch_size,
        'dataset': dataset,
        'seed': seed,
        'timestamp': datetime.now().isoformat(),
        **gpu_specs,
        **complexity,
        **inference_stats,
        **pipeline_stats,
        **memory_stats,
        **training_stats
    }
    
    print(f"\n✅ Profiling complete for {model_name}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Runtime and memory profiling for DAE-KAN models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Profile single model
  python profile_runtime.py --model-name dae_kan_attention --batch-size 8
  
  # Compare batch sizes
  python profile_runtime.py --model-name dae_kan_attention --batch-sizes 4 8 16
  
  # Profile all models
  python profile_runtime.py --all-models --output outputs/profiling/all_models.csv
  
  # Include clustering overhead
  python profile_runtime.py --model-name dae_kan_attention --profile-clustering
        """
    )
    
    # Model selection
    parser.add_argument(
        '--model-name',
        type=str,
        default='dae_kan_attention',
        help='Model to profile'
    )
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Profile all model variants'
    )
    
    # Data settings
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for profiling'
    )
    parser.add_argument(
        '--batch-sizes',
        type=int,
        nargs='+',
        default=None,
        help='Multiple batch sizes to test'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='HeparUnifiedPNG',
        help='Dataset name'
    )
    
    # Profiling options
    parser.add_argument(
        '--profile-clustering',
        action='store_true',
        help='Include clustering in pipeline timing'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    
    # Output
    parser.add_argument(
        '--output',
        type=str,
        default='outputs/profiling/runtime_report.csv',
        help='Output CSV path'
    )
    
    args = parser.parse_args()
    
    # Determine models to profile
    if args.all_models:
        models_to_profile = [
            'dae_kan_attention', 'baseline', 'bam_only', 'kan_only',
            'no_bam', 'no_kan', 'no_eka'
        ]
    else:
        models_to_profile = [args.model_name]
    
    # Determine batch sizes
    if args.batch_sizes:
        batch_sizes = args.batch_sizes
    else:
        batch_sizes = [args.batch_size]
    
    print("🔬 DAE-KAN Runtime & Memory Profiling")
    print("="*80)
    print(f"Models to profile: {models_to_profile}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Include clustering: {args.profile_clustering}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Run profiling
    all_results = []
    
    for model_name in models_to_profile:
        for batch_size in batch_sizes:
            try:
                results = run_full_profiling(
                    model_name=model_name,
                    batch_size=batch_size,
                    dataset=args.dataset,
                    include_clustering=args.profile_clustering,
                    seed=args.seed
                )
                all_results.append(results)
            except Exception as e:
                print(f"❌ Failed to profile {model_name} (batch_size={batch_size}): {e}")
                all_results.append({
                    'model_name': model_name,
                    'batch_size': batch_size,
                    'error': str(e)
                })
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Create output directory
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"\n✅ Results saved to: {output_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print("📊 PROFILING SUMMARY")
        print("="*80)
        
        # Select key columns
        summary_cols = [
            'model_name', 'batch_size', 'params_millions', 'macs_giga',
            'inference_time_ms_mean', 'full_pipeline_ms_mean',
            'memory_peak_mb', 'epoch_time_s'
        ]
        
        available_cols = [col for col in summary_cols if col in df.columns]
        
        if available_cols:
            print(df[available_cols].to_string(index=False))
        
        print("="*80)


if __name__ == "__main__":
    main()
