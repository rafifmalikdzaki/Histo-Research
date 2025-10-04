import torch
import torch.nn as nn
import time
import gc
from typing import Dict, List, Tuple
import numpy as np

# Import both original and optimized models
from src.models.model import DAE_KAN_Attention as OriginalModel
from src.models.model_optimized import OptimizedDAE_KAN_Attention as OptimizedModel


def get_gpu_memory_info():
    """Get GPU memory information"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


def benchmark_model(model, input_tensor, num_runs=10, warmup_runs=3):
    """
    Benchmark a model's performance

    Returns:
        Dict with timing, memory, and throughput metrics
    """
    model.eval()
    device = input_tensor.device

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    # Reset memory counters
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Timing runs
    times = []
    memory_usage = []

    with torch.no_grad():
        for _ in range(num_runs):
            # Clear cache before each run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Record start memory
            start_memory = get_gpu_memory_info()

            # Time the forward pass
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

                outputs = model(input_tensor)

                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event)  # ms
            else:
                start_time = time.time()
                outputs = model(input_tensor)
                elapsed_time = (time.time() - start_time) * 1000  # ms

            # Record end memory
            end_memory = get_gpu_memory_info()

            times.append(elapsed_time)
            memory_usage.append(end_memory['allocated'])

    # Calculate statistics
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'mean_memory': np.mean(memory_usage),
        'max_memory': np.max(memory_usage),
        'throughput': input_tensor.size(0) / (np.mean(times) / 1000),  # samples per second
    }


def compare_models(batch_sizes: List[int] = [2, 4, 8, 16]):
    """
    Compare original vs optimized models across different batch sizes
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    print("🔬 Starting Model Performance Comparison")
    print(f"Device: {device}")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available. Using CPU for benchmarking.")

    for batch_size in batch_sizes:
        print(f"\n📊 Testing batch size: {batch_size}")

        # Create input tensor
        input_tensor = torch.randn(batch_size, 3, 128, 128).to(device)

        # Test original model
        try:
            print("  🔵 Testing Original Model...")
            original_model = OriginalModel().to(device)

            # Adjust input size if needed for original model
            try:
                original_metrics = benchmark_model(original_model, input_tensor)
                original_success = True
                print(f"    ✅ Original: {original_metrics['mean_time']:.2f}ms ± {original_metrics['std_time']:.2f}ms")
                print(f"    📈 Throughput: {original_metrics['throughput']:.2f} samples/sec")
                print(f"    💾 Memory: {original_metrics['mean_memory']:.2f} GB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    ❌ Original model OOM at batch size {batch_size}")
                    original_metrics = None
                    original_success = False
                else:
                    raise e
            finally:
                del original_model
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"    ❌ Original model failed: {str(e)}")
            original_metrics = None
            original_success = False

        # Test optimized model
        try:
            print("  🟢 Testing Optimized Model...")
            optimized_model = OptimizedModel().to(device)

            try:
                optimized_metrics = benchmark_model(optimized_model, input_tensor)
                optimized_success = True
                print(f"    ✅ Optimized: {optimized_metrics['mean_time']:.2f}ms ± {optimized_metrics['std_time']:.2f}ms")
                print(f"    📈 Throughput: {optimized_metrics['throughput']:.2f} samples/sec")
                print(f"    💾 Memory: {optimized_metrics['mean_memory']:.2f} GB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"    ❌ Optimized model OOM at batch size {batch_size}")
                    optimized_metrics = None
                    optimized_success = False
                else:
                    raise e
            finally:
                del optimized_model
                torch.cuda.empty_cache()
                gc.collect()

        except Exception as e:
            print(f"    ❌ Optimized model failed: {str(e)}")
            optimized_metrics = None
            optimized_success = False

        # Calculate speedup if both succeeded
        if original_success and optimized_success and original_metrics and optimized_metrics:
            speedup = original_metrics['mean_time'] / optimized_metrics['mean_time']
            memory_reduction = (original_metrics['mean_memory'] - optimized_metrics['mean_memory']) / original_metrics['mean_memory'] * 100
            throughput_improvement = (optimized_metrics['throughput'] - original_metrics['throughput']) / original_metrics['throughput'] * 100

            print(f"  🚀 Performance Improvement:")
            print(f"    ⚡ Speedup: {speedup:.2f}x")
            print(f"    💾 Memory reduction: {memory_reduction:.1f}%")
            print(f"    📈 Throughput improvement: {throughput_improvement:.1f}%")

            results[batch_size] = {
                'original': original_metrics,
                'optimized': optimized_metrics,
                'speedup': speedup,
                'memory_reduction': memory_reduction,
                'throughput_improvement': throughput_improvement
            }
        else:
            results[batch_size] = {
                'original': original_metrics,
                'optimized': optimized_metrics,
                'success': {
                    'original': original_success,
                    'optimized': optimized_success
                }
            }

    return results


def print_summary(results: Dict):
    """Print a summary of all benchmark results"""
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)

    successful_comparisons = 0
    total_speedup = 0
    total_memory_reduction = 0
    total_throughput_improvement = 0

    for batch_size, result in results.items():
        if 'speedup' in result:
            successful_comparisons += 1
            total_speedup += result['speedup']
            total_memory_reduction += result['memory_reduction']
            total_throughput_improvement += result['throughput_improvement']

            print(f"Batch {batch_size:2d}: {result['speedup']:.2f}x faster, "
                  f"{result['memory_reduction']:+.1f}% memory, "
                  f"{result['throughput_improvement']:+.1f}% throughput")

    if successful_comparisons > 0:
        avg_speedup = total_speedup / successful_comparisons
        avg_memory_reduction = total_memory_reduction / successful_comparisons
        avg_throughput_improvement = total_throughput_improvement / successful_comparisons

        print("\n" + "-" * 40)
        print("📈 AVERAGE IMPROVEMENTS:")
        print(f"⚡ Average Speedup: {avg_speedup:.2f}x")
        print(f"💾 Average Memory Reduction: {avg_memory_reduction:+.1f}%")
        print(f"📈 Average Throughput Improvement: {avg_throughput_improvement:+.1f}%")

        # Find maximum batch size for each model
        max_original_batch = 0
        max_optimized_batch = 0

        for batch_size, result in results.items():
            if 'success' in result:
                if result['success']['original']:
                    max_original_batch = max(max_original_batch, batch_size)
                if result['success']['optimized']:
                    max_optimized_batch = max(max_optimized_batch, batch_size)

        print(f"\n📏 MAXIMUM BATCH SIZES:")
        print(f"🔵 Original Model: {max_original_batch}")
        print(f"🟢 Optimized Model: {max_optimized_batch}")

        if max_optimized_batch > max_original_batch:
            improvement = (max_optimized_batch - max_original_batch) / max_original_batch * 100
            print(f"🚀 Batch size improvement: {improvement:.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    # Set up optimized environment
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"🔧 Using GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Run comparison
    batch_sizes = [2, 4, 6, 8, 12, 16] if torch.cuda.is_available() else [1, 2, 4]
    results = compare_models(batch_sizes)

    # Print summary
    print_summary(results)

    # Save results to file
    import json
    with open('performance_results.json', 'w') as f:
        # Convert numpy types to native types for JSON serialization
        json_results = {}
        for batch_size, result in results.items():
            json_results[str(batch_size)] = {}
            for key, value in result.items():
                if isinstance(value, dict):
                    json_results[str(batch_size)][key] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64, np.int32, np.int64)) else v
                        for k, v in value.items() if v is not None
                    }
                elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                    json_results[str(batch_size)][key] = float(value)
                else:
                    json_results[str(batch_size)][key] = value

        json.dump(json_results, f, indent=2)

    print(f"\n💾 Results saved to performance_results.json")