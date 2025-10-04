"""
Comprehensive Performance Analysis for DAE-KAN Model

This module provides tools to analyze and compare the performance of different
KAN implementations, including training speed, memory usage, and model complexity.
"""

import torch
import time
import psutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from thop import profile
from torch.profiler import profile as torch_profile, record_function, ProfilerActivity

from models.model import DAE_KAN_Attention
from histodata import create_dataset, ImageDataset
from torch.utils.data import DataLoader


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for DAE-KAN models
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.results = {}

    def measure_model_complexity(self, model: torch.nn.Module, input_size: Tuple[int, ...]) -> Dict:
        """
        Measure FLOPs, parameters, and memory usage of the model
        """
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, *input_size).to(self.device)

        # Measure FLOPs using thop
        with torch.no_grad():
            flops, params = profile(model, inputs=(dummy_input,))

        # Calculate memory usage
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
        total_memory = param_memory + buffer_memory

        return {
            'flops': flops,
            'parameters': params,
            'param_memory_mb': param_memory / (1024**2),
            'buffer_memory_mb': buffer_memory / (1024**2),
            'total_memory_mb': total_memory / (1024**2)
        }

    def measure_inference_time(self, model: torch.nn.Module,
                              dataloader: DataLoader,
                              num_batches: int = 50) -> Dict:
        """
        Measure inference speed and memory usage
        """
        model.eval()
        times = []
        memory_usage = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                x, _ = batch
                x = x.to(self.device)

                # Measure memory before
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_memory = torch.cuda.memory_allocated()

                # Measure inference time
                start_time = time.time()
                _ = model(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()

                # Measure memory after
                if torch.cuda.is_available():
                    end_memory = torch.cuda.memory_allocated()
                    memory_usage.append((end_memory - start_memory) / (1024**2))  # MB

                times.append(end_time - start_time)

        return {
            'mean_inference_time_ms': np.mean(times) * 1000,
            'std_inference_time_ms': np.std(times) * 1000,
            'throughput_ips': 1 / np.mean(times),
            'peak_memory_mb': max(memory_usage) if memory_usage else 0,
            'avg_memory_mb': np.mean(memory_usage) if memory_usage else 0
        }

    def measure_training_performance(self, model: torch.nn.Module,
                                   train_loader: DataLoader,
                                   val_loader: DataLoader,
                                   num_epochs: int = 3) -> Dict:
        """
        Measure training performance including speed and memory usage
        """
        import torch.optim as optim

        optimizer = optim.AdamW(model.parameters(), lr=0.0001)
        criterion = torch.nn.MSELoss()

        training_times = []
        training_memory = []

        model.train()

        for epoch in range(num_epochs):
            epoch_times = []
            epoch_memory = []

            for batch_idx, batch in enumerate(train_loader):
                x, _ = batch
                x = x.to(self.device)

                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    start_memory = torch.cuda.memory_allocated()

                # Forward pass
                start_time = time.time()
                encoded, decoded, z = model(x)
                loss = criterion(decoded, x)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()

                # Measure memory after
                if torch.cuda.is_available():
                    end_memory = torch.cuda.memory_allocated()
                    epoch_memory.append((end_memory - start_memory) / (1024**2))

                epoch_times.append(end_time - start_time)

                # Limit measurement to first 100 batches per epoch
                if batch_idx >= 100:
                    break

            training_times.extend(epoch_times)
            training_memory.extend(epoch_memory)

        return {
            'mean_step_time_ms': np.mean(training_times) * 1000,
            'std_step_time_ms': np.std(training_times) * 1000,
            'training_throughput_ips': 1 / np.mean(training_times),
            'peak_training_memory_mb': max(training_memory) if training_memory else 0,
            'avg_training_memory_mb': np.mean(training_memory) if training_memory else 0
        }

    def compare_kan_configurations(self,
                                  grid_sizes: List[int] = [3, 5],
                                  spline_orders: List[int] = [1, 2, 3],
                                  input_size: Tuple[int, ...] = (3, 128, 128)) -> pd.DataFrame:
        """
        Compare different KAN configurations
        """
        results = []

        for grid_size in grid_sizes:
            for spline_order in spline_orders:
                print(f"Testing grid_size={grid_size}, spline_order={spline_order}")

                # Create model with specific configuration
                import sys
                sys.path.append('../src')

                # Patch the model creation with current configuration
                from models import model
                original_grid_size = getattr(model, 'current_grid_size', 3)
                original_spline_order = getattr(model, 'current_spline_order', 2)

                model.current_grid_size = grid_size
                model.current_spline_order = spline_order

                try:
                    model_instance = DAE_KAN_Attention().to(self.device)

                    # Measure complexity
                    complexity = self.measure_model_complexity(model_instance, input_size)

                    # Create small dataset for performance testing
                    train_ds = ImageDataset(*create_dataset('train'))
                    small_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

                    # Measure inference performance
                    inference_perf = self.measure_inference_time(model_instance, small_loader, num_batches=20)

                    # Combine results
                    result = {
                        'grid_size': grid_size,
                        'spline_order': spline_order,
                        **complexity,
                        **inference_perf
                    }
                    results.append(result)

                    print(f"  FLOPs: {complexity['flops']:,}")
                    print(f"  Parameters: {complexity['parameters']:,}")
                    print(f"  Inference time: {inference_perf['mean_inference_time_ms']:.2f}ms")
                    print(f"  Memory: {inference_perf['peak_memory_mb']:.1f}MB")

                    # Clean up
                    del model_instance
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error with grid_size={grid_size}, spline_order={spline_order}: {e}")
                    continue

        return pd.DataFrame(results)

    def plot_performance_comparison(self, df: pd.DataFrame, save_path: str = None):
        """
        Create performance comparison plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('KAN Configuration Performance Analysis', fontsize=16)

        # Plot 1: FLOPs vs Configuration
        ax = axes[0, 0]
        pivot_flops = df.pivot(index='spline_order', columns='grid_size', values='flops')
        sns.heatmap(pivot_flops, annot=True, fmt='.0e', ax=ax, cmap='viridis')
        ax.set_title('FLOPs (Lower is Better)')

        # Plot 2: Parameters vs Configuration
        ax = axes[0, 1]
        pivot_params = df.pivot(index='spline_order', columns='grid_size', values='parameters')
        sns.heatmap(pivot_params, annot=True, fmt=',.0f', ax=ax, cmap='viridis')
        ax.set_title('Number of Parameters')

        # Plot 3: Inference Time vs Configuration
        ax = axes[0, 2]
        pivot_time = df.pivot(index='spline_order', columns='grid_size', values='mean_inference_time_ms')
        sns.heatmap(pivot_time, annot=True, fmt='.1f', ax=ax, cmap='viridis_r')
        ax.set_title('Mean Inference Time (ms, Lower is Better)')

        # Plot 4: Memory Usage vs Configuration
        ax = axes[1, 0]
        pivot_memory = df.pivot(index='spline_order', columns='grid_size', values='peak_memory_mb')
        sns.heatmap(pivot_memory, annot=True, fmt='.1f', ax=ax, cmap='viridis_r')
        ax.set_title('Peak Memory Usage (MB, Lower is Better)')

        # Plot 5: Throughput vs Configuration
        ax = axes[1, 1]
        pivot_throughput = df.pivot(index='spline_order', columns='grid_size', values='throughput_ips')
        sns.heatmap(pivot_throughput, annot=True, fmt='.1f', ax=ax, cmap='viridis')
        ax.set_title('Throughput (images/sec, Higher is Better)')

        # Plot 6: Efficiency Score (Throughput / Memory)
        ax = axes[1, 2]
        df['efficiency_score'] = df['throughput_ips'] / (df['total_memory_mb'] + 1)
        pivot_efficiency = df.pivot(index='spline_order', columns='grid_size', values='efficiency_score')
        sns.heatmap(pivot_efficiency, annot=True, fmt='.3f', ax=ax, cmap='viridis')
        ax.set_title('Efficiency Score (Higher is Better)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_performance_report(self, df: pd.DataFrame, save_path: str = None):
        """
        Generate a comprehensive performance report
        """
        report = []
        report.append("# KAN Performance Analysis Report\n")

        # Summary statistics
        report.append("## Summary Statistics\n")
        report.append(f"- **Configurations Tested**: {len(df)}")
        report.append(f"- **FLOPs Range**: {df['flops'].min():,.0f} - {df['flops'].max():,.0f}")
        report.append(f"- **Parameter Count Range**: {df['parameters'].min():,.0f} - {df['parameters'].max():,.0f}")
        report.append(f"- **Inference Time Range**: {df['mean_inference_time_ms'].min():.1f} - {df['mean_inference_time_ms'].max():.1f} ms")
        report.append(f"- **Memory Usage Range**: {df['peak_memory_mb'].min():.1f} - {df['peak_memory_mb'].max():.1f} MB\n")

        # Best configurations
        report.append("## Best Configurations\n")

        # Fastest inference
        fastest = df.loc[df['mean_inference_time_ms'].idxmin()]
        report.append(f"### Fastest Inference")
        report.append(f"- Grid Size: {fastest['grid_size']}, Spline Order: {fastest['spline_order']}")
        report.append(f"- Time: {fastest['mean_inference_time_ms']:.2f} ms")
        report.append(f"- Throughput: {fastest['throughput_ips']:.1f} images/sec\n")

        # Most memory efficient
        most_efficient = df.loc[df['peak_memory_mb'].idxmin()]
        report.append(f"### Most Memory Efficient")
        report.append(f"- Grid Size: {most_efficient['grid_size']}, Spline Order: {most_efficient['spline_order']}")
        report.append(f"- Memory: {most_efficient['peak_memory_mb']:.1f} MB")
        report.append(f"- Parameters: {most_efficient['parameters']:,}\n")

        # Best overall efficiency
        best_efficiency = df.loc[df['efficiency_score'].idxmax()]
        report.append(f"### Best Overall Efficiency")
        report.append(f"- Grid Size: {best_efficiency['grid_size']}, Spline Order: {best_efficiency['spline_order']}")
        report.append(f"- Efficiency Score: {best_efficiency['efficiency_score']:.3f}")
        report.append(f"- FLOPs: {best_efficiency['flops']:,}")
        report.append(f"- Memory: {best_efficiency['total_memory_mb']:.1f} MB\n")

        # Trade-off analysis
        report.append("## Trade-off Analysis\n")
        report.append("### Performance vs Complexity")
        correlation = df[['flops', 'mean_inference_time_ms']].corr().iloc[0, 1]
        report.append(f"- FLOPs vs Inference Time correlation: {correlation:.3f}")

        correlation = df[['parameters', 'peak_memory_mb']].corr().iloc[0, 1]
        report.append(f"- Parameters vs Memory correlation: {correlation:.3f}\n")

        # Recommendations
        report.append("## Recommendations\n")

        # For real-time applications
        real_time_best = df.loc[df['mean_inference_time_ms'] < 100].sort_values('mean_inference_time_ms').iloc[0] if len(df[df['mean_inference_time_ms'] < 100]) > 0 else fastest
        report.append("### For Real-time Applications (<100ms inference)")
        report.append(f"- Recommended: Grid Size {real_time_best['grid_size']}, Spline Order {real_time_best['spline_order']}")
        report.append(f"- Expected inference time: {real_time_best['mean_inference_time_ms']:.1f} ms\n")

        # For memory-constrained environments
        memory_best = df.sort_values('peak_memory_mb').iloc[0]
        report.append("### For Memory-Constrained Environments")
        report.append(f"- Recommended: Grid Size {memory_best['grid_size']}, Spline Order {memory_best['spline_order']}")
        report.append(f"- Expected memory usage: {memory_best['peak_memory_mb']:.1f} MB\n")

        # For best accuracy (assuming higher complexity = better accuracy)
        accuracy_best = df.loc[df['flops'].idxmax()]
        report.append("### For Maximum Accuracy")
        report.append(f"- Recommended: Grid Size {accuracy_best['grid_size']}, Spline Order {accuracy_best['spline_order']}")
        report.append(f"- FLOPs: {accuracy_best['flops']:,}")
        report.append(f"- Note: Higher complexity may improve accuracy but reduces speed\n")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text


def main():
    """
    Main function to run performance analysis
    """
    analyzer = PerformanceAnalyzer()

    print("Starting KAN performance analysis...")

    # Compare different configurations
    df = analyzer.compare_kan_configurations(
        grid_sizes=[3, 5],
        spline_orders=[1, 2, 3],
        input_size=(3, 128, 128)
    )

    # Save results
    df.to_csv('../analysis/reports/performance_results.csv', index=False)

    # Create plots
    analyzer.plot_performance_comparison(
        df,
        save_path='../analysis/visualizations/performance_comparison.png'
    )

    # Generate report
    report = analyzer.generate_performance_report(
        df,
        save_path='../analysis/reports/performance_report.md'
    )

    print("\nPerformance Analysis Complete!")
    print("Results saved to:")
    print("- ../analysis/reports/performance_results.csv")
    print("- ../analysis/visualizations/performance_comparison.png")
    print("- ../analysis/reports/performance_report.md")

    return df


if __name__ == "__main__":
    results = main()