"""
Model Complexity and Performance Analysis for DAE-KAN

This module provides comprehensive analysis of model complexity, computational cost,
and performance characteristics to determine if performance improvements are worth
the extra computation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import gc
from typing import Dict, List, Tuple, Optional, Union
import json
from thop import profile
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
import warnings
warnings.filterwarnings('ignore')

from models.model import DAE_KAN_Attention, Autoencoder_Encoder, Autoencoder_Decoder, Autoencoder_Bottleneck


class ModelComplexityAnalyzer:
    """
    Comprehensive analysis of model complexity and computational requirements
    """

    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.complexity_metrics = {}

    def count_parameters(self, model: Optional[nn.Module] = None) -> Dict[str, int]:
        """
        Count parameters in the model with detailed breakdown
        """
        if model is None:
            model = self.model

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Detailed parameter breakdown by component
        param_breakdown = {}

        if hasattr(model, 'ae_encoder'):
            encoder_params = sum(p.numel() for p in model.ae_encoder.parameters())
            param_breakdown['encoder'] = encoder_params

        if hasattr(model, 'bottleneck'):
            bottleneck_params = sum(p.numel() for p in model.bottleneck.parameters())
            param_breakdown['bottleneck'] = bottleneck_params

        if hasattr(model, 'ae_decoder'):
            decoder_params = sum(p.numel() for p in model.ae_decoder.parameters())
            param_breakdown['decoder'] = decoder_params

        # KAN-specific parameters
        kan_params = 0
        for name, module in model.named_modules():
            if 'kan' in name.lower() and hasattr(module, 'base_weight'):
                kan_params += sum(p.numel() for p in module.parameters())
        param_breakdown['kan_layers'] = kan_params

        # Attention mechanism parameters
        attention_params = 0
        for name, module in model.named_modules():
            if 'attn' in name.lower() or 'bam' in name.lower():
                attention_params += sum(p.numel() for p in module.parameters())
        param_breakdown['attention_mechanisms'] = attention_params

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'breakdown': param_breakdown
        }

    def compute_flops_macs(self, input_size: Tuple[int, ...] = (1, 3, 128, 128)) -> Dict[str, Union[int, float]]:
        """
        Compute FLOPs and MACs (Multiply-Accumulate operations)
        """
        try:
            # Create dummy input
            dummy_input = torch.randn(input_size).to(self.device)

            # Use thop for FLOPs counting
            flops, params = profile(self.model, inputs=(dummy_input,), verbose=False)

            # Use fvcore for more detailed analysis
            try:
                flop_analysis = FlopCountAnalysis(self.model, dummy_input)
                activation_analysis = ActivationCountAnalysis(self.model, dummy_input)

                flop_analysis = flop_analysis.unsupported_ops_warnings()

                detailed_flops = {}
                for module_name, module_flops in flop_analysis.by_operator().items():
                    detailed_flops[module_name] = module_flops

                activation_counts = {}
                for module_name, acts in activation_analysis.by_operator().items():
                    activation_counts[module_name] = acts

            except Exception as e:
                print(f"fvcore analysis failed: {e}")
                detailed_flops = {}
                activation_counts = {}

            return {
                'total_flops': int(flops),
                'total_macs': int(flops / 2),  # Approximate MACs
                'parameters': int(params),
                'flops_per_parameter': flops / params if params > 0 else 0,
                'detailed_flops': detailed_flops,
                'activation_counts': activation_counts
            }

        except Exception as e:
            print(f"FLOPs analysis failed: {e}")
            return {
                'total_flops': 0,
                'total_macs': 0,
                'parameters': 0,
                'flops_per_parameter': 0,
                'detailed_flops': {},
                'activation_counts': {}
            }

    def measure_memory_usage(self, input_size: Tuple[int, ...] = (1, 3, 128, 128),
                           batch_sizes: List[int] = [1, 2, 4, 8, 16]) -> Dict[str, List]:
        """
        Measure memory usage at different batch sizes
        """
        memory_stats = {
            'batch_sizes': batch_sizes,
            'gpu_memory_mb': [],
            'cpu_memory_mb': [],
            'peak_gpu_memory_mb': [],
            'model_size_mb': 0
        }

        # Model size
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        memory_stats['model_size_mb'] = model_size / (1024 * 1024)

        if not torch.cuda.is_available():
            return memory_stats

        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

        for batch_size in batch_sizes:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()

                # Measure initial memory
                initial_mem = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

                # Create batch and run forward pass
                dummy_input = torch.randn(batch_size, *input_size[1:]).to(self.device)

                with torch.no_grad():
                    _ = self.model(dummy_input)

                # Measure peak memory
                peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                current_mem = torch.cuda.memory_allocated(self.device) / (1024 * 1024)

                memory_stats['gpu_memory_mb'].append(current_mem - initial_mem)
                memory_stats['peak_gpu_memory_mb'].append(peak_mem - initial_mem)

                # Clean up
                del dummy_input
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e):
                    memory_stats['gpu_memory_mb'].append(float('inf'))
                    memory_stats['peak_gpu_memory_mb'].append(float('inf'))
                    torch.cuda.empty_cache()
                else:
                    raise e
            except Exception as e:
                print(f"Error measuring memory for batch size {batch_size}: {e}")
                memory_stats['gpu_memory_mb'].append(0)
                memory_stats['peak_gpu_memory_mb'].append(0)

        # CPU memory measurement
        for batch_size in batch_sizes:
            try:
                process = psutil.Process()
                initial_mem = process.memory_info().rss / (1024 * 1024)

                dummy_input = torch.randn(batch_size, *input_size[1:])

                with torch.no_grad():
                    # Move model to CPU temporarily
                    if self.device.type == 'cuda':
                        model_cpu = self.model.cpu()
                        _ = model_cpu(dummy_input)
                        self.model.to(self.device)
                    else:
                        _ = self.model(dummy_input)

                final_mem = process.memory_info().rss / (1024 * 1024)
                memory_stats['cpu_memory_mb'].append(final_mem - initial_mem)

                del dummy_input

            except Exception as e:
                print(f"Error measuring CPU memory for batch size {batch_size}: {e}")
                memory_stats['cpu_memory_mb'].append(0)

        return memory_stats

    def benchmark_inference_time(self, input_size: Tuple[int, ...] = (1, 3, 128, 128),
                                batch_sizes: List[int] = [1, 2, 4, 8, 16],
                                num_runs: int = 100,
                                warmup_runs: int = 10) -> Dict[str, List]:
        """
        Benchmark inference time across different batch sizes
        """
        timing_stats = {
            'batch_sizes': batch_sizes,
            'mean_times_ms': [],
            'std_times_ms': [],
            'min_times_ms': [],
            'max_times_ms': [],
            'throughput_images_per_sec': []
        }

        self.model.eval()

        for batch_size in batch_sizes:
            try:
                dummy_input = torch.randn(batch_size, *input_size[1:]).to(self.device)

                # Warmup runs
                with torch.no_grad():
                    for _ in range(warmup_runs):
                        _ = self.model(dummy_input)

                # Timing runs
                times = []
                with torch.no_grad():
                    for _ in range(num_runs):
                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()

                        start_time = time.perf_counter()
                        _ = self.model(dummy_input)

                        if self.device.type == 'cuda':
                            torch.cuda.synchronize()

                        end_time = time.perf_counter()
                        times.append((end_time - start_time) * 1000)  # Convert to ms

                times = np.array(times)
                timing_stats['mean_times_ms'].append(float(np.mean(times)))
                timing_stats['std_times_ms'].append(float(np.std(times)))
                timing_stats['min_times_ms'].append(float(np.min(times)))
                timing_stats['max_times_ms'].append(float(np.max(times)))
                timing_stats['throughput_images_per_sec'].append(batch_size / (np.mean(times) / 1000))

                del dummy_input

            except RuntimeError as e:
                if "out of memory" in str(e):
                    timing_stats['mean_times_ms'].append(float('inf'))
                    timing_stats['std_times_ms'].append(0)
                    timing_stats['min_times_ms'].append(float('inf'))
                    timing_stats['max_times_ms'].append(float('inf'))
                    timing_stats['throughput_images_per_sec'].append(0)
                    torch.cuda.empty_cache()
                else:
                    raise e
            except Exception as e:
                print(f"Error benchmarking batch size {batch_size}: {e}")
                timing_stats['mean_times_ms'].append(0)
                timing_stats['std_times_ms'].append(0)
                timing_stats['min_times_ms'].append(0)
                timing_stats['max_times_ms'].append(0)
                timing_stats['throughput_images_per_sec'].append(0)

        return timing_stats

    def analyze_layer_complexity(self, input_size: Tuple[int, ...] = (1, 3, 128, 128)) -> Dict[str, Dict]:
        """
        Analyze complexity of individual layers
        """
        layer_analysis = {}
        dummy_input = torch.randn(input_size).to(self.device)

        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                # Count parameters
                params = sum(p.numel() for p in module.parameters())

                # Estimate output size
                if isinstance(output, tuple):
                    output_size = output[0].numel() if len(output[0].shape) > 0 else 0
                else:
                    output_size = output.numel() if len(output.shape) > 0 else 0

                # Estimate FLOPs (rough approximation)
                if hasattr(module, 'kernel_size'):
                    # Convolutional layer
                    if hasattr(module, 'groups') and module.groups > 1:
                        # Depthwise convolution
                        flops = output_size * module.kernel_size[0] * module.kernel_size[1]
                    else:
                        # Standard convolution
                        flops = output_size * module.in_channels * module.kernel_size[0] * module.kernel_size[1] / (module.stride[0] * module.stride[1])
                elif hasattr(module, 'in_features'):
                    # Linear layer
                    flops = module.in_features * module.out_features
                else:
                    # Other layers (rough estimate)
                    flops = output_size

                layer_analysis[layer_name] = {
                    'parameters': params,
                    'output_size': output_size,
                    'estimated_flops': flops,
                    'module_type': type(module).__name__,
                    'input_shape': list(input[0].shape) if input and len(input) > 0 else [],
                    'output_shape': list(output.shape) if hasattr(output, 'shape') else []
                }

            return hook_fn

        # Register hooks for all layers
        hooks = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(create_hook_fn(name))
                hooks.append(hook)

        # Forward pass
        with torch.no_grad():
            _ = self.model(dummy_input)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return layer_analysis

    def comprehensive_complexity_analysis(self, input_size: Tuple[int, ...] = (1, 3, 128, 128)) -> Dict:
        """
        Run comprehensive complexity analysis
        """
        print("Running comprehensive model complexity analysis...")

        analysis_results = {
            'model_name': type(self.model).__name__,
            'input_size': input_size,
            'device': str(self.device),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # Parameter analysis
        print("Analyzing parameters...")
        analysis_results['parameters'] = self.count_parameters()

        # FLOPs analysis
        print("Analyzing FLOPs...")
        analysis_results['flops'] = self.compute_flops_macs(input_size)

        # Memory analysis
        print("Analyzing memory usage...")
        analysis_results['memory'] = self.measure_memory_usage(input_size)

        # Timing analysis
        print("Benchmarking inference time...")
        analysis_results['timing'] = self.benchmark_inference_time(input_size)

        # Layer analysis
        print("Analyzing layer complexity...")
        analysis_results['layer_analysis'] = self.analyze_layer_complexity(input_size)

        return analysis_results


class PerformanceBenchmark:
    """
    Benchmark model performance against baseline models
    """

    def __init__(self, device: str = "cuda"):
        self.device = device

    def create_baseline_models(self) -> Dict[str, nn.Module]:
        """
        Create baseline models for comparison
        """
        models = {}

        # Simple CNN Autoencoder
        class SimpleCNN_AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded, encoded.flatten(1)

        # CNN with attention
        class CNN_Attention_AE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(),
                    # Add attention here
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU()
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded, encoded.flatten(1)

        models['simple_cnn'] = SimpleCNN_AE()
        models['cnn_attention'] = CNN_Attention_AE()
        models['dae_kan'] = DAE_KAN_Attention()

        return models

    def benchmark_models(self, input_size: Tuple[int, ...] = (1, 3, 128, 128),
                        num_runs: int = 50) -> pd.DataFrame:
        """
        Benchmark multiple models and return comparison results
        """
        models = self.create_baseline_models()
        results = []

        for model_name, model in models.items():
            print(f"Benchmarking {model_name}...")
            model.to(self.device)
            model.eval()

            analyzer = ModelComplexityAnalyzer(model, self.device)

            # Get complexity metrics
            params = analyzer.count_parameters()
            flops = analyzer.compute_flops_macs(input_size)
            timing = analyzer.benchmark_inference_time(input_size, num_runs=num_runs)
            memory = analyzer.measure_memory_usage(input_size, batch_sizes=[1])

            # Store results
            result = {
                'model': model_name,
                'total_parameters': params['total_parameters'],
                'trainable_parameters': params['trainable_parameters'],
                'flops': flops['total_flops'],
                'inference_time_ms': timing['mean_times_ms'][0] if timing['mean_times_ms'] else 0,
                'throughput_imgs_per_sec': timing['throughput_images_per_sec'][0] if timing['throughput_images_per_sec'] else 0,
                'gpu_memory_mb': memory['gpu_memory_mb'][0] if memory['gpu_memory_mb'] else 0,
                'model_size_mb': memory['model_size_mb']
            }

            # Compute efficiency metrics
            result['params_per_flop'] = result['total_parameters'] / result['flops'] if result['flops'] > 0 else 0
            result['flops_per_second'] = result['flops'] / (result['inference_time_ms'] / 1000) if result['inference_time_ms'] > 0 else 0
            result['memory_efficiency'] = result['total_parameters'] / result['gpu_memory_mb'] if result['gpu_memory_mb'] > 0 else 0

            results.append(result)

            # Clean up
            del model
            torch.cuda.empty_cache()

        return pd.DataFrame(results)

    def create_comparison_visualization(self, benchmark_df: pd.DataFrame,
                                     save_path: Optional[str] = None):
        """
        Create comprehensive comparison visualization
        """
        fig = plt.figure(figsize=(20, 12))

        # 1. Parameters comparison
        ax1 = plt.subplot(2, 3, 1)
        sns.barplot(data=benchmark_df, x='model', y='total_parameters', ax=ax1)
        ax1.set_title('Total Parameters by Model')
        ax1.set_ylabel('Parameters')
        ax1.tick_params(axis='x', rotation=45)

        # 2. FLOPs comparison
        ax2 = plt.subplot(2, 3, 2)
        sns.barplot(data=benchmark_df, x='model', y='flops', ax=ax2)
        ax2.set_title('FLOPs by Model')
        ax2.set_ylabel('FLOPs')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Inference time comparison
        ax3 = plt.subplot(2, 3, 3)
        sns.barplot(data=benchmark_df, x='model', y='inference_time_ms', ax=ax3)
        ax3.set_title('Inference Time by Model')
        ax3.set_ylabel('Time (ms)')
        ax3.tick_params(axis='x', rotation=45)

        # 4. Throughput comparison
        ax4 = plt.subplot(2, 3, 4)
        sns.barplot(data=benchmark_df, x='model', y='throughput_imgs_per_sec', ax=ax4)
        ax4.set_title('Throughput by Model')
        ax4.set_ylabel('Images/Second')
        ax4.tick_params(axis='x', rotation=45)

        # 5. Memory usage comparison
        ax5 = plt.subplot(2, 3, 5)
        sns.barplot(data=benchmark_df, x='model', y='gpu_memory_mb', ax=ax5)
        ax5.set_title('GPU Memory Usage by Model')
        ax5.set_ylabel('Memory (MB)')
        ax5.tick_params(axis='x', rotation=45)

        # 6. Efficiency scatter plot
        ax6 = plt.subplot(2, 3, 6)
        scatter = ax6.scatter(benchmark_df['total_parameters'], benchmark_df['inference_time_ms'],
                             s=benchmark_df['flops']/1e6, alpha=0.6, c=range(len(benchmark_df)))
        ax6.set_xlabel('Parameters')
        ax6.set_ylabel('Inference Time (ms)')
        ax6.set_title('Parameter-Time Efficiency\n(Bubble size = FLOPs)')

        # Add model labels to scatter plot
        for i, row in benchmark_df.iterrows():
            ax6.annotate(row['model'], (row['total_parameters'], row['inference_time_ms']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def generate_efficiency_report(self, benchmark_df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive efficiency analysis report
        """
        report = []
        report.append("# Model Complexity and Efficiency Analysis Report\n")

        # Summary table
        report.append("## Model Comparison Summary\n")
        report.append("| Model | Parameters | FLOPs | Inference Time (ms) | Throughput (img/s) | Memory (MB) |")
        report.append("|-------|------------|-------|---------------------|-------------------|-------------|")

        for _, row in benchmark_df.iterrows():
            report.append(f"| {row['model']} | {row['total_parameters']:,} | {row['flops']:,} | "
                         f"{row['inference_time_ms']:.2f} | {row['throughput_imgs_per_sec']:.1f} | "
                         f"{row['gpu_memory_mb']:.1f} |")

        report.append("")

        # Findings
        report.append("## Key Findings\n")

        # Most efficient model
        best_throughput = benchmark_df.loc[benchmark_df['throughput_imgs_per_sec'].idxmax()]
        report.append(f"### Highest Throughput")
        report.append(f"- **Model**: {best_throughput['model']}")
        report.append(f"- **Throughput**: {best_throughput['throughput_imgs_per_sec']:.1f} images/second")
        report.append(f"- **Inference Time**: {best_throughput['inference_time_ms']:.2f} ms\n")

        # Most parameter efficient
        best_params = benchmark_df.loc[benchmark_df['total_parameters'].idxmin()]
        report.append(f"### Fewest Parameters")
        report.append(f"- **Model**: {best_params['model']}")
        report.append(f"- **Parameters**: {best_params['total_parameters']:,}")
        report.append(f"- **Model Size**: {best_params['model_size_mb']:.1f} MB\n")

        # Most compute efficient
        best_flops_per_sec = benchmark_df.loc[benchmark_df['flops_per_second'].idxmax()]
        report.append(f"### Highest Compute Efficiency")
        report.append(f"- **Model**: {best_flops_per_sec['model']}")
        report.append(f"- **FLOPs/Second**: {best_flops_per_sec['flops_per_second']:.2e}")
        report.append(f"- **FLOPs**: {best_flops_per_sec['flops']:,}\n")

        # DAE-KAN specific analysis
        dae_kan_row = benchmark_df[benchmark_df['model'] == 'dae_kan'].iloc[0]
        simple_cnn_row = benchmark_df[benchmark_df['model'] == 'simple_cnn'].iloc[0]

        report.append("## DAE-KAN vs Simple CNN Analysis\n")

        # Parameter overhead
        param_overhead = (dae_kan_row['total_parameters'] / simple_cnn_row['total_parameters'] - 1) * 100
        report.append(f"### Parameter Overhead")
        report.append(f"- **DAE-KAN**: {dae_kan_row['total_parameters']:,} parameters")
        report.append(f"- **Simple CNN**: {simple_cnn_row['total_parameters']:,} parameters")
        report.append(f"- **Overhead**: +{param_overhead:.1f}%\n")

        # Performance trade-off
        time_overhead = (dae_kan_row['inference_time_ms'] / simple_cnn_row['inference_time_ms'] - 1) * 100
        report.append(f"### Performance Trade-off")
        report.append(f"- **DAE-KAN**: {dae_kan_row['inference_time_ms']:.2f} ms")
        report.append(f"- **Simple CNN**: {simple_cnn_row['inference_time_ms']:.2f} ms")
        report.append(f"- **Time Overhead**: +{time_overhead:.1f}%\n")

        # Memory efficiency
        memory_overhead = (dae_kan_row['gpu_memory_mb'] / simple_cnn_row['gpu_memory_mb'] - 1) * 100
        report.append(f"### Memory Usage")
        report.append(f"- **DAE-KAN**: {dae_kan_row['gpu_memory_mb']:.1f} MB")
        report.append(f"- **Simple CNN**: {simple_cnn_row['gpu_memory_mb']:.1f} MB")
        report.append(f"- **Memory Overhead**: +{memory_overhead:.1f}%\n")

        # Recommendations
        report.append("## Recommendations\n")

        if param_overhead > 200:
            report.append("⚠️ **High Parameter Overhead**: DAE-KAN uses significantly more parameters than baseline")
        elif param_overhead > 100:
            report.append("⚡ **Moderate Parameter Overhead**: DAE-KAN uses more parameters but may be justified by improved performance")
        else:
            report.append("✅ **Reasonable Parameter Overhead**: DAE-KAN parameter increase is manageable")

        if time_overhead > 200:
            report.append("⚠️ **High Computational Overhead**: DAE-KAN is significantly slower than baseline")
        elif time_overhead > 100:
            report.append("⚡ **Moderate Computational Overhead**: DAE-KAN is slower but may be acceptable for improved quality")
        else:
            report.append("✅ **Reasonable Computational Overhead**: DAE-KAN performance impact is manageable")

        report.append("\n### When to Use DAE-KAN")
        report.append("✅ **Use DAE-KAN when:")
        report.append("- Interpretability is crucial (attention maps, pathological correlation)")
        report.append("- Performance improvements justify computational costs")
        report.append("- Batch processing is acceptable (not real-time constraints)")
        report.append("- Memory constraints are not severe")

        report.append("\n❌ **Consider alternatives when:")
        report.append("- Real-time performance is required")
        report.append("- Memory/compute resources are severely limited")
        report.append("- Simple reconstruction task without need for interpretation")

        report_text = "\n".join(report)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)

        return report_text


def main():
    """
    Main function to run comprehensive model complexity analysis
    """
    print("Initializing Model Complexity Analysis...")

    # Create analyzer
    model = DAE_KAN_Attention()
    analyzer = ModelComplexityAnalyzer(model, device="cuda")
    benchmark = PerformanceBenchmark(device="cuda")

    # Run comprehensive analysis
    print("Running comprehensive analysis...")
    analysis_results = analyzer.comprehensive_complexity_analysis()

    # Save detailed results
    with open('../analysis/reports/model_complexity_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    # Run benchmark comparison
    print("Running benchmark comparison...")
    benchmark_df = benchmark.benchmark_models()

    # Save benchmark results
    benchmark_df.to_csv('../analysis/reports/model_benchmark_comparison.csv', index=False)

    # Create visualization
    print("Creating comparison visualization...")
    benchmark.create_comparison_visualization(
        save_path='../analysis/visualizations/model_comparison.png'
    )

    # Generate report
    print("Generating efficiency report...")
    report = benchmark.generate_efficiency_report(
        benchmark_df,
        save_path='../analysis/reports/efficiency_analysis_report.md'
    )

    print("\nModel Complexity Analysis Complete!")
    print("Results saved to:")
    print("- ../analysis/reports/model_complexity_analysis.json")
    print("- ../analysis/reports/model_benchmark_comparison.csv")
    print("- ../analysis/visualizations/model_comparison.png")
    print("- ../analysis/reports/efficiency_analysis_report.md")

    return analyzer, benchmark_df


if __name__ == "__main__":
    analyzer, results = main()