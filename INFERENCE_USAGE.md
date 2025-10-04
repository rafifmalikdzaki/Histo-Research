# DAE-KAN Inference Usage Guide

This guide explains how to use the comprehensive DAE-KAN inference system with pathological correlation validation.

## 🚀 Quick Start

### Basic Usage

```bash
# Single image inference
python inference_dae_kan.py --model_path models/best_model.pth --input_image test_image.png

# Batch inference on directory
python inference_dae_kan.py --model_path models/best_model.pth --input_dir test_images/ --batch_mode

# With custom output directory
python inference_dae_kan.py --model_path models/best_model.pth --input_dir test_images/ --batch_mode --output_dir results/
```

### Advanced Usage

```bash
# With custom configuration
python inference_dae_kan.py --model_path models/best_model.pth --input_dir test_images/ --batch_mode --config inference_config.json

# Fast inference (no analysis)
python inference_dae_kan.py --model_path models/best_model.pth --input_dir test_images/ --batch_mode --disable_analysis

# GPU-specific inference
python inference_dae_kan.py --model_path models/best_model.pth --input_dir test_images/ --batch_mode --device cuda
```

## 📊 Analysis Workflow During Training

### **Training Phase** (`pl_training_with_analysis.py`)

```python
# Every batch:
- Training time tracking (ms)
- Inference time tracking (ms)
- GPU memory usage (MB)
- Loss and reconstruction metrics
- W&B logging of all metrics

# Every N batches (configurable, default 100):
- Enhanced attention visualization
- Quantitative saliency metrics (entropy, concentration, sparsity)
- Attention region identification
- Pathology correlation analysis
- Comprehensive analysis reports

# Every epoch:
- Model complexity metrics logging
- Attention pattern evolution tracking
- Performance trend analysis

# Every 5 epochs:
- Full performance benchmark comparison
- Model complexity vs baseline analysis
- Clinical readiness assessment
- Timing analysis visualizations
```

### **Validation Phase**

```python
# Every validation batch:
- Validation timing and memory tracking
- Comprehensive attention analysis
- Pathological validation framework
- Annotation interface generation
- Expert validation metrics

# Validation analysis includes:
- Attention-pathology overlap metrics (IoU, Dice, Precision, Recall)
- Spatial correlation with pathological features
- Clinical relevance assessment
- Expert annotation interfaces
```

### **Post-Training**

```python
# Final comprehensive analysis:
- Multi-sample attention comparison
- Complete performance benchmarking
- Final timing and memory analysis
- Clinical deployment recommendations
- Comprehensive validation report
- Pathologist collaboration tools
```

## 🔬 Analysis Components

### **1. Attention Analysis**
- **BAM attention patterns**: Spatial and channel attention visualization
- **KAN activation patterns**: Spline basis and activation analysis
- **Layer-wise evolution**: Feature progression through encoder-decoder
- **Quantitative metrics**: Entropy, concentration, sparsity, spatial analysis
- **Region identification**: Attention clustering and region analysis

### **2. Pathological Correlation**
- **Feature extraction**: Nuclear features, tissue architecture, color patterns
- **Correlation analysis**: Statistical correlation with attention maps
- **Overlap metrics**: IoU, Dice coefficient, precision-recall
- **Expert validation**: Annotation interfaces for pathologists
- **Clinical assessment**: Deployment readiness evaluation

### **3. Model Complexity Analysis**
- **Parameter counting**: Detailed breakdown by component
- **FLOPs computation**: Computational cost analysis
- **Memory usage**: GPU memory profiling across batch sizes
- **Benchmarking**: Performance comparison with baseline models
- **Efficiency metrics**: FLOPs per parameter, inference speed analysis

### **4. Performance Profiling**
- **Timing analysis**: Training and inference time tracking
- **Memory profiling**: GPU and CPU memory usage
- **Throughput analysis**: Images per second measurement
- **Scalability testing**: Performance across batch sizes
- **Bottleneck identification**: Performance optimization insights

## 📁 Output Structure

```
inference_results/
├── sample_1/
│   ├── original.png
│   ├── reconstructed.png
│   ├── error_map.png
│   ├── reconstruction_comparison.png
│   ├── enhanced_attention.png
│   ├── annotation_interface.png
│   ├── pathology_correlation.png
│   ├── attention_report.md
│   ├── attention_data.json
│   └── inference_results.json
├── sample_2/
│   └── ...
├── batch_summary.png
├── batch_report.md
├── final_report.md
├── model_complexity_analysis.json
├── performance_benchmark.csv
├── benchmark_comparison.png
├── efficiency_report.md
└── inference_results.json
```

## 📋 Key Metrics and Reports

### **Attention Metrics**
- **Attention Entropy**: Distribution spread of attention (higher = more distributed)
- **Attention Concentration**: Focus on top-k regions (higher = more focused)
- **Spatial Variance**: Spread of attention from center point
- **Number of Peaks**: Distinct attention focus regions
- **Peak Coverage**: Area covered by high-attention regions

### **Pathological Validation Metrics**
- **IoU (Intersection over Union)**: Overlap between attention and annotations
- **Dice Coefficient**: Similarity measure for region overlap
- **Precision/Recall**: False positive/negative rates
- **F1-Score**: Harmonic mean of precision and recall
- **Weighted Overlap**: Intensity-weighted attention overlap

### **Performance Metrics**
- **Inference Time**: Time per forward pass (ms)
- **Throughput**: Images processed per second
- **Memory Usage**: GPU memory consumption (MB)
- **Parameter Efficiency**: FLOPs per parameter
- **Model Size**: Disk and memory footprint

### **Clinical Readiness Assessment**
- **Inference Speed**: <100ms (real-time), <1000ms (batch), >1000ms (slow)
- **Memory Requirements**: <1GB (low), <4GB (moderate), >4GB (high)
- **Attention Quality**: Correlation with pathological features
- **Interpretability**: Expert validation score
- **Deployment Recommendation**: Ready, Needs optimization, Not suitable

## 🎯 Pathologist Collaboration Workflow

### **Step 1: Attention Review**
```python
# Review generated attention visualizations
# Check if attention focuses on clinically relevant areas
# Assess attention patterns for diagnostic value
```

### **Step 2: Annotation Validation**
```python
# Use provided annotation interfaces
# Mark correct attention focus areas
# Identify false positives and false negatives
# Score attention quality (1-5 scale)
```

### **Step 3: Correlation Analysis**
```python
# Review attention-pathology correlation metrics
# Validate statistical significance
# Assess clinical relevance of correlations
# Provide expert feedback on patterns
```

### **Step 4: Clinical Assessment**
```python
# Evaluate model readiness for clinical deployment
# Consider workflow integration requirements
# Assess performance for diagnostic support
# Recommend improvements or optimizations
```

## ⚙️ Configuration Options

### **Analysis Configuration** (`inference_config.json`)

```json
{
  "enable_attention_analysis": true,      // Enable attention extraction and analysis
  "enhanced_visualization": true,        // Create enhanced multi-panel visualizations
  "pathology_correlation": true,         // Run pathology correlation analysis
  "complexity_analysis": true,           // Perform model complexity benchmarking
  "timing_analysis": true,               // Track timing and performance metrics
  "save_intermediate": true,             // Save intermediate analysis results
  "batch_analysis": true,                // Enable batch-level analysis
  "output_format": ["png", "pdf"],       // Output formats for visualizations
  "dpi": 300,                           // Resolution for saved images
  "create_report": true                  // Generate comprehensive reports
}
```

### **Performance Optimization**

```python
# For faster inference (no analysis)
python inference_dae_kan.py --model_path model.pth --input_dir images/ --batch_mode --disable_analysis

# For maximum analysis (comprehensive)
python inference_dae_kan.py --model_path model.pth --input_dir images/ --batch_mode --config full_analysis_config.json

# For GPU optimization
python inference_dae_kan.py --model_path model.pth --input_dir images/ --batch_mode --device cuda --precision fp16
```

## 🔍 Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use CPU
   python inference_dae_kan.py --model_path model.pth --input_image image.png --device cpu
   ```

2. **Model Loading Error**
   ```bash
   # Check model file format and path
   python inference_dae_kan.py --model_path /absolute/path/to/model.pth --input_image image.png
   ```

3. **Analysis Taking Too Long**
   ```bash
   # Disable analysis for faster inference
   python inference_dae_kan.py --model_path model.pth --input_dir images/ --batch_mode --disable_analysis
   ```

4. **Memory Issues During Analysis**
   ```bash
   # Use minimal analysis configuration
   python inference_dae_kan.py --model_path model.pth --input_dir images/ --batch_mode --config minimal_config.json
   ```

### **Performance Tips**

1. **Use GPU for faster inference**: `--device cuda`
2. **Batch processing**: Use `--batch_mode` for multiple images
3. **Disable analysis**: Use `--disable_analysis` for speed-critical applications
4. **Custom configuration**: Tailor analysis to your specific needs
5. **Memory management**: Close unnecessary applications during inference

## 📈 Integration with Clinical Workflows

### **DICOM Integration**
```python
# Future enhancement: Add DICOM support
python inference_dae_kan.py --model_path model.pth --input_dicom patient_study/ --batch_mode
```

### **Hospital Integration**
```python
# Future enhancement: Hospital system integration
python inference_dae_kan.py --model_path model.pth --hospital_system his --patient_id 12345
```

### **Real-time Analysis**
```python
# Future enhancement: Real-time analysis
python inference_dae_kan.py --model_path model.pth --real_time --camera_input
```

## 📚 API Reference

### **DAEKANInference Class**

```python
class DAEKANInference:
    def __init__(self, model_path: str, device: str = "auto", analysis_config: Optional[Dict] = None)
    def load_model(self) -> bool
    def run_single_inference(self, image_path: str, output_dir: str, sample_name: Optional[str] = None) -> Dict
    def run_batch_inference(self, input_dir: str, output_dir: str, file_extensions: List[str] = ['.png', '.jpg', '.jpeg', '.tiff']) -> Dict
    def run_complexity_analysis(self, output_dir: str) -> Dict
    def generate_final_report(self, output_dir: str, batch_results: Optional[Dict] = None)
```

### **Command Line Arguments**

- `--model_path`: Path to trained model (.pth file) **[Required]**
- `--input_image`: Path to single input image
- `--input_dir`: Path to directory containing images
- `--output_dir`: Output directory for results (default: inference_results)
- `--batch_mode`: Run batch inference on all images in input directory
- `--device`: Computation device (auto, cuda, cpu)
- `--config`: Path to analysis configuration JSON file
- `--enable_analysis`: Enable comprehensive analysis (default: True)
- `--disable_analysis`: Disable analysis for faster inference

This comprehensive inference system provides everything needed for clinical deployment and validation of your DAE-KAN model, with extensive analysis capabilities for pathological correlation validation.