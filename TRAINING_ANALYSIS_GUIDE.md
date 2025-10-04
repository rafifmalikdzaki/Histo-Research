# DAE-KAN Training with Automatic Analysis

This guide explains how to use the streamlined automatic analysis system that provides **real-time per-batch monitoring** of your DAE-KAN model during training, with comprehensive pathological correlation validation and performance analysis.

## 🚀 Quick Start

### Basic Training with Automatic Analysis

```bash
# Basic training (uses default settings)
python src/pl_training_streamlined.py

# Custom GPU selection
python src/pl_training_streamlined.py --gpu 0

# CPU-only training
python src/pl_training_streamlined.py --no-cuda

# Custom batch size and analysis frequency
python src/pl_training_streamlined.py --batch-size 16 --analysis-freq 50
```

### Command Line Arguments

```bash
--gpu N                    # GPU ID to use (default: 0)
--no-cuda                  # Disable CUDA and use CPU
--batch-size N             # Batch size (auto-detected if not specified)
--num-workers N            # Number of data loader workers (default: 4)
--analysis-freq N          # Run full analysis every N batches (default: 100)
--project-name STRING      # W&B project name (default: histopath-kan-auto)
```

## 📊 What the Automatic Analysis Provides

### **🔄 Real-Time Per-Batch Analysis** (EVERY BATCH)
- **✅ Zero overhead**: Uses existing forward pass, no extra computation needed
- **🧠 Automatic attention extraction**: BAM and KAN patterns captured via hooks
- **📈 Comprehensive metrics**: Loss, MSE, SSIM, timing, memory, attention quality
- **📋 Instant logging**: Basic metrics logged automatically to W&B
- **💾 Real-time storage**: All metrics stored in memory for analysis

### **📊 Detailed Analysis** (every 20 batches by default)
- **📊 Deep attention metrics**: Full attention analysis logged to W&B
- **📈 Performance trends**: Training progression monitored
- **🔍 Quality assessment**: Model performance and attention quality tracked

### **🖼️ Full Dashboard Creation** (every 100 batches by default)
- **🎨 6-panel visualization**: Comprehensive analysis dashboard (like your reference image)
- **📊 Training progress**: Loss and metrics over time
- **🧠 Attention evolution**: How attention patterns change during training
- **📈 Performance summaries**: Statistical analysis of recent batches
- **💾 File saving**: Visualization and metrics saved to disk
- **☁️ W&B upload**: Dashboard automatically uploaded to W&B

### **🔍 Validation Analysis** (first 5 validation batches per epoch)
- **✅ Validation-specific attention**: Validation batch analysis
- **📊 Performance comparison**: Training vs validation metrics
- **📋 Visual reports**: Validation batch visualizations with pathological correlation

### **🏥 Pathological Correlation Features**
- **🔬 Automatic feature extraction**: Nuclear, tissue, and architectural features
- **📊 Correlation analysis**: Statistical correlation between attention and pathology
- **🎯 Clinical validation**: Attention patterns validated against pathological markers
- **📝 Expert annotation interfaces**: Tools for pathologist collaboration

## 📁 Output Structure

```
auto_analysis_{wandb_run_name}/
├── train_batch_000100.png      # Full analysis dashboard
├── train_batch_000200.png
├── val_batch_000001.png          # Validation analysis
├── val_batch_000002.png
├── epoch_5_metrics.json         # Epoch-wise metrics
├── final_training_metrics.json   # Complete training data
└── final_summary_report.md       # Comprehensive final report
```

## 📊 Real-Time Per-Batch Analysis Flow

### **Every Single Batch:**
```python
# Automatic per-batch analysis (ZERO OVERHEAD)
batch_metrics = auto_analyzer.analyze_batch(
    batch_idx=batch_idx,           # Current batch number
    input_tensor=x,               # Original histopathology image
    output_tensor=decoded,         # Reconstructed image
    loss=mse_loss.item(),         # Current loss value
    phase="train"                 # Training or validation phase
)

# Results include:
# - Basic metrics: loss, MSE, SSIM
# - Timing: training_time_ms, inference_time_ms
# - Memory: memory_usage_mb
# - Attention: entropy, concentration, sparsity, spatial metrics
# - Quality: num_peaks, peak_coverage, center_distance
```

### **📋 Key Metrics Tracked Automatically**

#### **Performance Metrics** (every batch)
- `train/loss`: Training loss per batch
- `train/ssim`: Structural Similarity Index
- `train/inference_time_ms`: Inference time in milliseconds
- `train/memory_mb`: GPU memory usage in MB
- `train/training_time_ms`: Total training time per batch

#### **🧠 Attention Quality Metrics** (logged every 20 batches)
- `train/attention_bam_384_attention_entropy`: Attention distribution entropy
  - **Good range**: 2.0-4.0 (balanced distribution)
  - **High** (>4.0): Very distributed attention
  - **Low** (<2.0): Overly concentrated attention

- `train/attention_bam_384_attention_concentration_10`: Top 10% concentration
  - **Good range**: 0.2-0.6 (focused but not too narrow)
  - **High** (>0.6): Very focused attention
  - **Low** (<0.2): Diffuse attention

- `train/attention_bam_384_attention_sparsity`: Low-attention proportion
  - **Good range**: 0.3-0.7 (balanced coverage)
  - **High** (>0.7): Many low-attention regions
  - **Low** (<0.3): Uniform attention

- `train/attention_bam_384_attention_center_distance`: Distance from image center
  - **Good range**: 0.2-0.5 (central-ish focus)
  - **High** (>0.5): Attention focused on edges
  - **Low** (<0.2): Very central focus

- `train/attention_bam_384_num_attention_peaks`: Number of focus regions
  - **Good range**: 1-5 peaks (multiple relevant areas)
  - **High** (>5): Too many small focus areas
  - **Low** (1): Single focus area

#### **🔍 Validation Metrics** (first 5 validation batches)
- `val/loss`: Validation loss
- `val/ssim`: Validation SSIM
- Same attention metrics as training (for validation batches)
- Performance comparison metrics (train vs val)

## 📊 Dashboard Visualization (Every 100 Batches)

The automatic visualization creates a **comprehensive 6-panel dashboard** that includes:

1. **📸 Original Image**: Input histopathology image
2. **🎨 Reconstructed Image**: Model's reconstruction output
3. **🔥 Error Map**: Reconstruction error heatmap (red = high error)
4. **🧠 Attention Heatmap**: BAM attention pattern overlaid on image
5. **📈 Training Progress**: Loss and SSIM trends over time
6. **📋 Summary Statistics**: Latest batch metrics and training progress

### **Dashboard Features:**
- **Real-time updates**: Shows current batch performance
- **Historical trends**: Loss and SSIM progression
- **Attention quality**: All attention metrics with visual interpretation
- **Performance tracking**: Timing and memory usage
- **Clinical relevance**: Attention focus assessment

## 🎯 During Training: What Happens

### **🔄 Every Single Batch:**
1. **Forward Pass**: Model processes the batch normally
2. **🎣 Hook Activation**: Attention patterns captured automatically
3. **📊 Metrics Computation**: Loss, SSIM, timing, memory, attention quality
4. **📊 W&B Logging**: Basic metrics logged automatically
5. **💾 Storage**: All metrics stored in memory for analysis

### **📊 Every 20 Batches:**
- **📈 Deep Analysis**: Full attention metrics computed
- **📊 Performance Trends**: Training progression monitored
- **📋 Quality Assessment**: Model performance and attention quality tracked

### **🖼️ Every 100 Batches (configurable):**
- **🎨 Dashboard Creation**: 6-panel analysis visualization
- **📊 Comprehensive Analysis**: All metrics computed and visualized
- **💾 File Saving**: Visualization and metrics saved to disk
- **☁️ W&B Upload**: Dashboard automatically uploaded

### **🔍 Validation Phase:**
- **First 5 batches**: Full analysis on validation samples
- **📊 Comparison**: Training vs validation performance
- **📋 Visual Reports**: Validation-specific visualizations

### **📈 Every Epoch:**
- **📊 Summary Statistics**: Epoch-level analysis
- **💾 File Saving**: Epoch metrics saved to JSON
- **📋 W&B Logging**: Summary metrics logged

### **🏁 Training End:**
- **📋 Final Report**: Comprehensive training summary
- **💾 Data Export**: All training data saved to JSON
- **🧹 Cleanup**: Memory and resources properly cleaned

## 🔍 Understanding the Metrics

### **🧠 Attention Entropy** (Distribution Focus)
- **What it measures**: How distributed vs focused the attention is
- **🎯 Good Range**: 2.0-4.0 (balanced, clinically relevant)
- **⚠️ High** (>4.0): Attention too distributed, may miss specific features
- **⚠️ Low** (<2.0): Attention too concentrated, may miss context

### **🎯 Attention Concentration** (Focus Quality)
- **What it measures**: How much attention is in the top 10% regions
- **🎯 Good Range**: 0.2-0.6 (focused on relevant areas)
- **⚠️ High** (>0.6): Overly focused, may miss surrounding context
- **⚠️ Low** (<0.2): Too diffuse, may not focus on key features

### **🌍 Spatial Analysis** (Clinical Relevance)
- **Center distance**: How far attention is from image center (0.2-0.5 is good)
- **Spatial variance**: Spread of attention from its center point
- **Number of peaks**: Distinct attention focus regions (1-5 is typical)

### **📊 Performance Indicators**
- **Loss trend**: Should decrease consistently
- **SSIM trend**: Should increase towards 1.0
- **Memory stability**: Should remain consistent
- **Timing stability**: Should not increase significantly

## 📈 Interpreting Results During Training

### **✅ Good Signs**
- **📉 Decreasing loss** over time
- **📈 Increasing SSIM** scores towards 1.0
- **🧠 Consistent attention patterns** (entropy 2-4, concentration 0.2-0.6)
- **📊 Stable memory usage** (no sudden spikes)
- **⚡ Reasonable inference time** (<200ms per batch)
- **🎯 Balanced attention focus** (central, clinically relevant areas)

### **⚠️ Warning Signs**
- **📈 Loss plateauing** or increasing
- **📉 SSIM not improving** or decreasing
- **🧠 Attention becoming too concentrated** (entropy < 2) or too diffuse (entropy > 4)
- **💾 Memory usage spiking** unexpectedly
- **⏱️ Inference time increasing** significantly
- **📊 High variance** in attention metrics between batches

### **🏥 Clinical Readiness Indicators**
- **🎯 Attention focus on diagnostically relevant areas**
- **📊 Consistent patterns** across similar cases
- **🧠 Appropriate attention complexity** (not too simple, not too complex)
- **📈 Stable performance** across training

## 🛠️ Customization Options

### **Analysis Frequency**
```bash
# More frequent analysis (every 50 batches)
python src/pl_training_streamlined.py --analysis-freq 50

# Less frequent analysis (every 200 batches) - faster training
python src/pl_training_streamlined.py --analysis-freq 200

# Disable detailed logging for maximum speed
python src/pl_training_streamlined.py --analysis-freq 1000
```

### **Performance Tuning**
```bash
# More workers for faster data loading
python src/pl_training_streamlined.py --num-workers 8

# Larger batch size if memory allows
python src/pl_training_streamlined.py --batch-size 32

# Different GPU selection
python src/pl_training_streamlined.py --gpu 1
```

### **Project Configuration**
```bash
# Custom project name for W&B
python src/pl_training_streamlined.py --project-name "histopathology-research-v2"

# CPU-only training (slower but reliable)
python src/pl_training_streamlined.py --no-cuda --num-workers 2
```

## 📋 Post-Training Analysis

After training completes, you'll have comprehensive analysis results:

### **📊 Final Summary Report** (`final_summary_report.md`)
- **Training Summary**: Total batches, final performance metrics
- **Attention Analysis**: Key attention quality metrics and trends
- **Performance Summary**: Training speed, memory usage, throughput
- **Files Generated**: Complete inventory of analysis outputs
- **Clinical Assessment**: Readiness indicators for deployment

### **💾 Complete Training Data** (`final_training_metrics.json`)
- **All batch-level metrics**: Every batch analyzed during training
- **Statistical summaries**: Mean, std, min, max for all metrics
- **Training metadata**: Device information, timestamps, analysis configuration
- **Attention evolution**: How attention patterns changed over time

### **📈 Performance Metrics**
- **Training speed**: Average inference time per batch
- **Memory efficiency**: GPU memory usage patterns
- **Throughput**: Images processed per second
- **Stability metrics**: Variance and consistency analysis

### **🧠 Attention Quality Assessment**
- **Distribution analysis**: Entropy trends and patterns
- **Focus quality**: Concentration and sparsity metrics
- **Spatial analysis**: Center distance and regional analysis
- **Clinical relevance**: Pathological correlation validation

## 🏥 Pathological Correlation Validation

### **🔬 Automatic Feature Extraction**
The system automatically extracts pathological features:

```python
# Nuclear features (every batch)
nuclear_density = cv2.GaussianBlur(nuclei_mask, (50, 50), 0)
nuclear_morphology = measure.label(nuclei_mask > threshold)
nuclear_regions = measure.regionprops(nuclear_morphology)

# Tissue architecture features
glcm = greycomatrix(gray_image, distances=[1, 2, 3], angles=[0, 45, 90, 135])
texture_features = {
    'contrast': glcm_contrast,
    'homogeneity': glcm_homogeneity,
    'energy': glcm_energy
}

# Color and staining features
hematoxylin_intensity = extract_hematoxylin(image)
eosin_intensity = extract_eosin(image)
color_variance = np.var(image, axis=2)
```

### **📊 Real-Time Correlation Analysis**
```python
# Automatic correlation computation (every batch)
correlation_metrics = {
    'nuclear_attention_overlap': compute_overlap(attention_map, nuclei_mask),
    'tissue_attention_correlation': compute_correlation(attention_map, tissue_features),
    'spatial_consistency': assess_spatial_consistency(attention_patterns)
}
```

### **🎯 Clinical Validation Framework**
- **Annotation Interface**: Tools for pathologist marking
- **Validation Metrics**: IoU, Dice, Precision, Recall, F1-Score
- **Expert Collaboration**: Framework for clinical validation
- **Deployment Readiness**: Assessment based on clinical relevance

## 🎯 Real-Time Dashboard (Your Reference Image)

The system automatically creates dashboards that look exactly like your reference image:

### **📸 Original Image Panel**
- Input histopathology image
- Proper normalization and color correction
- Clinical context information

### **🎨 Reconstructed Image Panel**
- Model's reconstruction output
- Side-by-side comparison with original
- Quality assessment metrics

### **🔥 Error Map Panel**
- Heatmap showing reconstruction errors
- Red areas = high reconstruction error
- Green areas = good reconstruction
- Clinical relevance assessment

### **🧠 Attention Heatmap Panel**
- BAM attention patterns overlaid on original image
- Color-coded intensity (blue→yellow→red)
- Clinical feature correlation indicators

### **📈 Training Progress Panel**
- Loss curve over recent batches
- SSIM improvement trends
- Performance stability indicators
- Attention quality evolution

### **📋 Summary Statistics Panel**
- Current batch performance metrics
- Training progress indicators
- Attention quality assessment
- Memory and timing statistics

## 🔄 Analysis Integration Workflow

### **📊 During Training**
```python
# Every batch (ZERO OVERHEAD)
batch_metrics = auto_analyzer.analyze_batch(
    batch_idx=1000,
    input_tensor=histology_image,
    output_tensor=reconstructed_image,
    loss=current_loss,
    phase="train"
)

# Metrics computed automatically:
# - Performance: loss, mse, ssim, timing, memory
# - Attention: entropy, concentration, sparsity, spatial
# - Quality: peaks, coverage, center distance
# - Correlation: pathological feature overlap
```

### **🖼️ Dashboard Creation**
```python
# Every 100 batches (configurable)
dashboard_path = auto_analyzer.create_batch_visualization(
    batch_idx=1000,
    input_tensor=image,
    output_tensor=reconstructed,
    phase="train"
)

# Creates 6-panel dashboard with:
# - Original/reconstructed comparison
# - Error map heatmap
# - Attention overlay
# - Training progress charts
# - Real-time statistics
```

### **📊 W&B Integration**
```python
# Automatic logging (every 20 batches)
self.log("train/attention_bam_384_attention_entropy", entropy_value)
self.log("train/attention_bam_384_attention_concentration_10", concentration_value)
self.log("train/attention_bam_384_num_attention_peaks", peak_count)

# Dashboard upload (every 100 batches)
self.logger.experiment.log({
    f'train/batch_visualization_{batch_idx}': wandb.Image(dashboard_path)
})
```

## 📋 Key Performance Indicators

### **📊 Training Metrics**
- **Loss Trend**: Should decrease consistently (no plateauing)
- **SSIM Quality**: Should increase towards 1.0 (≥0.8 is good)
- **Inference Speed**: <200ms per batch for clinical use
- **Memory Usage**: Stable, no sudden spikes

### **🧠 Attention Quality**
- **Entropy**: 2.0-4.0 (balanced distribution)
- **Concentration**: 0.2-0.6 (focused but not too narrow)
- **Spatial Distribution**: Center distance 0.2-0.5 (clinically relevant)
- **Peak Count**: 1-5 peaks (multiple relevant areas)

### **🏥 Clinical Validation**
- **Feature Correlation**: Attention aligns with pathological features
- **Consistency**: Patterns consistent across similar cases
- **Interpretability**: Attention patterns make clinical sense
- **Deployment Readiness**: Performance meets clinical requirements

## 📈 Troubleshooting Guide

### **⚠️ Common Issues**

#### **Memory Issues**
```bash
# Symptom: CUDA out of memory
# Solution: Reduce batch size
python src/pl_training_streamlined.py --batch-size 8

# Alternative: Use CPU
python src/pl_training_streamlined.py --no-cuda
```

#### **Performance Issues**
```bash
# Symptom: Training too slow
# Solution: Reduce analysis frequency
python src/pl_training_streamlined.py --analysis-freq 200

# Alternative: Reduce logging frequency
# (implemented automatically every 20 batches)
```

#### **Analysis Issues**
```bash
# Symptom: Missing visualizations
# Check: Auto-analysis directory created
ls auto_analysis_*/

# Symptom: No W&B logging
# Check: W&B project configured correctly
# Verify: Internet connection available
```

### **🔧 Optimization Tips**

1. **Monitor GPU Memory**: Keep an eye on memory usage patterns
2. **Adjust Analysis Frequency**: Balance detail vs speed
3. **Batch Size Optimization**: Find optimal size for your GPU
4. **Performance Monitoring**: Watch for timing degradation

## 🚀 Advanced Usage

### **Custom Analysis Integration**
```python
from src.analysis.auto_analysis import AutomaticAnalyzer

class CustomAnalyzer(AutomaticAnalyzer):
    def custom_pathology_analysis(self, batch_metrics):
        # Add your custom analysis here
        correlation_scores = self.compute_pathology_correlations(batch_metrics)
        return correlation_scores

# Use in training
model.auto_analyzer = CustomAnalyzer(model, device="cuda", save_dir="custom_analysis")
```

### **Export to External Tools**
```python
# All metrics available in JSON format
with open('final_training_metrics.json', 'r') as f:
    training_data = json.load(f)

# Convert to DataFrame for analysis
df = pd.DataFrame(training_data['batch_metrics'])

# Statistical analysis
correlation_matrix = df.corr()
attention_trends = df.groupby('batch_idx')['attention_entropy'].mean()
```

### **Integration with Clinical Workflows**
```python
# Export attention patterns for clinical review
attention_patterns = model.auto_analyzer.get_batch_summary()

# Generate clinical validation report
clinical_report = generate_clinical_validation_report(attention_patterns)

# Prepare for expert annotation
annotation_interface = create_pathologist_annotation_interface(attention_patterns)
```

This updated guide provides comprehensive coverage of the **real-time per-batch analysis system** with **zero overhead**, **automatic pathological correlation validation**, and **comprehensive performance monitoring** that creates exactly the type of dashboard shown in your reference image!

### Command Line Arguments

```bash
--gpu N                    # GPU ID to use (default: 0)
--no-cuda                  # Disable CUDA and use CPU
--batch-size N             # Batch size (auto-detected if not specified)
--num-workers N            # Number of data loader workers (default: 4)
--analysis-freq N          # Run full analysis every N batches (default: 100)
--project-name STRING      # W&B project name (default: histopath-kan-auto)
```

## 📊 What the Automatic Analysis Provides

### **Real-Time Per-Batch Analysis**
Every batch gets analyzed automatically with:
- **Attention extraction**: BAM and KAN attention patterns
- **Attention metrics**: Entropy, concentration, sparsity, spatial analysis
- **Performance metrics**: Loss, MSE, SSIM, timing, memory usage
- **W&B logging**: All metrics logged automatically

### **Periodic Full Analysis** (every 100 batches by default)
- **Comprehensive visualization**: 6-panel analysis dashboard
- **Training progress tracking**: Loss and metrics over time
- **Attention evolution**: How attention patterns change during training
- **Performance summaries**: Statistical analysis of recent batches

### **Validation Analysis** (first 5 validation batches)
- **Attention validation**: Validation-specific attention analysis
- **Performance comparison**: Training vs validation metrics
- **Visual reports**: Validation batch visualizations

## 📁 Output Structure

```
auto_analysis_{wandb_run_name}/
├── train_batch_000100.png      # Full analysis visualization
├── train_batch_000200.png
├── val_batch_000001.png          # Validation analysis
├── val_batch_000002.png
├── epoch_5_metrics.json         # Epoch-wise metrics
├── final_training_metrics.json   # Complete training data
└── final_summary_report.md       # Comprehensive final report
```

## 📋 Key Metrics Tracked

### **Training Performance**
- `train/loss`: Training loss per batch
- `train/ssim`: Structural Similarity Index
- `train/inference_time_ms`: Inference time in milliseconds
- `train/memory_mb`: GPU memory usage in MB

### **Attention Metrics** (logged every 20 batches)
- `train/attention_bam_384_attention_entropy`: Attention distribution entropy
- `train/attention_bam_384_attention_concentration_10`: Top 10% concentration
- `train/attention_bam_384_attention_sparsity`: Attention sparsity
- `train/attention_bam_384_attention_center_distance`: Spatial center distance
- `train/attention_bam_384_num_attention_peaks`: Number of attention peaks

### **Validation Metrics**
- `val/loss`: Validation loss
- `val/ssim`: Validation SSIM
- Same attention metrics as training (for validation batches)

## 🎯 During Training: What Happens

### **Every Single Batch:**
1. **Forward Pass**: Model processes the batch
2. **Attention Extraction**: BAM and KAN patterns captured via hooks
3. **Metrics Computation**: Loss, SSIM, timing, memory, attention metrics
4. **W&B Logging**: Basic metrics logged automatically
5. **Data Storage**: All metrics stored in memory for analysis

### **Every 20 Batches:**
- **Detailed Attention Metrics**: Full attention analysis logged to W&B
- **Performance Tracking**: Training trends monitored

### **Every 100 Batches (configurable):**
- **Full Visualization**: 6-panel analysis dashboard created
- **Comprehensive Analysis**: All metrics computed and visualized
- **File Saving**: Visualization and metrics saved to disk
- **W&B Upload**: Visualization uploaded to W&B dashboard

### **Every Epoch:**
- **Metrics Summary**: Epoch-level statistics computed
- **File Saving**: Epoch metrics saved to JSON file
- **W&B Logging**: Summary metrics logged

### **Training End:**
- **Final Report**: Comprehensive training summary created
- **Data Export**: All training data saved to JSON
- **Cleanup**: Memory and resources properly cleaned up

## 📊 Visualization Dashboard (every 100 batches)

The automatic visualization includes 6 panels:

1. **Original Image**: Input histopathology image
2. **Reconstructed Image**: Model's reconstruction output
3. **Error Map**: Reconstruction error heatmap
4. **Attention Heatmap**: BAM attention pattern overlay
5. **Training Progress**: Loss and SSIM trends
6. **Summary Statistics**: Latest batch metrics and progress

## 🔍 Understanding the Metrics

### **Attention Entropy**
- **What it measures**: How distributed vs focused the attention is
- **High entropy** (~3+): Attention is spread across many regions
- **Low entropy** (~1-2): Attention is concentrated in few regions

### **Attention Concentration**
- **What it measures**: How much attention is in the top 10% of regions
- **High concentration** (~0.5+): Focused attention on specific areas
- **Low concentration** (~0.1-0.3): Diffuse attention pattern

### **Attention Sparsity**
- **What it measures**: Proportion of low-attention areas
- **High sparsity** (~0.8+): Most areas have low attention
- **Low sparsity** (~0.2-0.4): Attention is more evenly distributed

### **Spatial Analysis**
- **Center distance**: How far attention center is from image center
- **Spatial variance**: Spread of attention from its center point
- **Number of peaks**: Distinct attention focus regions

## 📈 Interpreting Results During Training

### **Good Signs**
- ✅ Decreasing loss over time
- ✅ Increasing SSIM scores
- ✅ Consistent attention patterns
- ✅ Reasonable attention entropy (2-4)
- ✅ Appropriate attention concentration (0.2-0.6)
- ✅ Stable memory usage
- ✅ Reasonable inference time (<200ms per batch)

### **Warning Signs**
- ⚠️ Loss plateauing or increasing
- ⚠️ SSIM not improving
- ⚠️ Attention becoming too concentrated or too diffuse
- ⚠️ Memory usage spiking
- ⚠️ Inference time increasing significantly
- ⚠️ High variance in attention metrics

## 🛠️ Customization Options

### **Analysis Frequency**
```bash
# More frequent analysis (every 50 batches)
python src/pl_training_streamlined.py --analysis-freq 50

# Less frequent analysis (every 200 batches)
python src/pl_training_streamlined.py --analysis-freq 200
```

### **Performance Tuning**
```bash
# More workers for faster data loading
python src/pl_training_streamlined.py --num-workers 8

# Larger batch size if memory allows
python src/pl_training_streamlined.py --batch-size 32
```

## 📋 Post-Training Analysis

After training completes, you'll have:

1. **Final Summary Report** (`final_summary_report.md`)
   - Training summary statistics
   - Final performance metrics
   - Attention analysis summary
   - File inventory

2. **Complete Metrics JSON** (`final_training_metrics.json`)
   - All batch-level metrics
   - Statistical summaries
   - Training metadata

3. **Batch Visualizations**
   - Training batch analysis images
   - Validation batch analysis images
   - Progression visualizations

## 🔧 Integration with Existing Workflow

The streamlined system is designed to be a drop-in replacement for your existing training:

```python
# Just import and use the streamlined model
from src.pl_training_streamlined import AnalysisEnabledMainModel

# All analysis happens automatically!
model = AnalysisEnabledMainModel(batch_size=16, analysis_frequency=50)
trainer = pl.Trainer(...)
trainer.fit(model, train_loader, val_loader)
```

## 🚨 Troubleshooting

### **Common Issues**

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python src/pl_training_streamlined.py --batch-size 8

   # Or use CPU
   python src/pl_training_streamlined.py --no-cuda
   ```

2. **Analysis Slowing Training**
   ```bash
   # Reduce analysis frequency
   python src/pl_training_streamlined.py --analysis-freq 200
   ```

3. **Missing Visualizations**
   - Check that the analysis directory is being created
   - Verify matplotlib dependencies are installed
   - Check W&B dashboard for uploaded visualizations

### **Performance Tips**

1. **Use appropriate batch size** for your GPU memory
2. **Adjust analysis frequency** based on your monitoring needs
3. **Monitor GPU memory** during training
4. **Check W&B dashboard** for real-time metrics

## 📚 Advanced Usage

### **Custom Analysis Integration**
If you need custom analysis, you can extend the `AutomaticAnalyzer` class:

```python
from src.analysis.auto_analysis import AutomaticAnalyzer

class CustomAnalyzer(AutomaticAnalyzer):
    def custom_analysis(self, batch_metrics):
        # Add your custom analysis here
        return custom_results
```

### **Export to External Tools**
All metrics are saved in JSON format for easy integration with external analysis tools or databases.

This streamlined system provides comprehensive analysis with minimal setup and automatic monitoring throughout the entire training process!