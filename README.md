# DAE-KAN: Attention-Enhanced Dual Autoencoder for Liver Histopathological Image Clustering

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Official implementation of:** "An Attention-Enhanced Dual Autoencoder Clustering for Liver Histopathological Images"

## 📋 Overview

This repository contains the official PyTorch implementation of our attention-enhanced dual autoencoder framework that integrates:

- **Kolmogorov-Arnold Networks (KAN)** - Learnable activation functions based on B-splines
- **Efficient Channel Attention (ECA)** - Lightweight channel attention mechanism
- **Bottleneck Attention Module (BAM)** - Spatial and channel attention for feature refinement
- **Dual Autoencoder Architecture** - Two-stage encoding-decoding for hierarchical feature learning
- **Multiple Clustering Algorithms** - KMeans, Bisecting KMeans, and Gaussian Mixture Models

### Key Features

✅ **Reproducible experiments** - Global seed setting, configuration management, timestamped outputs  
✅ **Statistical significance** - Multi-seed evaluation (N=5) with mean ± std and hypothesis testing  
✅ **Comprehensive metrics** - Internal (DBI, CHI, Silhouette) and external (AMI, ARI) clustering metrics  
✅ **Bootstrap stability** - Resampling-based cluster stability analysis  
✅ **Cross-domain transfer** - PanNuke ↔ IMI dataset transfer learning evaluation  
✅ **Runtime profiling** - GPU memory, training/inference timing analysis  
✅ **Interpretability metrics** - Attention entropy, activation sparsity, reconstruction fidelity  
✅ **Baseline comparisons** - SimCLR, BYOL, VAE, and single autoencoder baselines  

---

## 🗂️ Repository Structure

```
histodae/
├── src/
│   ├── models/
│   │   ├── model.py                    # Main DAE-KAN-Attention architecture
│   │   ├── factory.py                  # Model factory for different variants
│   │   ├── model_baseline.py           # Baseline (no KAN, no BAM)
│   │   ├── model_bam_only.py           # BAM attention only
│   │   ├── model_kan_only.py           # KAN only (no attention)
│   │   ├── model_no_bam.py             # No BAM variant
│   │   ├── model_no_kan.py             # No KAN variant
│   │   ├── model_no_eka.py             # No Efficient KAN Attention variant
│   │   ├── attention_mechanisms/
│   │   │   ├── bam.py                  # Bottleneck Attention Module
│   │   │   ├── eca.py                  # Efficient Channel Attention
│   │   │   └── cbam.py                 # Convolutional Block Attention Module
│   │   └── kan_convolutional/
│   │       ├── efficient_kan.py        # KAN linear layer implementation
│   │       └── efficient_kan_conv_fixed.py  # KAN convolutional layer
│   │
│   ├── analysis/
│   │   └── auto_analysis.py            # Automatic analysis during training
│   │
│   ├── utils/
│   │   ├── __init__.py                 # Utilities package
│   │   └── reproducibility.py          # Seed setting, config management
│   │
│   ├── histodata.py                    # Dataset loading utilities
│   └── pl_training_with_analysis_and_optimization.py  # Main training script
│
├── config/
│   └── experiment_config.yaml          # Centralized hyperparameter configuration
│
├── baselines/
│   ├── simclr_baseline.py              # SimCLR encoder + KMeans
│   ├── byol_baseline.py                # BYOL encoder + KMeans
│   ├── vae_baseline.py                 # VAE + KMeans
│   └── single_ae_baseline.py           # Single autoencoder (ablation)
│
├── outputs/                            # Experiment outputs (auto-generated)
│   └── exp_<timestamp>_<name>/
│       ├── checkpoints/                # Model checkpoints
│       ├── metrics/                    # Training metrics
│       ├── visualizations/             # Attention maps, reconstructions
│       ├── configs/                    # Experiment configurations
│       ├── embeddings/                 # Extracted latent features
│       └── logs/                       # Training logs
│
├── results/                            # Aggregated results
│   └── summary_stats.csv               # Multi-seed summary table
│
├── run_multi_seed_experiments.py       # Multi-seed experiment runner
├── evaluate_stats.py                   # Statistical significance analysis
├── sensitivity_analysis.py             # Hyperparameter sensitivity (Task 4)
├── cross_domain_eval.py                # Cross-domain transfer evaluation (Task 5)
├── profile_runtime.py                  # Runtime & memory profiling (Task 6)
├── extract_tile_embeddings.py          # Embedding extraction for clustering
├── run_ablation_study.py               # Ablation study runner
└── ablation_study_comparison.py        # Ablation study comparison
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/histodae.git
cd histodae

# Create virtual environment (using uv or venv)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or using uv
uv pip install -r requirements.txt
```

### 2. Data Preparation

Place your datasets in the `data/processed/` directory:

```bash
data/
└── processed/
    ├── PANnuke/              # PanNuke dataset
    │   ├── train_pannuke.csv
    │   └── test_pannuke.csv
    └── HeparUnifiedPNG/      # IMI liver histopathology dataset
        ├── train_heparunifiedpng.csv
        └── test_heparunifiedpng.csv
```

Create CSV files using the provided scripts:

```bash
# For PanNuke
python create_pannuke_csv.py --data-dir data/processed/PANnuke

# For IMI/Hepar
python create_hepar_csv.py --data-dir data/processed/HeparUnifiedPNG
```

### 3. Train the Model

**Single run (default seed = 42):**

```bash
python src/pl_training_with_analysis_and_optimization.py \
    --model-name dae_kan_attention \
    --dataset hepar \
    --max-epochs 30 \
    --batch-size 8 \
    --seed 42
```

**Multi-seed evaluation (for statistical significance):**

```bash
# Run with 5 seeds (42, 123, 456, 789, 1011)
python run_multi_seed_experiments.py \
    --model-name dae_kan_attention \
    --dataset HeparUnifiedPNG \
    --epochs 30 \
    --batch-size 8
```

### 4. Evaluate Results

**Statistical significance analysis:**

```bash
# Aggregate multi-seed results
python evaluate_stats.py \
    --model dae_kan_attention \
    --dataset HeparUnifiedPNG \
    --run-tests \
    --output results/summary_stats.csv
```

---

## 📊 Reproducing All Experiments

### Main Results (Table X in paper)

```bash
# Proposed method (DAE-KAN-Attention) - 5 seeds
python run_multi_seed_experiments.py --model-name dae_kan_attention --epochs 30

# Baseline variants
python run_multi_seed_experiments.py --model-name baseline --epochs 30
python run_multi_seed_experiments.py --model-name bam_only --epochs 30
python run_multi_seed_experiments.py --model-name kan_only --epochs 30
python run_multi_seed_experiments.py --model-name no_bam --epochs 30
python run_multi_seed_experiments.py --model-name no_kan --epochs 30
```

### Ablation Study (Table Y in paper)

```bash
# Run ablation study with variance reporting
python run_ablation_study.py \
    --mode different-architecture \
    --models dae_kan_attention baseline bam_only kan_only no_bam no_kan \
    --max-epochs 30

# Compare ablation results
python ablation_study_comparison.py --base_dir auto_analysis --all_runs
```

### Hyperparameter Sensitivity Analysis

```bash
python sensitivity_analysis.py \
    --latent-dims 32 64 128 256 \
    --n-clusters 3 4 5 6 7 \
    --kan-spline-orders 3 5 7 \
    --eca-kernel-sizes 3 5 7 \
    --output-dir outputs/sensitivity
```

### Cross-Domain Transfer Learning

```bash
python cross_domain_eval.py \
    --source PanNuke \
    --target HeparUnifiedPNG \
    --strategy zero_shot

# With fine-tuning
python cross_domain_eval.py \
    --source PanNuke \
    --target HeparUnifiedPNG \
    --strategy finetune \
    --finetune-epochs 10
```

### Runtime & Memory Profiling

```bash
python profile_runtime.py \
    --model-name dae_kan_attention \
    --batch-sizes 4 8 16 \
    --output outputs/profiling/runtime_report.csv
```

---

## 🧪 Model Variants

| Model Name | Description | KAN | BAM | ECA |
|------------|-------------|-----|-----|-----|
| `dae_kan_attention` | **Full proposed model** | ✓ | ✓ | ✓ |
| `baseline` | Standard convolutions only | ✗ | ✗ | ✗ |
| `bam_only` | BAM attention throughout | ✗ | ✓ | ✗ |
| `kan_only` | KAN layers only | ✓ | ✗ | ✗ |
| `no_bam` | KAN + simple attention | ✓ | ✗ | ✓ |
| `no_kan` | Standard conv + BAM | ✗ | ✓ | ✓ |
| `no_eka` | Efficient standard conv + BAM | ✗ | ✓ | ✓ |

---

## 📈 Expected Output Structure

After running experiments, your `outputs/` directory will contain:

```
outputs/
└── exp_20260331_143052_dae_kan_attention_seed42/
    ├── checkpoints/
    │   ├── epoch=29-val_loss=0.0234.ckpt
    │   └── last.ckpt
    ├── configs/
    │   ├── experiment.yaml
    │   └── experiment.json
    ├── metrics/
    │   ├── training_metrics.csv
    │   └── clustering_metrics.csv
    ├── visualizations/
    │   ├── batch_visualizations/
    │   ├── attention_maps/
    │   └── reconstructions/
    ├── embeddings/
    │   ├── train_embeddings.npy
    │   └── train_metadata.csv
    └── logs/
        └── training.log
```

---

## 🔬 Key Hyperparameters

Default hyperparameters (from `config/experiment_config.yaml`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latent_dim` | 128 | Latent representation dimension |
| `n_clusters` | 5 | Number of clusters |
| `kan_grid_size` | 3 | KAN B-spline grid intervals |
| `kan_spline_order` | 2 | KAN polynomial order |
| `eca_kernel_size` | 3 | ECA convolution kernel size |
| `bam_reduction` | 16 | BAM channel reduction ratio |
| `learning_rate` | 0.002 | Initial learning rate |
| `batch_size` | 8 | Training batch size |
| `epochs` | 30 | Number of training epochs |
| `seed` | 42 | Random seed |

---

## 📊 Clustering Metrics

### Internal Validation (no ground truth needed)

- **Davies-Bouldin Index (DBI)** - Lower is better
- **Calinski-Harabasz Index (CHI)** - Higher is better
- **Silhouette Score** - Higher is better (-1 to 1)
- **Xie-Beni Index (XBI)** - Lower is better

### External Validation (when labels available)

- **Adjusted Mutual Information (AMI)** - Higher is better (0 to 1)
- **Adjusted Rand Index (ARI)** - Higher is better (-1 to 1)

### Stability Analysis

- **Bootstrap Stability** - Variance of DBI across 10 resamples (80% data each)

---

## 🖥️ Hardware Requirements

**Minimum:**
- GPU: 8GB VRAM (e.g., RTX 3060)
- RAM: 16GB
- Storage: 50GB free space

**Recommended:**
- GPU: 24GB VRAM (e.g., RTX 3090/4090, A100)
- RAM: 32GB
- Storage: 100GB SSD

**Training time (30 epochs, batch size 8):**
- PanNuke (~20k images): ~4 hours on RTX 3090
- IMI (~200 images): ~30 minutes on RTX 3090

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{muflikhah2026attention,
  title={An Attention-Enhanced Dual Autoencoder Clustering for Liver Histopathological Images},
  author={Muflikhah, Lailil and others},
  journal={Computer Methods in Biomechanics and Biomedical Engineering: Imaging \& Visualization},
  year={2026},
  publisher={Taylor \& Francis}
}
```

---

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **PanNuke Dataset** - [Link](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke)
- **IMI Liver Dataset** - Provided by collaborating institution
- **PyTorch Lightning** - For streamlined training
- **Weights & Biases** - For experiment tracking

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

**Corresponding Author:** Dr. Lailil Muflikhah  
**Email:** lailil@ub.ac.id  
**Affiliation:** Universitas Brawijaya, Indonesia

---

## 📅 Revision History

- **v1.0.0** (March 2026) - Initial release with reviewer revisions:
  - ✅ Multi-seed statistical significance (N=5)
  - ✅ Comprehensive clustering metrics
  - ✅ Bootstrap stability analysis
  - ✅ Hyperparameter sensitivity
  - ✅ Cross-domain evaluation
  - ✅ Runtime profiling
  - ✅ Interpretability metrics
  - ✅ Baseline comparisons
