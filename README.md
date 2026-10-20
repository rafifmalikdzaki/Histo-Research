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
│   └── histodae/                       # the one installed package
│       ├── data/                       # dataset resolution, splits, metadata
│       │   ├── datasets.py             # ImageDataset, create_dataset, load_metadata
│       │   └── metadata.py             # Hepar filename taxonomy parsing
│       ├── clustering/                 # metrics + DEC/IDEC objective
│       │   ├── metrics.py              # robust internal/external metrics, normalization
│       │   └── dec.py                  # LatentProjectionHead, DECClusteringLoss
│       ├── models/                     # the dual autoencoder family
│       │   ├── dae.py                  # Unified dual-autoencoder + 7 variant configs
│       │   ├── blocks.py               # Shared KAN/attention building blocks
│       │   ├── factory.py              # Model factory for different variants
│       │   ├── attention_mechanisms/   # bam.py (+ eca.py, cbam.py when added)
│       │   └── kan_convolutional/      # efficient_kan.py, efficient_kan_conv_fixed.py
│       ├── training/                   # Lightning module and its collaborators
│       │   ├── module.py               # the LightningModule (forward, train/val steps)
│       │   ├── losses.py               # objective computation
│       │   ├── optim.py                # optimizer + head-registration safety net
│       │   ├── analysis_hook.py        # periodic in-training analysis
│       │   ├── config.py               # experiment YAML -> CLI defaults
│       │   ├── embedding_io.py         # embedding collection and dumping
│       │   ├── lifecycle.py            # epoch / end-of-training hooks
│       │   ├── reports.py              # report and figure generation
│       │   └── hparams.py              # ablation config, batch-size probe
│       ├── analysis/                   # in-training analysis
│       │   ├── auto_analysis.py        # AutomaticAnalyzer (assembles the mixins)
│       │   ├── attention.py            # attention hooks + attention metrics
│       │   ├── batch.py                # per-batch metrics and summaries
│       │   ├── wandb_logging.py        # exception-safe W&B logging
│       │   ├── locking.py              # cross-process lock on the output dir
│       │   ├── comparison.py           # cross-run ablation comparison figures
│       │   ├── embedding_cluster.py    # embedding clustering analysis
│       │   └── viz*.py                 # visualisations and dashboards
│       ├── stats/                      # effect sizes, seed pairing, result tables
│       ├── io/                         # run-artifact IO (configs, run discovery)
│       ├── utils/                      # seeding, experiment dirs, W&B integration
│       ├── config/                     # packaged YAML defaults
│       └── cli/                        # one module per pipeline stage
│
├── tests/                              # pytest suite (python -m pytest tests/)
│
├── archive/                            # Retired scripts, guides, legacy analysis
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
└── run_corrected_pipeline.py           # Thin shims for the CLI stages (prefer the
                                        # `histodae` console script)
```

Every module is importable as `histodae.<layer>.<module>` — that is the only
supported import path. The flat names (`histodata`, `models.factory`, ...)
still work as thin aliases that resolve to the *same* module objects.

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

# Install the project with its dependencies
uv pip install -e ".[dev]"

# Or with plain pip
pip install -e ".[dev]"

# Run the test suite
python -m pytest tests/
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
histodae data pannuke --data-path data/processed/PANnuke

# For IMI/Hepar
histodae data hepar --data-path data/processed/HeparUnifiedPNG
```

### 3. Train the Model

**Single run (default seed = 42):**

```bash
histodae train \
    --model-name dae_kan_attention \
    --dataset hepar \
    --max-epochs 30 \
    --batch-size 8 \
    --seed 42
```

**Corrected end-to-end sweep** (train → extract embeddings + metadata →
cluster → Holm-corrected statistics), which is the maintained orchestrator:

```bash
histodae pipeline \
    --models dae_kan_attention baseline bam_only kan_only no_bam no_kan no_eka \
    --seeds 42 123 456 789 1011 \
    --epochs 30 \
    --dataset hepar \
    --k 6
```

Add `--config-path config/experiment.yaml` to seed the hyperparameters from
the packaged experiment config (explicit CLI flags still win).

### 4. Evaluate Results

**Statistical significance analysis:**

```bash
# Aggregate multi-seed results
histodae evaluate \
    --model dae_kan_attention \
    --dataset HeparUnifiedPNG \
    --run-tests \
    --output results/summary_stats.csv
```

**Clustering metrics over extracted embeddings:**

```bash
histodae cluster --base-dir auto_analysis --k 6 --normalize standardize_l2 \
    --output results/raw_clustering_metrics.csv
```

**Figures and LaTeX tables:**

```bash
histodae plots  --input results/clustering_summary.csv \
    --tests results/statistical_tests.csv --output-dir figures
histodae tables --summary results/clustering_summary.csv \
    --tests results/statistical_tests.csv --output-dir tables
```

---

## 📊 Reproducing All Experiments

All experiments are driven by the corrected pipeline
(`histodae pipeline`), which trains, extracts test-split embeddings with
their biology metadata, clusters them, and runs the paired, Holm-corrected
statistics. The older per-experiment orchestrators
(`run_multi_seed_experiments.py`, `run_ablation_study.py`,
`sensitivity_analysis.py`, `cross_domain_eval.py`, `profile_runtime.py`)
were removed: they passed flags the training script no longer accepts and
had no metadata-extraction step, which is why their results were not
reproducible.

### Main results + ablation study

```bash
# 7 variants x 5 seeds x 30 epochs, one command
histodae pipeline \
    --models dae_kan_attention baseline bam_only kan_only no_bam no_kan no_eka \
    --seeds 42 123 456 789 1011 \
    --epochs 30 --dataset hepar --k 6
```

Ablation variants are configs of one implementation
(`src/histodae/models/dae.py`), so an ablation sweep differs from the main
run only in the `--models` list.

### Objective ablation (reconstruction vs clustering objective)

```bash
# clustering objective ON (DEC/IDEC); default is pure reconstruction
histodae pipeline --models dae_kan_attention --seeds 42 123 456 \
    --epochs 30 --cluster-weight 0.5 --latent-dim 128 --n-clusters 5 \
    --cluster-warmup-epochs 5
```

### Full analysis pipeline over existing runs

```bash
bash run_analysis_pipeline.sh --base-dir auto_analysis --k 6
```

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

Default hyperparameters (from `src/histodae/config/experiment.yaml`):

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
