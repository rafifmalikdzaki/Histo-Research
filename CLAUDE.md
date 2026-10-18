# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code for the paper *"An Attention-Enhanced Dual Autoencoder Clustering for Liver Histopathological Images"*. The core model (`DAE_KAN_Attention`) is a dual autoencoder combining Kolmogorov-Arnold Networks (KAN), Bottleneck Attention (BAM), and Efficient Channel Attention (ECA), trained unsupervised then evaluated by clustering the latent embeddings (KMeans / Bisecting KMeans / GMM). The repo is geared toward producing reproducible, paper-ready results: multi-seed runs, ablations, statistical significance tests, and LaTeX tables.

## Environment

Python ≥3.11, PyTorch ≥2.8, PyTorch Lightning. Dependencies are declared in `pyproject.toml` (single source of truth; see `uv.lock`). The project uses `uv`:

```bash
uv venv && source .venv/bin/activate.fish   # shell is fish
uv pip install -e ".[dev]"
```

## Commands

Training (main entrypoint — `main.py` is a stub, ignore it):

```bash
python src/pl_training_with_analysis_and_optimization.py \
    --model-name dae_kan_attention --dataset hepar --max-epochs 30 --batch-size 8 --seed 42
```

Corrected end-to-end sweep (train → extract test embeddings + biology metadata → cluster → Holm stats). This is the maintained orchestrator; the old `run_multi_seed_experiments.py` / `run_ablation_study.py` / `run_all_experiments.py` / `parallel_train.py` were stale (broken CLI, no metadata extraction) and have been removed:

```bash
python run_corrected_pipeline.py --models dae_kan_attention baseline bam_only kan_only no_bam no_kan no_eka \
    --seeds 42 123 456 789 1011 --epochs 30 --dataset hepar --k 6
python evaluate_stats.py --model dae_kan_attention --dataset HeparUnifiedPNG --run-tests --output results/summary_stats.csv
```

Full clustering analysis pipeline (extract metrics → significance tests → plots → LaTeX tables):

```bash
bash run_analysis_pipeline.sh --base-dir auto_analysis --k 6
```

Tests live in `tests/` (pytest; data-dependent cases skip cleanly without a data checkout):

```bash
.venv/bin/python -m pytest tests/
```

Clustering analysis with embedding normalization (new):

```bash
python run_clustering_analysis.py --base-dir auto_analysis --k 6 \
    --normalize standardize_l2 --output results/raw_clustering_metrics.csv
```

## Architecture

- **Model variants** share one implementation in `src/models/dae.py` (`DualAutoencoder`, built from `src/models/blocks.py`) and are registered in `src/models/factory.py` via the `MODELS` dict. `get_model(name)` returns the class. The variant names (`dae_kan_attention`, `baseline`, `bam_only`, `kan_only`, `no_bam`, `no_kan`, `no_eka`) are the ablation axes — each is a config (`kan` block × `bottleneck_attn` mode × `trailing_bam`) in `VARIANT_CONFIGS`. To add a variant, add a config plus a thin subclass in `dae.py` and register it in `factory.py`. Verified bit-identical to the old per-file variants (same state_dict keys, zero forward diff); pinned by `tests/test_model_variants.py`.
- **Building blocks**: `src/models/kan_convolutional/` (KAN linear & conv layers — `efficient_kan_conv_fixed.py` is the one in use; the other `efficient_kan_conv*` files are alternates) and `src/models/attention_mechanisms/` (`bam.py`, `eca.py`, `cbam.py`).
- **`dae.py`** structure: `DualEncoder` → `DualBottleneck` → `DualDecoder` composed into `DualAutoencoder` (dual-stage encode/decode producing the latent `z` that is later clustered). Shared pieces live in `blocks.py`; the 7 variants are configs in `VARIANT_CONFIGS`.
- **Data**: `src/histodata.py` resolves images from a flat per-dataset image dir (Hepar: `data/processed/HeparUnifiedPNG/tiles/` — the single source of truth; the legacy `train/`+`test/` physical split is leaky and deprecated) driven by CSVs named `{subset}_{csv_stem}.csv` (`hepar` → `train_hepar.csv`; see `_DATASET_REGISTRY`). CSVs carry biology metadata columns: `specimen_id`, `treatment_group`, `diet`, `agent`, `magnification`, `tile_id`, integer `label`. The Hepar train/test split is **specimen-level grouped** (StratifiedGroupKFold) — never per-tile random — to prevent intra-specimen leakage. Generate CSVs with `create_hepar_csv.py` / `create_pannuke_csv.py`; legacy CSVs backed up under `data/processed/_legacy_csv_backup/`. See `DATA_PATHS.md`.
- **Metadata parsing**: `src/metadata.py` (`parse_hepar_filename`) decodes the 7-group Hepar taxonomy + magnification + specimen id from tile filenames; `CANONICAL_GROUPS` is the canonical name table. Used by `create_hepar_csv.py` to populate real labels.
- **Clustering objective** (`src/clustering_objective.py`, opt-in): `LatentProjectionHead` projects the conv bottleneck `z` (actually 16×16×16=4096 for 128px input, not the 1024 once assumed) to a true unit-norm `latent_dim` vector; `DECClusteringLoss` is a DEC/IDEC KL loss over learnable centroids. Wired into the Lightning module at the training level (so all 7 variants share it) and gated by `--cluster-weight` — **default 0 keeps training byte-identical to pure reconstruction**. Enable with `--cluster-weight >0 --latent-dim 128 --n-clusters 5 --cluster-warmup-epochs 5`. Centroids are KMeans-initialised from a latent pool once warmup ends. Tested by `tests/test_clustering_objective.py`.
- **Clustering metrics**: `src/clustering_metrics.py` is the trusted module (robust Xie-Beni, NaN guards on degenerate clusters, `normalize_embeddings` with `standardize_l2` default, prefix-namespaced `external_metrics` for AMI/ARI/NMI/purity vs `treatment_group`/`diet`/`magnification`). `run_clustering_analysis.py` calls it; do not re-implement metrics inline.
- **Training script**: `src/pl_training_with_analysis_and_optimization.py` is the single maintained training entrypoint (the earlier `pl_training.py` / `pl_training_optimized.py` / `pl_training_streamlined.py` / `pl_training_with_analysis.py` versions were removed). It does in-training analysis, W&B logging, and periodic embedding dumps controlled by the many `--*-freq` flags (defaults are sensible; use `--fast-mode` for a 1-batch `fast_dev_run` smoke test). In-training auto-analysis is gated by `--analysis-freq` (every N batches). Set `WANDB_MODE=offline` for unattended/batch runs. The module is split via mixins: `src/training/embedding_io.py`, `src/training/reports.py`, `src/training/hparams.py`; `src/analysis/auto_analysis.py` (`AutomaticAnalyzer`) splits into `src/analysis/viz.py` + `src/analysis/embedding_cluster.py` mixins.
- **Console scripts**: after `uv pip install -e ".[dev]"`, entrypoints are `histodae-train`, `histodae-extract`, `histodae-pipeline`, `histodae-cluster`, `histodae-stats`, `histodae-plots`, `histodae-tables`, `histodae-evaluate`, `histodae-make-hepar-csv`, `histodae-make-pannuke-csv` (root `run_*.py` files are thin shims). The 10 pipeline scripts live in `src/`; `import histodae` exposes the package surface (`get_model`, metrics, data).
- **Config**: `config/experiment_config.yaml` is the centralized hyperparameter source (latent_dim 128, n_clusters 5, KAN grid/spline, ECA kernel, BAM reduction, lr, etc.). Also `wandb_config.yaml`. (`parallel_config.yaml` / `transfer_config.yaml` belonged to removed scripts; archived.)

## Outputs & data flow

Training writes timestamped dirs under `outputs/exp_<ts>_<name>_seed<n>/` (and ablation runs under `auto_analysis/`) containing `checkpoints/`, `embeddings/`, `metrics/`, `visualizations/`. The analysis/eval scripts consume those embeddings to compute clustering metrics — internal (DBI↓, CHI↑, Silhouette↑, Xie-Beni↓) and external (AMI, ARI) — then `compute_statistical_significance.py`, `generate_clustering_plots.py`, and `generate_latex_tables.py` produce paper artifacts into `results/`, `figures/`, `tables/`.

## Conventions & gotchas

- `.gitignore` excludes `data/`, `wandb/`, `auto_analysis/`, `checkpoints_*/`, and **`*.md`** — so the many `*.md` guide files (WORKFLOW.md, WALKTHROUGH.md, WANDB_GUIDE.md, ABLATION_*.md, etc.) and this CLAUDE.md are untracked. Don't assume markdown docs are in version control.
- `--dataset` takes `pannuke`/`hepar` in the single-run training script but full names like `HeparUnifiedPNG`/`PanNuke` in the multi-seed/eval scripts — check the script's argparse before passing values.
- Reproducibility (global seed, timestamped outputs) is a core requirement of this codebase; preserve seeding behavior when modifying training.
- W&B is used for experiment tracking (default project set per script); see `WANDB_GUIDE.md`.
- **Embeddings must be normalized before clustering** (default `standardize_l2`). Raw 1024-d AE features produce degenerate clusters → meaningless metrics. Always go through `clustering_metrics.normalize_embeddings`.
- **External validation depends on metadata sidecars.** Training currently discards labels via `x, _ = batch`, so embedding dumps before this rewrite have no biology metadata. New runs should write a `<phase>_metadata.csv` next to each `<phase>_embeddings.npy` carrying `treatment_group` / `diet` / `magnification` for AMI/ARI to compute.
- **Resubmission context** (CMBBE MS 252387151.R1, due 2026-08-14): see `/home/dzakirm/.claude/projects/-home-dzakirm-Research-histodae/memory/paper-resubmission-context.md` and `hepar-dataset-taxonomy.md`. Active plan: `/home/dzakirm/.claude/plans/forwarded-message-peaceful-teacup.md`.
