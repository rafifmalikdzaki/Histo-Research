# In-training analysis

`histodae.analysis` is the in-training analysis subsystem. It runs *while* a
model trains: it hooks the attention modules, summarises what they produce,
renders visualisations, and logs the results to Weights & Biases.

It is not a standalone analysis suite. To analyse a trained model, use the
CLI (see below) — that path reuses the same clustering metrics and writes the
biology metadata sidecar the external validation depends on.

## Architecture

`AutomaticAnalyzer` is a coordinator assembled from focused mixins. Each
concern lives in its own module so the class stays small and each piece can
be reasoned about on its own.

| Mixin | Module | Responsibility |
|-------|--------|----------------|
| `AttentionMetricsMixin` | `attention.py` | Forward hooks on the BAM/KAN submodules; attention map metrics (distribution, sparsity, concentration, spatial, peaks) |
| `BatchAnalysisMixin` | `batch.py` | Per-batch forward pass, reconstruction metrics, batch summaries, metric serialisation |
| `WandbLoggingMixin` | `wandb_logging.py` | Frequency-aware, exception-safe W&B logging |
| `FileLockMixin` | `locking.py` | Advisory lock so concurrent processes do not write the same metric file |
| `VizMixin` | `viz.py` | Basic visualisations (reconstructions, embedding plots) |
| `VizComponentsMixin` | `viz_components.py` | Individual attention components (original, reconstructed, error map, attention maps) |
| `VizDashboardMixin` | `viz_dashboard.py` | The comprehensive "paper dashboard" figure |
| `EmbeddingClusterMixin` | `embedding_cluster.py` | Embedding clustering analysis |

`comparison.py` holds `create_ablation_study_comparison()`, which compares
several runs side by side.

The analyzer is constructed by `histodae.training.module` in `on_fit_start`,
and driven by `histodae.training.analysis_hook`, which decides *when* to run
it. Both live in the training package, not here.

## What it is triggered by

During training, `histodae training` (or `histodae pipeline`) drives it:

- `analysis_hook.run_training_analysis` runs every `--analysis-freq` batches
  during training and for the first few validation batches.
- Every call is wrapped so that a failing visualisation **cannot** take the
  training loop down — failures are counted and logged to W&B instead.

## Offline analysis of a trained model

For anything that happens *after* training, use the CLI rather than this
package:

```bash
histodae extract --model-name dae_kan_attention --dataset HeparUnifiedPNG \
    --phase test --checkpoint <run>/checkpoints/last.ckpt --output-dir <run>
histodae cluster --base-dir auto_analysis --k 6 --normalize standardize_l2
histodae stats   --input results/raw_clustering_metrics.csv --baseline baseline
histodae per-magnification --embeddings <run>/embeddings/test_embeddings.npy \
    --metadata <run>/embeddings/test_metadata.csv --output results/per_mag.csv
histodae biology --embeddings <run>/embeddings/test_embeddings.npy \
    --metadata <run>/embeddings/test_metadata.csv --k 6 --method kmeans \
    --output-dir pathologist_review/<run>
```

Embeddings **must** be normalised before clustering (default
`standardize_l2`, applied by `histodae.clustering.metrics`). Raw bottleneck
features produce degenerate clusters and meaningless indices.

## Legacy

The original standalone analysis scripts (`performance_analysis.py`,
`gradcam_analysis.py`, `attention_visualizer.py`,
`pathology_correlation.py`, `comprehensive_analysis.ipynb`) are retired and
live in `archive/legacy-analysis/`. Their README, which recorded the
exploratory results of that pass, is kept alongside them.
