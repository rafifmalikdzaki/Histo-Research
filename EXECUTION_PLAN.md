# Execution Plan — corrected experiments (resubmission)

_Last updated: 2026-06-20. Owner: автоloop. Paper: CMBBE MS 252387151.R1, due 2026-08-14._

Plain-language map of what runs, in what order, and the exact commands. Scope per
user decision: **1-seed scouting first, all 7 variants, let the DEC run finish first.**

## Current state (as of writing)

- ✅ Corrected pipeline working end-to-end (leak-free split, biology labels, robust
  metrics, normalization, external validation, Holm stats, per-magnification control).
- ✅ Speed: `--epoch-viz dump` → ~14 min/epoch (~2.3 h per 10-epoch run incl. extraction).
- 🔄 **Running now:** DEC-objective experiment — `dae_kan_attention` + `--cluster-weight 0.5`,
  seed 42, 10 ep, epoch 3/10 (~2 h left). Holds the GPU.
- ✅ Already done (reuse, don't rerun): seed-42 / 10-epoch **recon** runs for
  `baseline` (`auto_analysis/baseline_baseline_20260620_022524`) and
  `dae_kan_attention` (`auto_analysis/baseline_dae_kan_attention_20260620_045037`),
  both with `test_embeddings.npy` + `test_metadata.csv`.

## Step-by-step

### Step 0 — wait for the DEC run (in progress)
Let the objective run finish (do not start another GPU job concurrently — 8 GB OOM).

### Step 1 — evaluate the DEC-objective run (objective ON vs OFF)
When the DEC run finishes, extract its test embeddings and compare to the recon-only
dae_kan to see whether the clustering loss improved separation.
```bash
python extract_embeddings_for_clustering.py --model-name dae_kan_attention \
    --dataset HeparUnifiedPNG --phase test \
    --checkpoint auto_analysis/<dec_run>/checkpoints/last.ckpt \
    --output-dir auto_analysis/<dec_run> --batch-size 32
```
Then internal + external metrics on it; record objective-ON vs objective-OFF in
`results/` and `REBUTTAL_R2.md` R2.1.

### Step 2 — scouting sweep: the 5 missing variants (1 seed, recon)
baseline + dae_kan already exist, so only run the other five. recon-only
(cluster_weight 0) so all 7 share the same objective for a fair architecture ablation.
```bash
python run_corrected_pipeline.py \
    --models bam_only kan_only no_bam no_kan no_eka \
    --seeds 42 --epochs 10 --batch-size 8 --dataset hepar --k 6 \
    --epoch-viz dump --analysis-freq 200
```
≈ 5 runs × ~2.3 h ≈ **~12 h**. Each step: train → locate run dir → extract test
embeddings + biology metadata sidecar.

### Step 3 — clean scoped aggregate over the 7 recon runs
The pipeline's built-in aggregate scans ALL of `auto_analysis/` (includes old leaky
runs) → pollution. Instead symlink only the 7 corrected recon runs into a scratch dir
and aggregate that:
```bash
mkdir -p /tmp/scout7 && ln -s <each of the 7 run dirs> /tmp/scout7/<name>_seed42
python run_clustering_analysis.py --base-dir /tmp/scout7 --k 6 \
    --normalize standardize_l2 --output results/scout_raw_metrics.csv
# (1 seed → skip Holm stats; report the metric table directly)
```
Produces the scouting table: internal (Silhouette/DBI/CHI/Xie-Beni) + external
(treatment/diet AMI/ARI, magnification confound) for all 7 variants.

### Step 4 — per-magnification confound control on the 7
```bash
for r in <7 run dirs>; do
  python per_magnification_analysis.py \
    --embeddings $r/embeddings/test_embeddings.npy \
    --metadata   $r/embeddings/test_metadata.csv \
    --output results/per_magnification_<name>.csv
done
```

### Step 5 — regenerate epoch-viz from the dumps (no GPU)
`--epoch-viz dump` saved per-epoch embeddings but skipped rendering. Rebuild the
evolution plots offline for any run worth a figure:
```bash
python regenerate_epoch_viz.py --run-dir auto_analysis/<run> --phase val
```

### Step 6 — decide contenders, then full seeds (later, heavier HW)
From the scouting table pick the variants that actually matter, then run **3 seeds**
(the significance floor) — ideally on a stronger GPU:
```bash
python run_corrected_pipeline.py --models <all 7> \
    --seeds 42 123 456 789 1011 --epochs 30 --epoch-viz dump
# 3-seed minimum: --seeds 42 123 456
```
This is the run that produces the Holm-corrected significance table the reviewers want.

### Step 7 — SSL baselines (R2)
SimCLR / BYOL / VAE / single-AE × 3 seeds through the same corrected eval path
(`baselines/`), then fold into the comparison table.

### Step 8 — paper artifacts + rebuttal finalization
```bash
bash run_analysis_pipeline.sh --base-dir <scoped dir> --k 6   # tables + figures
```
Fill remaining `REBUTTAL_R2.md` ⟨PENDING⟩ cells; regenerate `tables/`, `figures/`.

## Notes / guardrails
- One GPU job at a time (8 GB). Don't launch Step 2 until Step 0/1 frees the GPU.
- Always aggregate over a **scoped** dir, never raw `auto_analysis/` (old leaky runs).
- 1-seed results are **scouting only** — not submittable; final table needs ≥3 seeds.
- `WANDB_MODE=offline` for unattended runs.
