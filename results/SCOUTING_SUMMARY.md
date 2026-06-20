# Scouting Summary (1 seed, 10 epochs, recon-only, leak-free test set)

_Generated 2026-06-20. Basis: `scout_raw_metrics.csv`, `scout_per_magnification.csv`,
`objective_ablation.csv`. All 7 variants trained on the corrected pipeline; held-out test
= 4,536 tiles / 42 specimens; K-Means, k=6, standardize_l2._

## Headline findings

1. **Architecture → biology.** Every KAN/attention variant agrees with treatment group
   far better than the plain baseline: treatment-AMI **0.41–0.43 vs 0.34**, diet-AMI
   0.45–0.48 vs 0.35. This is the consistent architectural benefit.
2. **Internal compactness is NOT an architectural win** under pure reconstruction —
   baseline/no_eka are as good or better on Silhouette. We do not claim otherwise.
3. **The clustering objective drives separability.** Turning on the DEC loss (dae_kan)
   lifts Silhouette 0.033→0.046 (+39%), CHI 141.9→199.7 (+41%), Xie–Beni 17.2→5.9.
4. **Confound-robust.** Treatment-AMI within each magnification stratum (20×/40×) is
   0.39–0.50 for all 7 variants → the biological signal is not a magnification artifact.

## Ranking by treatment-AMI (biological agreement)

| Rank | Model | treat-AMI | treat-ARI | diet-AMI | Silhouette |
|---|---|--:|--:|--:|--:|
| 1 | kan_only | 0.428 | **0.306** | 0.482 | 0.015 |
| 2 | no_eka | 0.424 | 0.248 | 0.467 | 0.038 |
| 3 | dae_kan_attention | 0.423 | 0.246 | 0.462 | 0.033 |
| 4 | no_kan | 0.415 | 0.245 | 0.455 | 0.029 |
| 5 | no_bam | 0.412 | 0.247 | 0.455 | 0.022 |
| 6 | bam_only | 0.408 | 0.249 | 0.453 | 0.017 |
| 7 | baseline | 0.342 | 0.189 | 0.353 | 0.035 |

(The variants are tightly bunched on treatment-AMI; multi-seed runs are needed to tell
them apart with significance.)

## Recommendation for the final (submittable) sweep

- **Keep all 7 variants** — the ablation argument needs every arm; they're cheap relative
  to the value, and the scouting shows none is degenerate.
- **≥3 seeds** (42, 123, 456) — the significance floor the reviewers demanded; enables the
  Holm-corrected table you already have machinery for.
- **Run the proposed model both objective-OFF and objective-ON** so the headline §5.1
  separability result carries a multi-seed CI.
- **Epochs:** 10 is defensible *with* a convergence plot (recon loss plateaus very early);
  15–20 is a safer margin. 30 is heaviest.
- **Hardware:** at ~5–15 min/epoch on this laptop GPU the 3-seed × 7-variant × 10-epoch
  sweep is ~1 day; a 4090/A100 does the full 5-seed/30-epoch sweep in well under a day —
  prefer that if available (`run_corrected_pipeline.py --models <all 7> --seeds 42 123 456
  --epochs 30 --epoch-viz dump`).
- **SSL baselines** (SimCLR/BYOL/VAE + pretrained encoder) through the same path after.

## Honest framing for the manuscript (already in PAPER_DRAFT.md §6)

Architecture drives **biological** agreement; the clustering objective drives
**separability**. We do not claim an internal-compactness win for the architecture alone.
