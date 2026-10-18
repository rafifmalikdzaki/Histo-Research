# Response to reviewers — JOIV revision (working draft)

Manuscript: *Attention-Enhanced Dual Autoencoder Clustering for Liver
Histopathological Images*. All numbers below are frozen from the gpu-kc
multi-seed sweep (`joiv_revision/results/frozen_summary.md`).

We thank the editors and reviewers. The critiques were substantive and largely
correct. We rebuilt the experimental pipeline to remove the flaws (per-tile
leakage, placeholder labels, unnormalised embeddings), re-evaluated all models on
a leak-free benchmark with genuine biological validation, and moderated every
claim to what the evidence supports.

## To the Editor

- **Abstract.** Rewritten to 237 words covering objectives, materials,
  method, results, and implication for further research (was ~251 words,
  background-heavy).
- **References.** Now 38 sources, 27 (71%) from 2020–2026 (was exactly 30 at the
  70% threshold with no headroom); IEEE style via reference manager, DOI per entry;
  every in-text citation verified against the list and vice-versa.

## Reviewer A

**A1 — Datasets, splitting, leakage.** Correct: the original 70/15/15 per-tile
split leaked near-duplicate tiles from the same specimen across folds. Replaced
with a specimen-level grouped split stratified by treatment group
(`StratifiedGroupKFold`; `create_hepar_csv.py` asserts
`set(train.specimen) ∩ set(test.specimen) = ∅`): 16,632 tiles / 154 specimens
train, 4,536 tiles / 42 specimens test, all 7 treatment groups in both. Methods
now specify cohort, patch extraction (128×128, RGB, per-channel normalisation),
augmentation (rotations, flips, colour normalisation), and preprocessing.

**A2 — Dual vs conventional autoencoder; implementation details.** Added: the
dual architecture (Encoder → Bottleneck → Decoder with a second encode/decode
refinement stage producing latent `z`), how it differs from a single
encode–decode pass, full layer table (KAN convolution grid/spline, ECA kernel,
BAM reduction, projection head to a unit-norm `latent_dim` vector), optimiser,
schedule, early stopping, seeds, and hardware.

**A3 — Accuracy/precision/recall/F1 on an unsupervised task.** Removed. The
Table IV classification block had no defined ground truth or protocol; external
validity is now reported exclusively with clustering-appropriate measures
(AMI/ARI/NMI/purity vs treatment group and diet; Silhouette alongside internal
indices).

**A4 — k, initialisation, seeds, variability.** k = 6 (sensitivity sweep k = 3–8: treatment-AMI rises 0.35 at k = 3 to a
plateau 0.42 at k = 4–6); KPI initialisation specified; 3 seeds (42/123/456) ×
7 variants, mean ± SD, paired t + Wilcoxon + Cohen's d, Holm–Bonferroni across
the model×metric family.

**A5 — Biological meaning of clusters.** New external-validation table:
cluster↔treatment-group and cluster↔diet AMI/ARI/NMI, χ² with bias-corrected
Cramér's V, per-cluster enrichment. Scouting (1 seed, leak-free test):
treatment-AMI 0.41–0.43 for all six KAN/attention variants vs 0.34 baseline;
diet-AMI 0.45–0.48 vs 0.35 (3-seed frozen table).

**A6 — Expert validation, contemporary baselines, external data.** Per-cluster
montages (centroid-nearest + random) and a structured scoring template are
exported for pathologist review (`cluster_biology_analysis.py`); scores are
outstanding at submission and are stated as a limitation, with Cramér's V and
enrichment statistics reported in their place. Baselines: frozen pretrained-encoder + KMeans
through the identical eval path (minimum); SimCLR/BYOL/VAE (frozen ResNet-50 reported; full SSL training stated as future work). Evaluation on the independent PanNuke
split retained as the external-dataset arm (classification block removed per A3;
reported with clustering metrics only).

**A7 — DBI 3.633 vs ~0.4.** The two values came from different experimental
settings (leaky full-run vs a differently-normalised subset) that the text never
distinguished. All numbers are now regenerated from one frozen pipeline; every
reported value is cross-checked across abstract, body, tables, and figures by
script (see validation log).

**A8 — Compute measurements.** One protocol on named hardware
(gpu-kc, NVIDIA RTX 3060 12 GB): train ms/image,
inference ms/image, throughput, s/epoch, s/1000 patches, peak memory, parameter
count — with the arithmetic identities (`1000/(s per 1000) == images/s`)
verified before tabulation.

**A9 — Notation.** ELU equation rewritten in standard form; Calinski–Harabasz
typo fixed; all validity indices (Xie–Beni, Davies–Bouldin, Calinski–Harabasz,
Hartigan) defined with every variable introduced.

**A10 — Ablations.** Full 7-variant ablation isolating KAN, ECA/ECA-kernel
(`no_eka`), and BAM, each with repeated-run variability and a parameter-count
column so module effects are separable from capacity effects. (3-seed table in the revision).

**A11 — Claim moderation.** "Biologically coherent / reliable / robust /
treatment-related" now appear only where backed by A5 evidence and the
per-magnification control; elsewhere softened to "associated" / "consistent with".

**A12 — Language/formatting/figures.** Full pass; figures re-rendered at
readable resolution (the 6.9 MB EMF replaced); citations integrated into the
argument; JOIV/IEEE style applied.

## Reviewer B

**B1 — Unsupervised vs classification metrics.** As A3 + A5: internal
(Silhouette, DBI, CHI, Xie–Beni) vs external (ARI, NMI, AMI) validation
separated and defined; no classification metrics on the unsupervised task.

**B2 — DBI and throughput contradictions.** As A7 + A8.

**B3 — Specimen-level splitting.** As A1 (overlap = 0, asserted in code).

**B4 — UMAP/samples insufficient.** As A6.

**B5 — Component ablations, modern baselines, multi-run stats.** As A10 + A4 + A6.

## Magnification confound (A1-crosscutting)

Magnification (10×/20×/40×) is stratified across splits and reported as an
explicit confound: Cramér's V of clusters vs magnification alongside vs
treatment, plus per-magnification clustering (treatment agreement with
magnification held fixed). Scouting: within-stratum treatment-AMI 0.39–0.50
(≥ pooled) — the phenotype signal is not a magnification artifact (multi-seed per-mag table in the frozen supplement).
