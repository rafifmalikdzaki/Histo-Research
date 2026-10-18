# Response to reviewers — JOIV revision (working draft)

Manuscript: *Attention-Enhanced Dual Autoencoder Clustering for Liver
Histopathological Images*. All numbers below are frozen from the gpu-kc
multi-seed sweep (`joiv_revision/results/frozen_summary.md`).

We thank the editors and reviewers. The critiques were substantive and largely
correct. We rebuilt the experimental pipeline to remove the flaws (per-tile
leakage, placeholder labels, unnormalised embeddings), re-evaluated all models on
a specimen-disjoint benchmark with genuine biological validation, and moderated every
claim to what the evidence supports. Where records or experiments are still
missing (expert scoring, replicated contemporary baselines, external cohort,
raw compute measurements), we state the gap plainly instead of claiming it.

## To the Editor

- **Abstract.** Rewritten to 241 words covering objectives, materials,
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
Stated limit: disjointness is by filename-derived specimen IDs (overlap = 0 in
code); independence of the underlying animals/slides, the acquisition-to-tiling
record, and split-before-tiling ordering remain unverified from records, so we
claim specimen-disjointness, not a fully audited leak-free provenance.

**A2 — Dual vs conventional autoencoder; implementation details.** Added: the
dual architecture (Encoder → Bottleneck → Decoder with a second encode/decode
refinement stage producing latent `z`), how it differs from a single
encode–decode pass, full layer table (KAN convolution grid/spline, ECA kernel,
BAM reduction, projection head to a unit-norm `latent_dim` vector), optimiser,
schedule, early stopping, seeds, and hardware. Stated limit: historical
execution provenance (exact code state per reported run) is reconstructed from
the current pipeline and run logs, not from per-run versioned snapshots.

**A3 — Accuracy/precision/recall/F1 on an unsupervised task.** Removed. The
Table IV classification block had no defined ground truth or protocol; external
validity is now reported exclusively with clustering-appropriate measures
(AMI/ARI/NMI/purity vs treatment group and diet; Silhouette alongside internal
indices).

**A4 — k, initialisation, seeds, variability.** k = 6 (full sensitivity sweep
k = 3–8 for baseline and full model: plateau at k = 4–6 — baseline AMI
0.423/0.427/0.420, full 0.473/0.455/0.433 — declining at k = 7–8; k = 6 is the
pre-registered on-plateau choice, not the tuned peak at k = 4–5); KPI
initialisation specified; 3 seeds (42/123/456) × 7 variants, mean ± SD,
paired t + Wilcoxon + Cohen's d, Holm–Bonferroni across the model×metric
family. Stated limit: run→seed mapping follows launch records with row-level
aggregation by experiment directory; a per-row paired-run provenance manifest
is outstanding.

**A5 — Biological meaning of clusters.** New external-validation table:
cluster↔treatment-group and cluster↔diet AMI/ARI/NMI, χ² with bias-corrected
Cramér's V, per-cluster enrichment. Frozen 3-seed test results (KMeans, k = 6):
treatment-AMI 0.416–0.439 across all variants vs baseline 0.420 (all within
seed noise; the earlier 0.34 baseline figure was a 1-seed scouting value and
is retired); diet-AMI 0.454–0.489 vs baseline 0.459.

**A6 — Expert validation, contemporary baselines, external data.** Per-cluster
montages (centroid-nearest + random) and a structured scoring template are
exported for pathologist review (`cluster_biology_analysis.py`); scores are
outstanding at submission and are stated as a limitation, with Cramér's V and
enrichment statistics reported in their place. Baselines: frozen pretrained-encoder + KMeans
through the identical eval path (minimum); SimCLR/BYOL/VAE (frozen ResNet-50 reported; full SSL training stated as future work). No replicated contemporary-method comparison and no independent external-cohort clustering evaluation are available at submission; both are stated as limitations, not claimed. PanNuke is used strictly as multi-organ pre-training source, not as an external evaluation arm (the old classification block is removed per A3).

**A7 — DBI 3.633 vs ~0.4.** The two values came from different experimental
settings (leaky full-run vs a differently-normalised subset) that the text never
distinguished. All numbers are now regenerated from one frozen pipeline; every
reported value is cross-checked across abstract, body, tables, and figures by
script (see validation log). The DEC-objective arm figures are reported from
the completed 3-seed runs but flagged provisional pending
checkpoint-restoration verification.

**A8 — Compute measurements.** One protocol on named hardware
(gpu-kc, NVIDIA RTX 3060 12 GB): Table II reports inference ms/image,
throughput, s per 1,000 patches, and parameter count, with the arithmetic
identities (`1000/(ms per image) == images/s`) verified before tabulation;
training wall-clock (~35 min per 10-epoch run, ~210 s/epoch) is reported as
observed run time. Raw instrumented measurements (per-phase timers, peak
memory) are still outstanding and no broader benchmark is claimed.

**A9 — Notation.** ELU equation rewritten in standard form; Calinski–Harabasz
typo fixed; all validity indices (Xie–Beni, Davies–Bouldin, Calinski–Harabasz,
Hartigan) defined with every variable introduced.

**A10 — Ablations.** Full 7-variant ablation isolating KAN, ECA/ECA-kernel
(`no_eka`), and BAM, each with repeated-run variability and a parameter-count
column (baseline 4.9M vs full 5.25M — close but not matched, so capacity is a
reported caveat, not a controlled factor). A clean factorial design,
capacity-matched variants, and isolated pre-training comparisons remain
outstanding and are stated as future work. (3-seed table in the revision).

**A11 — Claim moderation.** "Biologically coherent / reliable / robust /
treatment-related" now appear only where backed by A5 evidence and the
per-magnification control; elsewhere softened to "associated" / "consistent with".

**A12 — Language/formatting/figures.** Full pass; citations integrated into the
argument; JOIV/IEEE style applied. Figure preservation: the **original
architecture figure is restored unchanged** (not redrawn) and retained in its
original placement with all 15 figure placements in the original layout;
tensor-level differences between the diagram and the executed implementation
(KAN/convolution blocks, attention placement, projection head) are explained
in the text alongside the figure rather than hidden by redrawing it.

## Reviewer B

**B1 — Unsupervised vs classification metrics.** As A3 + A5: internal
(Silhouette, DBI, CHI, Xie–Beni) vs external (ARI, NMI, AMI) validation
separated and defined; no classification metrics on the unsupervised task.

**B2 — DBI and throughput contradictions.** As A7 + A8.

**B3 — Specimen-level splitting.** As A1 (filename-derived specimen overlap = 0,
asserted in code; animal/slide independence and pre-tiling split order unverified — see A1 limit).

**B4 — UMAP/samples insufficient.** As A6.

**B5 — Component ablations, modern baselines, multi-run stats.** As A10 + A4 + A6.

## Magnification confound (A1-crosscutting)

Magnification (10×/20×/40×) is stratified across splits and reported as an
explicit confound: Cramér's V of clusters vs magnification alongside vs
treatment, plus per-magnification clustering (treatment agreement with
magnification held fixed). Within-stratum treatment-AMI 0.39–0.50 (≥ pooled)
is consistent with biology rather than pure magnification artifact — but a
repeated-run, specimen-aware magnification control is still outstanding, so
this stays an association result, not a causal claim (multi-seed per-mag table
in the frozen supplement).
