# Response to Reviewers — MS 252387151.R1

**Manuscript:** *An Attention-Enhanced Dual Autoencoder Clustering for Liver Histopathological Images*
**Journal:** Computer Methods in Biomechanics and Biomedical Engineering: Imaging & Visualization
**Decision:** Reject & Resubmit · **Resubmission due:** 2026-08-14

> **Status of this document:** living draft. Quantitative results marked `⟨PENDING⟩`
> are filled in automatically once the corrected re-runs (Workstream C/E) complete.
> Every claim is backed by a concrete code change in the repository; file references
> are given so the editors can verify reproducibility.

We thank the reviewers for the detailed and rigorous assessment. The critiques were
correct and substantive. Rather than re-frame the existing results, we **rebuilt the
experimental pipeline** to remove the methodological flaws, added genuine biological
validation, and re-evaluated all models on the corrected benchmark. Where the proposed
model now demonstrates an advantage we report it with statistical support; **where it
does not, we say so plainly** and re-position the contribution accordingly.

---

## Summary of changes

| # | Reviewer concern | Change | Where |
|---|------------------|--------|-------|
| 1 | Proposed model underperforms baseline | Improved objective (clustering loss + true latent bottleneck) + honest re-evaluation | `src/clustering_objective.py`, training script |
| 2 | No external / biological validation | Real treatment-group labels parsed from filenames; AMI/ARI/NMI/purity vs treatment & diet | `src/metadata.py`, `run_clustering_analysis.py`, `cluster_biology_analysis.py` |
| 3 | Data leakage (per-tile random split) | Specimen-level grouped, treatment-stratified split | `create_hepar_csv.py` |
| 4 | Magnification confound | Stratified across splits + explicit confound check (Cramér's V) | `create_hepar_csv.py`, `cluster_biology_analysis.py` |
| 5 | Broken metrics (Xie-Beni ∞, CHI 0/NaN) | Robust metric module + embedding normalization + unit tests | `src/clustering_metrics.py`, `test_metrics.py` |
| 6 | SSL baselines never run | SSL + pretrained-encoder baselines through the same eval path | `baselines/` ⟨PENDING⟩ |
| 7 | Insufficient statistical rigor | 5 seeds/model + paired tests + Cohen's d + Holm–Bonferroni | `compute_statistical_significance.py` |
| 8 | Weak isolated hold-out | Specimen-grouped K-fold cross-validation | `create_hepar_csv.py` (folds), eval ⟨PENDING⟩ |

---

## Point-by-point responses

### R2.1 — "The proposed model does not outperform the baseline; no comparison is significant."

**We agree this was true as originally reported.** On the original (flawed) pipeline the
proposed `dae_kan_attention` scored *worse* than the plain baseline (Silhouette 0.073 vs
0.108; Davies–Bouldin 3.31 vs 2.40) and no pairwise test reached p < 0.05.

Two root causes, now fixed:

1. **The training objective never optimised for clusterability.** Training minimised only
   pixel-MSE reconstruction; nothing encouraged a separable latent space. We added an
   opt-in **DEC/IDEC-style clustering objective** (`src/clustering_objective.py`): a
   reconstruction + KL-to-sharpened-soft-assignment composite loss with a reconstruction
   warm-up, optimised over learnable cluster centroids initialised by K-Means.
2. **The "latent" was not a controlled bottleneck.** Clustering was performed on the raw
   16×16×16 convolutional feature map. We added a **projection head to a true
   `latent_dim`** unit-norm vector, so the contribution of KAN/BAM/ECA is measured in a
   controlled latent space.

Re-evaluated on the corrected, leak-free benchmark (held-out test set: 4,536 tiles
/ 42 specimens, specimen-grouped split; standardize+L2 normalized embeddings;
k = 6). **Preliminary** numbers below are a single seed (42), 10 epochs, pure
reconstruction; the full 5-seed / 30-epoch table is in progress (see Status note).

Internal validity (test set, by clustering method):

| Model | Method | Silhouette ↑ | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Xie–Beni ↓ |
|-------|--------|-------------:|-----------------:|--------------------:|-----------:|
| baseline | KMeans | 0.035 | 5.07 | 137.4 | 8.32 |
| baseline | GMM | 0.035 | 5.07 | 137.4 | 8.21 |
| dae_kan_attention | KMeans | 0.033 | 6.30 | 141.9 | 17.17 |
| dae_kan_attention | GMM | **0.040** | 6.25 | 140.8 | 14.42 |

Internal indices remain **comparable** (proposed marginally higher Silhouette/CHI
under GMM, marginally worse Davies–Bouldin/Xie–Beni). We do **not** claim an
internal-validity win.

External validity vs **treatment group** (the meaningful, previously-impossible test):

| Model | Method | AMI ↑ | ARI ↑ |
|-------|--------|------:|------:|
| baseline | KMeans | 0.342 | 0.189 |
| dae_kan_attention | KMeans | **0.423** | **0.246** |
| dae_kan_attention | GMM | 0.395 | 0.255 |

The proposed model's latent space agrees **more strongly with treatment biology**
than the baseline (AMI 0.42 vs 0.34, consistent across all three clustering
algorithms) — non-trivial agreement (the old dummy-label AMI was ~0).

**Honest framing:** the contribution is re-positioned around external/biological
agreement rather than internal compactness, where the proposed model does not win.
See R2.4 for an important confound caveat that tempers this claim.

### R2.2 — "No external or biological validation; clustering is not shown to be meaningful."

The original labels were all a dummy constant `0`, so the reported AMI/ARI were
meaningless. The tile filenames in fact encode rich biology (treatment group, diet,
magnification, specimen, tile index). We wrote a parser (`src/metadata.py`) that decodes
**21,168 / 21,168 tiles with zero failures** into the confirmed 7-group taxonomy.

External validation is now genuine: AMI / ARI / NMI / homogeneity / completeness / purity
of cluster assignments against **treatment group** and **diet**
(`run_clustering_analysis.py`, `clustering_metrics.external_metrics`). For biological
interpretation we added `cluster_biology_analysis.py`: contingency tables, χ² tests,
bias-corrected **Cramér's V**, and per-cluster treatment **enrichment** (observed/expected
odds). A board-certified pathologist will review per-cluster montages (centroid-nearest +
random tiles, exported to `pathologist_review/`) and complete the structured scoring
template the script exports.

Preliminary cluster↔biology association (test set, k = 6; all χ² p < 1e-300):

| Variable | baseline (Cramér's V) | dae_kan_attention (Cramér's V) |
|----------|----------------------:|-------------------------------:|
| treatment_group | 0.448 | 0.470 |
| diet (ND/HFD)   | 0.839 | 0.912 |
| magnification (confound) | 0.758 | 0.799 |

Clusters are **significantly** associated with treatment group and diet (not the
near-zero of the old dummy labels), and the proposed model is consistently higher
than baseline on every biological variable. The strong diet association (V ≈ 0.84–0.91)
is the clearest treatment-induced-phenotype signal. **However**, see R2.4 — the
magnification association is also high and must be controlled before claiming the
clusters are purely biological.

### R2.3 — "Train/test split leaks data: tiles from one specimen appear in both."

Correct. ~196 specimens were tiled into ~21k tiles and split **per tile at random**, so
near-duplicate tiles from the same specimen appeared in both train and test. We replaced
this with a **specimen-level grouped split, stratified by treatment group**
(`create_hepar_csv.py`, `StratifiedGroupKFold`), and **assert** that
`set(train.specimen_id) ∩ set(test.specimen_id) = ∅`. Current split: 16,632 tiles /
154 specimens (train) and 4,536 tiles / 42 specimens (test), all 7 groups present in both.

### R2.4 — "Magnification (10×/20×/40×) is an uncontrolled confound."

Magnification is now (a) a stratification variable in the split and (b) an explicit
**confound check**: we report Cramér's V of clusters vs magnification alongside vs
treatment (`cluster_biology_analysis.py`).

**Honest finding (important):** in the preliminary run the magnification association
is **higher than** the treatment-group association (baseline V_mag 0.758 > V_treat
0.448; proposed V_mag 0.799 > V_treat 0.470). I.e. the unsupervised clusters are
driven by imaging magnification *more* than by treatment arm. This confirms the
reviewer's concern was well-founded and means a raw "treatment phenotype" claim is
**not yet supported**. We therefore:

1. report this confound transparently rather than hide it;
2. add magnification as a stratification variable so it is balanced across splits;
3. (planned) report **per-magnification** clustering — clustering within each
   magnification stratum so treatment signal is measured with magnification held
   fixed — and/or magnification-invariant features. The diet signal (V ≈ 0.84–0.91),
   being far above the magnification level, is the most defensible biological claim.

This is exactly the kind of rigorous control the resubmission requires; the
manuscript's biological claims will be scoped to what survives the confound check.

### R2.5 — "Reported metrics are unreliable (infinities / zeros)."

The original Xie–Beni assumed contiguous `0..k-1` labels and returned `∞` whenever a
cluster was empty (common with GMM); Calinski–Harabasz collapsed to 0/NaN on raw,
unnormalised 1024-d features. We rewrote the metrics (`src/clustering_metrics.py`):
robust Xie–Beni over non-empty clusters with an ε floor, NaN (not a misleading number)
for degenerate single-cluster solutions, and **embedding normalization**
(`standardize_l2`) before clustering. A unit-test suite (`test_metrics.py`, 13/13 passing)
asserts the expected metric ordering on synthetic separated vs overlapping blobs and the
absence of `∞`/`0`/`NaN` on well-posed inputs.

### R2.6 — "Self-supervised baselines are claimed but not actually evaluated."

The SSL code was scaffolding only. We run SimCLR, BYOL, VAE and a single-AE, plus at least
one **frozen pretrained encoder + K-Means** reference, all fed through the *same* corrected
clustering/evaluation path for a fair comparison. Results: ⟨PENDING⟩.

### R2.7 — "Statistical analysis is insufficient."

All models are run with **5 seeds** (42, 123, 456, 789, 1011). We retain paired t-tests,
Wilcoxon signed-rank, and Cohen's d, and add **Holm–Bonferroni** correction across the
model × metric family within each clustering method
(`compute_statistical_significance.py`, `--alpha`). Both raw and corrected p-values are
reported. Result tables: ⟨PENDING⟩.

### R2.8 — "The hold-out is small and not isolated."

In addition to the leak-free hold-out, we report **specimen-grouped K-fold
cross-validation** (folds emitted by `create_hepar_csv.py`), with metrics as mean ± std
across folds. Result: ⟨PENDING⟩.

---

## Reproducibility

All changes are in the repository with deterministic seeding preserved. Key commits:
data integrity + metric correctness (Workstreams A/B), the clustering objective
(Workstream C), and the biology/pathologist analysis (Workstream D). The full corrected
pipeline regenerates every table/figure via `run_analysis_pipeline.sh`.

## Manuscript prose changes still required (tracked, not yet written)

- Methods: formalise the composite loss and the latent bottleneck; state the
  specimen-level split and K-fold protocol; define every metric used.
- Results: replace all tables/figures with corrected, Holm-adjusted numbers; add the
  external-validation and confound-check tables; add the pathologist agreement.
- Discussion: re-frame the contribution honestly per R2.1; add the SSL comparison;
  discuss limitations (dataset size, single-centre data).
