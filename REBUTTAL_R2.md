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

Re-evaluated on the corrected, leak-free, treatment-validated benchmark
(5 seeds × {K-Means, Bisecting K-Means, GMM}):

- Internal validity (mean ± std, Holm-corrected p vs baseline): ⟨PENDING⟩
- External validity (AMI/ARI vs treatment group): ⟨PENDING⟩

**Honest framing:** if the proposed model does not win an internal index after these
fixes, we will not claim it does. The contribution is re-positioned around
(i) agreement with treatment biology, (ii) interpretability via attention, and
(iii) robustness across seeds/folds — each reported with significance and effect size.

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
random tiles) and complete the structured scoring template the script exports. Results:
⟨PENDING⟩.

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
treatment, and require clusters to track treatment more strongly than magnification
(`cluster_biology_analysis.py`). Result: ⟨PENDING⟩.

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
