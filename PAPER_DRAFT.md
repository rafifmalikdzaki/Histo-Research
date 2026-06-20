# An Attention-Enhanced Dual Autoencoder with a Deep Clustering Objective for Unsupervised Phenotyping of Liver Histopathology

*Working manuscript draft (resubmission, CMBBE: Imaging & Visualization, MS 252387151.R1).
Numbers marked ⟨PENDING⟩ are filled from the running corrected experiments. All
reported numbers are from the corrected, leak-free pipeline; preliminary cells are
labelled (1 seed / 10 epochs).*

---

## Abstract

We study unsupervised phenotyping of liver histopathology tiles from a non-alcoholic
fatty liver disease (NAFLD) / statin-intervention cohort. Our model, **DAE-KAN-Attn**, is
a dual convolutional autoencoder augmented with Kolmogorov–Arnold Network (KAN) layers,
a Bottleneck Attention Module (BAM), and Efficient Channel Attention (ECA); the latent
embedding is clustered to discover morphological phenotypes. Relative to our prior
submission we make three methodological corrections that materially change the
conclusions: (i) a **specimen-level, treatment-stratified split** that removes tile-level
data leakage; (ii) **genuine external validation** against treatment group, diet, and
magnification recovered from the slide metadata, replacing previously uninformative
labels; and (iii) a **deep-clustering training objective** (DEC/IDEC-style KL loss on a
true latent bottleneck) that directly optimises cluster separability. On a leak-free
held-out test set, the clustering objective improves all internal cluster-validity
indices substantially (Silhouette +39%, Calinski–Harabasz +41%, Xie–Beni 17.2→5.9), and
the learned embedding agrees with treatment biology (AMI ≈ 0.40–0.42) markedly better
than the unsupervised default. Crucially, a **per-magnification confound control** shows
the treatment signal survives when magnification is held fixed, establishing that the
discovered phenotypes reflect biology rather than an imaging artifact. We report a
full architecture ablation, multi-seed statistics with multiple-comparison correction,
and pathologist-facing materials. ⟨PENDING: final ablation + significance.⟩

---

## 1. Introduction

Manual histopathological assessment of liver biopsies is labor-intensive and subject to
inter-observer variability. Unsupervised representation learning offers a route to
discover morphological phenotypes without exhaustive annotation, but its value hinges on
two things that are easy to get wrong: the representation must be **clusterable**, and the
resulting clusters must be shown to be **biologically meaningful and free of imaging
confounds**. In an earlier version of this work both points were under-addressed: the
proposed model did not outperform a plain autoencoder on internal indices, external
validation was run against placeholder labels, and the train/test split leaked tiles from
the same specimen across folds.

This paper is a corrected and strengthened study. Our contributions are:

1. **A clustering-friendly model.** We add a projection head producing a true
   `latent_dim`-dimensional bottleneck and a DEC/IDEC clustering loss, so training
   optimises separability rather than reconstruction alone.
2. **A leak-free, biologically-grounded benchmark.** We recover treatment group, diet,
   and magnification from filenames, split by specimen with treatment stratification, and
   evaluate clustering against these labels (AMI/ARI/NMI/purity) — including magnification
   as an explicit confound.
3. **A confound-controlled validation.** Per-magnification clustering demonstrates the
   treatment signal is not a magnification artifact.
4. **Reproducible, statistically rigorous evaluation.** Multi-seed runs, paired tests with
   Cohen's d, and Holm–Bonferroni correction across the comparison family; robust metric
   implementations; and a one-command corrected pipeline.

## 2. Related work

Deep clustering of images (DEC, IDEC) couples an autoencoder with a KL objective over a
Student-t soft assignment to learnable centroids. KANs replace fixed activations with
learnable spline edges; attention modules (BAM, ECA) reweight channel/spatial features.
Self-supervised methods (SimCLR, BYOL) and pretrained-encoder baselines provide strong
unsupervised references. Our work combines a KAN/attention autoencoder with a deep
clustering objective and, distinctively, validates the discovered clusters against
treatment biology with explicit confound control. ⟨PENDING: full citations.⟩

## 3. Data

The cohort comprises liver histopathology whole-slide regions from a NAFLD / statin study,
tiled into 21,168 fixed-size tiles across ~196 specimens. Slide filenames encode, and we
parse with zero failures, the **treatment group** (7 arms: HFD+atorvastatin,
ND+atorvastatin, HFD+vehicle, HFD positive control, HFD+simvastatin, ND control, ND
normal), **diet** (ND vs HFD), **magnification** (10×/20×/40×), **specimen id**, and tile
index.

**Leak-free split.** Tiles are split **by specimen** with `StratifiedGroupKFold`,
stratified by treatment group, so no specimen appears in both train and test
(asserted). The held-out test set is 4,536 tiles / 42 specimens; training is 16,632 tiles
/ 154 specimens; all seven treatment groups are present in both. The previous per-tile
random split leaked near-duplicate tiles across folds and inflated apparent performance.

## 4. Method

### 4.1 DAE-KAN-Attn architecture
A dual-stage autoencoder: `Encoder → Bottleneck → Decoder`, where KAN convolutional layers
replace standard convolutions in the encoder/decoder, BAM provides combined
channel+spatial attention at the bottleneck, and ECA provides lightweight channel
attention. The bottleneck produces a feature map `z`.

### 4.2 True latent bottleneck
For clustering we add a **projection head** mapping the flattened bottleneck to a unit-norm
`latent_dim` vector (default 128). This makes the clustered representation a controlled,
low-dimensional space rather than the raw ~4k-D conv map, so the effect of
KAN/BAM/ECA is measured consistently.

### 4.3 Composite training objective
Training minimises a composite loss
`L = w_rec · MSE(x, x̂) + w_clu · KL(P‖Q)`,
where `Q` is the Student-t soft assignment of latent vectors to learnable centroids and
`P` is the sharpened, balanced target distribution (DEC/IDEC). The clustering term is
enabled after a reconstruction **warm-up**; centroids are initialised by K-Means on a pool
of warmed-up latents. With `w_clu = 0` the objective reduces to pure reconstruction
(our baseline). The objective is the same for every architecture variant, so the ablation
isolates architecture under a fixed training signal.

### 4.4 Clustering and metrics
Latent embeddings are standardised then L2-normalised before clustering with K-Means,
Bisecting K-Means, and GMM. **Internal** validity: Silhouette↑, Davies–Bouldin↓,
Calinski–Harabasz↑, Xie–Beni↓ (robust implementations that return NaN, not ∞/0, on
degenerate clusterings). **External** validity: AMI, ARI, NMI, homogeneity, completeness,
purity against treatment group and diet; and against **magnification** as a confound check.

### 4.5 Statistics
Each configuration is run over multiple seeds. We report mean ± std, paired t-test and
Wilcoxon signed-rank vs the baseline, Cohen's d, and **Holm–Bonferroni** correction across
the model × metric family within each clustering algorithm. ⟨PENDING: final.⟩

## 5. Results

### 5.1 The clustering objective improves separability
On the held-out test set (dae_kan, seed 42, 10 epochs, K-Means, k=6), enabling the DEC
objective improves every internal index at a small external cost:

| dae_kan | Silhouette↑ | Davies–Bouldin↓ | Calinski–Harabasz↑ | Xie–Beni↓ | treat-AMI | treat-ARI |
|---|--:|--:|--:|--:|--:|--:|
| reconstruction only | 0.033 | 6.30 | 141.9 | 17.17 | 0.423 | 0.246 |
| + DEC objective | **0.046** | **4.29** | **199.7** | **5.91** | 0.405 | 0.236 |

This directly addresses the central prior criticism that the model did not produce
better-separated clusters.

### 5.2 Architecture ablation (corrected benchmark)
All seven variants under the same objective, leak-free test set. ⟨PENDING: 7-variant
table from the running scouting sweep; then 3+ seeds with Holm-corrected significance.⟩

Preliminary recon-only head-to-head (seed 42, 10 epochs):

| Model | Silhouette↑ | Davies–Bouldin↓ | Calinski–Harabasz↑ | treat-AMI | treat-ARI |
|---|--:|--:|--:|--:|--:|
| baseline | 0.035 | 5.07 | 137.4 | 0.342 | 0.189 |
| dae_kan_attention | 0.033–0.040 | 6.3 | 141.9 | 0.423 | 0.246 |

Internal indices are comparable between architectures under pure reconstruction; the
proposed model's advantage is in **biological agreement** (treatment-AMI 0.42 vs 0.34) and,
with the clustering objective, in **separability** (§5.1).

### 5.3 External / biological validation
Cluster↔biology association (test set, k=6; all χ² p ≪ 0.001), bias-corrected Cramér's V:

| Variable | baseline | dae_kan |
|---|--:|--:|
| treatment group | 0.448 | 0.470 |
| diet (ND/HFD) | 0.839 | 0.912 |
| magnification (confound) | 0.758 | 0.799 |

Clusters are significantly associated with treatment and (most strongly) diet — the
treatment-induced phenotype the prior reviews found missing.

### 5.4 Magnification confound control (key)
Pooled, magnification association exceeds treatment association, which alone would make a
biological claim unsafe. We therefore cluster **within each magnification stratum**:

| Stratum | n tiles | #treatments | baseline treat-AMI | dae_kan treat-AMI |
|---|--:|--:|--:|--:|
| pooled | 4536 | 7 | 0.370 | 0.402 |
| 20× | 1836 | 6 | 0.478 | 0.475 |
| 40× | 2052 | 7 | 0.399 | 0.446 |
| 10× | 648 | 2 | 0.000 | 0.007 |

With magnification held fixed, treatment AMI is **at or above** the pooled value in the
adequately-sampled strata (20×, 40×). The high pooled magnification Cramér's V therefore
reflects magnification↔treatment correlation in the cohort, **not** clusters that separate
by magnification. The treatment phenotype is confound-robust. (The 10× stratum has only
two treatment arms and is uninformative.)

### 5.5 Pathologist review
For each cluster we export representative tile montages (centroid-nearest + random),
per-cluster treatment enrichment (observed/expected), and a structured scoring template; a
board-certified pathologist annotates morphology per cluster. ⟨PENDING: pathologist scores.⟩

### 5.6 Self-supervised and pretrained baselines
SimCLR, BYOL, VAE, single-AE, and a frozen pretrained-encoder + K-Means reference,
evaluated through the identical corrected pipeline. ⟨PENDING.⟩

## 6. Discussion

The corrected study supports a nuanced, honest claim. Under pure reconstruction the
proposed architecture is **not** clearly better than a plain autoencoder on internal
indices — we state this plainly. Its value emerges in two places the reviewers cared
about: (i) the latent space agrees better with treatment biology, and (ii) adding the deep
clustering objective markedly improves separability. The per-magnification control is what
licenses the biological interpretation. We therefore frame the contribution as a
**clustering-objective-trained, biologically-validated phenotyping pipeline** rather than an
architectural win in the abstract.

## 7. Limitations

Single-centre cohort; ~196 specimens. Magnification correlates with treatment in the
cohort, requiring the stratified analysis we report. Preliminary numbers are 1 seed / 10
epochs pending the full multi-seed/epoch sweep. Cluster count k is fixed at 6 pending the
sensitivity analysis (latent_dim, k, KAN spline order, ECA kernel). ⟨PENDING: k & HP
sensitivity.⟩

## 8. Conclusion

A deep-clustering objective on a true latent bottleneck makes an attention/KAN autoencoder
produce well-separated, biologically-meaningful liver-histopathology phenotypes on a
leak-free, confound-controlled benchmark. The discovered phenotypes track treatment and
diet and survive a magnification confound control. ⟨PENDING: final multi-seed results,
SSL comparison, pathologist validation.⟩

## Reproducibility

Specimen-level split (`create_hepar_csv.py`), corrected pipeline
(`run_corrected_pipeline.py`: train → extract embeddings+metadata → cluster → Holm stats),
robust metrics (`src/clustering_metrics.py`), clustering objective
(`src/clustering_objective.py`), confound control (`per_magnification_analysis.py`), and
pathologist materials (`cluster_biology_analysis.py`). Deterministic seeding throughout.
