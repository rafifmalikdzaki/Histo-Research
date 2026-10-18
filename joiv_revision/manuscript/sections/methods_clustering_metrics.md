# §II.C Clustering strategy and evaluation (replacement text)

## C. Clustering Strategy and Evaluation Metrics

Latent embeddings are standardised then L2-normalised before clustering; raw
high-dimensional autoencoder features otherwise produce degenerate partitions
and meaningless indices. We apply three algorithms with complementary
assumptions: KMeans (hard partitional baseline minimising within-cluster
variance), Bisecting KMeans (hierarchical splitting then partition), and
Gaussian Mixture Models (probabilistic soft assignment, suited to overlapping
and heteroscedastic histopathology distributions). Embeddings use k = 6
⟨selection justification at freeze⟩ with fixed initialisation protocols and
repeated seeds (see §III).

**Internal validity** (no labels required) uses four standard indices, with
robust implementations that return NaN — never ∞ or a misleading 0 — on
degenerate single-cluster solutions, and that handle empty clusters from GMM:

- Xie–Beni (Eq. 2): ratio of total within-cluster variance to N times the
  minimum squared inter-centroid distance; lower is better.
- Davies–Bouldin (Eq. 3): mean over clusters of the worst-case ratio of
  intra-cluster scatter to inter-centroid separation; lower is better.
- Calinski–Harabasz (Eq. 4): ratio of between-cluster to within-cluster
  dispersion, each normalised by its degrees of freedom; higher is better.
- Hartigan (Eq. 5): log-ratio form of the same dispersion comparison; lower
  is better.

**External validity** compares cluster assignments against biology recovered
from filenames: Adjusted Mutual Information (AMI), Adjusted Rand Index (ARI),
Normalised Mutual Information (NMI), homogeneity, completeness, and purity —
each versus treatment group and versus diet. Magnification is evaluated
identically as a confound check, and per-magnification clustering (treatment
agreement with magnification held fixed) is the primary guard against
imaging-artifact phenotypes. Statistical comparison across seeds uses paired
t-tests and Wilcoxon signed-rank tests with Cohen's d, Holm–Bonferroni
corrected across the model×metric family within each clustering method.

> Equation fixes for the docx (A9): Eq. 1 ELU as in §II.B above; Eq. 4 CHI
> numerator/denominator degrees-of-freedom typo corrected; every symbol
> (N, k, n_i, c_i, x_j, u_ij, m) defined at first use.
