# Scout20 verdict (20 epochs, seed 42, k=6, standardize_l2, test 4,536 tiles)

Computed 2026-10-18 from `code_scout20/auto_analysis/*/embeddings/test_embeddings.npy` on gpu-kc
(KMeans k=6, n_init=10, random_state=42 — single clustering seed).

| Model (20 epochs, seed 42) | treat-AMI | Silhouette |
|---|---|---|
| baseline | 0.412 | 0.049 |
| kan_only | 0.437 | 0.018 |
| dae_kan_attention | 0.421 | 0.031 |
| no_eka | 0.423 | 0.042 |

Reference (10-epoch frozen, 3-seed mean±SD): baseline treat-AMI 0.420±0.011,
dae_kan 0.433±0.012.

## Verdict

Longer training does **not** separate the full model from baseline on biological
agreement, and the 4-model ranking at 20 epochs reproduces the 10-epoch frozen
pattern (kan_only nominally ahead, all others within noise of baseline).
The architecture story stays weak; the defensible contributions remain
(i) the leak-free/biology-validated pipeline, (ii) DEC-objective separability,
(iii) confound control. No 3-seed 20-epoch confirmation launched: ~14 GPU-hours
for an outcome this scout predicts would not change the conclusion. Report the
scout as a training-length robustness check (n=1 seed) in the revision.
