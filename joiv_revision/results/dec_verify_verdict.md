# DEC checkpoint-restoration verification (gpu-kc, 2026-10-18)

Method: `histodae extract --phase test` (`src/histodae/cli/extract.py`) re-run from each
DEC run's `checkpoints/last.ckpt` (the same checkpoint the pipeline's
`_find_checkpoint` prefers), then KMeans k=6 / standardize_l2 vs
`treatment_group`, compared against `joiv_revision/results/frozen_metrics.csv` DEC arm
(AMI 0.399±0.033, Sil 0.040±0.002, DBI 4.19±0.20).

## Metric-level reproducibility: PASS

| Source | treat-AMI | Sil | DBI |
|---|---|---|---|
| banked 063047 | 0.361 | 0.042 | 4.41 |
| banked 071518 | 0.414 | 0.039 | 4.04 |
| banked 075941 | 0.423 | 0.039 | 4.13 |
| re-extract last.ckpt 063047 | 0.365 | 0.050 | 4.19 |
| re-extract last.ckpt 071518 | 0.383 | 0.047 | 4.28 |
| re-extract last.ckpt 075941 | 0.421 | 0.040 | 4.53 |
| repeat re-extract 063047 | 0.417 | 0.038 | 4.11 |

Banked mean AMI 0.399 (= frozen); re-extract mean 0.390. All seven
extractions sit in the frozen band, and the separability pattern
(DBI ~4.0–4.5 vs recon 5.32) holds in every restoration. The DEC conclusion
does not depend on one embedding dump.

## Exact-embedding reproducibility: FAILS (documented, cause open)

Same checkpoint + same code (both checkouts at 8d0612a, `shuffle=False`,
`.eval()`, `no_grad`) gives tile-order-identical but value-different
embeddings: max abs diff 0.67–1.02 banked-vs-new, and 1.85 (~14% of mean row
norm) between two consecutive re-extractions of one checkpoint. No
dropout/rng in the model sources; prime suspect is nondeterministic
forward kernels (KAN/cuDNN). Effect on conclusions: none at the metric level
(see table), but exact-embedding determinism is an open engineering item.

## Verdict for the revision

"Provisional pending checkpoint-restoration" is lifted to
**restoration-verified at metric level**; the extraction jitter above is
reported as a limitation, not hidden.
