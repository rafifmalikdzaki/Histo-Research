# Table / figure replacement plan (freeze-time checklist)

Source of all numbers: `joiv_revision/results/` (pulled from Hub `sweep/`).
No hand-typed values; regeneration via `run_analysis_pipeline.sh`-equivalent.

| Old artifact | Problem | Replacement |
|---|---|---|
| Abstract DBI 3.633 | irreproducible setting | frozen multi-seed headline (mean±std) |
| TABLE I ablation (DBI/XB only, no variance) | no isolation, no stats | 7-variant × (Sil, DBI, CHI, XB, treat-AMI/ARI, diet-AMI) mean±std + params column |
| TABLE II ms/image + throughput | arithmetic contradiction | one-protocol profiling table (train ms/img, infer ms/img, img/s, s/epoch, s/1000, MB, params) |
| TABLE III params/time variants | unverifiable | same protocol, or merged into TABLE II |
| TABLE IV Acc/Prec/Rec/F1 + DBI 0.398/0.406 | undefined labels; DBI split | external-validation table (AMI/ARI/NMI/purity vs treatment+diet) + per-mag control table; classification block deleted |
| Fig. 6–7 loss curves | fine if regenerated | convergence plot from dumped epoch embeddings |
| Fig. 8 k-sweep metrics | k justification | k-sensitivity (k=3–7) or fixed-k rationale |
| Fig. 9 workflow | ok | keep / refresh |
| Fig. 10 UMAP | qualitative only | keep, labelled as qualitative support + per-cluster enrichment |
| Fig. 11 cluster samples | ok | keep + pathologist montage appendix if scored |
| Fig. 12–15 recon/attention/cluster training viz | old runs | regenerate from frozen runs or drop |

Validation at freeze: scripted grep of every numeric claim across
abstract/body/tables/letter (`joiv_revision/scripts/check_numbers.py`, to write);
abstract word count 230–250; refs ≥30, ≥70% recent, DOI each.
