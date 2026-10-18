# JOIV comment matrix — 5-day revision

Source: `revission_req.pdf` (JOIV decisions 2026-07-08, 2026-08-19, 2026-10-14).
Manuscript working copy: `joiv_revision/manuscript/JOIV_revised_v1.docx`.
Evidence repo files are the corrected pipeline (scouting done; multi-seed sweep pending on gpu-kc).

## Editor round 1 (2026-07-08)

| # | Point | Manuscript change | Status |
|---|-------|-------------------|--------|
| E1 | Abstract 230–250 words, with objectives / materials / method / results / implication | Rewrite abstract (current draft ≈251 words, background-heavy) | TODO |
| E2 | ≥30 references, ≥70% from 2020–2026 | Audit + add/replace refs (current exactly 30, several pre-2020) | TODO |
| E3 | IEEE style, Mendeley/Zotero, DOI per reference | Re-export reference list | TODO |

## Reviewer A (2026-08-19)

| # | Point | Manuscript change | Evidence |
|---|-------|-------------------|----------|
| A1 | Clarify datasets, sample sizes, treatment groups, patching, patient-level splitting, preprocessing, augmentation; exclude leakage | Replace §II.A: specimen-level stratified split w/ disjointness assertion; full dataset spec | `create_hepar_csv.py`, `src/metadata.py` |
| A2 | Explain how dual AE differs from conventional AE; full architecture/optimisation/implementation details | Dual-vs-conventional paragraph + architecture table + optimizer/seeds | `src/models/` |
| A3 | Unsupervised task but reports accuracy/precision/recall/F1 without ground truth | Remove Table IV classification block (or define supervised probe) | — |
| A4 | Report k, selection procedure, init, seeds, repeated-run stats, CIs for all clustering metrics | k=6 justification + multi-seed mean±std + Holm tests | `compute_statistical_significance.py` |
| A5 | Internal indices + visuals alone don't prove biology/treatment relevance | External validation table (AMI/ARI vs treatment/diet) + confound control | `run_clustering_analysis.py`, `per_magnification_analysis.py` |
| A6 | Add expert quantitative validation, contemporary-method comparison, external dataset eval | Pathologist montages+template (scores if available); pretrained-encoder baseline minimum; SSL stretch | `cluster_biology_analysis.py`, `baselines/` |
| A7 | DBI 3.633 vs ~0.4 in Table IV; cross-check all tables/figures/text | One frozen number system; retire old numbers | corrected CSVs |
| A8 | Perf numbers ill-defined/inconsistent (ms/image, throughput, memory, s/epoch, s/1000) | One profiling protocol on named hardware; arithmetic check | `profile_runtime.py` |
| A9 | Math notation: ELU malformed, CHI typo, incomplete index/variable definitions | Fix equations 1–5, define all variables | — |
| A10 | Stronger ablations: isolate all components, report variability, control capacity | 7-variant ablation w/ variance + params column | `run_corrected_pipeline.py` |
| A11 | Moderate "biologically coherent / reliable / robust / treatment-related" | Soften unless backed by A5 evidence | — |
| A12 | Language, formatting, figure readability, reference consistency, journal style | Full pass | — |

## Reviewer B (2026-08-19) — maps to A-points

| # | Point | Maps to |
|---|-------|---------|
| B1 | Unsupervised claim vs classification metrics; need internal vs external distinction + ARI/NMI/AMI/Silhouette | A3 + A5 |
| B2 | DBI 3.633 vs 0.398/0.406; 43.75 ms/img (≈22.9/s) vs 4.9 s/1000 (≈204/s) | A7 + A8 |
| B3 | Specimen/slide-level split before patching | A1 |
| B4 | UMAP+samples insufficient; need expert assessment / quantitative morphology | A6 |
| B5 | Systematic KAN/ECA/BAM/PanNuke-TL ablation + modern deep-clustering/SSL baselines + multi-run stats | A10 + A4 + A6 |

## Number-freeze decision (record outcome here when made)

- [ ] Track chosen: multi-seed sweep (gpu-kc) or labelled 1-seed scouting
- [ ] DBI headline value + location consistency verified
- [ ] Throughput arithmetic verified
