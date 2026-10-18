# Latest revision document — JOIV (audit + status + results)

**Manuscript:** *Attention-Enhanced Dual Autoencoder Clustering for Liver
Histopathological Images* (JOIV, decision letters 14 Oct 2026; 5-day window).
**Prepared:** 18 Oct 2026. **Package dir:** `joiv_revision/`. All HF repos
**private**; nothing submitted or made public.

---

## 1. What has been done (verified)

### Infrastructure & reproducibility — DONE
- New isolated work dir `joiv_revision/` (manuscript / response / hf / results).
- Private Hub dataset repo `pytholove/hepar-pannuke-joiv-rev` (33 files:
  22 Hepar + 4 PanNuke tar shards + 5 split CSVs + card), model repo
  `pytholove/dae-kan-attn-joiv-rev` (168 files: `sweep/recon`, `sweep/dec_on`,
  `sweep/frozen`, code bundle), and a private Collection linking both
  (see `joiv_revision/hf/HF_MANIFEST.md`).
- gpu-kc (`labkc-comm1`, 2× RTX 3060 12 GB) provisioned: torch 2.8.0+cu128,
  pytorch-lightning 2.6.6, Hub-authenticated, dataset downloaded + extracted
  flat (21,168 Hepar + 7,901 PanNuke tiles), code cloned from the Hub bundle.
- Split integrity **asserted in the sweep**: 16,632 tiles/154 specimens train,
  4,536 tiles/42 specimens test, `|train ∩ test| specimens = 0`.

### Experiments — DONE (gpu-kc, 3 seeds 42/123/456, 10 epochs, k = 6)
- 21/21 reconstruction runs: 7 variants × 3 seeds, each with test embeddings +
  biology metadata sidecars (`test_embeddings.npy`, `test_metadata.csv`).
- DEC-objective arm: 3/3 seeds complete (frozen).
- Numbers frozen: `results/frozen_metrics.csv` (72 rows), `frozen_summary.md`,
  `frozen_stats.csv` (48 paired tests; 0 Holm-significant in full-model
  families, 1/60 in the variant family).
- k-sensitivity sweep complete (12/12: baseline + full model × k = 3–8):
  `results/ksens/`. Plateau at k = 4–6, decline at k = 7–8; k = 6 justified
  as the pre-registered on-plateau choice near the 7-arm treatment structure.
- 20-epoch training-length scout complete (4/4: baseline, kan_only, full,
  no_eka; seed 42): `results/scout20_verdict.md`. Same ranking as 10-epoch —
  longer training does not separate the full model.

### Writing — DONE (in the .docx)
- `manuscript/JOIV_revised_v1.docx` (4,909 words plain) built from
  `manuscript_master.md`: abstract 233 words; retired numbers absent by grep;
  38 refs (27 recent, 71%); ELU/CHI equations fixed; specimen-level split;
  undefined Accuracy/F1 removed; Table II single-protocol profiling;
  §III-F frozen-encoder + biology; DEC-instability limitation stated.
  Now includes the k-sensitivity justification (§II-C) and the 20-epoch
  scout (§III-A).
- Response letter `response/response_letter_draft.md` filled (no placeholders);
  comment matrix, reference audit, tables/figures plan staged.

### Gap results — banked
Frozen ResNet-50 (treat-AMI 0.245 KMeans / 0.354 GMM), Cramér's V
(treatment 0.477 / diet 0.939 / magnification 0.726), single-protocol
profiling (baseline 29.4 ms/img, 34.1 fps, 4.9M vs full 37.3 ms/img,
26.8 fps, 5.25M), multi-seed per-magnification control, DEC 128-d latent
eval (seed-unstable — reported as limitation, not headline).

## 2. Verified results snapshot (recon, 3 seeds, standardize_l2, k = 6)

KMeans, held-out test (4,536 tiles). Mean ± SD across 3 seeds:

| Model | Silhouette↑ | DBI↓ | treat-AMI↑ | treat-ARI↑ | diet-AMI↑ |
|---|---|---|---|---|---|
| baseline | 0.036±0.010 | 5.41±1.22 | 0.420±0.011 | 0.249±0.007 | 0.459±0.017 |
| bam_only | 0.030±0.009 | 5.60±0.85 | 0.419±0.001 | 0.248±0.003 | 0.459±0.002 |
| kan_only | 0.024±0.012 | 5.98±0.45 | **0.439±0.007** | **0.289±0.036** | **0.489±0.014** |
| no_bam | 0.023±0.005 | 5.96±0.23 | 0.426±0.001 | 0.256±0.008 | 0.467±0.000 |
| no_kan | 0.033±0.007 | 5.33±0.19 | 0.422±0.002 | 0.248±0.002 | 0.462±0.004 |
| no_eka | 0.039±0.005 | 4.94±0.35 | 0.416±0.015 | 0.243±0.010 | 0.454±0.020 |
| **dae_kan_attention** | 0.036±0.006 | 5.32±0.72 | 0.433±0.012 | 0.279±0.034 | 0.478±0.022 |

DEC-objective arm (frozen): KMeans Sil 0.040±0.002, DBI 4.19±0.20 (recon
5.32), CHI 215±15 (166), Xie-Beni 5.16±1.01 (11.44), treat-AMI 0.399±0.033
(recon 0.433). Evaluation embeddings are the 4096-d bottleneck z for all
arms; the 128-d head exists only in DEC training.

20-epoch scout (seed 42): baseline 0.412, kan_only 0.437, full 0.421,
no_eka 0.423 — same ranking, no separation from longer training.

Honest reading (drives the manuscript):
- **All variants sit within noise of baseline on biological agreement**
  (0.416–0.439). Only kan_only is nominally ahead. After Holm correction,
  1 of 60 model×metric cells is significant.
- The defensible contribution is (i) the leak-free/biology-validated pipeline,
  (ii) DEC-objective separability, (iii) confound control — not an
  internal-index win. No "architecture >> baseline" claims.

## 3. k-sensitivity (KMeans treat-AMI, mean over 3 staged runs per k)

| Model | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---|---|---|---|---|---|
| baseline | 0.346 | 0.423 | 0.427 | 0.420 | 0.404 | 0.392 |
| dae_kan_attention | 0.383 | 0.473 | 0.455 | 0.433 | 0.408 | 0.396 |

Plateau k = 4–6 both models; k = 6 pre-registered, not tuned (peaks at k = 4–5).

## 4. What is left (must-do before submission)

1. **Hub manifest check.** Confirm `sweep/` bulk checksums vs local
   (`HF_MANIFEST.md` update). Mechanical; bulk already landed (168 files).
2. **DOI resolution.** Verify all 38 refs resolve (conference items need care).
3. **Production figures.** Print-resolution UMAP/montages + JOIV-template
   tracked-changes copy.
4. **User-owned.** Confirm JOIV portal status (5-day window from 14 Oct may
   have lapsed — contact editor first if closed), pathologist scoring,
   final read + submit. Agent stops at package-ready.

## 5. Risks & blockers

| Risk | Status |
|---|---|
| JOIV 5-day window (→ 19 Oct) | **High** — confirm portal status before submission |
| Weak multi-seed separation (1/60 Holm-sig) | Framed honestly; DEC + pipeline contribution |
| No pathologist / SSL suite | Limitation language in manuscript |
| IMI licence for Hub hosting (even private) | Unconfirmed; repos stay private |

## 6. Deliverables index

- `joiv_revision/LATEST_REVISION.md` (+ `.pdf`) — this document.
- `joiv_revision/response/{comment_matrix,response_letter_draft,reference_audit}.md`
- `joiv_revision/manuscript/{manuscript_master.md,manuscript_text.txt,JOIV_revised_v1.docx}`
- `joiv_revision/results/{frozen_metrics.csv,frozen_summary.md,frozen_stats.csv,ksens/,scout20_verdict.md,dae_kan_recon_metrics.csv,dec_metrics_fresh.csv,gpukc_pull/}`
- Hub: `pytholove/hepar-pannuke-joiv-rev` (dataset), `pytholove/dae-kan-attn-joiv-rev` (model), Collection `pytholove/joiv-revision-dae-kan-liver-histopathology-6ad38ddcc996ae6086599c23` — all private.
