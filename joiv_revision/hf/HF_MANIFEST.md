# HF manifest — JOIV revision (PRIVATE, do not make public)

- Collection: `pytholove/joiv-revision-dae-kan-liver-histopathology-6ad38ddcc996ae6086599c23`
  ([open](https://huggingface.co/collections/pytholove/joiv-revision-dae-kan-liver-histopathology-6ad38ddcc996ae6086599c23)) — private.
- Dataset repo (type dataset, private): `pytholove/hepar-pannuke-joiv-rev`
  ([open](https://huggingface.co/datasets/pytholove/hepar-pannuke-joiv-rev))
  — `hepar_tiles/` (21,168 canonical tiles) + `pannuke/` + `csvs/` (specimen-grouped
  train/test/folds; overlap verified = 0) + `README.md` card.
- Model repo (type model, private): `pytholove/dae-kan-attn-joiv-rev`
  ([open](https://huggingface.co/pytholove/dae-kan-attn-joiv-rev))
  — `code/histodae_code.bundle` (git bundle of this repo at sweep commit) +
  `runs/<variant>_seed<n>/` (checkpoints, embeddings, metadata sidecars, configs) +
  `results/` (aggregated CSVs, Holm stats).
- gpu-kc project dir: `/mnt/home/dzakirm/histodae_joiv/` (venv `.venv`, torch 2.8.0+cu128,
  `HF_HOME=/mnt/home/dzakirm/hf_hub_cache`). Data dir on gpu-kc: staged Hub download.

Transfer log (append revisions):
- dataset upload: complete — 33 files (22 hepar tars + 4 pannuke tars + 5 CSVs + README).
  Note: first attempt with 29k flat files hit the Hub 10k-files/dir cap; repo was
  recreated and re-uploaded tar-sharded. Flat per-file layout lives only in
  `joiv_revision/hf/dataset_staging/` (hardlinks, local).
- code bundle: `code/histodae_code.bundle` (8.4 MB) in model repo.
- sweep: launched 2026-10-17 ~22:35 WIB on gpu-kc (`gpukc_sweep.sh`, recon 7×3×10ep
  on 2 GPUs + DEC-ON 1×3×10ep in `code_dec/`); curated artifacts → model repo `sweep/`.
- frozen numbers uploaded 2026-10-18: `sweep/frozen/` (frozen_metrics.csv 72 rows,
  frozen_summary.md, dae_kan + DEC scoped aggregates, summary + Holm tests).
  Key result: dae_kan recon treat-AMI 0.433±0.012 vs baseline 0.420±0.011;
  DEC arm DBI 5.32→4.19, Xie–Beni 11.44→5.16, treat-AMI 0.399±0.033.
  Evaluation embeddings are the 4096-d bottleneck z for all arms.
