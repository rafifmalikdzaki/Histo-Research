# Archive (2026-10-18 cleanup)

Retired files moved out of the repo root. Nothing here is imported by the
live pipeline — verified with repo-wide grep before moving. (One correction:
`extract_embeddings_for_clustering.py` turned out to be called by the live
`run_corrected_pipeline.py` orchestrator, so it was restored to `src/` and
wired as the `histodae-extract` console script.)

- `scripts/` — superseded one-off analysis scripts (`cross_domain_eval`,
  `sensitivity_analysis`, `per_magnification_analysis`,
  `compute_interpretability_metrics`,
  `regenerate_epoch_viz`, `cluster_biology_analysis`,
  `ablation_study_comparison`, `profile_runtime`, `inference_dae_kan` +
  `inference_config.json`, and the old `baselines/` package).
- `docs/` — superseded/historical guides (see `DOCUMENTATION_INDEX.md`
  "Superseded / historical" and "Guides" sections).
- `legacy-analysis/` — retired `src/analysis/old_modules/` (kept for
  provenance; nothing imports it).
- `binaries/` — `checkpoints_with_analysis/` (121MB, Oct 2025),
  `results_handover_20260621.zip`, `PROGRESS_REPORT.*`, `RESULTS_REPORT.*`.
- `deps/` — superseded `requirements.txt` (duplicated `pyproject.toml`) and
  `environment.yml` (stale conda snapshot). Install with
  `uv pip install -e ".[dev]"` instead.
- `joiv-handover/` — handover snapshot (`JOIV_revision_package.zip`,
  next-agent README, Hub `HF_MANIFEST.md`); superseded by `joiv_revision/`.
- `original-documents/` — submission originals (manuscript docx + revision
  request); provenance only, nothing references them.

Live entrypoints stay at root: `run_corrected_pipeline.py`,
`run_clustering_analysis.py`, `compute_statistical_significance.py`,
`generate_clustering_plots.py`, `generate_latex_tables.py`,
`evaluate_stats.py`, `create_hepar_csv.py`, `create_pannuke_csv.py`,
`run_analysis_pipeline.sh`. Live docs: `README.md`, `CLAUDE.md`,
`DATA_PATHS.md`, `HANDOVER.md`, `DOCUMENTATION_INDEX.md`.
