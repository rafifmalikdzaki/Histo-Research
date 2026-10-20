# Archive (2026-10-18 cleanup)

Retired files moved out of the repo root. Nothing here is imported by the
live pipeline — verified with repo-wide grep before moving. (One correction:
`extract_embeddings_for_clustering.py` turned out to be called by the live
`run_corrected_pipeline.py` orchestrator, so it was restored to `src/` and
wired as the `histodae-extract` console script.)

- `scripts/` — superseded one-off analysis scripts. Five of them turned out
  to be live reviewer evidence (cited in `joiv_revision/response/
  comment_matrix.md` for A5/A6/A8 and in the response letter) and were
  restored to the package as `histodae.cli.{per_magnification,biology,
  interpretability,profile,regenerate_viz}` — see `src/histodae/cli/`.
  What remains here: `cross_domain_eval`, `sensitivity_analysis`,
  `inference_dae_kan` + `inference_config.json`, and the old `baselines/`
  package (cited as the A6 pretrained-encoder comparison).
- `docs/` — superseded/historical guides (see `DOCUMENTATION_INDEX.md`
  "Superseded / historical" and "Guides" sections).
- `legacy-analysis/` — retired `src/analysis/old_modules/` (kept for
  provenance; nothing imports it). Their README, which recorded that pass's
  exploratory numbers, now carries a banner marking it retired and pointing
  at the frozen JOIV results, because two of its claims contradicted them.
- `binaries/` — `checkpoints_with_analysis/` (121MB, Oct 2025),
  `results_handover_20260621.zip`, `PROGRESS_REPORT.*`, `RESULTS_REPORT.*`.
- `deps/` — superseded `requirements.txt` (duplicated `pyproject.toml`) and
  `environment.yml` (stale conda snapshot). Install with
  `uv pip install -e ".[dev]"` instead.
- `joiv-handover/` — handover snapshot (`JOIV_revision_package.zip`,
  next-agent README, Hub `HF_MANIFEST.md`); superseded by `joiv_revision/`.
- `original-documents/` — submission originals (manuscript docx + revision
  request); provenance only, nothing references them.

Live entrypoints are the unified `histodae` CLI (`histodae train|extract|
pipeline|cluster|stats|plots|tables|evaluate|per-magnification|biology|
interpretability|profile|regenerate-viz|data`) and the `histodae-*`
console scripts. The root-level `run_*.py` files are thin shims kept so
that `python run_corrected_pipeline.py` still works from a clone root — the
remote JOIV sweep invokes the pipeline that way. Live docs: `README.md`,
`CLAUDE.md`, `DATA_PATHS.md`, `HANDOVER.md`, `DOCUMENTATION_INDEX.md`.
