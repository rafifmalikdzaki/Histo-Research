# Documentation index

Entry point for maintained documentation and historical research material.
Live inventory lives in `RESEARCH_MANIFEST.yaml` (`python scripts/researcher.py status`).

## Current (JOIV revision — start here)

| Doc | What |
|---|---|
| `joiv_revision/LATEST_REVISION.md` | Audit: done / should-do / could-do + verified results |
| `joiv_revision/response/response_letter_draft.md` | Point-by-point Editor + A + B response |
| `joiv_revision/response/comment_matrix.md` | Every JOIV point → section → evidence |
| `joiv_revision/manuscript/manuscript_master.md` | Revised manuscript source (→ JOIV_revised_v1.docx) |
| `HANDOVER.md` | Local historical handover; verify commands against the current CLI |
| `DATA_PATHS.md` | Local dataset notes; tiles/ is the source of truth |
| `CLAUDE.md` | Repo guide for AI assistants |

## Pipeline & analysis (maintained)

| Doc | What |
|---|---|
| `README.md` | Project overview + reproduction |
| `src/histodae/analysis/README.md` | Current in-training analyzer and offline CLI commands |
| `pyproject.toml` | Dependencies, console entry points, pytest and mypy configuration |
| `Makefile` | Quality checks: `make check` |

## Superseded / historical (kept for provenance, do not follow)

The following filenames are relative to `archive/docs/`:

`CLUSTERING_ANALYSIS_README.md`, `WORKFLOW.md`, `EXECUTION_PLAN.md`,
`REVISION_SUMMARY.md` (Mar 2026 CMBBE round; references removed scripts),
`REBUTTAL_R2.md` (CMBBE R2 response; JOIV letter supersedes),
`PAPER_DRAFT.md` (CMBBE draft; JOIV master supersedes §prose),
`PROGRESS_REPORT.md`, `PAPER_READY_ANALYSIS_SUMMARY.md`, `FIXED_ANALYSIS_SUMMARY.md`,
`FILES_CREATED.md`, `TRAINING_STATUS.md`.

## Guides (reference, may drift — verify against code)

These filenames are also relative to `archive/docs/`:

`ABLATION_ORGANIZATION_GUIDE.md`, `ABLATION_STUDY_GUIDE.md`, `DATASET_SWITCHING.md`,
`INFERENCE_USAGE.md`, `optimizations_README.md`, `PARALLEL_EXECUTION.md`,
`PARALLEL_EXECUTION_GUIDE.md`, `QUICK_REFERENCE.md`, `QUICK_START.md`,
`TRAINING_ANALYSIS_GUIDE.md`, `WALKTHROUGH.md`,
`WANDB_GUIDE.md`, `WANDB_INTEGRATION_COMPLETE.md`, `WANDB_LOGGING_IMPROVEMENTS.md`,
`WANDB_QUICK_START.md`.

Note: `*.md` is gitignored, but explicitly tracked Markdown files (including
this index, README.md and CLAUDE.md) remain versioned. Historical guides and
local notes may be absent from a fresh checkout; check `git ls-files` rather
than assuming all documents listed here are tracked.

Cleanup 2026-10-18: superseded guides and drift-prone references now live in
`archive/docs/`; superseded scripts in `archive/scripts/`; old checkpoints,
handover zip and report PDFs in `archive/binaries/`. Root keeps only
`README.md`, `CLAUDE.md`, `DATA_PATHS.md`, `HANDOVER.md`, and this index.
