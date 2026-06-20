# Execution Plan — AUTONOMOUS RUNBOOK (resubmission experiments)

_Paper: CMBBE MS 252387151.R1, due 2026-08-14. Last updated 2026-06-20._

This is a **self-driving runbook**: the autonomous loop reads the "STATE DETECTION"
rule each tick, finds the first incomplete step whose preconditions are met, and
executes it — no user prompt needed. Decision rules are pre-committed below.
**Hard stop / notify gates** mark the few points that need a human (big compute /
resource commitments). Everything before those runs unattended.

## Pre-committed decisions (do NOT re-ask)
- Scope: **1-seed scouting, all 7 variants, 10 epochs, recon-only** (cluster_weight 0).
- Speed flags on every run: `--epoch-viz dump --analysis-freq 200`, `WANDB_MODE=offline`.
- **One GPU job at a time** (8 GB → OOM otherwise). Never launch training while another trains.
- Always aggregate over a **scoped symlink dir**, never raw `auto_analysis/` (old leaky runs).
- Commit after every step; append a dated line to the Notion log.
- 1-seed = scouting only, never presented as final.

## Timings (measured, GPU to itself)
- ~5 min/epoch training loop (AMP+TF32 on; instrumentation + epoch-viz gated).
- ~1–1.3 h per 10-epoch run incl. test extraction.
- 5-variant scouting ≈ ~5–6 h. 3-seed×7×10ep ≈ ~1 day. (earlier 24-min/2.3-h figures were
  inflated by concurrent CPU load — disregard.)

## Known run dirs (reuse, don't rerun)
- baseline recon, seed42, 10ep: `auto_analysis/baseline_baseline_20260620_022524` (has test sidecar)
- dae_kan recon, seed42, 10ep: `auto_analysis/baseline_dae_kan_attention_20260620_045037` (has test sidecar)
- dae_kan + DEC objective, seed42, 10ep: `auto_analysis/<dec_run>` (in progress)

---

## STATE DETECTION (run each loop tick)
1. If a `pl_training`/`run_corrected_pipeline` process is alive → a job is running; **wait**
   (check progress, reschedule). Do not launch anything.
2. Else find the first step below whose **Done-check** is false and **Precondition** is true; do its **Action**.
3. If all steps through the STOP gate are done → push-notify the gate and end the loop.

---

## STEP 1 — Evaluate DEC-objective run
- **Precondition:** DEC training finished (no live process; `<dec_run>/checkpoints/last.ckpt` exists).
- **Done-check:** `<dec_run>/embeddings/test_metadata.csv` exists AND results recorded.
- **Action:**
  ```bash
  python extract_embeddings_for_clustering.py --model-name dae_kan_attention \
    --dataset HeparUnifiedPNG --phase test \
    --checkpoint <dec_run>/checkpoints/last.ckpt --output-dir <dec_run> --batch-size 32
  ```
  Then compute internal+external (KMeans, k=6, standardize_l2) on `<dec_run>` and on the
  recon dae_kan run; write `results/objective_ablation.csv` (objective ON vs OFF). Commit.
  Update `REBUTTAL_R2.md` R2.1 with the ON-vs-OFF deltas + Notion line.

## STEP 2 — Scouting sweep: 5 missing variants
- **Precondition:** Step 1 done; GPU free.
- **Done-check:** all 5 dirs (`bam_only`,`kan_only`,`no_bam`,`no_kan`,`no_eka` seed42) have `test_embeddings.npy`.
- **Action (single command, sequential, auto-extracts each):**
  ```bash
  WANDB_MODE=offline python run_corrected_pipeline.py \
    --models bam_only kan_only no_bam no_kan no_eka \
    --seeds 42 --epochs 10 --batch-size 8 --dataset hepar --k 6 \
    --epoch-viz dump --analysis-freq 200
  ```
  Launch in background; on first launch, **measure and record the true epoch time** here.
  (Ignore the pipeline's built-in aggregate — Step 3 does a clean scoped one.)

## STEP 3 — Clean scoped aggregate over the 7 recon runs
- **Precondition:** Step 2 done.
- **Done-check:** `results/scout_raw_metrics.csv` exists with 7 models.
- **Action:** symlink the 7 recon run dirs into `/tmp/scout7/<name>_seed42`, then:
  ```bash
  python run_clustering_analysis.py --base-dir /tmp/scout7 --k 6 \
    --normalize standardize_l2 --output results/scout_raw_metrics.csv
  ```
  (1 seed → no Holm stats; report the table.) Commit `results/scout_raw_metrics.csv`.

## STEP 4 — Per-magnification confound control (7 variants)
- **Precondition:** Step 3 done. **Done-check:** `results/per_magnification_<name>.csv` for all 7.
- **Action:** loop `per_magnification_analysis.py` over the 7 run dirs. Commit.

## STEP 5 — Epoch-viz regeneration (figures)
- **Precondition:** Step 3 done. **Done-check:** `epoch_metric_evolution_*.png` in the dae_kan run.
- **Action:** `python regenerate_epoch_viz.py --run-dir <dae_kan run> --phase val` (+ baseline). Commit.

## STEP 6 — Update rebuttal + scouting summary
- **Precondition:** Steps 3–4 done.
- **Action:** fill `REBUTTAL_R2.md` with the 7-variant scouting table + confound results;
  write `results/SCOUTING_SUMMARY.md` ranking variants by treatment-AMI (with baseline as
  reference). Commit + Notion line.

## 🛑 STOP GATE A — before the full multi-seed sweep
- After Step 6, **do not** auto-launch the 3-seed×7×30-epoch sweep (≈1 day GPU) or SSL
  baselines — these are large resource commits and a human decision.
- **Action:** PushNotification with the scouting verdict + recommended contenders, then
  **end the loop**. Wait for the user to choose seeds/epochs/contenders.

## STEP 7 (post-approval only) — full sweep
```bash
python run_corrected_pipeline.py --models <approved> \
  --seeds 42 123 456 --epochs 30 --epoch-viz dump   # or 5 seeds if approved
```
then clean scoped aggregate → `compute_statistical_significance.py` (Holm) → tables.

## STEP 8 (post-approval) — SSL baselines + paper artifacts
`baselines/` SSL × 3 seeds through the same eval path; then
`bash run_analysis_pipeline.sh` for `tables/` + `figures/`; finalize `REBUTTAL_R2.md`.

---

## Loop bookkeeping
- **MANDATORY: update the Notion work-log page after EVERY step and EVERY commit** —
  page `3846b247-440f-8144-ae0f-efc912265426`. One dated line/section per action: what
  ran, key numbers, what's next. Never skip it, even for small steps.
- Heartbeat: while a job runs, reschedule ~1800 s; when idle and a step is ready, act immediately.
- On any step failure: capture the log tail, record the error in this file under the step,
  update Notion with the failure, push-notify, and stop (do not thrash).
- Three idle ticks with nothing actionable and no running job → update Notion + push-notify + end loop.
