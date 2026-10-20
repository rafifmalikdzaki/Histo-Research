#!/usr/bin/env python3
"""
End-to-end corrected experiment pipeline (resubmission).

The repository's older orchestrators (run_multi_seed_experiments.py,
run_ablation_study.py) pass flags the current training script no longer accepts
(--dataset-name, no metadata extraction step), which is part of why the
experiments were never reproducible. This script is the corrected glue:

  for each (model, seed):
      1. train pl_training_with_analysis_and_optimization.py (leak-free Hepar
         split; optional clustering objective via --cluster-weight)
      2. locate the run directory it created under auto_analysis/
      3. extract TEST-split embeddings + biology metadata sidecar
         (extract_embeddings_for_clustering.py) into that run dir
  then once, over all runs:
      4. run_clustering_analysis.py  (normalized, external validation)
      5. compute_statistical_significance.py (5-seed stats + Holm)

This makes the corrected, biology-validated benchmark reproducible with one
command. It is hardware-bound: a full 7x5x30-epoch sweep will not finish on a
laptop GPU overnight — scope it with --models/--seeds/--epochs.

Examples:
    # Architecture ablation, pure reconstruction, all variants, 5 seeds:
    python run_corrected_pipeline.py --models baseline dae_kan_attention bam_only \
        kan_only no_bam no_kan no_eka --seeds 42 123 456 789 1011 --epochs 30

    # Objective ablation: proposed model with vs without clustering loss:
    python run_corrected_pipeline.py --models dae_kan_attention --seeds 42 \
        --epochs 30 --cluster-weight 0.5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AUTO_DIR = Path("auto_analysis")
PYTHON = sys.executable  # use the same interpreter (the venv) for children

# The training script accepts the short dataset alias (hepar/pannuke) and maps it
# internally, but extract_embeddings_for_clustering.py / histodata resolve images
# by the FULL dataset name. Map here so extraction finds data/processed/<Full>/tiles.
DATASET_FULL = {"hepar": "HeparUnifiedPNG", "pannuke": "PANnuke"}


def _source_root() -> Path | None:
    """The checkout's ``src`` directory, when running from a source tree.

    ``None`` when histodae is used as an installed package (no source tree to
    prefer over the installed copy).
    """
    candidate = Path(__file__).resolve().parents[3] / "src"
    return candidate if (candidate / "histodae").is_dir() else None


def _child_env(env: dict | None = None) -> dict:
    """Environment for child processes, with clone-local code taking priority.

    A clone (e.g. the remote JOIV sweep) must run *its own* code even when a
    different copy of ``histodae`` is installed in the shared virtualenv.
    Prepending this checkout's ``src`` to ``PYTHONPATH`` guarantees that.
    """
    child = dict(env if env is not None else os.environ)
    src = _source_root()
    if src is not None:
        existing = child.get("PYTHONPATH")
        child["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)
    return child


def _full_dataset(name: str) -> str:
    return DATASET_FULL.get(name.lower(), name)


def _run(cmd: list[str], log: Path, env: dict | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(cmd)}")
    with open(log, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=_child_env(env))
    return proc.returncode


def _snapshot_runs() -> set:
    return {p.name for p in AUTO_DIR.iterdir()} if AUTO_DIR.exists() else set()


def _newest_new_run(before: set) -> Path | None:
    if not AUTO_DIR.exists():
        return None
    new = [p for p in AUTO_DIR.iterdir() if p.is_dir() and p.name not in before]
    if not new:
        return None
    return max(new, key=lambda p: p.stat().st_mtime)


def _stamp_run_seed(run_dir: Path, model: str, seed: int, dataset: str) -> None:
    """Record the training seed inside the auto_analysis run dir.

    The in-training analyzer names run dirs without the seed and its
    experiment_config.json carries none, so without this stamp the
    clustering/stats stages cannot pair runs by seed (they used to treat
    every row as seed -1, collapsing all pairs). The stats stage reads this
    file via run_clustering_analysis.extract_seed; evaluate_stats also picks
    it up through its configs/experiment.yaml scan.
    """
    from histodae.utils.reproducibility import load_config, save_config
    cfg_path = run_dir / 'configs' / 'experiment.yaml'
    try:
        meta = dict(load_config(cfg_path))
    except Exception:
        meta = {}
    meta['seed'] = seed
    meta.setdefault('model_name', model)
    meta.setdefault('dataset', dataset)
    save_config(meta, cfg_path)
    print(f"✓ stamped seed{seed} into {cfg_path}")


def _find_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = list(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        return None
    # Prefer a 'last'/'best' checkpoint, else the most recent.
    for key in ("last", "best"):
        for c in ckpts:
            if key in c.name.lower():
                return c
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def train_one(model: str, seed: int, epochs: int, batch_size: int, dataset: str,
              cluster_weight: float, latent_dim: int, n_clusters: int,
              warmup: int, logs_dir: Path, env: dict,
              analysis_freq: int = 200, epoch_viz: str = "dump") -> Path | None:
    before = _snapshot_runs()
    cmd = [
        PYTHON, "-m", "histodae.cli.train",
        "--model-name", model, "--dataset", dataset, "--seed", str(seed),
        "--max-epochs", str(epochs), "--batch-size", str(batch_size),
        "--analysis-freq", str(analysis_freq), "--epoch-viz", epoch_viz,
    ]
    if cluster_weight > 0:
        cmd += ["--cluster-weight", str(cluster_weight),
                "--latent-dim", str(latent_dim),
                "--n-clusters", str(n_clusters),
                "--cluster-warmup-epochs", str(warmup)]
    rc = _run(cmd, logs_dir / f"train_{model}_seed{seed}.log", env=env)
    run_dir = _newest_new_run(before)
    if rc != 0:
        print(f"⚠️  training failed (rc={rc}) for {model} seed{seed}; see log")
        return None
    if run_dir is None:
        print(f"⚠️  could not locate run dir for {model} seed{seed}")
        return None
    _stamp_run_seed(run_dir, model, seed, dataset)
    print(f"✓ trained {model} seed{seed} -> {run_dir}")
    return run_dir


def extract_one(model: str, run_dir: Path, dataset: str, batch_size: int,
                logs_dir: Path, env: dict) -> bool:
    ckpt = _find_checkpoint(run_dir)
    if ckpt is None:
        print(f"⚠️  no checkpoint in {run_dir}; skipping extraction")
        return False
    cmd = [
        PYTHON, "-m", "histodae.cli.extract",
        "--model-name", model, "--dataset", _full_dataset(dataset), "--phase", "test",
        "--checkpoint", str(ckpt), "--output-dir", str(run_dir),
        "--batch-size", str(batch_size),
    ]
    rc = _run(cmd, logs_dir / f"extract_{run_dir.name}.log", env=env)
    return rc == 0


def aggregate(base_dir: Path, k: int, logs_dir: Path, env: dict) -> None:
    results = Path("results")
    results.mkdir(exist_ok=True)
    raw = results / "raw_clustering_metrics_corrected.csv"
    rc = _run([
        PYTHON, "-m", "histodae.cli.cluster", "--base-dir", str(base_dir),
        "--k", str(k), "--normalize", "standardize_l2", "--output", str(raw),
    ], logs_dir / "clustering_analysis.log", env=env)
    if rc != 0 or not raw.exists():
        print("⚠️  clustering analysis failed; skipping stats")
        return
    _run([
        PYTHON, "-m", "histodae.cli.stats", "--input", str(raw),
        "--baseline", "baseline",
        "--output", str(results / "clustering_summary_corrected.csv"),
        "--tests-output", str(results / "statistical_tests_corrected.csv"),
        "--latex-output", "tables/table_clustering_corrected.tex",
    ], logs_dir / "stats.log", env=env)
    print(f"✓ aggregate results in {results}/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="+", default=["baseline", "dae_kan_attention"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dataset", default="hepar")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--cluster-weight", type=float, default=0.0)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--n-clusters", type=int, default=5)
    ap.add_argument("--cluster-warmup-epochs", type=int, default=5)
    ap.add_argument("--analysis-freq", type=int, default=200,
                    help="Run in-training auto-analysis every N batches (higher = faster).")
    ap.add_argument("--epoch-viz", default="dump", choices=["render", "dump", "off"],
                    help="Per-epoch embedding viz mode (default dump: save data, skip "
                         "rendering for ~40%% faster epochs; rebuild with `histodae regenerate-viz`).")
    ap.add_argument("--skip-train", action="store_true",
                    help="Only run the aggregate eval over existing auto_analysis runs.")
    ap.add_argument("--offline-wandb", action="store_true", default=True,
                    help="Run W&B offline (default; avoids auth hangs in batch).")
    args = ap.parse_args(argv)

    env = dict(os.environ)
    if args.offline_wandb:
        env["WANDB_MODE"] = "offline"

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path("logs") / f"pipeline_{stamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logs -> {logs_dir}")

    if not args.skip_train:
        total = len(args.models) * len(args.seeds)
        i = 0
        for model in args.models:
            for seed in args.seeds:
                i += 1
                print(f"\n=== [{i}/{total}] {model} seed{seed} ===")
                run_dir = train_one(
                    model, seed, args.epochs, args.batch_size, args.dataset,
                    args.cluster_weight, args.latent_dim, args.n_clusters,
                    args.cluster_warmup_epochs, logs_dir, env,
                    analysis_freq=args.analysis_freq, epoch_viz=args.epoch_viz)
                if run_dir is not None:
                    extract_one(model, run_dir, args.dataset, args.batch_size,
                                logs_dir, env)

    aggregate(AUTO_DIR, args.k, logs_dir, env)
    print("\n✅ corrected pipeline finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
