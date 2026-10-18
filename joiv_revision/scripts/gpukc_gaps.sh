#!/bin/bash
# Gap-fill batch: k-sensitivity, per-mag multi-seed, biology, profiling.
# Runs on gpu-kc inside histodae_joiv/code (recon clone). Logs to gaps.log.
set -u
PROJ=/mnt/home/dzakirm/histodae_joiv
export HF_HOME=/mnt/home/dzakirm/hf_hub_cache
export WANDB_MODE=offline
cd "$PROJ/code" || exit 1
. ../.venv/bin/activate
LOG="$PROJ/gaps.log"
OUT=/tmp/joiv_gaps
mkdir -p "$OUT"/{kmag,permag,bio,profile}
echo "GAPS START $(date)" >>"$LOG" 2>&1

# pick 3 baseline + 3 dae_kan recon run dirs (sorted)
mapfile -t BASE < <(ls -d auto_analysis/baseline_baseline_* | head -n 3)
mapfile -t FULL < <(ls -d auto_analysis/baseline_dae_kan_attention_* | head -n 3)
echo "baseline runs: ${BASE[*]}" >>"$LOG" 2>&1
echo "full runs: ${FULL[*]}" >>"$LOG" 2>&1

# 1) k-sensitivity: scoped dirs per model (3 runs), k = 3..8
for spec in "baseline|${BASE[*]}" "full|${FULL[*]}"; do
  name="${spec%%|*}"; dirs="${spec##*|}"
  sd="/tmp/ksens_$name"; rm -rf "$sd"; mkdir -p "$sd"
  i=0
  for d in $dirs; do
    i=$((i+1))
    mkdir -p "$sd/run$i/embeddings"
    cp "$d/embeddings/test_embeddings.npy" "$d/embeddings/test_metadata.csv" "$sd/run$i/embeddings/"
  done
  for k in 3 4 5 6 7 8; do
    python run_clustering_analysis.py --base-dir "$sd" --k "$k" \
      --normalize standardize_l2 --output "$OUT/kmag/${name}_k${k}.csv" >>"$LOG" 2>&1
  done
done
echo "k-sensitivity done $(date)" >>"$LOG" 2>&1

# 2) per-magnification (3 runs each, baseline + full)
for d in "${BASE[@]}" "${FULL[@]}"; do
  b=$(basename "$d")
  python per_magnification_analysis.py --embeddings "$d/embeddings/test_embeddings.npy" \
    --metadata "$d/embeddings/test_metadata.csv" \
    --output "$OUT/permag/${b}.csv" >>"$LOG" 2>&1
done
echo "per-mag done $(date)" >>"$LOG" 2>&1

# 3) cluster biology (full model, 3 runs, no montages)
for d in "${FULL[@]}"; do
  b=$(basename "$d")
  python cluster_biology_analysis.py --embeddings "$d/embeddings/test_embeddings.npy" \
    --metadata "$d/embeddings/test_metadata.csv" --k 6 --method kmeans \
    --output-dir "$OUT/bio/$b" --no-montages >>"$LOG" 2>&1
done
echo "biology done $(date)" >>"$LOG" 2>&1

# 4) profiling (baseline + full)
CUDA_VISIBLE_DEVICES=0 python profile_runtime.py --model-name baseline \
  --batch-size 8 --output "$OUT/profile/baseline.csv" >>"$LOG" 2>&1
CUDA_VISIBLE_DEVICES=0 python profile_runtime.py --model-name dae_kan_attention \
  --batch-size 8 --output "$OUT/profile/full.csv" >>"$LOG" 2>&1
echo "GAPS DONE $(date)" >>"$LOG" 2>&1
