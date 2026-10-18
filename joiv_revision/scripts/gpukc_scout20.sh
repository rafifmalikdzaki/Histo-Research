#!/bin/bash
# 20-epoch seed-42 scout: does longer training separate the full model?
# GPU0: baseline + kan_only | GPU1: dae_kan recon + no_eka (sequential per card)
set -u
PROJ=/mnt/home/dzakirm/histodae_joiv
export HF_HOME=/mnt/home/dzakirm/hf_hub_cache
export WANDB_MODE=offline
cd "$PROJ/code_scout20" 2>/dev/null || { echo "clone missing"; exit 1; }
. ../.venv/bin/activate
LOG="$PROJ/scout20.log"
echo "SCOUT20 START $(date)" >>"$LOG" 2>&1
run_one() { local gpu="$1"; shift; ( CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=offline \
  python run_corrected_pipeline.py "$@" --dataset hepar --k 6 \
    --epoch-viz dump --analysis-freq 200 >>"$LOG" 2>&1 ); }
run_one 0 --models baseline kan_only --seeds 42 --epochs 20 --batch-size 8 &
P1=$!
run_one 1 --models dae_kan_attention no_eka --seeds 42 --epochs 20 --batch-size 8 &
P2=$!
wait $P1 $P2
echo "SCOUT20 DONE $(date)" >>"$LOG" 2>&1
