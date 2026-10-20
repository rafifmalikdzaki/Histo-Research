#!/bin/bash
# JOIV sweep on gpu-kc. Two clones separate recon vs DEC-objective aggregates.
# Recon: 7 variants x 3 seeds x 10 epochs (sharded over 2 GPUs).
# DEC: dae_kan_attention objective-ON x 3 seeds (separate clone).
# Usage: setsid -f ./gpukc_sweep.sh </dev/null >>sweep.log 2>&1  (from histodae_joiv/)
set -u
# Overridable so the clone-preparation logic can be exercised locally.
PROJ="${PROJ:-/mnt/home/dzakirm/histodae_joiv}"
export HF_HOME=/mnt/home/dzakirm/hf_hub_cache
export WANDB_MODE=offline
cd "$PROJ" || exit 1
. .venv/bin/activate
LOG="$PROJ/sweep.log"

# The Hub bundle is the single source of truth for the code this sweep runs.
# It is re-downloaded on every sweep, and an existing clone is reset onto it,
# so publishing a new bundle is enough to roll the sweep forward.
BUNDLE_REPO="pytholove/dae-kan-attn-joiv-rev"
BUNDLE_REL="code/histodae_code.bundle"
BUNDLE="$PROJ/hub_code_tmp/$BUNDLE_REL"

prep_clone() {  # <dir>
  local d="$1"
  rm -rf "$PROJ/hub_code_tmp"
  hf download "$BUNDLE_REPO" --repo-type model "$BUNDLE_REL" \
    --local-dir "$PROJ/hub_code_tmp" >>"$LOG" 2>&1
  [ -f "$BUNDLE" ] || { echo "FATAL: bundle download failed for $d" >>"$LOG" 2>&1; exit 1; }

  if [ ! -d "$d/.git" ]; then
    rm -rf "$d"
    git clone -q "$BUNDLE" "$d" >>"$LOG" 2>&1
  else
    # Roll the existing clone onto the bundle's branch. `reset --hard` leaves
    # untracked files alone, so the run outputs and the data symlinks set up
    # below survive.
    local ref
    ref="$(git bundle list-heads "$BUNDLE" | awk '/refs\/heads\//{print $2; exit}')"
    [ -n "$ref" ] || { echo "FATAL: bundle has no branch for $d" >>"$LOG" 2>&1; exit 1; }
    git -C "$d" fetch -q "$BUNDLE" "$ref:refs/remotes/bundle/main" >>"$LOG" 2>&1
    git -C "$d" reset -q --hard refs/remotes/bundle/main >>"$LOG" 2>&1
    echo "updated $d to $(git -C "$d" rev-parse --short HEAD 2>/dev/null)" >>"$LOG"
  fi

  [ -f "$d/run_corrected_pipeline.py" ] || { echo "FATAL: clone failed for $d" >>"$LOG" 2>&1; exit 1; }
  mkdir -p "$d/data/processed/HeparUnifiedPNG" "$d/data/processed"
  rm -rf "$d/data/processed/PANnuke"
  ln -sfn "$PROJ/tiles_hepar" "$d/data/processed/HeparUnifiedPNG/tiles"
  ln -sfn "$PROJ/tiles_pannuke" "$d/data/processed/PANnuke"
  cp -n "$PROJ"/hub_data/*.csv "$d/data/processed/" 2>/dev/null || true
}

[ -d hub_data ] || hf download pytholove/hepar-pannuke-joiv-rev --repo-type dataset \
  --local-dir hub_data >>"$LOG" 2>&1
# Extract tar shards flat (Hub caps a dir at 10k files; tiles must be flat for histodata)
if [ ! -d tiles_hepar ]; then
  mkdir -p tiles_hepar tiles_pannuke
  for t in hub_data/hepar_*.tar; do tar -xf "$t" -C tiles_hepar; done >>"$LOG" 2>&1
  for t in hub_data/pannuke_*.tar; do tar -xf "$t" -C tiles_pannuke; done >>"$LOG" 2>&1
  echo "tiles: $(ls tiles_hepar | wc -l) hepar, $(ls tiles_pannuke | wc -l) pannuke" >>"$LOG" 2>&1
fi
prep_clone code
prep_clone code_dec
python -c "
import csv
def specs(f):
    with open(f) as fh: return set(r['specimen_id'] for r in csv.DictReader(fh))
tr = specs('code/data/processed/train_hepar.csv'); te = specs('code/data/processed/test_hepar.csv')
assert len(tr & te) == 0, 'LEAK'
print(f'split ok: train {len(tr)} specimens, test {len(te)} specimens', flush=True)" >>"$LOG" 2>&1

run_one() {  # <clone> <gpu> <args...>
  local c="$1" gpu="$2"; shift 2
  ( cd "$c" && export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
    CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=offline \
    python -m histodae.cli.pipeline "$@" --dataset hepar --k 6 \
      --epoch-viz dump --analysis-freq 200 >>"$LOG" 2>&1 )
}

echo "SWEEP START $(date)" >>"$LOG" 2>&1
run_one code 0 --models baseline bam_only kan_only --seeds 42 123 456 --epochs 10 --batch-size 8 &
P1=$!
run_one code 1 --models no_bam no_kan no_eka --seeds 42 123 456 --epochs 10 --batch-size 8 &
P2=$!
wait $P1 $P2
run_one code 0 --models dae_kan_attention --seeds 42 123 456 --epochs 10 --batch-size 8 &
P3=$!
run_one code_dec 1 --models dae_kan_attention --seeds 42 123 456 --epochs 10 --batch-size 8 \
  --cluster-weight 0.5 --latent-dim 128 --n-clusters 5 --cluster-warmup-epochs 5 &
P4=$!
wait $P3 $P4

# Curate: last.ckpt + newest best ckpt + test embeddings/metadata + config per run
stage() {  # <clone> <prefix>
  local c="$1" p="$2" rd
  for rd in "$c"/auto_analysis/*/; do
    [ -d "$rd" ] || continue
    local dest="$PROJ/hf_out/$p/$(basename "$rd")"
    mkdir -p "$dest"
    cp "$rd"/checkpoints/last.ckpt "$dest/" 2>/dev/null || true
    ls -t "$rd"/checkpoints/best-checkpoint-*.ckpt 2>/dev/null | head -n 1 | xargs -r -I{} cp {} "$dest/"
    cp "$rd"/embeddings/test_embeddings.npy "$rd"/embeddings/test_metadata.csv "$dest/" 2>/dev/null || true
    cp "$rd"/experiment_config.json "$rd"/final_summary_report.md "$dest/" 2>/dev/null || true
  done
  mkdir -p "$PROJ/hf_out/$p"
  cp "$c"/results/*.csv "$PROJ/hf_out/$p/" 2>/dev/null || true
  cp "$c"/tables/*.tex "$PROJ/hf_out/$p/" 2>/dev/null || true
}
rm -rf "$PROJ/hf_out"
stage code recon
stage code_dec dec_on
du -sh "$PROJ/hf_out" >>"$LOG" 2>&1
hf upload pytholove/dae-kan-attn-joiv-rev "$PROJ/hf_out" sweep --repo-type model \
  --commit-message "JOIV sweep: curated runs + results" >>"$LOG" 2>&1
echo "SWEEP DONE $(date)" >>"$LOG" 2>&1
