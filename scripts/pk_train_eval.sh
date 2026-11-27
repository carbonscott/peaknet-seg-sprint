#!/bin/bash
# One Slurm step: train, full-test eval (+ overlays), copy best/final outputs to durable storage.
# usage (inside srun, from $PK/work/<you>): bash scripts/pk_train_eval.sh <model> <config> <exp> <time_min> [seg.train args...]
set -u
MODEL=$1; CFG=$2; EXP=$3; TMIN=$4; shift 4
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
ME=$(basename "$PWD")
R=$PK/runs/$ME/$EXP
mkdir -p "$R"
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1
echo "[$(date +%T)] host $(hostname) cvd=$CUDA_VISIBLE_DEVICES step=${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-}"
python -m seg.train --model "$MODEL" --config "$CFG" --out "$R" --time-min "$TMIN" "$@" > "$R/train.log" 2>&1
echo "[$(date +%T)] train exit $?"
python -m seg.evaluate --model "$MODEL" --config "$CFG" --ckpt "$R/best.pt" --split test --workers 8 \
    --save-overlays "$R/overlays" --n-overlays 4 --out "$R/test_metrics.json" > "$R/eval.log" 2>&1
echo "[$(date +%T)] eval exit $?"
for d in checkpoints logs evals figures; do mkdir -p "$DUR/$d/$ME/$EXP"; done
cp "$R/best.pt" "$DUR/checkpoints/$ME/$EXP/"
cp "$R/train.log" "$R/metrics.jsonl" "$DUR/logs/$ME/$EXP/"
cp "$R/eval.log" "$R/test_metrics.json" "$DUR/evals/$ME/$EXP/"
cp -r "$R/overlays/." "$DUR/figures/$ME/$EXP/" 2>/dev/null
echo "[$(date +%T)] copied to $DUR/*/$ME/$EXP"
