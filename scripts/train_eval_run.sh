#!/bin/bash
# One job step: train with seg.train, then evaluate best.pt on the WHOLE test split (one process start each),
# then copy the run's important outputs to durable storage.
# usage (from $PK/work/<you>, inside srun with CUDA_VISIBLE_DEVICES set):
#   bash scripts/train_eval_run.sh <model> <config> <exp> <time_min> [extra seg.train args...]
set -uo pipefail
MODEL=$1; CFG=$2; EXP=$3; TMIN=$4; shift 4
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
ME=$(basename "$PWD")
R=$PK/runs/$ME/$EXP
SPLIT=results/split/result.json
mkdir -p "$R" "$DUR/checkpoints/$ME/$EXP" "$DUR/logs/$ME/$EXP" "$DUR/evals/$ME/$EXP" "$DUR/figures/$ME/$EXP"
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1
echo "[run] $(date) host $(hostname) cvd=$CUDA_VISIBLE_DEVICES step=${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-} model=$MODEL cfg=$CFG"
python -u -m seg.train --model "$MODEL" --config "$CFG" --out "$R" --time-min "$TMIN" "$@"
echo "[run] train exit $? at $(date)"
[ -f "$R/best.pt" ] || { echo "[run] no best.pt: abort"; exit 1; }
cp "$R/best.pt" "$DUR/checkpoints/$ME/$EXP/best.pt"
cp "$R/metrics.jsonl" "$DUR/logs/$ME/$EXP/metrics.jsonl"
cp "$R/train.log" "$DUR/logs/$ME/$EXP/train.log" 2>/dev/null
echo "[run] eval start $(date)"
python -u -m seg.evaluate --model "$MODEL" --config "$CFG" --ckpt "$R/best.pt" --split test \
  --split-json "$SPLIT" --peak-label 1 --workers 8 --out "$R/test_metrics.json" \
  --save-overlays "$R/overlays" --n-overlays 4 2>&1 | tee "$R/eval.log"
echo "[run] eval exit ${PIPESTATUS[0]} at $(date)"
cp "$R/test_metrics.json" "$R/eval.log" "$DUR/evals/$ME/$EXP/"
cp -r "$R/overlays/." "$DUR/figures/$ME/$EXP/" 2>/dev/null
cp "$R/train.log" "$DUR/logs/$ME/$EXP/train.log" 2>/dev/null
echo "[run] done $(date)"
