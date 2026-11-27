#!/bin/bash
# Pick the threshold on VAL (sweep incl. > 0.9), then evaluate the WHOLE test split at that threshold.
# usage (from $PK/work/<you>, inside srun with CUDA_VISIBLE_DEVICES set):
#   bash scripts/thr_eval_run.sh <model> <config> <ckpt> <exp> [val_limit=0 (all)] [extra seg.evaluate args, e.g. --tta]
# Outputs: $PK/runs/<you>/<exp>/{val_sweep.json,test_metrics.json,eval_val.log,eval.log,overlays/}
#          copied to results/peaknet-seg-sprint/{evals,figures}/<you>/<exp>/
set -uo pipefail
MODEL=$1; CFG=$2; CKPT=$3; EXP=$4; VLIM=${5:-0}; shift 4; [ $# -gt 0 ] && shift
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
ME=$(basename "$PWD")
R=$PK/runs/$ME/$EXP
THRS=0.5,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.98,0.99,0.995,0.998,0.999
mkdir -p "$R" "$DUR/evals/$ME/$EXP" "$DUR/figures/$ME/$EXP"
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1
echo "[thr] $(date) host $(hostname) cvd=$CUDA_VISIBLE_DEVICES step=${SLURM_JOB_ID:-}.${SLURM_STEP_ID:-} ckpt=$CKPT"
python -u -m seg.evaluate --model "$MODEL" --config "$CFG" --ckpt "$CKPT" --split val \
  --split-json results/split/result.json --peak-label 1 --workers 8 --thr $THRS --limit "$VLIM" \
  --out "$R/val_sweep.json" "$@" > "$R/eval_val.log" 2>&1
THR=$(python -c "import json;print(json.load(open('$R/val_sweep.json'))['threshold'])") || { echo "[thr] val sweep failed"; tail "$R/eval_val.log"; exit 1; }
echo "[thr] val-best thr $THR at $(date)"
python -u -m seg.evaluate --model "$MODEL" --config "$CFG" --ckpt "$CKPT" --split test \
  --split-json results/split/result.json --peak-label 1 --workers 8 --thr "$THR" \
  --out "$R/test_metrics.json" --save-overlays "$R/overlays" --n-overlays 4 "$@" > "$R/eval.log" 2>&1
echo "[thr] test exit $? at $(date)"
tail -n 30 "$R/eval.log"
cp "$R/val_sweep.json" "$R/eval_val.log" "$R/test_metrics.json" "$R/eval.log" "$DUR/evals/$ME/$EXP/"
cp -r "$R/overlays/." "$DUR/figures/$ME/$EXP/" 2>/dev/null
echo "[thr] done $(date)"
