#!/bin/bash
# lucid_unet pipeline, run INSIDE one srun step (one GPU): train -> durable copies -> full-test eval
# with overlays -> durable copies.
# usage: bash scripts/lucid_unet_run.sh <exp> <time_min> [extra seg.train args, e.g. --set train.lr=5e-4]
# env: CUDA_VISIBLE_DEVICES set by the caller.  Run from $PK/work/h6.
set -uo pipefail
EXP=$1; TMIN=$2; shift 2
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
RUN=$PK/runs/h6/$EXP
mkdir -p $RUN $DUR/checkpoints/h6/$EXP $DUR/logs/h6/$EXP $DUR/evals/h6/$EXP $DUR/figures/h6/$EXP
source $PK/env/venv/bin/activate
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
echo "slurm_step ${SLURM_JOB_ID:-?}.${SLURM_STEP_ID:-?} host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} start $(date -Is)"

if [ "${SKIP_TRAIN:-0}" != 1 ]; then
  python -m seg.train --model lucid_unet --config configs/lucid_unet.yaml --out $RUN --time-min $TMIN "$@" \
    > $RUN/train.log 2>&1
  echo "train exit $? $(date -Is)"
  cp $RUN/train.log $RUN/metrics.jsonl $DUR/logs/h6/$EXP/ 2>/dev/null
  cp $RUN/best.pt $DUR/checkpoints/h6/$EXP/best.pt
  cp configs/lucid_unet.yaml $DUR/logs/h6/$EXP/config.yaml
fi

CK=$DUR/checkpoints/h6/$EXP/best.pt
EVAL_ARGS=${EVAL_ARGS:-}
python -m seg.evaluate --model lucid_unet --config configs/lucid_unet.yaml --ckpt $CK --split test \
  --out $DUR/evals/h6/$EXP/test_metrics.json --save-overlays $DUR/figures/h6/$EXP --n-overlays 4 \
  --workers 6 $EVAL_ARGS > $DUR/evals/h6/$EXP/eval.log 2>&1
echo "eval exit $? $(date -Is)"
tail -n 30 $DUR/evals/h6/$EXP/eval.log
