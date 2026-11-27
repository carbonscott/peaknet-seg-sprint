#!/bin/bash
# lucid_unet round-2 pipeline, INSIDE one srun step (one GPU): [train/fine-tune] -> durable copies ->
# whole-test eval with the threshold picked on VAL (seg.evaluate --val-thr) -> durable copies.
# usage: CFG=configs/lucid_unet_ft.yaml [INIT=/sdf/.../best.pt] [SKIP_TRAIN=1 CKPT=...] \
#        bash scripts/lucid_unet_run2.sh <exp> <time_min> [extra seg.train args]
# env for eval: VAL_THR (default .5,...,.999), VAL_EVERY (default 4), TILE (default 1024), EVAL_EXTRA (e.g. --tta)
set -uo pipefail
EXP=$1; TMIN=$2; shift 2
CFG=${CFG:-configs/lucid_unet.yaml}
PK=/sdf/data/lcls/ds/prj/prjcwang31/scratch/main/peaknet-seg
DUR=/sdf/data/lcls/ds/prj/prjcwang31/results/peaknet-seg-sprint
RUN=$PK/runs/h6/$EXP
mkdir -p $RUN $DUR/checkpoints/h6/$EXP $DUR/logs/h6/$EXP $DUR/evals/h6/$EXP $DUR/figures/h6/$EXP
source $PK/env/venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$PK/hf_home HF_HUB_OFFLINE=1 PYTHONPATH=$PWD PYTHONUNBUFFERED=1
echo "slurm_step ${SLURM_JOB_ID:-?}.${SLURM_STEP_ID:-?} host $(hostname) gpus ${CUDA_VISIBLE_DEVICES:-?} cfg $CFG start $(date -Is)"

if [ "${SKIP_TRAIN:-0}" != 1 ]; then
  INIT_ARGS=""
  [ -n "${INIT:-}" ] && INIT_ARGS="--init-ckpt $INIT"
  python -m seg.train --model lucid_unet --config $CFG --out $RUN --time-min $TMIN $INIT_ARGS "$@" \
    > $RUN/train.log 2>&1
  echo "train exit $? $(date -Is)"
  cp $RUN/train.log $RUN/metrics.jsonl $DUR/logs/h6/$EXP/ 2>/dev/null
  cp $RUN/best.pt $DUR/checkpoints/h6/$EXP/best.pt
  cp $CFG $DUR/logs/h6/$EXP/config.yaml
fi
CK=${CKPT:-$DUR/checkpoints/h6/$EXP/best.pt}
python -m seg.evaluate --model lucid_unet --config $CFG --ckpt $CK --split test \
  --val-thr ${VAL_THR:-0.5,0.7,0.8,0.9,0.95,0.97,0.98,0.99,0.995,0.998,0.999} --val-every ${VAL_EVERY:-4} \
  --tile ${TILE:-1024} --out $DUR/evals/h6/$EXP/test_metrics.json --save-overlays $DUR/figures/h6/$EXP \
  --n-overlays 4 --workers 4 ${EVAL_EXTRA:-} > $DUR/evals/h6/$EXP/eval.log 2>&1
echo "eval exit $? $(date -Is)"
tail -n 30 $DUR/evals/h6/$EXP/eval.log
